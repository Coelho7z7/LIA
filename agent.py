"""
O "cérebro" do assistente: junta modelo + tokenizador + ferramentas.

Ciclo de uma resposta:
  1. A fala do usuário entra no histórico.
  2. O modelo gera a fala do assistente. O PRIMEIRO token decide o caminho:
     - <|call|>  -> é uma chamada de ferramenta ("abrir_app: spotify")
     - outro     -> é texto normal, transmitido token a token
  3. Se for chamada: valida, (pede confirmação), executa, coloca o resultado no
     histórico como <|result|> e volta ao passo 2 para o modelo comentar.

O KV cache é mantido entre as falas: cada turno novo só processa os tokens novos,
em vez de reler a conversa inteira.
"""

import codecs
from pathlib import Path

import torch

import tools
from model import GPT, GPTConfig, apply_repetition_penalty, sample_next
from tokenizer import BPETokenizer

FALLBACK_TEXT = "Hmm, me enrolei aqui. Fala de outro jeito?"

# Ferramentas cujo resultado é um VALOR que a resposta tem que repetir certinho.
# Se o modelo errar um dígito, a resposta é trocada por esta frase segura.
FAITHFUL = {"calcular": "Dá {v}.", "hora": "São {v}.", "data": "Hoje é {v}.", "clima": "Olha só: {v}.",
            "recordar": "Pelo que você me contou: {v}.", "ver_notas": "Suas últimas notas: {v}."}


class Assistant:
    def __init__(self, ckpt_path="out/chat.pt", device=None, temperature=0.7, top_k=40, top_p=0.9,
                 repetition_penalty=1.1, max_reply_tokens=160, max_tool_steps=4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"{ckpt_path} não existe. Treine primeiro: python train.py --stage sft")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.tok = BPETokenizer.from_dict(ckpt["tokenizer"])
        self.model = GPT(GPTConfig(**ckpt["config"])).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.info = {"params": self.model.num_params(), "iter": ckpt["iter"], "val_loss": ckpt["best_val"],
                     "stage": ckpt.get("stage", "?"), "block_size": self.model.cfg.block_size}
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.repetition_penalty = repetition_penalty
        self.max_reply_tokens, self.max_tool_steps = max_reply_tokens, max_tool_steps
        t = self.tok
        # Tokens que nunca fazem sentido no meio de uma fala do assistente.
        self.banned_in_text = [t.bos, t.user, t.result, t.call, t.assistant]
        self.reset()

    # ------------------------------------------------------------ histórico
    def reset(self):
        self.turns = []  # [(papel, texto, ids_do_turno)]
        self.cache = self.model.new_cache()
        self.cache_ids = []
        self.pending = None  # chamada esperando confirmação: (nome, argumento)

    def add_user(self, text):
        ids, _ = self.tok.encode_turn("user", text)
        self.turns.append(("user", text, ids))

    def _context(self):
        return [self.tok.bos] + [i for _, _, ids in self.turns for i in ids]

    def _prefill(self, prompt):
        """Garante que o cache contém `prompt` e devolve os logits do último token."""
        limit = self.model.cfg.block_size - self.max_reply_tokens
        if len(prompt) > limit:
            # Conversa longa: esquece os turnos mais antigos (mantendo a estrutura).
            while len(self._context()) + 1 > limit and len(self.turns) > 1:
                self.turns.pop(0)
            prompt = self._context() + [self.tok.assistant]
            if len(prompt) > limit:  # uma única mensagem gigante: corta o começo
                prompt = [self.tok.bos] + prompt[-(limit - 1):]
            self.cache.reset()
            self.cache_ids = []
        n = len(self.cache_ids)
        if prompt[:n] != self.cache_ids or n == len(prompt):
            self.cache.reset()
            self.cache_ids, n = [], 0
        new = prompt[n:]
        logits, _ = self.model(torch.tensor([new], device=self.device), cache=self.cache)
        self.cache_ids = list(prompt)
        return logits[0, -1]

    def _feed(self, token):
        logits, _ = self.model(torch.tensor([[token]], device=self.device), cache=self.cache)
        self.cache_ids.append(token)
        return logits[0, -1]

    # -------------------------------------------------------------- geração
    @torch.no_grad()
    def step(self):
        """
        Gera UMA fala do assistente. É um gerador de eventos:
          {"type": "token", "text": ...}                      pedaço de texto (streaming)
          {"type": "text", "text": ...}                       fala de texto terminada
          {"type": "call", "tool", "arg", "confirm": bool}    chamada de ferramenta pronta
        """
        tok = self.tok
        prompt = self._context() + [tok.assistant]
        logits = self._prefill(prompt)

        # Decisão texto x ferramenta: se <|call|> é o token mais provável, é ferramenta.
        if int(logits.argmax()) == tok.call:
            gen = [tok.call]
            logits = self._feed(tok.call)
            for _ in range(48):  # chamadas são curtas e saem em modo determinístico
                logits[self.banned_in_text] = float("-inf")
                t = int(logits.argmax())
                if t == tok.end:
                    break
                gen.append(t)
                logits = self._feed(t)
            text = tok.decode(gen[1:]).strip()
            parsed = tools.parse_call(text)
            if parsed:
                name, arg = parsed
                self.turns.append(("call", text, [tok.assistant] + gen + [tok.end]))
                self.pending = (name, arg)
                yield {"type": "call", "tool": name, "arg": arg, "confirm": tools.TOOLS[name].confirm}
                return
            # Chamada inválida (ferramenta que não existe): responde algo seguro.
            ids, _ = tok.encode_turn("assistant", FALLBACK_TEXT)
            self.turns.append(("assistant", FALLBACK_TEXT, ids))
            self.cache.reset()
            self.cache_ids = []
            yield {"type": "token", "text": FALLBACK_TEXT}
            yield {"type": "text", "text": FALLBACK_TEXT}
            return

        # Logo depois de uma ferramenta de valor, a resposta é conferida antes de
        # aparecer (é curta, então não transmitir token a token não pesa).
        must = None
        if len(self.turns) >= 2 and self.turns[-1][0] == "result" and self.turns[-2][0] == "call":
            name = self.turns[-2][1].partition(":")[0].strip()
            res = self.turns[-1][1]
            if name in FAITHFUL and res.startswith("ok: ") and "nenhuma nota" not in res:
                must = (name, res[4:])

        gen = []
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for _ in range(self.max_reply_tokens):
            logits[self.banned_in_text] = float("-inf")
            if not gen:
                logits[tok.end] = float("-inf")  # nada de resposta vazia
            # Penaliza só o que ela mesma acabou de dizer (evita "ok ok ok ok"),
            # não o contexto: copiar o resultado da ferramenta é desejado.
            logits = apply_repetition_penalty(logits, gen[-48:], self.repetition_penalty)
            t = sample_next(logits, self.temperature, self.top_k, self.top_p)
            if t == tok.end:
                break
            gen.append(t)
            piece = decoder.decode(tok.vocab[t])
            if piece and not must:
                yield {"type": "token", "text": piece}
            logits = self._feed(t)
        tail = decoder.decode(b"", final=True)
        if tail and not must:
            yield {"type": "token", "text": tail}
        text = tok.decode(gen).strip()
        ids = [tok.assistant] + gen + [tok.end]
        if must:
            name, value = must
            if tools.norm(value) not in tools.norm(text):
                text = FAITHFUL[name].format(v=value)
                ids, _ = tok.encode_turn("assistant", text)
                self.cache.reset()  # o cache tem a resposta errada; refaz na próxima fala
                self.cache_ids = []
            yield {"type": "token", "text": text}
        self.turns.append(("assistant", text, ids))
        yield {"type": "text", "text": text}

    def resolve(self, approved=True):
        """Executa (ou cancela) a chamada pendente e coloca o resultado no histórico."""
        name, arg = self.pending
        self.pending = None
        result = tools.run(name, arg) if approved else "cancelado: você não autorizou"
        ids, _ = self.tok.encode_turn("result", result)
        self.turns.append(("result", result, ids))
        return result

    def chat(self, text, confirm=lambda name, arg: True, auto=False):
        """
        Laço completo de uma mensagem do usuário, já executando ferramentas.
        `confirm(nome, arg) -> bool` é chamado para ferramentas sensíveis (a menos que auto=True).
        Gera os mesmos eventos de step() mais {"type": "result", "tool", "text"}.
        """
        self.add_user(text)
        for _ in range(self.max_tool_steps):
            call = None
            for ev in self.step():
                yield ev
                if ev["type"] == "call":
                    call = ev
            if call is None:
                return
            approved = auto or not call["confirm"] or confirm(call["tool"], call["arg"])
            result = self.resolve(approved)
            yield {"type": "result", "tool": call["tool"], "text": result}
        # Muitas ferramentas seguidas: encerra com uma frase em vez de entrar em loop.
        msg = "Fiz o que deu por aqui. Quer mais alguma coisa?"
        ids, _ = self.tok.encode_turn("assistant", msg)
        self.turns.append(("assistant", msg, ids))
        yield {"type": "token", "text": msg}
        yield {"type": "text", "text": msg}
