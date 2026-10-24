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

Com o conselho ligado (conselho.py), a primeira fala depois de uma mensagem do usuário
é decidida por vários papéis em vez de sair direto do modelo (com a mesma trava de
confiança e o mesmo porteiro).
"""

import codecs
import copy
import random
import re
from pathlib import Path

import torch

import tools
from conselho import Conselho
from revisao import MOTIVO, SEGURA, revisar_fala
from copia import Copia, carregar_fixas
from model import GPT, GPTConfig, apply_repetition_penalty, sample_next
from tokenizer import BPETokenizer

FALLBACK_TEXT = "Hmm, me enrolei aqui. Fala de outro jeito?"
# Trava de confiança: em resposta a uma fala do usuário, só chama ferramenta se p(<|call|>) >= TRAVA. Chamadas
# certas ficam em ~1,00; as à toa (conversa virando ferramenta) ficam espalhadas abaixo (ver testes/trava.py).
TRAVA = 0.9
# Porteiro (porteiro.py): modelo pequeno e separado que decide ferramenta x conversa. Seguro (>= SIM ou <= NAO), ele
# manda; em dúvida, vale a decisão do modelo principal com a trava.
PORTEIRO_SIM, PORTEIRO_NAO = 0.8, 0.2

# Mensagem delicada (saúde mental). Um modelo pequeno sorteando palavras pode responder "Que bom saber!" a quem diz
# que está com depressão; então: risco à vida -> sempre a resposta de apoio com o CVV; mensagem delicada -> se a
# resposta sair comemorando, troca pela de apoio. (Mesma regra em webapp/engine/lia.js.)
APOIO = ("Sinto muito que você esteja passando por isso. Você não precisa enfrentar sozinho: conversar com alguém de "
         "confiança ou com um profissional ajuda muito. Se quiser falar com alguém agora, o CVV atende 24 horas pelo 188, "
         "de graça.")
RISCO = re.compile(r"\b(me matar|suicid\w*|tirar (a )?minha (propria )?vida|vontade de morrer|quero morrer|me machucar|"
                   r"nao quero mais viver|acabar com a minha vida)")
DELICADO = re.compile(r"\b(depress\w*|(com|muita|minha) ansiedade|crise de (choro|panico|ansiedade)|nao aguento mais|sem vontade de viver|"
                      r"me sinto um lixo|muito pra baixo|sindrome do panico)")
ALEGRE = ("que bom", "parabens", "que demais", "merecido", "comemora", "que otimo", "aproveita", "adoro ouvir")


# Triagem: onde um pedido composto pode ser quebrado
DIVISOR = re.compile(r"\s*(?:,?\s*e depois\s+|,?\s*depois\s+|;\s*|,?\s+e\s+(?=\S))", re.IGNORECASE)
FEITO = {"youtube": "busquei {a} no YouTube", "abrir_app": "abri o {a}", "abrir_site": "abri o site {a}",
         "pesquisar": "pesquisei {a}", "lembrete": "criei o lembrete", "anotar": "anotei {a}", "clima": "vi o clima",
         "hora": "vi a hora", "data": "vi a data", "calcular": "fiz a conta", "bloquear": "bloqueei a tela",
         "abrir_pasta": "abri a pasta {a}", "consultar": "consultei {a} na Wikipédia", "lembrar": "guardei na memória",
         "recordar": "consultei a memória", "ver_notas": "li suas notas", "midia": "mexi na música"}
VOLUME = {"mais": "aumentei o volume", "menos": "abaixei o volume", "mudo": "mexi no mudo"}


def resumo_feitos(feitos):
    """Entrega única de um pedido composto: "Pronto: busquei forró no YouTube e abaixei o volume."."""
    ok = [VOLUME.get(a, "mexi no volume") if n == "volume" else FEITO.get(n, n).format(a=a)
          for n, a, r in feitos if r.startswith("ok")]
    falhou = [FEITO.get(n, n).format(a=a) for n, a, r in feitos if not r.startswith("ok")]
    junta = lambda xs: ", ".join(xs[:-1]) + " e " + xs[-1] if len(xs) > 1 else xs[0]  # noqa: E731
    if not ok:
        return "Não consegui fazer nenhuma das partes, foi mal."
    txt = "Pronto: " + junta(ok) + "."
    if falhou:
        txt += " Só não deu: " + junta(falhou) + "."
    return txt


def apoio_se_preciso(fala, resposta):
    """A resposta segura quando a fala é delicada e a resposta não serve (ou None: a resposta fica)."""
    f, r = tools.norm(fala), tools.norm(resposta)
    if RISCO.search(f) and "188" not in resposta:
        return APOIO
    if DELICADO.search(f) and any(a in r for a in ALEGRE):
        return APOIO
    return None

# Ferramentas cujo resultado é um VALOR que a resposta tem que repetir certinho.
# Se o modelo errar um dígito, a resposta é trocada por esta frase segura.
FAITHFUL = {"calcular": "Dá {v}.", "hora": "São {v}.", "data": "Hoje é {v}.", "clima": "Olha só: {v}.",
            "recordar": "Pelo que você me contou: {v}.", "ver_notas": "Suas últimas notas: {v}.",
            "consultar": "Segundo a Wikipédia: {v}"}


class Assistant:
    def __init__(self, ckpt_path="out/chat.pt", device=None, temperature=0.7, top_k=40, top_p=0.9,
                 repetition_penalty=1.1, max_reply_tokens=160, max_tool_steps=4, porteiro=None, conselho=False):
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
        self.porteiro = None
        if porteiro:
            from porteiro import Porteiro
            self.porteiro = Porteiro(porteiro, self.device)
        t = self.tok
        # Tokens que nunca fazem sentido no meio de uma fala do assistente.
        self.banned_in_text = [t.bos, t.user, t.result, t.call, t.assistant]
        # Cópia forçada nas chamadas (copia.py): vocabulário fixo gravado ao lado do modelo.
        self.fixas = carregar_fixas(ckpt_path)
        self.info["copia"] = self.fixas is not None
        self.usar_conselho = conselho
        self.ferramentas = None  # None = pode tudo; um agente (agentes.py) só pode as dele
        self.conselho = Conselho(self.model, t, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                 max_reply_tokens=max_reply_tokens, fixas=self.fixas)
        self.reset()

    # ------------------------------------------------------------ histórico
    def reset(self):
        self.turns = []  # [(papel, texto, ids_do_turno)]
        self.cache = self.model.new_cache()
        self.cache_ids = []
        self.pending = None  # chamada esperando confirmação: (nome, argumento)
        self.fila, self.feitos = [], []  # pedido composto: partes que faltam e o que já foi feito (triagem)

    def add_user(self, text):
        ids, _ = self.tok.encode_turn("user", text)
        self.turns.append(("user", text, ids))

    def _context(self):
        return [self.tok.bos] + [i for _, _, ids in self.turns for i in ids]

    def _prefill(self, prompt):
        """Garante que o cache contém `prompt` e devolve os logits do último token."""
        limit = self.model.cfg.block_size - self.max_reply_tokens
        if len(prompt) > limit:
            # Conversa longa: esquece os turnos mais antigos de uma vez, até sobrar ~60% do espaço (mantendo a
            # estrutura). Esquecer o mínimo obrigava a reler a conversa inteira em toda fala.
            keep = int(limit * 0.6)
            while len(self._context()) + 1 > keep and len(self.turns) > 1:
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
          {"type": "conselho", "ev": ...}                      só com o conselho: o debate ao vivo (conselho.py)
          {"type": "debate", "propostas", "veredito"}          só com o conselho: resumo, antes da fala
        """
        tok = self.tok
        prompt = self._context() + [tok.assistant]
        logits = self._prefill(prompt)
        depois_do_usuario = bool(self.turns) and self.turns[-1][0] == "user"
        p = self.porteiro.p_ferramenta(prompt) if self.porteiro and depois_do_usuario else None

        if self.usar_conselho and depois_do_usuario:
            # O conselho só lê o cache (já com o prompt); a fala escolhida entra no histórico e o próximo
            # _prefill processa só ela. O porteiro, quando seguro, manda; senão vale a trava de confiança.
            forcar = "call" if p is not None and p >= PORTEIRO_SIM else "text" if p is not None and p <= PORTEIRO_NAO else None
            self.conselho.temperature = self.temperature
            win, props, veredito = yield from self.conselho.rodada(self.cache, logits, self.turns[-1][1],
                                                                   trava=TRAVA, forcar=forcar)
            yield {"type": "debate", "propostas": [x.resumo() for x in props], "veredito": veredito}
            if win.kind == "call":
                yield from self._call(win.ids)
                return
            text, ids = win.text, [tok.assistant] + win.ids + [tok.end]
            problema = revisar_fala(self.turns[-1][1], text)
            if problema:  # a revisora da conversa barra: vale a próxima proposta de texto que passar, ou a segura
                yield {"type": "revisao", "motivo": MOTIVO[problema], "antes": text}
                outra = next((x for x in props if x.kind == "text" and not revisar_fala(self.turns[-1][1], x.text)), None)
                text = outra.text if outra else random.choice(SEGURA[problema])
                ids = [tok.assistant] + outra.ids + [tok.end] if outra else tok.encode_turn("assistant", text)[0]
            seguro = apoio_se_preciso(self.turns[-1][1], text)
            if seguro:
                text, ids = seguro, tok.encode_turn("assistant", seguro)[0]
            self.turns.append(("assistant", text, ids))
            yield {"type": "token", "text": text}
            yield {"type": "text", "text": text}
            return

        # Decisão texto x ferramenta: se <|call|> é o token mais provável, é ferramenta, desde que, logo depois de
        # uma fala do usuário, a confiança passe da trava (na dúvida, conversa).
        quer = int(logits.argmax()) == tok.call
        if depois_do_usuario:
            if p is not None and p >= PORTEIRO_SIM:
                quer = True
            elif p is not None and p <= PORTEIRO_NAO:
                quer = False
            elif quer:
                quer = torch.softmax(logits.float(), -1)[tok.call].item() >= TRAVA
        if quer:
            gen = [tok.call]
            logits = self._feed(tok.call)
            # Palavras do argumento só da conversa (ou do vocabulário fixo da ferramenta).
            copia = Copia(self.fixas, " ".join(text for _, text, _ in self.turns)) if self.fixas else None
            raw = b""
            for _ in range(48):  # chamadas são curtas e saem em modo determinístico
                logits[self.banned_in_text] = float("-inf")
                t = copia.escolhe(logits, raw, tok) if copia else int(logits.argmax())
                if t is None or t == tok.end:
                    break
                gen.append(t)
                raw += tok.vocab[t]
                logits = self._feed(t)
            yield from self._call(gen)
            return

        # Logo depois de uma ferramenta de valor, a resposta é conferida antes de
        # aparecer (é curta, então não transmitir token a token não pesa).
        must = None
        if len(self.turns) >= 2 and self.turns[-1][0] == "result" and self.turns[-2][0] == "call":
            name = self.turns[-2][1].partition(":")[0].strip()
            res = self.turns[-1][1]
            if name in FAITHFUL and res.startswith("ok: ") and "nenhuma nota" not in res:
                must = (name, res[4:])

        # Respostas recentes dela: se sair igualzinha a uma delas, gera de novo (até 2 vezes).
        # O evento "text" do fim troca o que já apareceu na tela.
        # Resumo da Wikipédia é longo: aparece enquanto é gerado (se a cópia sair errada, o "text" do fim troca).
        mostra = not must or must[0] == "consultar"
        recentes = {tools.norm(t) for r, t, _ in self.turns[-6:] if r == "assistant"}
        start, last = len(self.cache_ids), self.cache_ids[-1]  # o prompt; para refazer, o cache volta até aqui
        proibidas = []  # 1ª palavra das tentativas que repetiram: a próxima começa diferente
        for tentativa in range(3):
            gen = []
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            temperature = self.temperature if not tentativa else max(self.temperature, 0.8) + 0.2 * tentativa
            for _ in range(self.max_reply_tokens):
                logits[self.banned_in_text] = float("-inf")
                if not gen:
                    logits[tok.end] = float("-inf")  # nada de resposta vazia
                    logits[proibidas] = float("-inf")
                # Penaliza só o que ela mesma acabou de dizer (evita "ok ok ok ok"),
                # não o contexto: copiar o resultado da ferramenta é desejado.
                logits = apply_repetition_penalty(logits, gen[-48:], self.repetition_penalty)
                t = sample_next(logits, temperature, self.top_k, self.top_p)
                if t == tok.end:
                    break
                gen.append(t)
                piece = decoder.decode(tok.vocab[t])
                if piece and mostra and not tentativa:
                    yield {"type": "token", "text": piece}
                logits = self._feed(t)
            tail = decoder.decode(b"", final=True)
            if tail and mostra and not tentativa:
                yield {"type": "token", "text": tail}
            text = tok.decode(gen).strip()
            problema = revisar_fala(self.turns[-1][1], text) if depois_do_usuario and not must else None
            if problema:
                yield {"type": "revisao", "motivo": MOTIVO[problema], "antes": text}
            if must or (tools.norm(text) not in recentes and not problema) or tentativa == 2:
                break
            # repetiu: volta o cache para o fim do prompt e sorteia de novo
            proibidas.append(gen[0])
            self.cache.pos = start - 1
            self.cache_ids = self.cache_ids[:start - 1]
            logits = self._feed(last)
        ids = [tok.assistant] + gen + [tok.end]
        if must:
            name, value = must
            if tools.norm(value) not in tools.norm(text):
                text = FAITHFUL[name].format(v=value)
                ids, _ = tok.encode_turn("assistant", text)
                # o cache tem a resposta errada: volta só até o começo dela (a conversa antes continua valendo)
                self.cache.pos = start
                self.cache_ids = self.cache_ids[:start]
            if not mostra:
                yield {"type": "token", "text": text}
        fala = next((t for r, t, _ in reversed(self.turns) if r == "user"), "")
        if depois_do_usuario and not must and revisar_fala(fala, text):  # 3 sorteios e o tom continua errado
            text = random.choice(SEGURA[revisar_fala(fala, text)])
            ids, _ = tok.encode_turn("assistant", text)
            self.cache.pos = start
            self.cache_ids = self.cache_ids[:start]
        seguro = apoio_se_preciso(fala, text)
        if seguro:  # o "text" do fim troca o que apareceu na tela
            text = seguro
            ids, _ = tok.encode_turn("assistant", text)
            self.cache.pos = start
            self.cache_ids = self.cache_ids[:start]
        self.turns.append(("assistant", text, ids))
        yield {"type": "text", "text": text}

    def _call(self, gen):
        """Fecha uma chamada gerada (`gen` começa com <|call|>): valida e deixa pendente."""
        tok = self.tok
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

    def irma(self, ferramentas=None):
        """Outro assistente com o MESMO modelo (sem carregar de novo) e conversa própria: um agente."""
        b = copy.copy(self)
        b.ferramentas = set(ferramentas) if ferramentas is not None else None
        b.reset()
        return b

    @torch.no_grad()
    def prever_ferramenta(self, text):
        """Que ferramenta o modelo usaria para `text`, sem histórico nem cache da conversa? None = conversa.
        Serve para o grupo decidir qual agente responde (com a mesma trava de confiança)."""
        tok = self.tok
        prompt = [tok.bos] + tok.encode_turn("user", text)[0] + [tok.assistant]
        prompt = prompt[-(self.model.cfg.block_size - self.max_reply_tokens):]
        cache = self.model.new_cache()
        logits, _ = self.model(torch.tensor([prompt], device=self.device), cache=cache)
        win, _, _ = self.conselho.decidir(cache, logits[0, -1], text, trava=TRAVA)
        parsed = tools.parse_call(win.text) if win.kind == "call" else None
        return parsed[0] if parsed else None

    def pode(self, ferramenta):
        return self.ferramentas is None or ferramenta in self.ferramentas

    def desviar(self, texto):
        """Desfaz a chamada pendente (fora da função deste agente) e responde `texto` no lugar."""
        self.pending = None
        if self.turns and self.turns[-1][0] == "call":
            self.turns.pop()
        ids, _ = self.tok.encode_turn("assistant", texto)
        self.turns.append(("assistant", texto, ids))

    def substituir_ultima(self, texto):
        """Troca a última fala do assistente (o agente fala de si pelo perfil, não pelo que o modelo decorou)."""
        if self.turns and self.turns[-1][0] == "assistant":
            ids, _ = self.tok.encode_turn("assistant", texto)
            self.turns[-1] = ("assistant", texto, ids)
            self.cache.reset()  # o cache tem a fala antiga; a próxima mensagem relê a conversa
            self.cache_ids = []

    def resolve(self, approved=True):
        """Executa (ou cancela) a chamada pendente e coloca o resultado no histórico."""
        name, arg = self.pending
        self.pending = None
        if name == "lembrete":  # horário que o modelo perdeu ao copiar ("às 15h", "amanhã") vem da fala
            fala = next((t for r, t, _ in reversed(self.turns) if r == "user"), "")
            arg = tools.ancorar_lembrete(arg, fala)
        result = tools.run(name, arg) if approved else "cancelado: você não autorizou"
        ids, _ = self.tok.encode_turn("result", result)
        self.turns.append(("result", result, ids))
        self.feitos.append((name, arg, result))
        return result

    # ------------------------------------------------------------ triagem
    def _p_call_sozinha(self, parte):
        """p(<|call|>) se `parte` fosse a próxima fala do usuário (sem mexer no cache nem no histórico)."""
        ids = self._context() + self.tok.encode_turn("user", parte)[0] + [self.tok.assistant]
        ids = ids[-(self.model.cfg.block_size - 1):]
        with torch.no_grad():
            logits, _ = self.model(torch.tensor([ids], device=self.device))
        return torch.softmax(logits[0, -1].float(), -1)[self.tok.call].item()

    def triagem(self, text):
        """
        Pedido composto ("toca um forró e abaixa o volume") -> partes, uma ferramenta cada; senão [text].
        Só quebra se CADA parte sozinha passar da trava de confiança: "anota arroz, feijão e óleo" não vira três
        pedidos ("óleo" sozinho não é pedido).
        """
        partes = [x.strip(" ,.") for x in DIVISOR.split(text) if x.strip(" ,.")]
        if not 2 <= len(partes) <= 3 or any(len(x.split()) < 2 for x in partes):
            return [text]
        if all(self._p_call_sozinha(x) >= TRAVA for x in partes):
            return partes
        return [text]

    def chat(self, text, confirm=lambda name, arg: True, auto=False):
        """
        Laço completo de uma mensagem do usuário, já executando ferramentas.
        `confirm(nome, arg) -> bool` é chamado para ferramentas sensíveis (a menos que auto=True).
        Gera os mesmos eventos de step() mais {"type": "result", "tool", "text"}; pedido composto também gera
        {"type": "triagem", "partes"} no começo e {"type": "resumo", "text"} no fim.
        """
        partes = self.triagem(text)
        if len(partes) > 1:
            yield {"type": "triagem", "partes": partes}
        self.feitos = []
        for parte in partes:
            yield from self._uma(parte, confirm, auto)
        if len(partes) > 1 and self.feitos:
            yield {"type": "resumo", "text": resumo_feitos(self.feitos)}

    def _uma(self, text, confirm, auto):
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
