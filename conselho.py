"""
Conselho da Lia: em vez de uma fala só, vários papéis propõem a próxima fala, uma
revisora aponta os problemas de cada proposta e a Lia junta tudo e decide.

A ideia vem do modo multiagente do Grok, adaptada a um modelo de ~28M parâmetros: ela
não consegue "debater em voz alta", então cada papel é um jeito diferente de usar o
MESMO modelo, e a crítica é feita por regras que dá para conferir.

  Direta       o que o modelo faria sozinho (o caminho mais provável)
  Cética       defende o caminho contrário: se a Direta quer ferramenta, ela propõe só
               conversar, e vice-versa
  Exploradora  sorteia outra versão no mesmo caminho da Direta (outra ferramenta, outro
               argumento, outra frase)
  Copista      testa cada trecho do que você digitou como argumento da chamada e fica com
               o que o modelo acha mais provável: assim não sai "descoriu" quando você
               escreveu "descobriu"
  Revisora     não propõe nada: confere cada proposta (a ferramenta existe? o argumento
               está no pedido? o pedido fala de volume mesmo?)
  Lia          junta tudo: chance que o modelo dá a cada caminho, votos e notas da revisora

Só a primeira fala depois de uma mensagem sua passa pelo conselho, porque é ali que se
decide entre conversar e usar ferramenta. O comentário depois de um resultado segue o
caminho normal.

O debate é narrado como um chat em grupo (rodada()): cada papel manda mensagens na sala,
responde e reage às mensagens dos outros, e guarda os bastidores (chances, placar) no
próprio chat. A interface mostra a sala e o chat de cada papel.
"""

import difflib
import math
import random
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import tools
from revisao import MOTIVO, revisar_fala
from copia import Copia
from model import KVCache, apply_repetition_penalty, sample_next

# Ferramentas cujo argumento é sempre um pedaço do pedido (nos dados de treino, 100%).
COPIA = {"abrir_app", "abrir_site", "abrir_pasta", "pesquisar", "youtube", "anotar", "lembrete", "calcular", "clima",
         "consultar"}

# Ferramentas que não copiam o argumento: o pedido precisa falar do assunto. Aceita erro
# de digitação ("blouqeia"), porque as conversas de treino vêm com eles.
GATILHOS = {
    "volume": ["som", "volume", "alto", "alta", "baixo", "baixa", "baixinho", "mudo", "muta", "mute", "silencia",
               "silencio", "aumenta", "aumentada", "diminui", "abaixa", "abaixada", "sobe", "audio", "ouvindo",
               "ouvir", "desmuta"],
    "midia": ["musica", "faixa", "pula", "passa", "pausa", "pause", "pausar", "play", "toca", "tocar", "tocando",
              "anterior", "proxima", "troca", "continua", "continuar", "retoma", "segue", "solta", "video", "volta",
              "despausa"],
    "hora": ["hora", "horas", "horario"],
    "data": ["dia", "data", "hoje"],
    "clima": ["tempo", "clima", "previsao", "frio", "calor", "quente", "temperatura", "chuva", "chover", "chovendo",
              "sol", "graus", "casaco", "sombrinha", "guarda-chuva", "blusa", "protetor", "chove"],
    "bloquear": ["bloqueia", "bloquear", "bloqueio", "tranca", "trava", "travar"],
    "ver_notas": ["nota", "notas", "anotei", "anotacao", "anotacoes", "lista"],
}
# Argumento só com estas palavras não diz nada ("abrir_app: pra"): nenhum pedido do treino é assim.
VAZIAS = set("a o as os e de do da dos das em no na nos nas um uma pra pro por que q se me eu vc voce ai la ja so "
             "mais pfv favor lia isso essa esse".split())
PONTUACAO = "?!.,;:\"'()"

# Pesos da Lia na hora de juntar (em "log de chance", a mesma escala do modelo).
# Nos dados de treino, o argumento dessas ferramentas é SEMPRE um trecho seguido do pedido e o
# pedido das outras quase sempre (>99%) fala do assunto. Então fugir disso pesa muito.
PESO_FORA = 6.0  # argumento inteiro fora do pedido
PESO_SALTEADO = 3.0  # palavras do pedido, mas não um trecho seguido ("melhores de 2025")
PESO_GATILHO = 10.0  # pedido não fala do assunto da ferramenta
PESO_REPETE = 1.0  # lembrete com o tempo repetido dentro do texto
PESO_VOTO = 0.5  # cada papel a mais que chegou na mesma proposta
PESO_FLUENCIA = 1.0  # quanto a fluência conta entre propostas do mesmo caminho


@dataclass
class Proposta:
    kind: str  # "call" ou "text"
    text: str  # "abrir_app: Spotify" ou a fala
    ids: list  # tokens da fala (com <|call|> na frente, se for chamada), sem <|end|>
    papeis: list
    rota: float = 0.0  # log da chance que o modelo dá ao caminho (ferramenta x conversa)
    fluencia: float = 0.0  # log-prob média por token, dentro do caminho
    ajuste: float = 0.0  # notas da revisora
    notas: list = field(default_factory=list)
    score: float = 0.0

    def resumo(self):
        return {"kind": self.kind, "text": self.text, "papeis": self.papeis, "notas": self.notas,
                "score": round(self.score, 2) if math.isfinite(self.score) else None}


def palavras(text):
    return tools.norm(text).split()


# Como os papéis falam de cada ferramenta no debate.
ACOES = {"abrir_app": "abrir o app", "abrir_site": "abrir o site", "pesquisar": "pesquisar no Google",
         "youtube": "buscar no YouTube", "hora": "ver a hora", "data": "ver a data", "calcular": "calcular",
         "lembrete": "criar um lembrete", "anotar": "anotar", "ver_notas": "ler as notas", "lembrar": "guardar na memória",
         "recordar": "consultar a memória", "volume": "mexer no volume", "midia": "controlar a música",
         "abrir_pasta": "abrir a pasta", "clima": "ver o clima", "bloquear": "bloquear a tela",
         "consultar": "consultar a Wikipédia"}


# Reações de acordo com o papo (não sempre as mesmas): o tom e o assunto da sua fala, a dúvida real de cada papel e
# o tipo de problema que a Revisora achou.
_RUIM = re.compile(r"\b(cague\w*|fora|vacuo|perdi|morreu|faleceu|demiti\w*|reprov\w*|roubaram|quebr\w*|doente|triste|"
                   r"brig\w*|terminou|trai\w*|deu ruim|deu errado|chorando|mal|sozinh\w*|esqueci|atrasad\w*)")
_BOM = re.compile(r"\b(passei|consegui|promovid\w*|ganhei|aprovad\w*|formei|casar|campe\w*|sextou|feliz|aceitou|nasceu)")
_ASSUNTOS = [(r"\b(toca|musica|som|play|pausa|volume|youtube|ouvir)\b", ["🎵", "🎧"]),
             (r"\b(chuva|chover|clima|tempo|frio|calor|sol)\b", ["🌦️", "☔"]),
             (r"\b(hora|horas|data|dia|lembra|lembrete|avisa|acorda)\b", ["⏰", "📅"]),
             (r"(\d\s*[-+x*/]\s*\d|\b(quanto e|calcula|porcento|%)\b)", ["🧮", "🔢"]),
             (r"\b(abre|abrir|app|programa|site|pasta)\b", ["🖥️", "📂"]),
             (r"\b(pesquisa|procura|google|quem foi|o que e)\b", ["🔎", "📚"]),
             (r"\b(oi|ola|bom dia|boa tarde|boa noite|eae|salve)\b", ["👋", "😄"]),
             (r"\b(anota|nota|notas)\b", ["📝", "✏️"])]


# Falas de cada papel: várias por situação (tom da sua fala, dúvida real da Cética), para não repetir sempre igual
FALA_DIRETA_CALL = ["Eu iria de {d}.", "Pra mim é {d}.", "Fácil: {d}.", "Pedido claro, bora: {d}.", "Isso é com a gente: {d}.",
                    "Nem precisa pensar muito: {d}.", "Tá pedindo, então {d}."]
FALA_DIRETA_TEXTO = {
    "ruim": ["Tá chateado, nada de ferramenta agora. Eu diria: “{t}”", "Momento de acolher. Minha sugestão: “{t}”",
             "Isso pede carinho, não comando. Eu responderia: “{t}”"],
    "bom": ["Notícia boa! Eu comemoraria assim: “{t}”", "Isso é pra celebrar. Eu diria: “{t}”", "Que alegria! Minha resposta: “{t}”"],
    "pergunta": ["É uma pergunta pra gente, não um pedido. Eu responderia: “{t}”", "Ele quer saber da gente. Eu diria: “{t}”"],
    "papo": ["Acho que é só papo. Eu diria: “{t}”", "Não vejo pedido aqui. Eu responderia: “{t}”", "Conversa normal. Minha sugestão: “{t}”",
             "Só bater papo mesmo: “{t}”"]}
FALA_EXPLORADORA = ["Sorteei outra possibilidade: {d}.", "Tentei outro caminho e saiu isso: {d}.", "Que tal variar? {d}.",
                    "Olha essa alternativa: {d}."]
FALA_EXPLORADORA_IGUAL = ["Tentei de novo e deu a mesma coisa. Concordo.", "Mesmo sorteando diferente, chego no mesmo lugar.",
                          "Fechado com a Direta."]
FALA_CETICA_TEXTO = {
    "alta": ["Calma: pode não ser pedido. Eu só responderia: “{t}”", "Tô na dúvida de verdade. Talvez seja só: “{t}”"],
    "media": ["E se não for pra fazer nada? Eu só responderia: “{t}”", "Será que ele só quer conversar? Tipo: “{t}”"],
    "baixa": ["Sei que é improvável, mas e se for só papo? “{t}”", "Só pra constar a outra opção: “{t}”"]}
FALA_CETICA_CALL = {
    "alta": ["Pera, isso parece pedido sim. Eu tentaria {d}.", "Tô vendo um pedido escondido: {d}."],
    "media": ["E se for um pedido? Eu tentaria {d}.", "Não descartaria uma ferramenta: {d}."],
    "baixa": ["Duvido, mas alguém tem que testar: {d}.", "Advogada do diabo aqui: e se fosse {d}?"]}
FALA_CETICA_TOM = ["Pera, Direta: {m}. Eu responderia: “{t}”", "Discordo do tom: {m}. Melhor assim: “{t}”",
                   "Não dá pra mandar isso: {m}. Que tal: “{t}”"]
FALA_CETICA_CONCORDA = ["Dessa vez não tenho o que discordar.", "Tô procurando defeito e não acho. Pode ir.", "Concordo, sem ressalvas."]
FALA_COPISTA_IGUAL = ["“{a}” tá igualzinho no pedido.", "Conferi letra por letra: “{a}”. Certo.", "O trecho “{a}” bate com o que você escreveu."]
FALA_REVISORA_OK = ["Conferi tudo: nenhuma proposta quebra as regras.", "Nada a corrigir por aqui.", "Tudo certo, pode seguir.",
                    "Passei o pente fino: sem problema."]


def _faixa(chance):
    return "alta" if chance > 0.4 else "media" if chance > 0.05 else "baixa"


def tom_da_fala(user_text):
    f = tools.norm(user_text)
    ruim, bom = bool(_RUIM.search(f)), bool(_BOM.search(f))
    return "ruim" if ruim and not bom else "bom" if bom and not ruim else None


def reacao_a_fala(user_text, rng):
    """A primeira reação da rodada, à sua mensagem: tom primeiro, depois o assunto."""
    tom = tom_da_fala(user_text)
    if tom == "ruim":
        return rng.choice(["😢", "🫂", "😬"])
    if tom == "bom":
        return rng.choice(["🎉", "🥳", "👏"])
    f = tools.norm(user_text)
    for padrao, emojis in _ASSUNTOS:
        if re.search(padrao, f):
            return rng.choice(emojis)
    return "🤔" if user_text.strip().endswith("?") else "👀"


def reacao_da_revisora(notas):
    """Que tipo de problema ela achou."""
    t = " ".join(notas)
    if "chance" in t or "porteiro" in t:
        return "📉"
    if "não existe" in t or "nao existe" in t:
        return "❌"
    if "não fala" in t or "nao fala" in t or "não está no pedido" in t:
        return "🙅"
    return "⚠️"


def _curto(text, n=140):
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _pct(x):
    return f"{x:.1%}".replace(".", ",")


def descrever(p):
    """Uma proposta em palavras: "pesquisar no Google: “quem descobriu o Brasil”"."""
    if p.kind == "text":
        return f"só responder “{_curto(p.text, 80)}”"
    name, _, arg = p.text.partition(":")
    acao = ACOES.get(name.strip(), name.strip())
    return f"{acao}: “{arg.strip()}”" if arg.strip() else acao


def _parecida(w, g):
    """Mesma palavra, ou com duas letras vizinhas trocadas (o erro de digitação dos dados), ou quase igual."""
    if w == g:
        return True
    if len(w) < 5 or abs(len(w) - len(g)) > 1:
        return False
    if len(w) == len(g):
        diff = [i for i in range(len(w)) if w[i] != g[i]]
        if len(diff) == 2 and diff[1] == diff[0] + 1 and w[diff[0]] == g[diff[1]] and w[diff[1]] == g[diff[0]]:
            return True
    return difflib.SequenceMatcher(None, w, g).ratio() >= 0.85


def _menciona(user_words, gatilhos):
    return any(_parecida(w, g) for w in user_words for g in gatilhos)


def _trechos(text, max_palavras=12, max_inicio=40):
    """Todos os trechos de palavras seguidas do texto, sem pontuação nas pontas e sem os que não dizem nada."""
    ws = text.split()
    out = []
    for i in range(min(len(ws), max_inicio)):
        for j in range(i + 1, min(len(ws), i + max_palavras) + 1):
            s = " ".join(ws[i:j]).strip(PONTUACAO)
            if s and s not in out and not all(w in VAZIAS for w in palavras(s)):
                out.append(s)
    return out


def _clone(cache):
    c = KVCache.__new__(KVCache)
    c.k, c.v, c.max_len, c.pos = cache.k.clone(), cache.v.clone(), cache.max_len, cache.pos
    return c


def _expand(cache, batch, extra):
    """Cópia do cache repetida `batch` vezes, só com espaço para mais `extra` tokens (economiza memória)."""
    n = cache.pos + extra
    c = KVCache.__new__(KVCache)
    c.k = cache.k[:, :1, :, :n].expand(-1, batch, -1, -1, -1).clone()
    c.v = cache.v[:, :1, :, :n].expand(-1, batch, -1, -1, -1).clone()
    c.max_len, c.pos = n, cache.pos
    return c


class Conselho:
    def __init__(self, model, tok, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1,
                 max_reply_tokens=160, fixas=None):
        self.model, self.tok = model, tok
        self.fixas = fixas  # vocabulário fixo da cópia forçada (copia.py), se veio junto do modelo
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.repetition_penalty, self.max_reply_tokens = repetition_penalty, max_reply_tokens
        t = tok
        self.banned_in_text = [t.bos, t.user, t.result, t.call, t.assistant]
        self.dev = model.tok_emb.weight.device

    # ------------------------------------------------------------ geração
    def _feed(self, cache, token):
        logits, _ = self.model(torch.tensor([[token]], device=self.dev), cache=cache)
        return logits[0, -1]

    def _call(self, cache, logits, temperature=0.0, contexto=None):
        """Gera uma chamada a partir do prompt (gulosa, ou sorteada se temperature > 0). Gulosa e com
        `contexto`, usa a mesma cópia forçada do agente (copia.py), quando o modelo traz fixas.json."""
        tok = self.tok
        cache = _clone(cache)
        logits = self._feed(cache, tok.call)
        copia = Copia(self.fixas, contexto) if self.fixas and contexto is not None and temperature <= 0 else None
        gen, raw = [], b""
        for _ in range(48):
            logits[self.banned_in_text] = float("-inf")
            if copia:
                t = copia.escolhe(logits, raw, tok)
            else:
                t = sample_next(logits, temperature, 20, 1.0) if temperature > 0 else int(logits.argmax())
            if t is None or t == tok.end:
                break
            gen.append(t)
            raw += tok.vocab[t]
            logits = self._feed(cache, t)
        return [tok.call] + gen

    def _copista(self, cache, name, arg, user_text):
        """
        Mantém a ferramenta e troca o argumento pelo trecho do pedido que o modelo acha mais provável
        (a chamada inteira, até o <|end|>). No lembrete ("tempo | texto"), refaz só o texto, sem
        repetir o tempo.
        Devolve os trechos [(trecho, ids, log-prob)] do mais provável para o menos.
        """
        fixo, sep, _ = arg.rpartition("|")
        prefixo = f"{name}: " + (fixo.strip() + " | " if sep else "")
        trechos = _trechos(user_text)
        if sep:
            trechos = [t for t in trechos if tools.norm(fixo) not in tools.norm(t)]
        return sorted(self._pontuar(cache, prefixo, trechos), key=lambda n: n[2], reverse=True)

    def _pontuar(self, cache, prefixo, trechos):
        """[(trecho, ids da chamada, log-prob total)] de cada trecho como argumento, em lotes."""
        tok = self.tok
        cands = [[tok.call] + tok.encode(prefixo + t) for t in trechos]
        out = []
        for i in range(0, len(cands), 16):
            chunk = cands[i:i + 16]
            T = max(map(len, chunk))
            if cache.pos + T > self.model.cfg.block_size:
                break
            inp = torch.full((len(chunk), T), tok.end, device=self.dev)
            tgt = torch.full((len(chunk), T), -100, device=self.dev)
            for b, ids in enumerate(chunk):
                inp[b, :len(ids)] = torch.tensor(ids)
                tgt[b, :len(ids)] = torch.tensor(ids[1:] + [tok.end])
            logits, _ = self.model(inp, targets=tgt, cache=_expand(cache, len(chunk), T))
            valid = tgt != -100
            lp = F.log_softmax(logits.float(), dim=-1).gather(1, tgt[valid][:, None])[:, 0]
            tot = torch.zeros(len(chunk), device=self.dev).index_add_(0, valid.nonzero()[:, 0], lp)
            out += [(trechos[i + b], ids, float(tot[b])) for b, ids in enumerate(chunk)]
        return out

    def _text(self, cache, logits, temperature):
        """Fala de texto sorteada, com as mesmas travas do agent.py (sem token especial, sem repetir)."""
        tok = self.tok
        cache = _clone(cache)
        logits = logits.clone()
        gen = []
        for _ in range(self.max_reply_tokens):
            logits[self.banned_in_text] = float("-inf")
            if not gen:
                logits[tok.end] = float("-inf")
            logits = apply_repetition_penalty(logits, gen[-48:], self.repetition_penalty)
            t = sample_next(logits, temperature, self.top_k, self.top_p)
            if t == tok.end:
                break
            gen.append(t)
            logits = self._feed(cache, t)
        return gen

    def _fluencia(self, cache, logits, p):
        """Log-prob média dos tokens da proposta, sem contar a escolha do caminho."""
        tok = self.tok
        logp = F.log_softmax(logits.float(), dim=-1)
        seq = p.ids + [tok.end]
        if p.kind == "call":  # o <|call|> é a escolha do caminho; conta só o que vem depois
            first, seq_in, n = 0.0, seq[:-1], len(seq) - 1
        else:
            first, seq_in, n = float(logp[seq[0]]) - p.rota, seq[:-1], len(seq)
        if len(seq) < 2:
            return first
        _, loss = self.model(torch.tensor([seq_in], device=self.dev), targets=torch.tensor([seq[1:]], device=self.dev),
                             cache=_clone(cache))
        return (first - float(loss) * (len(seq) - 1)) / n

    # ------------------------------------------------------------- revisão
    def revisar(self, p, user_text):
        user_words = set(palavras(user_text))
        if p.kind == "text":
            if not p.text:
                p.ajuste, p.notas = float("-inf"), ["fala vazia"]
            return
        parsed = tools.parse_call(p.text)
        if not parsed:
            p.ajuste, p.notas = float("-inf"), ["essa ferramenta não existe"]
            return
        name, arg = parsed
        if name in COPIA and arg:
            ws = palavras(arg)
            fora = [w for w in ws if w not in user_words]
            if fora:
                p.ajuste -= PESO_FORA * len(fora) / len(ws)
                p.notas.append("“" + " ".join(fora) + "” não está no que você escreveu")
            elif all(w in VAZIAS for w in ws):
                p.ajuste -= PESO_FORA
                p.notas.append(f"“{arg}” não diz o que fazer")
            elif not all(f" {s} " in f" {tools.norm(user_text)} " for s in map(tools.norm, arg.split("|")) if s):
                p.ajuste -= PESO_SALTEADO
                p.notas.append("pulou palavras do que você escreveu")
        if name == "lembrete" and "|" in arg:
            tempo, _, texto = (tools.norm(x) for x in arg.partition("|"))
            if tempo and tempo in texto:
                p.ajuste -= PESO_REPETE
                p.notas.append("o texto do lembrete repete o tempo")
        if name in GATILHOS and not _menciona(user_words, GATILHOS[name]):
            p.ajuste -= PESO_GATILHO
            p.notas.append("o pedido não fala disso")


    # -------------------------------------------------------------- decisão
    def decidir(self, cache, logits, user_text, **kw):
        """Igual a rodada(), sem a narração: devolve (vencedora, propostas, veredito)."""
        gen = self.rodada(cache, logits, user_text, **kw)
        while True:
            try:
                next(gen)
            except StopIteration as fim:
                return fim.value

    @torch.no_grad()
    def rodada(self, cache, logits, user_text, trava=None, forcar=None, antes=""):
        """
        O debate inteiro. `cache` já contém o prompt (até <|assistant|>) e `logits` são do último
        token; nenhum dos dois é alterado. É um gerador de eventos para a interface, que saem na
        hora em que cada papel termina o trabalho:
          {"type": "conselho", "ev": "digitando", "de"}                      quem está trabalhando
          {"type": "conselho", "ev": "msg", "id", "de", "texto", "resp", "privado"}
                    mensagem na sala; `resp` = id da mensagem respondida; `privado` = só no chat do papel
          {"type": "conselho", "ev": "reacao", "msg", "de", "emoji"}
        No fim devolve (vencedora, propostas em ordem de placar, veredito); use com `yield from`.

        `trava`: como no agent.py, só vale chamar ferramenta se o modelo der p(<|call|>) >= trava.
        `forcar`: o porteiro (porteiro.py) estava seguro: "call" ou "text" manda no caminho.

        As falas são frases prontas preenchidas com o que cada papel fez de verdade (a proposta, a
        crítica, o placar): o modelo é pequeno demais para escrever o debate sozinho.
        """
        tok = self.tok
        rng = random.Random(user_text)  # mesmo pedido, mesmas frases
        p_call = float(F.softmax(logits.float(), dim=-1)[tok.call])
        rota = {"call": math.log(max(p_call, 1e-9)), "text": math.log(max(1 - p_call, 1e-9))}
        direta_call = int(logits.argmax()) == tok.call
        props, origem, falas = {}, {}, []

        def chave(p):
            return p.kind, tools.norm(p.text)

        def add(kind, ids, papel):
            """(proposta, é nova?). Proposta repetida vira um voto a mais."""
            text = tok.decode(ids[1:] if kind == "call" else ids).strip()
            k = (kind, tools.norm(text))
            if k in props:
                if papel not in props[k].papeis:
                    props[k].papeis.append(papel)
                return props[k], False
            props[k] = Proposta(kind, text, ids, [papel], rota=rota[kind])
            return props[k], True

        def msg(de, texto, resp=None, privado=False):
            ev = {"type": "conselho", "ev": "msg", "id": len(falas) + 1, "de": de, "texto": texto, "resp": resp,
                  "privado": privado}
            falas.append(ev)
            return ev

        def reacao(msg_id, de, emoji):
            return {"type": "conselho", "ev": "reacao", "msg": msg_id, "de": de, "emoji": emoji}

        def digitando(de):
            return {"type": "conselho", "ev": "digitando", "de": de}

        text_logits = logits.clone()
        text_logits[tok.call] = float("-inf")

        # Direta: o caminho mais provável.
        yield digitando("Direta")
        if direta_call:
            p, _ = add("call", self._call(cache, logits, contexto=user_text), "Direta")
            m = msg("Direta", rng.choice(FALA_DIRETA_CALL).format(d=descrever(p)))
        else:
            p, _ = add("text", self._text(cache, text_logits, self.temperature), "Direta")
            tom = tom_da_fala(user_text) or ("pergunta" if user_text.strip().endswith("?") else "papo")
            m = msg("Direta", rng.choice(FALA_DIRETA_TEXTO[tom]).format(t=_curto(p.text)))
        origem[chave(p)] = direta = m["id"]
        prop_direta = p
        yield reacao(0, "Direta", reacao_a_fala(user_text, rng))  # a mensagem 0 é a sua, que abre a rodada
        yield m
        yield msg("Direta", f"Chance de usar ferramenta, segundo o modelo: {_pct(p_call)}. "
                            "Fui sempre no token mais provável.", privado=True)

        # Exploradora: outro sorteio no mesmo caminho.
        yield digitando("Exploradora")
        temp = 0.8 if direta_call else max(self.temperature, 0.9)
        ids = self._call(cache, logits, temperature=temp) if direta_call else self._text(cache, text_logits, temp)
        p, nova = add("call" if direta_call else "text", ids, "Exploradora")
        if nova:
            m = msg("Exploradora", rng.choice(FALA_EXPLORADORA).format(d=descrever(p)), resp=direta)
            origem[chave(p)] = m["id"]
            yield m
        else:
            yield reacao(direta, "Exploradora", "💯" if direta_call else "🤝")  # concorda: só reage
            if rng.random() < 0.25:
                yield msg("Exploradora", rng.choice(FALA_EXPLORADORA_IGUAL), resp=direta)
        yield msg("Exploradora", f"Sorteio com temperatura {f'{temp:.1f}'.replace('.', ',')}: às vezes o segundo "
                                 "caminho é o certo.",
                  privado=True)

        # Cética: só discorda quando tem motivo. Se a Direta errou o tom, ela aponta e propõe outra resposta; se há
        # dúvida real entre conversar e usar ferramenta (>= 5% pro outro lado), defende o outro caminho; senão concorda.
        yield digitando("Cética")
        chance = (1 - p_call) if direta_call else p_call
        problema = None if direta_call else revisar_fala(user_text, prop_direta.text, antes)
        if problema:
            for _ in range(3):
                ids = self._text(cache, text_logits, max(self.temperature, 0.9))
                if not revisar_fala(user_text, tok.decode(ids).strip(), antes):
                    break
            p, _ = add("text", ids, "Cética")
            yield reacao(direta, "Cética", "🙅")
            m = msg("Cética", rng.choice(FALA_CETICA_TOM).format(m=MOTIVO[problema], t=_curto(p.text)), resp=direta)
            origem.setdefault(chave(p), m["id"])
            yield m
        elif chance >= 0.05:
            if direta_call:
                p, _ = add("text", self._text(cache, text_logits, self.temperature), "Cética")
                m = msg("Cética", rng.choice(FALA_CETICA_TEXTO[_faixa(chance)]).format(t=_curto(p.text)), resp=direta)
            else:
                p, _ = add("call", self._call(cache, logits, contexto=user_text), "Cética")
                m = msg("Cética", rng.choice(FALA_CETICA_CALL[_faixa(chance)]).format(d=descrever(p)), resp=direta)
            yield reacao(0, "Cética", "🧐" if chance > 0.4 else "🤔")
            origem.setdefault(chave(p), m["id"])
            yield m
        else:
            yield reacao(direta, "Cética", "👌")
            if rng.random() < 0.3:
                yield msg("Cética", rng.choice(FALA_CETICA_CONCORDA), resp=direta)
        yield msg("Cética", f"O modelo dá {_pct(chance)} de chance pro outro caminho.", privado=True)

        # Copista: refaz cada ferramenta proposta que copia o argumento do pedido.
        feitas = set()
        for base in [p for p in props.values() if p.kind == "call"]:
            parsed = tools.parse_call(base.text)
            if not parsed or parsed[0] not in COPIA or parsed[0] in feitas:
                continue
            feitas.add(parsed[0])
            yield digitando("Copista")
            ranking = self._copista(cache, *parsed, user_text)
            if not ranking:
                continue
            p, nova = add("call", ranking[0][1], "Copista")
            top = "\n".join(f"{i + 1}. “{t}” ({lp:.1f})" for i, (t, _, lp) in enumerate(ranking[:3]))
            if nova:
                m = msg("Copista", f"Montei só com pedaços do que você escreveu: {descrever(p)}.",
                        resp=origem[chave(base)])
                origem[chave(p)] = m["id"]
                yield m
            else:
                yield reacao(origem[chave(p)], "Copista", "🎯")  # o argumento já estava certo: só reage
                if rng.random() < 0.3:
                    yield msg("Copista", rng.choice(FALA_COPISTA_IGUAL).format(a=tools.parse_call(p.text)[1]),
                              resp=origem[chave(p)])
            yield msg("Copista", f"Testei {len(ranking)} trechos do pedido como argumento de {parsed[0]}. "
                                 f"Os favoritos do modelo (log da chance):\n{top}", privado=True)

        # Revisora: confere cada proposta.
        yield digitando("Revisora")
        ordem = list(props.values())
        checklist = []
        for p in ordem:
            self.revisar(p, user_text)
            tom = revisar_fala(user_text, p.text, antes) if p.kind == "text" else None
            if tom:  # resposta no tom errado (revisao.py) não pode ganhar
                p.ajuste, p.notas = float("-inf"), p.notas + [MOTIVO[tom]]
            # a trava de confiança e o porteiro valem também aqui
            if forcar == "text" and p.kind == "call":
                p.ajuste, p.notas = float("-inf"), p.notas + ["o porteiro disse que é conversa"]
            elif forcar == "call" and p.kind == "text":
                p.ajuste, p.notas = float("-inf"), p.notas + ["o porteiro disse que é pedido"]
            elif forcar is None and trava and p.kind == "call" and p_call < trava:
                p.ajuste, p.notas = float("-inf"), p.notas + [f"o modelo só deu {_pct(p_call)} de chance de ser pedido"]
            alvo = origem[chave(p)]
            if p.notas:
                yield reacao(alvo, "Revisora", reacao_da_revisora(p.notas))
                yield msg("Revisora", "; ".join(p.notas)[:1].upper() + "; ".join(p.notas)[1:] + ".", resp=alvo)
            else:
                yield reacao(alvo, "Revisora", "✅")
            checklist.append(f"• {descrever(p)}: {'; '.join(p.notas) if p.notas else 'ok'}")
        if not any(p.notas for p in ordem):
            yield msg("Revisora", rng.choice(FALA_REVISORA_OK))
        yield msg("Revisora", "O que eu conferi:\n" + "\n".join(checklist), privado=True)

        # Lia: junta tudo e decide.
        yield digitando("Lia")
        for p in ordem:
            p.fluencia = self._fluencia(cache, logits, p) if math.isfinite(p.ajuste) else float("-inf")
        best_flu = {k: max((p.fluencia for p in ordem if p.kind == k), default=0.0) for k in rota}
        for p in ordem:
            if not math.isfinite(p.ajuste):
                p.score = float("-inf")
                continue
            p.score = (p.rota + p.ajuste + PESO_VOTO * (len(p.papeis) - 1)
                       + PESO_FLUENCIA * (p.fluencia - best_flu[p.kind]))
        ranking = sorted(ordem, key=lambda p: p.score, reverse=True)
        win = ranking[0]
        veredito = self._veredito(win, ranking)
        placar = "\n".join(
            f"{i + 1}. {descrever(p)} → {p.score:.2f}" if math.isfinite(p.score) else f"{i + 1}. {descrever(p)} → descartada"
            for i, p in enumerate(ranking))
        yield msg("Lia", "Placar (caminho + revisora + votos + fluência, em log de chance):\n" + placar, privado=True)
        m = msg("Lia", veredito, resp=origem[chave(win)])  # a Lia não reage: decide respondendo à escolhida
        yield m
        tom = tom_da_fala(user_text)
        festa = "🫂" if tom == "ruim" else "🥳" if tom == "bom" else "🙌" if win.kind == "call" else "😊"
        for papel in win.papeis:
            yield reacao(m["id"], papel, festa)
        if "Cética" not in win.papeis:
            dela = next((x for x in ranking if "Cética" in x.papeis), None)
            yield reacao(m["id"], "Cética", "😅" if dela is not None and not math.isfinite(dela.score) else "🤷")
        return win, ranking, veredito

    def _veredito(self, win, props):
        if not math.isfinite(win.score):
            return "Ninguém chegou numa resposta boa."
        direta = next(p for p in props if "Direta" in p.papeis)
        quem = " e ".join(win.papeis)
        r = random.Random(win.text)
        if win is direta:
            if len(win.papeis) > 1:
                return r.choice(["Fiquei com a Direta: {q} chegaram na mesma resposta.", "Quando {q} concordam, nem discuto. Vai essa.",
                                 "Unanimidade entre {q}: fechado."]).format(q=quem)
            return r.choice(["Fiquei com a Direta: as outras propostas tinham mais problemas.", "A Direta acertou de primeira. Vai essa.",
                             "Ouvi todo mundo, mas a primeira ideia era a melhor."])
        motivo = direta.notas[0] if direta.notas else "a proposta dela era menos provável"
        return r.choice(["Fiquei com a {q}: na Direta, {m}.", "Dessa vez vou de {q}. É que na Direta, {m}.",
                         "{q} me convenceu: na proposta da Direta, {m}."]).format(q=quem, m=motivo)
