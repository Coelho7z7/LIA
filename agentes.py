"""
Agentes que você cria, no modelo do Grok Bot: cada um é um colega com nome, um rótulo (o
trabalho principal, em poucas palavras), uma descrição (como ele trabalha e as regras que
sempre valem), um avatar (cor e forma), as ferramentas que ele pode usar e rotinas com
horário. Ficam em memoria/agentes.json, fora do git, porque são seus.

    {"id": "musica-x1", "nome": "Música", "rotulo": "Cuida do som do PC",
     "descricao": "Só mexe no volume e na música; nunca abre apps.", "cor": 2, "forma": "nuvem",
     "ferramentas": ["midia", "volume", "youtube"], "permitidos": ["youtube"],
     "rotinas": [{"id": "r1", "nome": "Bom dia", "hora": "08:00", "dias": "dias úteis",
                  "passos": ["data:", "clima: Recife"], "ativa": true, "ultima": "2026-10-22",
                  "historico": [{"quando": "2026-10-22T08:00", "texto": "..."}]}]}

O modelo é o mesmo para todos (a Lia): o que muda é o que cada agente PODE fazer e o que ele
sabe sobre si e sobre os colegas (sobre()). Pedido fora da função dele vai para o agente que
tem a ferramenta, ou para a Lia, que tem todas. "permitidos" são as ferramentas que você
liberou com "Sempre permitir" no cartão de aprovação. As rotinas rodam as ferramentas direto,
sem passar pelo modelo: o horário e os passos são seus, então fazem exatamente o configurado.
"""

import re
import secrets
import threading
import time
from datetime import datetime, timedelta

import tools

ARQUIVO = "agentes.json"
DIAS = {"todo dia": (0, 1, 2, 3, 4, 5, 6), "dias úteis": (0, 1, 2, 3, 4), "fim de semana": (5, 6)}
N_CORES = 8
FORMAS = ("circulo", "gota", "nuvem", "quadrado", "triangulo")
FORMA_DA_COR = ("circulo", "gota", "nuvem", "quadrado", "triangulo", "gota", "circulo", "nuvem")  # igual à página
RESERVADOS = ("lia", "todos", "conselho", "equipe")
LOCK = threading.Lock()  # o arquivo é lido/escrito pelo servidor e pelo agendador

# O que cada ferramenta faz, do jeito que um agente fala de si ("eu cuido de abrir apps")
O_QUE = {"abrir_app": "abrir apps", "abrir_site": "abrir sites", "pesquisar": "pesquisar no Google",
         "youtube": "buscar no YouTube", "hora": "dizer a hora", "data": "dizer a data", "calcular": "fazer contas",
         "lembrete": "criar lembretes", "anotar": "anotar coisas", "ver_notas": "ler suas notas",
         "lembrar": "guardar o que você me conta", "recordar": "lembrar o que você contou",
         "volume": "mexer no volume", "midia": "controlar a música", "abrir_pasta": "abrir pastas",
         "clima": "ver o clima", "bloquear": "bloquear a tela"}
# Palavras que, numa pergunta "quem faz...?", apontam para a ferramenta
ASSUNTO = {"abrir_app": ["app", "apps", "aplicativo", "aplicativos", "programa", "programas"],
           "abrir_site": ["site", "sites"], "pesquisar": ["pesquisa", "pesquisar", "google", "busca", "buscar"],
           "youtube": ["youtube", "video", "videos"], "hora": ["hora", "horas"], "data": ["data"],
           "calcular": ["conta", "contas", "calcula", "calcular", "calculo"],
           "lembrete": ["lembrete", "lembretes", "timer", "alarme"], "anotar": ["anota", "anotar", "anotacao"],
           "ver_notas": ["nota", "notas"], "lembrar": ["memoria", "guarda", "guardar"], "recordar": ["memoria"],
           "volume": ["volume", "som"], "midia": ["musica", "musicas", "midia", "spotify"],
           "abrir_pasta": ["pasta", "pastas"], "clima": ["clima", "tempo", "chuva", "previsao"],
           "bloquear": ["bloquear", "bloqueia", "tela"]}


def _normalizar(a):
    """Agentes salvos por versões antigas da página ("funcao", "autonomo") viram o formato atual."""
    if "rotulo" not in a:
        a["rotulo"] = a.pop("funcao", "")
    a.pop("funcao", None)
    a.setdefault("descricao", "")
    a.setdefault("forma", FORMA_DA_COR[a.get("cor", 0) % N_CORES])
    perm = a.setdefault("permitidos", [])
    if a.pop("autonomo", False):
        perm += [f for f in a["ferramentas"] if tools.TOOLS[f].confirm and f not in perm]
    return a


def carregar():
    lista = tools._load_json(ARQUIVO, [])
    return [_normalizar(a) for a in lista] if isinstance(lista, list) else []


def salvar(lista):
    tools._save_json(ARQUIVO, lista)


def achar(agente_id, lista=None):
    return next((a for a in (carregar() if lista is None else lista) if a["id"] == agente_id), None)


def mencao(texto, lista=None):
    """
    "@Música abaixa o som" -> ("musica-x1", "abaixa o som"). "@todos ..." -> ("todos", ...).
    "@Lia ..." -> ("lia", ...). Sem menção a quem existe -> (None, texto). Aceita sem acento e em qualquer caixa.
    """
    m = re.match(r"\s*@(\S.*)", texto)
    if not m:
        return None, texto
    resto = m.group(1)
    nomes = [("todos", "todos"), ("lia", "Lia")] + [(a["id"], a["nome"]) for a in (carregar() if lista is None else lista)]
    for aid, nome in sorted(nomes, key=lambda x: -len(x[1])):  # o nome mais comprido primeiro ("Música Alta")
        n = len(nome)
        if tools.norm(resto[:n]) == tools.norm(nome) and (len(resto) == n or not resto[n].isalnum()):
            return aid, resto[n:].lstrip(" ,:")
    return None, texto


def quem_faz(ferramenta, sem=None, lista=None):
    """Primeiro agente (fora `sem`) que pode usar a ferramenta; None = fica com a Lia."""
    return next((a for a in (carregar() if lista is None else lista)
                 if ferramenta in a["ferramentas"] and a["id"] != sem), None)


def _texto(v, limite, campo, obrigatorio=True, linhas=False):
    v = str(v or "")
    v = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", v)).strip() if linhas else re.sub(r"\s+", " ", v).strip()
    if obrigatorio and not v:
        raise ValueError(f"{campo} não pode ficar vazio")
    return v[:limite]


def validar(dados, antigo=None):
    """Confere e limpa um agente vindo da interface. Erro vira ValueError com uma frase."""
    ferramentas = [f for f in dict.fromkeys(dados.get("ferramentas") or []) if f in tools.TOOLS]
    if not ferramentas:
        raise ValueError("escolha pelo menos uma ferramenta")
    ultimas = {r["id"]: r.get("ultima") for r in (antigo or {}).get("rotinas", [])}
    historicos = {r["id"]: r.get("historico", []) for r in (antigo or {}).get("rotinas", [])}
    rotinas = []
    for r in (dados.get("rotinas") or [])[:10]:
        nome = _texto(r.get("nome"), 40, "o nome da rotina")
        hora = str(r.get("hora", "")).strip()
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", hora):
            raise ValueError(f"horário inválido em “{nome}” (use 08:00)")
        dias = r.get("dias") if r.get("dias") in DIAS else "todo dia"
        passos = []
        for linha in (r.get("passos") or [])[:8]:
            linha = str(linha).strip()
            if not linha:
                continue
            parsed = tools.parse_call(linha if ":" in linha else linha + ":")
            if not parsed:
                raise ValueError(f"“{linha}” não é uma ferramenta (use ferramenta: argumento)")
            if parsed[0] not in ferramentas:
                raise ValueError(f"“{parsed[0]}” não está nas ferramentas deste agente")
            passos.append(f"{parsed[0]}: {parsed[1]}".rstrip())
        if not passos:
            raise ValueError(f"a rotina “{nome}” não tem nenhum passo")
        rid = r.get("id") if r.get("id") in ultimas else "r" + secrets.token_hex(3)
        rotinas.append({"id": rid, "nome": nome, "hora": hora, "dias": dias, "passos": passos,
                        "ativa": bool(r.get("ativa", True)), "ultima": ultimas.get(rid),
                        "historico": historicos.get(rid, [])})
    nome = _texto(dados.get("nome"), 40, "o nome")
    if tools.norm(nome) in RESERVADOS:
        raise ValueError(f"“{nome}” já é usado pela própria interface")
    try:
        cor = int(dados.get("cor", 0)) % N_CORES
    except (TypeError, ValueError):
        cor = 0
    forma = dados.get("forma") if dados.get("forma") in FORMAS else FORMA_DA_COR[cor]
    permitidos = [f for f in (dados.get("permitidos", (antigo or {}).get("permitidos", [])) or []) if f in ferramentas]
    slug = re.sub(r"[^a-z0-9]+", "-", tools.strip_accents(nome).lower()).strip("-")[:20] or "agente"
    return {"id": antigo["id"] if antigo else f"{slug}-{secrets.token_hex(2)}", "nome": nome,
            "rotulo": _texto(dados.get("rotulo"), 60, "o rótulo", obrigatorio=False),
            "descricao": _texto(dados.get("descricao"), 600, "a descrição", obrigatorio=False, linhas=True),
            "cor": cor, "forma": forma, "ferramentas": ferramentas, "permitidos": permitidos, "rotinas": rotinas}


def gravar(dados):
    """Cria ou atualiza um agente (pelo id). Devolve o agente salvo."""
    with LOCK:
        lista = carregar()
        antigo = achar(dados.get("id"), lista) if dados.get("id") else None
        a = validar(dados, antigo)
        if any(tools.norm(x["nome"]) == tools.norm(a["nome"]) and x["id"] != a["id"] for x in lista):
            raise ValueError(f"já existe um agente chamado “{a['nome']}”")
        lista = [a if x["id"] == a["id"] else x for x in lista] if antigo else lista + [a]
        salvar(lista)
        return a


def apagar(agente_id):
    """Apaga o agente e, com ele, as rotinas (como no Grok Bot)."""
    with LOCK:
        salvar([a for a in carregar() if a["id"] != agente_id])


def _mudar(agente_id, fn):
    with LOCK:
        lista = carregar()
        a = achar(agente_id, lista)
        if a is None:
            raise ValueError("esse agente não existe mais")
        fn(a)
        salvar(lista)
        return a


def permitir(agente_id, ferramenta, sim=True):
    """"Sempre permitir": o agente passa a usar a ferramenta sem pedir aprovação (sim=False desfaz)."""
    def fn(a):
        if ferramenta not in a["ferramentas"]:
            raise ValueError(f"{a['nome']} não usa essa ferramenta")
        a["permitidos"] = [f for f in a["permitidos"] if f != ferramenta] + ([ferramenta] if sim else [])
    return _mudar(agente_id, fn)


def ativar_rotina(agente_id, rotina_id, ativa):
    """Pausa ou volta a ativar uma rotina."""
    def fn(a):
        for r in a["rotinas"]:
            if r["id"] == rotina_id:
                r["ativa"] = bool(ativa)
    return _mudar(agente_id, fn)


# ------------------------------------------------------- quem é quem na equipe
LIA = {"id": "lia", "nome": "Lia", "rotulo": "faz de tudo um pouco",
       "descricao": "Uso todas as ferramentas e cuido do que nenhum agente cuida."}


def _e(itens):
    itens = list(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1] if itens else ""


def _quando(r):
    return f"{r['dias']} às {r['hora']}" + ("" if r.get("ativa", True) else ", pausada")


def _resumo(a):
    """“Música (controlar a música e mexer no volume)”"""
    return f"{a['nome']} ({_e(O_QUE.get(f, f) for f in a['ferramentas'])})"


def apresentar(agente_id, lista=None):
    """O agente (ou a Lia) fala de si e dos colegas, a partir do perfil, não do modelo."""
    lista = carregar() if lista is None else lista
    if agente_id in ("lia", "lia-equipe"):
        colegas = "; ".join(_resumo(a) for a in lista)
        return ("Eu sou a Lia: faço de tudo um pouco e cuido do que nenhum agente cuida. "
                + (f"Na equipe tem {colegas}. Quando um pedido é de alguém, passo para quem cuida." if lista
                   else "Ainda não tem nenhum agente na equipe; dá para criar em “Novo”."))
    a = achar(agente_id, lista)
    if a is None:
        return None
    partes = [f"Eu sou {a['nome']}" + (f" ({a['rotulo'].rstrip('.')})" if a["rotulo"] else "") + "."]
    if a["descricao"]:
        partes.append(a["descricao"].replace("\n", " "))
    partes.append(f"Cuido de {_e(O_QUE.get(f, f) for f in a['ferramentas'])}.")
    if a["rotinas"]:
        partes.append(f"Tenho {'a rotina' if len(a['rotinas']) == 1 else 'as rotinas'} "
                      + _e(f"“{r['nome']}” ({_quando(r)})" for r in a["rotinas"]) + ".")
    outros = [_resumo(x) for x in lista if x["id"] != a["id"]]
    partes.append("O resto eu passo para quem cuida: " + _e(outros + ["a Lia para o que ninguém cuida"]) + "."
                  if outros else "O que não é comigo eu passo para a Lia.")
    return " ".join(partes)


def curta(agente_id, lista=None):
    """Uma linha no lugar do "sou um Transformer treinado no seu PC" que o modelo decorou."""
    a = achar(agente_id, lista)
    if a is None:
        return "Aqui é a Lia: faço de tudo um pouco. No que posso ajudar?"
    return f"Aqui é {a['nome']}: cuido de {_e(O_QUE.get(f, f) for f in a['ferramentas'])}. No que posso ajudar?"


def descrever(a):
    """Um colega na terceira pessoa ("o que o Apps faz?")."""
    if a["id"] == "lia":
        return "A Lia faz de tudo um pouco: usa todas as ferramentas e cuida do que nenhum agente cuida."
    texto = f"{a['nome']}" + (f" ({a['rotulo'].rstrip('.')})" if a["rotulo"] else "") + "."
    texto += f" Cuida de {_e(O_QUE.get(f, f) for f in a['ferramentas'])}."
    if a["rotinas"]:
        texto += " Rotinas: " + _e(f"“{r['nome']}” ({_quando(r)})" for r in a["rotinas"]) + "."
    return texto


QUEM_SOU = re.compile(r"(quem (e|eh|é) (voce|vc|tu)|o que (voce|vc|tu) (faz|sabe fazer|e|eh)|"
                      r"(voce|vc) faz o que|qual (e |eh )?(a )?(sua|tua) (funcao|tarefa|trabalho)|"
                      r"pra que (voce|vc) serve|para que (voce|vc) serve|(se|te) apresenta|o que (voce|vc) (pode|consegue) fazer)")
EQUIPE = re.compile(r"(quem (sao|e|eh) (voces|os agentes|a equipe|seus colegas|os outros)|quais (sao )?(os )?agentes|"
                    r"quem (esta|ta) (na equipe|no grupo|aqui)|seus colegas|quem faz o que|"
                    r"(lista|fala|me fala) (os|dos|quem sao os) agentes|quantos agentes)")
SOBRE_OUTRO = re.compile(r"\b(faz|serve|funcao|quem (e|eh)|cuida|sabe|trabalho|tarefa)\b")
QUEM_FAZ = re.compile(r"\b(quem|qual agente|com quem|pra quem|para quem)\b.*\b(faz|faria|cuida|abre|toca|pesquisa|ve|"
                      r"mostra|anota|lembra|mexe|controla|sabe|falo|peco|pede|bloqueia|calcula)\b")


def sobre(agente_id, texto, lista=None):
    """
    Perguntas sobre a equipe, respondidas pelo perfil de cada um (o modelo só sabe dizer que é
    "um Transformer treinado do zero"): "quem é você?", "o que o Apps faz?", "quem toca música?",
    "quem são vocês?". Devolve a resposta, ou None se não for uma pergunta dessas.
    """
    lista = carregar() if lista is None else lista
    t = tools.norm(texto)
    eu = "lia" if agente_id in ("lia", "lia-equipe") else agente_id
    if EQUIPE.search(t):
        if not lista:
            return "Por enquanto é só a Lia. Dá para criar agentes em “Novo”."
        return "A equipe: " + "; ".join(f"{a['nome']} cuida de {_e(O_QUE.get(f, f) for f in a['ferramentas'])}"
                                         for a in lista) + ". E a Lia faz o que ninguém cuida."
    if QUEM_SOU.search(t):
        return apresentar(eu, lista)
    for a in sorted(lista + [dict(LIA, ferramentas=[])], key=lambda x: -len(x["nome"])):
        if a["id"] != eu and re.search(r"(^|\W)" + re.escape(tools.norm(a["nome"])) + r"($|\W)", t) and SOBRE_OUTRO.search(t):
            return descrever(a)
    if QUEM_FAZ.search(t):
        palavras = set(t.split())
        ferramentas = [f for f, ws in ASSUNTO.items() if palavras & set(ws)]
        if not ferramentas:
            return None
        donos = {}
        for f in ferramentas:
            dono = quem_faz(f, lista=lista)
            donos.setdefault(dono["id"] if dono else "lia", []).append(O_QUE[f])
        frases = []
        for did, coisas in donos.items():
            o_que = _e(dict.fromkeys(coisas))
            if did == eu:
                frases.append(f"Sou eu que cuido de {o_que}.")
            elif did == "lia":
                frases.append(f"Ninguém da equipe cuida de {o_que}; isso fica com a Lia.")
            else:
                frases.append(f"Quem cuida de {o_que} é {achar(did, lista)['nome']}.")
        return " ".join(frases)
    return None


# ------------------------------------------------------------------ rotinas
def rodar(rotina):
    """Executa os passos de uma rotina e devolve o relatório que vai para o chat do agente."""
    linhas = []
    for passo in rotina["passos"]:
        name, arg = tools.parse_call(passo)
        res = tools.run(name, arg)
        linhas.append(f"• {passo} → {res[4:] if res.startswith('ok: ') else res}")
    return "\n".join(linhas)


def registrar(agente_id, rotina_id, texto, quando=None):
    """Guarda a execução no histórico da rotina (as 20 mais recentes, a mais nova primeiro)."""
    with LOCK:
        lista = carregar()
        for a in lista:
            for r in a["rotinas"] if a["id"] == agente_id else []:
                if r["id"] == rotina_id:
                    quando = quando or datetime.now().isoformat(timespec="minutes")
                    r["historico"] = ([{"quando": quando, "texto": texto}] + r.get("historico", []))[:20]
        salvar(lista)


def devidas(lista, agora):
    """Rotinas que chegaram na hora e ainda não rodaram hoje (até 1h de atraso, se o PC estava dormindo)."""
    hoje = agora.date().isoformat()
    for a in lista:
        for r in a["rotinas"]:
            if not r.get("ativa", True) or r.get("ultima") == hoje or agora.weekday() not in DIAS[r["dias"]]:
                continue
            h, m = map(int, r["hora"].split(":"))
            alvo = agora.replace(hour=h, minute=m, second=0, microsecond=0)
            if alvo <= agora < alvo + timedelta(hours=1):
                yield a, r


def proxima(rotina, agora=None):
    """Quando a rotina roda de novo (para mostrar na interface)."""
    agora = agora or datetime.now()
    h, m = map(int, rotina["hora"].split(":"))
    for d in range(8):
        dia = agora + timedelta(days=d)
        alvo = dia.replace(hour=h, minute=m, second=0, microsecond=0)
        ja_rodou = d == 0 and rotina.get("ultima") == agora.date().isoformat()
        if alvo.weekday() in DIAS[rotina["dias"]] and alvo > agora - timedelta(hours=1) and not ja_rodou:
            return alvo
    return None


def agendador(ao_rodar, intervalo=20):
    """Thread que confere as rotinas a cada `intervalo` segundos. `ao_rodar(agente, rotina, texto)`."""

    def laco():
        while True:
            try:
                with LOCK:
                    lista = carregar()
                    agora = datetime.now()
                    fila = list(devidas(lista, agora))
                    for _, r in fila:  # marca antes de rodar: nunca roda duas vezes no mesmo dia
                        r["ultima"] = agora.date().isoformat()
                    if fila:
                        salvar(lista)
                for a, r in fila:
                    texto = rodar(r)
                    registrar(a["id"], r["id"], texto)
                    ao_rodar(a, r, texto)
            except Exception as e:  # o agendador nunca pode morrer
                print("agendador:", e)
            time.sleep(intervalo)

    t = threading.Thread(target=laco, daemon=True)
    t.start()
    return t
