"""
Agentes que você cria: cada um tem nome, uma função (o que ele faz, em palavras), as
ferramentas que ele pode usar e rotinas com horário. Ficam em memoria/agentes.json, fora
do git, porque são seus.

    {"id": "musica-x1", "nome": "Música", "funcao": "cuida do som", "cor": 2,
     "ferramentas": ["midia", "volume", "youtube"],
     "rotinas": [{"id": "r1", "nome": "Bom dia", "hora": "08:00", "dias": "dias úteis",
                  "passos": ["data:", "clima: Recife"], "ativa": true, "ultima": "2026-10-22"}]}

O modelo é o mesmo para todos (a Lia): o que muda é o que cada agente PODE fazer. Pedido
fora da função dele vai para o agente que tem a ferramenta, ou para a Lia, que tem todas.
As rotinas rodam as ferramentas direto, sem passar pelo modelo: o horário e os passos são
seus, então o resultado é exatamente o que você configurou.
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
LOCK = threading.Lock()  # o arquivo é lido/escrito pelo servidor e pelo agendador


def carregar():
    lista = tools._load_json(ARQUIVO, [])
    return lista if isinstance(lista, list) else []


def salvar(lista):
    tools._save_json(ARQUIVO, lista)


def achar(agente_id, lista=None):
    return next((a for a in (carregar() if lista is None else lista) if a["id"] == agente_id), None)


def quem_faz(ferramenta, sem=None, lista=None):
    """Primeiro agente (fora `sem`) que pode usar a ferramenta; None = fica com a Lia."""
    return next((a for a in (carregar() if lista is None else lista)
                 if ferramenta in a["ferramentas"] and a["id"] != sem), None)


def _texto(v, limite, campo, obrigatorio=True):
    v = re.sub(r"\s+", " ", str(v or "")).strip()
    if obrigatorio and not v:
        raise ValueError(f"{campo} não pode ficar vazio")
    return v[:limite]


def validar(dados, antigo=None):
    """Confere e limpa um agente vindo da interface. Erro vira ValueError com uma frase."""
    ferramentas = [f for f in dict.fromkeys(dados.get("ferramentas") or []) if f in tools.TOOLS]
    if not ferramentas:
        raise ValueError("escolha pelo menos uma ferramenta")
    ultimas = {r["id"]: r.get("ultima") for r in (antigo or {}).get("rotinas", [])}
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
                        "ativa": bool(r.get("ativa", True)), "ultima": ultimas.get(rid)})
    nome = _texto(dados.get("nome"), 40, "o nome")
    if nome.lower() in ("lia", "conselho"):
        raise ValueError(f"“{nome}” já é usado pela própria interface")
    try:
        cor = int(dados.get("cor", 0)) % N_CORES
    except (TypeError, ValueError):
        cor = 0
    slug = re.sub(r"[^a-z0-9]+", "-", tools.strip_accents(nome)).strip("-")[:20] or "agente"
    return {"id": antigo["id"] if antigo else f"{slug}-{secrets.token_hex(2)}", "nome": nome,
            "funcao": _texto(dados.get("funcao"), 300, "a função", obrigatorio=False), "cor": cor,
            "ferramentas": ferramentas, "rotinas": rotinas}


def gravar(dados):
    """Cria ou atualiza um agente (pelo id). Devolve o agente salvo."""
    with LOCK:
        lista = carregar()
        antigo = achar(dados.get("id"), lista) if dados.get("id") else None
        a = validar(dados, antigo)
        if any(x["nome"].lower() == a["nome"].lower() and x["id"] != a["id"] for x in lista):
            raise ValueError(f"já existe um agente chamado “{a['nome']}”")
        lista = [a if x["id"] == a["id"] else x for x in lista] if antigo else lista + [a]
        salvar(lista)
        return a


def apagar(agente_id):
    with LOCK:
        salvar([a for a in carregar() if a["id"] != agente_id])


# ------------------------------------------------------------------ rotinas
def rodar(rotina):
    """Executa os passos de uma rotina e devolve o relatório que vai para o chat do agente."""
    linhas = []
    for passo in rotina["passos"]:
        name, arg = tools.parse_call(passo)
        res = tools.run(name, arg)
        linhas.append(f"• {passo} → {res[4:] if res.startswith('ok: ') else res}")
    return "\n".join(linhas)


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
                    ao_rodar(a, r, rodar(r))
            except Exception as e:  # o agendador nunca pode morrer
                print("agendador:", e)
            time.sleep(intervalo)

    t = threading.Thread(target=laco, daemon=True)
    t.start()
    return t
