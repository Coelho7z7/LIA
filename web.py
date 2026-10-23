"""
Interface web local (só biblioteca padrão do Python, nada para instalar).

    python web.py                  # abre http://127.0.0.1:7860 no navegador
    python web.py --porta 8000 --simular
    python web.py --sem-conselho   # responde direto, sem o debate entre papéis (conselho.py)

A página segue o Grok Bot: a Lia e os agentes que você cria (agentes.py) são colegas com
conversa própria, grupos de 2 a 6, @menções, repasses entre eles, aprovações com "Permitir
uma vez / Sempre permitir / Negar" e rotinas com horário, que este servidor roda.

Segurança: o servidor só escuta em 127.0.0.1 (nenhum outro aparelho da rede
acessa) e toda chamada da API exige um token aleatório que só a página servida
por ele conhece. Assim, um site malicioso aberto no seu navegador não consegue
mandar a IA abrir coisas no seu PC.
"""

import argparse
import json
import re
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import agentes
import tools
from agent import Assistant
from conselho import ACOES

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
# O modelo mais novo (release lia-v26, com o fixas.json da cópia forçada ao lado); sem ele, o de sempre.
CKPT = "out_v26/chat.pt" if (Path(__file__).resolve().parent / "out_v26" / "chat.pt").exists() else "out/chat.pt"
# Conversas marcadas como "errou?": ficam para revisão. NÃO vão direto pro treino (a resposta errada
# está nelas); depois de corrigidas, viram exemplos em data/conversas/.
FEEDBACK = Path(__file__).resolve().parent / "data" / "feedback" / "erros.jsonl"
STATIC = {"/manifest.json": ("manifest.json", "application/manifest+json"), "/sw.js": ("sw.js", "text/javascript"),
          "/icon-192.png": ("icon-192.png", "image/png"), "/icon-512.png": ("icon-512.png", "image/png")}
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
STATE = {"auto": False, "permitidas": set()}  # ferramentas que não pedem mais aprovação a ninguém ("Executar sempre")
BOT = None  # a Lia
BOTS = {}  # id do agente (ou "lia-equipe", a Lia nos grupos) -> conversa dele, com o mesmo modelo
EVENTS = []  # lembretes e rotinas disparados, entregues para a página na próxima consulta
EVENTS_LOCK = threading.Lock()
# Fala do modelo sobre si mesmo ("sou um Transformer treinado do zero no seu PC"): num agente,
# vira a apresentação dele, tirada do perfil.
META = re.compile(r"(transformer|parametros?\b|treinad[ao]|do zero|neuronio|\bcpu\b|\bgpu\b|processador|modelo pronto)")


def on_reminder(text):
    with EVENTS_LOCK:
        EVENTS.append({"type": "reminder", "text": text})


def on_rotina(agente, rotina, texto):
    with EVENTS_LOCK:
        EVENTS.append({"type": "rotina", "agente": agente["id"], "nome": rotina["nome"], "texto": texto})


def bot_de(agente_id):
    """A conversa da Lia, da Lia nos grupos ("lia-equipe") ou de um agente. None se o agente não existe."""
    if not agente_id or agente_id == "lia":
        return BOT
    if agente_id == "lia-equipe":  # nos grupos a Lia tem conversa própria (não mistura com o chat dela)
        if agente_id not in BOTS:
            BOTS[agente_id] = BOT.irma()
        return BOTS[agente_id]
    a = agentes.achar(agente_id)
    if a is None:
        BOTS.pop(agente_id, None)
        return None
    if agente_id not in BOTS:
        BOTS[agente_id] = BOT.irma(a["ferramentas"])
    BOTS[agente_id].ferramentas = set(a["ferramentas"])  # pode ter sido editado
    return BOTS[agente_id]


def nome_de(agente_id):
    return "Lia" if agente_id in ("lia", "lia-equipe") else (agentes.achar(agente_id) or {}).get("nome", "?")


def undo_info(name, arg, result):
    """O que o cartão pode desfazer depois de uma ação que deu certo."""
    if not result.startswith("ok"):
        return None
    if name == "lembrar":
        return {"tipo": "fato", "ref": arg, "rotulo": "desfazer"}
    if name == "anotar":
        return {"tipo": "nota", "ref": arg, "rotulo": "apagar"}
    if name == "lembrete" and tools.LAST_REMINDER_ID and not tools.DRY_RUN:
        return {"tipo": "lembrete", "ref": tools.LAST_REMINDER_ID, "rotulo": "cancelar"}
    return None


def resolver(bot, approve):
    """Executa (ou nega) a chamada pendente; o evento leva o "desfazer" e, na Wikipédia, o link do artigo."""
    name, arg = bot.pending
    tools.LAST_REMINDER_ID = None
    text = bot.resolve(approve)
    ev = {"type": "result", "tool": name, "text": text, "undo": undo_info(name, arg, text)}
    if name == "consultar" and text.startswith("ok") and tools.LAST_WIKI:
        ev["link"], ev["titulo"] = tools.LAST_WIKI["url"], tools.LAST_WIKI["titulo"]  # o cartão mostra o artigo
    return ev


def falar(texto):
    yield {"type": "token", "text": texto}
    yield {"type": "text", "text": texto}


def repassar(bot, agente_id, call, pedido):
    """Pedido fora da função deste agente: passa para quem tem a ferramenta (ou para a Lia, que tem todas)."""
    alvo = agentes.quem_faz(call["tool"], sem=agente_id)
    para, nome = (alvo["id"], alvo["nome"]) if alvo else ("lia", "Lia")
    acao = ACOES.get(call["tool"], call["tool"])
    texto = f"{acao[:1].upper() + acao[1:]} não é comigo: mandei uma mensagem para {nome}."
    bot.desviar(texto)
    yield {"type": "handoff", "para": para, "nome": nome, "tool": call["tool"], "arg": call["arg"], "texto": pedido}
    yield from falar(texto)


def drive(bot, agente_id="lia", text=None, approve=None, em_grupo=False, retry=False):
    """
    Roda o laço do assistente até uma resposta em texto, um pedido de aprovação ou um repasse.
    Num grupo, a aprovação não aparece ali (como no Grok Bot): vira {"type": "aprovacao"} e o
    cartão vai para o chat privado do agente.
    """
    eu = "lia" if agente_id in ("lia", "lia-equipe") else agente_id
    perfil = agentes.achar(eu) if eu != "lia" else None
    permitidos = set(perfil["permitidos"]) if perfil else set()
    if text is not None:
        # "@Apps abre o spotify": você mesmo escolheu quem faz, então passa direto, sem o modelo decidir
        alvo, resto = agentes.mencao(text)
        if alvo == eu:
            text = resto
        elif alvo not in (None, "todos") and resto:
            if bot.pending is not None:
                bot.resolve(False)
            bot.add_user(text)
            texto = f"Passei para {nome_de(alvo)}."
            bot.desviar(texto)
            yield {"type": "handoff", "para": alvo, "nome": nome_de(alvo), "tool": None, "arg": "", "texto": resto}
            yield from falar(texto)
            return
        # "quem é você?", "o que o Apps faz?", "quem toca música?": responde pelo perfil de cada um
        resposta = agentes.sobre(agente_id, text)
        if resposta:
            if bot.pending is not None:
                bot.resolve(False)
            bot.add_user(text)
            bot.desviar(resposta)
            yield from falar(resposta)
            return
    if approve is not None:
        if bot.pending is None:
            return
        yield resolver(bot, approve)
    if retry:  # "tentar de novo": tira a última fala de texto e gera outra
        if bot.pending is not None or not bot.turns or bot.turns[-1][0] != "assistant":
            return
        bot.turns.pop()
    if text is not None:
        if bot.pending is not None:  # mensagem nova com aprovação em aberto = negar
            bot.resolve(False)
        bot.add_user(text)
    for _ in range(bot.max_tool_steps):
        call = None
        for ev in bot.step():
            if ev["type"] == "call" and not bot.pode(ev["tool"]):
                yield from repassar(bot, agente_id, ev, text if text is not None else ev["arg"])
                return
            if perfil and ev["type"] == "token":
                continue  # o agente fala de uma vez: dá para trocar o "sou um Transformer" antes de aparecer
            if perfil and ev["type"] == "text":
                fala = ev["text"]
                if META.search(tools.norm(fala)):
                    fala = agentes.curta(eu)
                    bot.substituir_ultima(fala)
                yield from falar(fala)
                continue
            yield ev
            if ev["type"] == "call":
                call = ev
        if call is None:
            return
        if call["confirm"] and not (STATE["auto"] or call["tool"] in STATE["permitidas"] or call["tool"] in permitidos):
            yield {"type": "aprovacao" if em_grupo else "confirm", "tool": call["tool"], "arg": call["arg"],
                   "de": agente_id, "quem": eu}
            return
        yield resolver(bot, True)


def grupo(text, membros, responder_a=None):
    """
    Um grupo de 2 a 6 colegas, como no Grok Bot: escrevendo normalmente, responde quem tem a
    ferramenta que o pedido precisa; "@nome" ou "Responder" numa mensagem escolhe quem; "@todos"
    chama todo mundo. Repasses acontecem dentro do grupo. Cada resposta vem depois de
    {"type": "falante", "de"}.
    """
    lista = [a for a in agentes.carregar() if a["id"] in membros]
    ids = (["lia"] if "lia" in membros else []) + [a["id"] for a in lista]
    if not ids:
        return
    alvo, resto = agentes.mencao(text, lista)
    if alvo not in ids + ["todos"]:
        alvo, resto = None, text
    if alvo is None and responder_a in ids:
        alvo = responder_a
    if alvo == "todos":
        for q in ids:
            yield {"type": "falante", "de": q}
            yield from falar(agentes.apresentar(q, agentes.carregar()))
        return
    if alvo is None:
        ferramenta = BOT.prever_ferramenta(resto)
        dono = agentes.quem_faz(ferramenta, lista=lista) if ferramenta else None
        alvo = dono["id"] if dono else ("lia" if "lia" in ids else ids[0])
    for _ in range(3):  # no máximo dois repasses seguidos dentro do grupo
        chave = "lia-equipe" if alvo == "lia" else alvo
        bot = bot_de(chave)
        if bot is None:
            return
        yield {"type": "falante", "de": alvo}
        repasse = None
        for ev in drive(bot, chave, text=resto, em_grupo=True):
            if ev["type"] == "handoff":
                repasse = ev
            yield ev
        if not repasse:
            return
        alvo, resto = repasse["para"], repasse["texto"]


def com_proximas(a):
    """O agente com a próxima execução de cada rotina (para a interface mostrar)."""
    out = dict(a)
    out["rotinas"] = [dict(r, proxima=(p.isoformat(timespec="minutes") if (p := agentes.proxima(r)) else None))
                      for r in a["rotinas"]]
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _allowed(self):
        host = self.headers.get("Host", "").split(":")[0]
        return host in ("127.0.0.1", "localhost") and secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for ev in events:
                self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", TOKEN).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path in STATIC:
            body, ctype = (PAGE.parent / STATIC[url.path][0]).read_bytes(), STATIC[url.path][1]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if url.path == "/sw.js":
                self.send_header("Service-Worker-Allowed", "/")
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/info" and self._allowed():
            i = BOT.info
            self._json({"params": i["params"], "iter": i["iter"], "val_loss": i["val_loss"], "copia": i.get("copia", False),
                        "modelo": CKPT_EM_USO, "auto": STATE["auto"], "permitidas": sorted(STATE["permitidas"]),
                        "conselho": BOT.usar_conselho,
                        "temperature": BOT.temperature, "apps": len(tools.APPS.apps or []),
                        "simulado": tools.DRY_RUN,
                        "tools": [{"name": t.name, "desc": t.desc, "confirm": t.confirm} for t in tools.TOOLS.values()]})
        elif url.path == "/api/turns" and self._allowed():
            bot = bot_de(parse_qs(url.query).get("agente", ["lia"])[0])
            with LOCK:
                turns = [{"role": r, "text": t} for r, t, _ in bot.turns] if bot else []
            self._json({"turns": turns})
        elif url.path == "/api/agentes" and self._allowed():
            self._json([com_proximas(a) for a in agentes.carregar()])
        elif url.path == "/api/memoria" and self._allowed():
            self._json(tools.memoria())
        elif url.path == "/api/lembretes" and self._allowed():
            self._json(tools.lembretes_ativos())
        elif url.path == "/api/events" and self._allowed():
            with EVENTS_LOCK:
                out, EVENTS[:] = list(EVENTS), []
            self._json(out)
        else:
            self._json({"erro": "não encontrado"}, 404)

    def do_POST(self):
        if not self._allowed():
            return self._json({"erro": "proibido"}, 403)
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}") if n < 1_000_000 else {}
        except ValueError:
            return self._json({"erro": "json inválido"}, 400)
        agente_id = str(data.get("agente") or "lia")
        if self.path in ("/api/chat", "/api/confirm", "/api/retry", "/api/reset", "/api/load", "/api/feedback"):
            bot = bot_de(agente_id)
            if bot is None:
                return self._json({"erro": "esse agente não existe mais"}, 404)
        if self.path == "/api/chat":
            text = str(data.get("text", "")).strip()[:2000]
            if not text:
                return self._json({"erro": "mensagem vazia"}, 400)
            with LOCK:
                self._stream(drive(bot, agente_id, text=text))
        elif self.path == "/api/grupo":
            text = str(data.get("text", "")).strip()[:2000]
            membros = [str(m) for m in (data.get("membros") or [])][:6]
            if not text or not membros:
                return self._json({"erro": "mensagem ou grupo vazio"}, 400)
            with LOCK:
                self._stream(grupo(text, membros, data.get("responder")))
        elif self.path == "/api/confirm":
            with LOCK:
                if data.get("sempre") and bot.pending:  # "Sempre permitir" da Lia: não pergunta mais para essa ferramenta
                    STATE["permitidas"].add(bot.pending[0])
                self._stream(drive(bot, agente_id, approve=bool(data.get("approve"))))
        elif self.path == "/api/retry":
            with LOCK:
                self._stream(drive(bot, agente_id, retry=True))
        elif self.path == "/api/perfil":  # nome da boas-vindas vira memória: "meu nome é ..."
            nome = " ".join(str(data.get("nome", "")).split())[:40]
            if nome:
                with LOCK:
                    tools.run("lembrar", f"meu nome é {nome}")
            self._json({"ok": bool(nome)})
        elif self.path == "/api/desfazer":
            with LOCK:
                msg = tools.desfazer(str(data.get("tipo", "")), str(data.get("ref", "")))
            self._json({"ok": True, "msg": msg})
        elif self.path == "/api/memoria":
            tipo, i = data.get("tipo"), data.get("apagar")
            if tipo in ("fatos", "notas") and (i is None or isinstance(i, int)):
                tools.apagar_memoria(tipo, i)
            self._json(tools.memoria())
        elif self.path == "/api/lembretes":
            tools.cancelar_lembrete(str(data.get("cancelar", "")))
            self._json(tools.lembretes_ativos())
        elif self.path == "/api/reset":
            with LOCK:
                bot.reset()
            self._json({"ok": True})
        elif self.path == "/api/load":
            # Reabre uma conversa salva: refaz o contexto do modelo com as falas dela.
            # Só vira contexto; nenhuma ferramenta é executada aqui.
            turns = data.get("turns") if isinstance(data.get("turns"), list) else []
            with LOCK:
                bot.reset()
                for t in turns[-200:]:
                    role, text = (t or {}).get("role"), str((t or {}).get("text", ""))[:2000]
                    if role in ("user", "assistant", "call", "result"):
                        bot.turns.append((role, text, bot.tok.encode_turn(role, text)[0]))
                if bot.turns and bot.turns[-1][0] == "call":  # aprovação que ficou pendente: descarta
                    bot.turns.pop()
            self._json({"ok": True})
        elif self.path == "/api/feedback":
            FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
            lines = FEEDBACK.read_text(encoding="utf-8").splitlines() if FEEDBACK.exists() else []
            if data.get("undo"):
                lines = lines[:-1]
            else:
                with LOCK:
                    msgs = [{"role": r, "content": t} for r, t, _ in bot.turns]
                item = {"quando": time.strftime("%Y-%m-%d %H:%M"), "origem": "interface", "agente": agente_id, "messages": msgs}
                if str(data.get("esperado", "")).strip():
                    item["esperado"] = str(data["esperado"]).strip()[:500]
                lines.append(json.dumps(item, ensure_ascii=False))
            FEEDBACK.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            self._json({"ok": True, "total": len(lines)})
        elif self.path == "/api/agentes":
            try:
                a = agentes.gravar(data.get("agente") or {})
            except ValueError as e:
                return self._json({"erro": str(e)}, 400)
            if a["id"] in BOTS:
                BOTS[a["id"]].ferramentas = set(a["ferramentas"])
            self._json(com_proximas(a))
        elif self.path == "/api/agentes/apagar":
            agentes.apagar(agente_id)
            BOTS.pop(agente_id, None)
            self._json({"ok": True})
        elif self.path == "/api/agentes/permitir":
            # "Sempre permitir" no cartão de aprovação (ou desfazer, no painel de detalhes)
            try:
                a = agentes.permitir(agente_id, str(data.get("tool", "")), bool(data.get("sim", True)))
            except ValueError as e:
                return self._json({"erro": str(e)}, 400)
            self._json(com_proximas(a))
        elif self.path == "/api/rotina":
            # "Fazer um teste": executa os passos da rotina na hora (não conta como a execução do dia).
            a = agentes.achar(agente_id)
            r = next((r for r in (a or {}).get("rotinas", []) if r["id"] == data.get("rotina")), None)
            if r is None:
                return self._json({"erro": "rotina não encontrada"}, 404)
            texto = agentes.rodar(r)
            agentes.registrar(a["id"], r["id"], texto)
            self._json({"nome": r["nome"], "texto": texto})
        elif self.path == "/api/rotina/ativa":
            try:
                a = agentes.ativar_rotina(agente_id, str(data.get("rotina", "")), bool(data.get("ativa")))
            except ValueError as e:
                return self._json({"erro": str(e)}, 400)
            self._json(com_proximas(a))
        elif self.path == "/api/settings":
            todos = [BOT, *BOTS.values()]
            if "auto" in data:
                STATE["auto"] = bool(data["auto"])
            if isinstance(data.get("permitidas"), list):
                STATE["permitidas"] = {str(t) for t in data["permitidas"] if str(t) in tools.TOOLS}
            if "temperature" in data:
                for b in todos:
                    b.temperature = max(0.0, min(1.5, float(data["temperature"])))
            if "conselho" in data:
                for b in todos:
                    b.usar_conselho = bool(data["conselho"])
            self._json({"auto": STATE["auto"], "temperature": BOT.temperature, "conselho": BOT.usar_conselho,
                        "permitidas": sorted(STATE["permitidas"])})
        else:
            self._json({"erro": "não encontrado"}, 404)


CKPT_EM_USO = CKPT


def main():
    global BOT, CKPT_EM_USO
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help=f"modelo (padrão: {CKPT})")
    ap.add_argument("--porta", type=int, default=7860)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--simular", action="store_true")
    ap.add_argument("--sem-navegador", action="store_true")
    ap.add_argument("--sem-conselho", action="store_true", help="responde direto, sem o debate entre papéis")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    tools.DRY_RUN = args.simular
    if args.simular:  # testes nunca mexem na memória, nas notas e nos agentes de verdade
        tools.MEMORY_DIR = tools.MEMORY_DIR.with_name("memoria_simulada")
    STATE["auto"] = args.auto
    CKPT_EM_USO = args.ckpt
    BOT = Assistant(args.ckpt, conselho=not args.sem_conselho)
    tools.APPS.load()
    tools.REMINDER_LISTENERS.append(on_reminder)
    agentes.agendador(on_rotina)
    server = ThreadingHTTPServer(("127.0.0.1", args.porta), Handler)
    url = f"http://127.0.0.1:{args.porta}"
    print(f"Interface no ar: {url}  (Ctrl+C para parar)")
    if not args.sem_navegador:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
