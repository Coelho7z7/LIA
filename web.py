"""
Interface web local (só biblioteca padrão do Python, nada para instalar).

    python web.py                  # abre http://127.0.0.1:7860 no navegador
    python web.py --porta 8000 --simular
    python web.py --sem-conselho   # responde direto, sem o debate entre papéis (conselho.py)

Além da Lia, a página tem os agentes que você cria (agentes.py): cada um com conversa
própria, só as ferramentas da função dele e rotinas com horário, que este servidor roda.

Segurança: o servidor só escuta em 127.0.0.1 (nenhum outro aparelho da rede
acessa) e toda chamada da API exige um token aleatório que só a página servida
por ele conhece. Assim, um site malicioso aberto no seu navegador não consegue
mandar a IA abrir coisas no seu PC.
"""

import argparse
import json
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
# Conversas marcadas como "errou?": ficam para revisão. NÃO vão direto pro treino (a resposta errada
# está nelas); depois de corrigidas, viram exemplos em data/conversas/.
FEEDBACK = Path(__file__).resolve().parent / "data" / "feedback" / "erros.jsonl"
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
STATE = {"auto": False}
BOT = None  # a Lia
BOTS = {}  # id do agente -> conversa dele (mesmo modelo da Lia, ver Assistant.irma)
EVENTS = []  # lembretes e rotinas disparados, entregues para a página na próxima consulta
EVENTS_LOCK = threading.Lock()


def on_reminder(text):
    with EVENTS_LOCK:
        EVENTS.append({"type": "reminder", "text": text})


def on_rotina(agente, rotina, texto):
    with EVENTS_LOCK:
        EVENTS.append({"type": "rotina", "agente": agente["id"], "nome": rotina["nome"], "texto": texto})


def bot_de(agente_id):
    """A conversa da Lia ou de um agente (criada na primeira mensagem). None se o agente não existe."""
    if not agente_id or agente_id == "lia":
        return BOT
    a = agentes.achar(agente_id)
    if a is None:
        BOTS.pop(agente_id, None)
        return None
    if agente_id not in BOTS:
        BOTS[agente_id] = BOT.irma(a["ferramentas"])
    BOTS[agente_id].ferramentas = set(a["ferramentas"])  # pode ter sido editado
    return BOTS[agente_id]


def repassar(bot, agente_id, call):
    """Pedido fora da função deste agente: passa para quem tem a ferramenta (ou para a Lia, que tem todas)."""
    alvo = agentes.quem_faz(call["tool"], sem=agente_id)
    para, nome = (alvo["id"], alvo["nome"]) if alvo else ("lia", "Lia")
    acao = ACOES.get(call["tool"], call["tool"])
    texto = f"{acao[:1].upper() + acao[1:]} não é comigo: é com {nome}. Já passei o pedido."
    bot.desviar(texto)
    yield {"type": "handoff", "para": para, "nome": nome, "tool": call["tool"], "arg": call["arg"]}
    yield {"type": "token", "text": texto}
    yield {"type": "text", "text": texto}


def drive(bot, agente_id="lia", text=None, approve=None):
    """Roda o laço do assistente até uma resposta em texto, uma confirmação pendente ou um repasse."""
    if approve is not None:
        if bot.pending is None:
            return
        name = bot.pending[0]
        yield {"type": "result", "tool": name, "text": bot.resolve(approve)}
    if text is not None:
        if bot.pending is not None:  # mensagem nova com confirmação em aberto = cancelar
            bot.resolve(False)
        bot.add_user(text)
    for _ in range(bot.max_tool_steps):
        call = None
        for ev in bot.step():
            if ev["type"] == "call" and not bot.pode(ev["tool"]):
                yield from repassar(bot, agente_id, ev)
                return
            yield ev
            if ev["type"] == "call":
                call = ev
        if call is None:
            return
        if call["confirm"] and not STATE["auto"]:
            yield {"type": "confirm", "tool": call["tool"], "arg": call["arg"]}
            return
        yield {"type": "result", "tool": call["tool"], "text": bot.resolve(True)}


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
        elif url.path == "/api/info" and self._allowed():
            i = BOT.info
            self._json({"params": i["params"], "iter": i["iter"], "auto": STATE["auto"], "conselho": BOT.usar_conselho,
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
        if self.path in ("/api/chat", "/api/confirm", "/api/reset", "/api/load", "/api/feedback"):
            bot = bot_de(agente_id)
            if bot is None:
                return self._json({"erro": "esse agente não existe mais"}, 404)
        if self.path == "/api/chat":
            text = str(data.get("text", "")).strip()[:2000]
            if not text:
                return self._json({"erro": "mensagem vazia"}, 400)
            with LOCK:
                self._stream(drive(bot, agente_id, text=text))
        elif self.path == "/api/confirm":
            with LOCK:
                self._stream(drive(bot, agente_id, approve=bool(data.get("approve"))))
        elif self.path == "/api/reset":
            with LOCK:
                bot.reset()
            self._json({"ok": True})
        elif self.path == "/api/load":
            # Reabre uma conversa salva no histórico: refaz o contexto do modelo com as falas dela.
            # Só vira contexto; nenhuma ferramenta é executada aqui.
            turns = data.get("turns") if isinstance(data.get("turns"), list) else []
            with LOCK:
                bot.reset()
                for t in turns[-200:]:
                    role, text = (t or {}).get("role"), str((t or {}).get("text", ""))[:2000]
                    if role in ("user", "assistant", "call", "result"):
                        bot.turns.append((role, text, bot.tok.encode_turn(role, text)[0]))
                if bot.turns and bot.turns[-1][0] == "call":  # confirmação que ficou pendente: descarta
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
                lines.append(json.dumps({"quando": time.strftime("%Y-%m-%d %H:%M"), "messages": msgs},
                                        ensure_ascii=False))
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
        elif self.path == "/api/rotina":
            # "Rodar agora": executa os passos da rotina na hora (não conta como a execução do dia).
            a = agentes.achar(agente_id)
            r = next((r for r in (a or {}).get("rotinas", []) if r["id"] == data.get("rotina")), None)
            if r is None:
                return self._json({"erro": "rotina não encontrada"}, 404)
            self._json({"nome": r["nome"], "texto": agentes.rodar(r)})
        elif self.path == "/api/settings":
            todos = [BOT, *BOTS.values()]
            if "auto" in data:
                STATE["auto"] = bool(data["auto"])
            if "temperature" in data:
                for b in todos:
                    b.temperature = max(0.0, min(1.5, float(data["temperature"])))
            if "conselho" in data:
                for b in todos:
                    b.usar_conselho = bool(data["conselho"])
            self._json({"auto": STATE["auto"], "temperature": BOT.temperature, "conselho": BOT.usar_conselho})
        else:
            self._json({"erro": "não encontrado"}, 404)


def main():
    global BOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--porta", type=int, default=7860)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--simular", action="store_true")
    ap.add_argument("--sem-navegador", action="store_true")
    ap.add_argument("--sem-conselho", action="store_true", help="responde direto, sem o debate entre papéis")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    tools.DRY_RUN = args.simular
    STATE["auto"] = args.auto
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
