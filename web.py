"""
Interface web local (só biblioteca padrão do Python, nada para instalar).

    python web.py                  # abre http://127.0.0.1:7860 no navegador
    python web.py --porta 8000 --simular

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
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tools
from agent import Assistant

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
STATE = {"auto": False}
BOT = None
EVENTS = []  # lembretes disparados, entregues para a página na próxima consulta
EVENTS_LOCK = threading.Lock()


def on_reminder(text):
    with EVENTS_LOCK:
        EVENTS.append({"type": "reminder", "text": text})


def drive(text=None, approve=None):
    """Roda o laço do assistente até uma resposta em texto ou uma confirmação pendente."""
    if approve is not None:
        if BOT.pending is None:
            return
        name = BOT.pending[0]
        yield {"type": "result", "tool": name, "text": BOT.resolve(approve)}
    if text is not None:
        if BOT.pending is not None:  # mensagem nova com confirmação em aberto = cancelar
            BOT.resolve(False)
        BOT.add_user(text)
    for _ in range(BOT.max_tool_steps):
        call = None
        for ev in BOT.step():
            yield ev
            if ev["type"] == "call":
                call = ev
        if call is None:
            return
        if call["confirm"] and not STATE["auto"]:
            yield {"type": "confirm", "tool": call["tool"], "arg": call["arg"]}
            return
        yield {"type": "result", "tool": call["tool"], "text": BOT.resolve(True)}


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
        if self.path in ("/", "/index.html"):
            body = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", TOKEN).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info" and self._allowed():
            i = BOT.info
            self._json({"params": i["params"], "iter": i["iter"], "auto": STATE["auto"],
                        "temperature": BOT.temperature, "apps": len(tools.APPS.apps or []),
                        "simulado": tools.DRY_RUN,
                        "tools": [{"name": t.name, "desc": t.desc, "confirm": t.confirm} for t in tools.TOOLS.values()]})
        elif self.path == "/api/events" and self._allowed():
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
            data = json.loads(self.rfile.read(n) or b"{}") if n < 100_000 else {}
        except ValueError:
            return self._json({"erro": "json inválido"}, 400)
        if self.path == "/api/chat":
            text = str(data.get("text", "")).strip()[:2000]
            if not text:
                return self._json({"erro": "mensagem vazia"}, 400)
            with LOCK:
                self._stream(drive(text=text))
        elif self.path == "/api/confirm":
            with LOCK:
                self._stream(drive(approve=bool(data.get("approve"))))
        elif self.path == "/api/reset":
            with LOCK:
                BOT.reset()
            self._json({"ok": True})
        elif self.path == "/api/settings":
            if "auto" in data:
                STATE["auto"] = bool(data["auto"])
            if "temperature" in data:
                BOT.temperature = max(0.0, min(1.5, float(data["temperature"])))
            self._json({"auto": STATE["auto"], "temperature": BOT.temperature})
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
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    tools.DRY_RUN = args.simular
    STATE["auto"] = args.auto
    BOT = Assistant(args.ckpt)
    tools.APPS.load()
    tools.REMINDER_LISTENERS.append(on_reminder)
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
