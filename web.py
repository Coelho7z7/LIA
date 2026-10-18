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
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tools
from agent import Assistant

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
# Conversas marcadas como "errou?": ficam para revisão. NÃO vão direto pro treino (a resposta errada
# está nelas); depois de corrigidas, viram exemplos em data/conversas/.
FEEDBACK = Path(__file__).resolve().parent / "data" / "feedback" / "erros.jsonl"
STATIC = {"/manifest.json": ("manifest.json", "application/manifest+json"), "/sw.js": ("sw.js", "text/javascript"),
          "/icon-192.png": ("icon-192.png", "image/png"), "/icon-512.png": ("icon-512.png", "image/png")}
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
STATE = {"auto": False, "permitidas": set()}  # ferramentas que não pedem mais confirmação
BOT = None
EVENTS = []  # lembretes disparados, entregues para a página na próxima consulta
EVENTS_LOCK = threading.Lock()


def on_reminder(text):
    with EVENTS_LOCK:
        EVENTS.append({"type": "reminder", "text": text})


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


def resolve(approve):
    name, arg = BOT.pending
    tools.LAST_REMINDER_ID = None
    text = BOT.resolve(approve)
    ev = {"type": "result", "tool": name, "text": text, "undo": undo_info(name, arg, text)}
    if name == "consultar" and text.startswith("ok") and tools.LAST_WIKI:
        ev["link"], ev["titulo"] = tools.LAST_WIKI["url"], tools.LAST_WIKI["titulo"]  # o cartão mostra o artigo
    return ev


def drive(text=None, approve=None, retry=False):
    """Roda o laço do assistente até uma resposta em texto ou uma confirmação pendente."""
    if approve is not None:
        if BOT.pending is None:
            return
        yield resolve(approve)
    if retry:  # "tentar de novo": tira a última fala de texto dela e gera outra
        if BOT.pending is not None or not BOT.turns or BOT.turns[-1][0] != "assistant":
            return
        BOT.turns.pop()
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
        if call["confirm"] and not STATE["auto"] and call["tool"] not in STATE["permitidas"]:
            yield {"type": "confirm", "tool": call["tool"], "arg": call["arg"]}
            return
        yield resolve(True)


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
        elif self.path in STATIC:
            body, ctype = (PAGE.parent / STATIC[self.path][0]).read_bytes(), STATIC[self.path][1]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if self.path == "/sw.js":
                self.send_header("Service-Worker-Allowed", "/")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info" and self._allowed():
            i = BOT.info
            self._json({"params": i["params"], "iter": i["iter"], "val_loss": i["val_loss"], "auto": STATE["auto"],
                        "temperature": BOT.temperature, "apps": len(tools.APPS.apps or []),
                        "simulado": tools.DRY_RUN,
                        "tools": [{"name": t.name, "desc": t.desc, "confirm": t.confirm} for t in tools.TOOLS.values()]})
        elif self.path == "/api/turns" and self._allowed():
            with LOCK:
                turns = [{"role": r, "text": t} for r, t, _ in BOT.turns]
            self._json({"turns": turns})
        elif self.path == "/api/memoria" and self._allowed():
            self._json(tools.memoria())
        elif self.path == "/api/lembretes" and self._allowed():
            self._json(tools.lembretes_ativos())
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
            data = json.loads(self.rfile.read(n) or b"{}") if n < 1_000_000 else {}
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
                if data.get("sempre") and BOT.pending:  # "Executar sempre": não pergunta mais para essa ferramenta
                    STATE["permitidas"].add(BOT.pending[0])
                self._stream(drive(approve=bool(data.get("approve"))))
        elif self.path == "/api/retry":
            with LOCK:
                self._stream(drive(retry=True))
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
        elif self.path == "/api/reset":
            with LOCK:
                BOT.reset()
            self._json({"ok": True})
        elif self.path == "/api/load":
            # Reabre uma conversa salva no histórico: refaz o contexto do modelo com as falas dela.
            # Só vira contexto; nenhuma ferramenta é executada aqui.
            turns = data.get("turns") if isinstance(data.get("turns"), list) else []
            with LOCK:
                BOT.reset()
                for t in turns[-200:]:
                    role, text = (t or {}).get("role"), str((t or {}).get("text", ""))[:2000]
                    if role in ("user", "assistant", "call", "result"):
                        BOT.turns.append((role, text, BOT.tok.encode_turn(role, text)[0]))
                if BOT.turns and BOT.turns[-1][0] == "call":  # confirmação que ficou pendente: descarta
                    BOT.turns.pop()
            self._json({"ok": True})
        elif self.path == "/api/feedback":
            FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
            lines = FEEDBACK.read_text(encoding="utf-8").splitlines() if FEEDBACK.exists() else []
            if data.get("undo"):
                lines = lines[:-1]
            else:
                with LOCK:
                    msgs = [{"role": r, "content": t} for r, t, _ in BOT.turns]
                item = {"quando": time.strftime("%Y-%m-%d %H:%M"), "origem": "interface", "messages": msgs}
                if str(data.get("esperado", "")).strip():
                    item["esperado"] = str(data["esperado"]).strip()[:500]
                lines.append(json.dumps(item, ensure_ascii=False))
            FEEDBACK.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            self._json({"ok": True, "total": len(lines)})
        elif self.path == "/api/memoria":
            tipo, i = data.get("tipo"), data.get("apagar")
            if tipo in ("fatos", "notas") and (i is None or isinstance(i, int)):
                tools.apagar_memoria(tipo, i)
            self._json(tools.memoria())
        elif self.path == "/api/lembretes":
            tools.cancelar_lembrete(str(data.get("cancelar", "")))
            self._json(tools.lembretes_ativos())
        elif self.path == "/api/settings":
            if "auto" in data:
                STATE["auto"] = bool(data["auto"])
            if isinstance(data.get("permitidas"), list):
                STATE["permitidas"] = {str(t) for t in data["permitidas"] if str(t) in tools.TOOLS}
            if "temperature" in data:
                BOT.temperature = max(0.0, min(1.5, float(data["temperature"])))
            self._json({"auto": STATE["auto"], "temperature": BOT.temperature, "permitidas": sorted(STATE["permitidas"])})
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
    if args.simular:  # testes nunca mexem na memória e nas notas de verdade
        tools.MEMORY_DIR = tools.MEMORY_DIR.with_name("memoria_simulada")
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
