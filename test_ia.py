"""
Testes rápidos de tudo que não depende de um modelo treinado.

    python test_ia.py

Nada é aberto de verdade: as ferramentas rodam em modo simulado (DRY_RUN) e a
memória vai para uma pasta temporária.
"""

import pathlib
import sys
import tempfile

import torch

import tools as T
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

sys.stdout.reconfigure(encoding="utf-8")
T.DRY_RUN = True
T.MEMORY_DIR = pathlib.Path(tempfile.mkdtemp())
failures = 0


def check(desc, got, ok):
    global failures
    status = "ok " if ok else "FALHOU"
    if not ok:
        failures += 1
    print(f"  [{status}] {desc:<45} -> {got}")


def test_kv_cache():
    print("KV cache = forward completo")
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=4, n_embd=64)).eval()
    x = torch.randint(0, 300, (1, 40))
    full = [m(x[:, :t])[0][0, -1] for t in range(1, 41)]
    c = m.new_cache()
    outs = [(10, m(x[:, :10], cache=c)[0][0, -1]), (17, m(x[:, 10:17], cache=c)[0][0, -1])]
    outs += [(t + 1, m(x[:, t:t + 1], cache=c)[0][0, -1]) for t in range(17, 40)]
    err = max((full[t - 1] - o).abs().max().item() for t, o in outs)
    check("diferença máxima", f"{err:.2e}", err < 1e-4)


def test_tokenizer():
    print("Tokenizador")
    tok = BPETokenizer()
    tok.train("abre o spotify por favor. que horas são? ação, coração! " * 50, 320, verbose=False)
    s = "Olá, mundo! Ação_123 4567 ç ² 😀\n\n  x\tfim <|call|>"
    check("ida e volta sem perdas", repr(tok.decode(tok.encode(s))[:20]), tok.decode(tok.encode(s)) == s)
    check("texto nunca vira token especial", "", tok.call not in tok.encode("<|call|>"))
    ids, mask = tok.encode_chat([{"role": "user", "content": "oi"}, {"role": "call", "content": "hora:"},
                                 {"role": "result", "content": "ok: 10:00"}, {"role": "assistant", "content": "São 10:00."}])
    trained = tok.decode([i for i, k in zip(ids, mask) if k])
    check("só o assistente entra na perda", repr(trained), trained == "<|call|>hora:<|end|>São 10:00.<|end|>")
    t2 = BPETokenizer.from_dict(tok.to_dict())
    check("salvar/carregar", "", t2.encode_chat([{"role": "user", "content": "oi"}])[0] ==
          tok.encode_chat([{"role": "user", "content": "oi"}])[0])


def test_tools():
    print("Ferramentas (simuladas)")
    cases = [
        ("abrir_app", "bloco de notas", "ok:"), ("abrir_app", "calculadora", "ok:"),
        ("abrir_app", "configurações", "ok:"), ("abrir_app", "gerenciador de tarefas", "ok:"),
        ("abrir_app", "youtube", "ok: abri youtube.com"), ("abrir_app", "xyzabcqwe", "erro:"),
        ("abrir_site", "site do github", "ok: abri github.com"), ("abrir_site", "exemplo.com.br", "ok: abri exemplo.com.br"),
        ("abrir_site", "gmial", "ok: abri mail.google.com"), ("abrir_site", "blablabla", "erro:"),
        ("calcular", "23 * 47", "ok: 1081"), ("calcular", "15% de 200", "ok: 30"),
        ("calcular", "raiz de 144", "ok: 12"), ("calcular", "2 elevado a 10", "ok: 1024"),
        ("calcular", "3 x 4 + 1", "ok: 13"), ("calcular", "10 dividido por 4", "ok: 2,5"),
        ("calcular", "1,5 vezes 2", "ok: 3"), ("calcular", "12345*1000", "ok: 12.345.000"),
        ("calcular", "9**9**9", "erro:"), ("calcular", "__import__('os')", "erro:"), ("calcular", "1/0", "erro:"),
        ("lembrete", "10 minutos | pizza", "ok: em 10 minutos: pizza"), ("lembrete", "1h30 | x", "ok: em 1 hora e 30 minutos: x"),
        ("lembrete", "meia hora | y", "ok: em 30 minutos: y"), ("lembrete", "30 segundos", "ok: em 30 segundos: o timer acabou"),
        ("lembrete", "duas horas | z", "ok: em 2 horas: z"), ("lembrete", "abc | b", "erro:"),
        ("abrir_pasta", "downloads", "ok: abri a pasta Downloads"), ("abrir_pasta", "a pasta de documentos", "ok: abri a pasta Documentos"),
        ("abrir_pasta", "área de trabalho", "ok:"), ("abrir_pasta", "xyz", "erro:"),
        ("volume", "mais", "ok:"), ("midia", "proxima", "ok:"), ("bloquear", "", "ok:"),
        ("ferramenta_falsa", "x", "erro:"),
    ]
    for name, arg, expect in cases:
        got = T.run(name, arg)
        check(f"{name}: {arg}", got, got.startswith(expect))

    T.run("lembrar", "meu nome é Matheus")
    T.run("lembrar", "eu moro em Recife")
    T.run("lembrar", "meu nome é Ana")  # substitui o nome antigo
    got = T.run("recordar", "nome")
    check("recordar nome (fato atualizado)", got, got == "ok: meu nome é Ana")
    got = T.run("recordar", "onde eu moro")
    check("recordar cidade", got, "Recife" in got)
    got = T.run("recordar", "aniversario")
    check("recordar algo desconhecido", got, got.startswith("erro:"))
    T.run("anotar", "comprar leite")
    got = T.run("ver_notas")
    check("anotar + ver notas", got, got == "ok: comprar leite")
    check("parse_call válido", T.parse_call("abrir_app: spotify"), T.parse_call("abrir_app: spotify") == ("abrir_app", "spotify"))
    check("parse_call rejeita ferramenta inventada", T.parse_call("shell: dir"), T.parse_call("shell: dir") is None)


def _tiny_checkpoint():
    """Um modelo minúsculo e aleatório, só para exercitar a mecânica do agente."""
    from dataclasses import asdict
    tok = BPETokenizer()
    tok.train("abre o spotify. que horas são? ok: abri Spotify. Pronto! " * 30, 300, verbose=False)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=256, n_layer=2, n_head=2, n_embd=32)
    path = T.MEMORY_DIR / "tiny.pt"
    torch.save({"model": GPT(cfg).state_dict(), "config": asdict(cfg), "tokenizer": tok.to_dict(),
                "iter": 0, "best_val": 0.0, "stage": "sft"}, path)
    return path


def test_agent():
    print("Agente (modelo aleatório minúsculo)")
    from agent import Assistant
    bot = Assistant(str(_tiny_checkpoint()), max_reply_tokens=20)
    events = list(bot.chat("oi, tudo bem?"))
    check("conversa termina com uma fala", events[-1]["type"], events[-1]["type"] in ("text", "result"))
    # Simula uma chamada de ferramenta no histórico e executa.
    ids = [bot.tok.assistant, bot.tok.call] + bot.tok.encode("hora:") + [bot.tok.end]
    bot.turns.append(("call", "hora:", ids))
    bot.pending = ("hora", "")
    res = bot.resolve(True)
    check("resolve executa a ferramenta", res, res.startswith("ok: "))
    list(bot.step())
    bot.add_user("e agora?")
    prompt = bot._context() + [bot.tok.assistant]
    with torch.no_grad():
        cached = bot._prefill(prompt)
        full = bot.model(torch.tensor([prompt], device=bot.device))[0][0, -1]
    err = (cached - full).abs().max().item()
    check("cache do agente = conversa relida do zero", f"{err:.1e}", err < 1e-4)
    bot.max_reply_tokens = 200
    for i in range(30):  # conversa longa: tem que esquecer o começo sem quebrar
        bot.add_user(f"mensagem número {i} " * 3)
        bot.turns.append(("assistant", "ok", [bot.tok.assistant] + bot.tok.encode("ok") + [bot.tok.end]))
    list(bot.step())
    check("conversa longa cabe no contexto", len(bot._context()), len(bot._context()) <= 256)
    return bot


def test_conselho(bot):
    print("Conselho")
    from conselho import Proposta, _trechos

    def nota(user, call):
        p = Proposta("call", call, [], ["Direta"])
        bot.conselho.revisar(p, user)
        return p.ajuste, p.notas

    aj, notas = nota("quem descobriu o Brasil?", "pesquisar: quem descoriu o Brasil")
    check("revisora pega palavra fora do pedido", notas, aj < 0)
    check("argumento copiado certinho passa", nota("quem descobriu o Brasil?", "pesquisar: quem descobriu o Brasil")[1],
          nota("quem descobriu o Brasil?", "pesquisar: quem descobriu o Brasil")[0] == 0)
    check("argumento salteado é cobrado", nota("busca melhores filmes de 2025", "pesquisar: melhores de 2025")[1],
          nota("busca melhores filmes de 2025", "pesquisar: melhores de 2025")[0] < 0)
    check("papo não vira volume", nota("bem também", "volume: mais")[1], nota("bem também", "volume: mais")[0] < 0)
    check("pedido de volume passa", nota("aumenta o som aí", "volume: mais")[1], nota("aumenta o som aí", "volume: mais")[0] == 0)
    check("gatilho aceita letras trocadas", nota("blouqeia o pc", "bloquear:")[1], nota("blouqeia o pc", "bloquear:")[0] == 0)
    check("lembrete no formato tempo | texto", nota("lembrete pra dormir em 10 min", "lembrete: 10 min | dormir")[1],
          nota("lembrete pra dormir em 10 min", "lembrete: 10 min | dormir")[0] == 0)
    check("ferramenta inventada é descartada", nota("oi", "hackear: tudo")[1], nota("oi", "hackear: tudo")[0] == float("-inf"))
    tr = _trechos("então, quem descobriu o Brasil?")
    check("trechos do pedido sem pontuação nas pontas", tr[-1], "quem descobriu o Brasil" in tr and "então" in tr)

    bot.reset()
    bot.usar_conselho = True
    events = list(bot.chat("oi, tudo bem?"))
    sala = [e for e in events if e["type"] == "conselho"]
    msgs = [e for e in sala if e["ev"] == "msg"]
    quem = {m["de"] for m in msgs}
    check("todos os papéis falam na sala", sorted(quem), {"Direta", "Exploradora", "Cética", "Revisora", "Lia"} <= quem)
    ids = {m["id"] for m in msgs}
    ok = all(m["resp"] in ids for m in msgs if m["resp"]) and all(e["msg"] in ids for e in sala if e["ev"] == "reacao")
    check("respostas e reações apontam para mensagens que existem", len(msgs), ok)
    resumo = [i for i, e in enumerate(events) if e["type"] == "debate"]
    check("debate vem antes da resposta", resumo, resumo and all(e["type"] == "conselho" for e in events[:resumo[0]]))
    check("conselho termina com uma fala", events[-1]["type"], events[-1]["type"] in ("text", "result"))
    bot.add_user("e agora?")
    prompt = bot._context() + [bot.tok.assistant]
    with torch.no_grad():
        cached = bot._prefill(prompt)
        full = bot.model(torch.tensor([prompt], device=bot.device))[0][0, -1]
    err = (cached - full).abs().max().item()
    check("cache continua certo depois do conselho", f"{err:.1e}", err < 1e-4)
    bot.usar_conselho = False
    return bot


def test_agentes(bot):
    print("Agentes")
    from datetime import datetime
    import agentes
    import web
    antiga, T.MEMORY_DIR = T.MEMORY_DIR, pathlib.Path(tempfile.mkdtemp())  # nunca mexe na memória de verdade
    try:
        m = agentes.gravar({"nome": "Música", "ferramentas": ["midia", "volume"],
                            "rotinas": [{"nome": "Noite", "hora": "22:00", "dias": "dias úteis", "passos": ["volume: menos"]}]})
        check("cria agente com rotina", m["id"], m["rotinas"][0]["passos"] == ["volume: menos"])

        def recusa(dados):
            try:
                agentes.gravar(dados)
                return False
            except ValueError:
                return True

        check("recusa agente sem ferramenta", "", recusa({"nome": "X", "ferramentas": []}))
        check("recusa horário inválido", "", recusa({"nome": "Y", "ferramentas": ["hora"],
                                                     "rotinas": [{"nome": "a", "hora": "25:00", "passos": ["hora:"]}]}))
        check("recusa passo fora das ferramentas", "", recusa({"nome": "Z", "ferramentas": ["hora"],
                                                              "rotinas": [{"nome": "a", "hora": "08:00", "passos": ["volume: mais"]}]}))
        check("recusa nome repetido", "", recusa({"nome": "música", "ferramentas": ["hora"]}))
        lista = agentes.carregar()
        qui, sab = datetime(2026, 10, 22, 22, 30), datetime(2026, 10, 24, 22, 30)
        check("rotina roda no horário (dia útil)", "", len(list(agentes.devidas(lista, qui))) == 1)
        check("rotina não roda fora dos dias", "", not list(agentes.devidas(lista, sab)))
        check("rotina não roda antes da hora", "", not list(agentes.devidas(lista, qui.replace(hour=21))))
        check("relatório da rotina", agentes.rodar(m["rotinas"][0]), "volume: menos" in agentes.rodar(m["rotinas"][0]))

        p = agentes.gravar({"nome": "Pesquisa", "ferramentas": ["pesquisar", "hora"]})
        irma = bot.irma(p["ferramentas"])
        check("agente usa o mesmo modelo, com conversa própria", "", irma.model is bot.model and irma.turns is not bot.turns)
        irma.add_user("aumenta o volume")
        irma.turns.append(("call", "volume: mais", [irma.tok.assistant, irma.tok.call] + irma.tok.encode("volume: mais") + [irma.tok.end]))
        irma.pending = ("volume", "mais")
        evs = list(web.repassar(irma, p["id"], {"tool": "volume", "arg": "mais"}))
        check("pedido fora da função vai para quem cuida", evs[0].get("nome"), evs[0]["para"] == m["id"])
        check("repasse não executa nada", irma.turns[-1][0], irma.pending is None and irma.turns[-1][0] == "assistant")
        agentes.apagar(m["id"])
        evs = list(web.repassar(irma, p["id"], {"tool": "volume", "arg": "mais"}))
        check("sem agente para a ferramenta, vai para a Lia", evs[0]["para"], evs[0]["para"] == "lia")
    finally:
        T.MEMORY_DIR = antiga
    return bot


def test_web(bot):
    print("Servidor web")
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    import web
    web.BOT = bot
    bot.reset()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body, token=web.TOKEN, host=None):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-Token": token})
        if host:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, ""

    html = urllib.request.urlopen(base + "/").read().decode()
    check("página traz o token", "", web.TOKEN in html)
    check("API sem token é recusada", post("/api/chat", {"text": "oi"}, token="x")[0], post("/api/chat", {"text": "oi"}, token="x")[0] == 403)
    code = post("/api/chat", {"text": "oi"}, host="evil.example.com")[0]
    check("Host estranho é recusado (DNS rebinding)", code, code == 403)
    code, body = post("/api/chat", {"text": "oi"})
    evs = [json.loads(l) for l in body.splitlines() if l.strip()]
    check("chat responde em streaming", f"{len(evs)} eventos", code == 200 and evs and evs[-1]["type"] in ("text", "confirm", "result"))
    srv.shutdown()


if __name__ == "__main__":
    test_kv_cache()
    test_tokenizer()
    test_tools()
    test_web(test_agentes(test_conselho(test_agent())))
    print(f"\n{'Tudo certo!' if not failures else f'{failures} falha(s).'}")
    sys.exit(1 if failures else 0)
