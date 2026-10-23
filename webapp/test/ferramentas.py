"""
Respostas "certas" das ferramentas (tools.py) para comparar com webapp/ferramentas.js.

    python webapp/test/ferramentas.py     # -> webapp/test/ferramentas.json
    node webapp/test/ferramentas.js
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools  # noqa: E402
from make_dialogues import DURATIONS, Gen, eval_cases  # noqa: E402


HORARIOS = ["às 15h", "as 15:30", "amanhã", "amanhã às 9h", "amanha 9h", "hoje às 20h", "meio-dia", "meia noite",
            "às 3 da tarde", "às 8h", "daqui a 10 minutos", "1h30", "2 horas", "10 min", "em 15 horas", "às 25h",
            "amanhã às 7:45", "às 7h", "até as 18h", "hoje 22h", "às 0h30", "3 dias"]
ANCORAS = [("15h", "me avisa às 15h"), ("pagar a conta", "me lembra amanhã de pagar a conta"),
           ("18h | ligar pro joão", "me lembra de ligar pro joão às 18h"), ("10 minutos | beber água", "me lembra de beber água em 10 minutos"),
           ("às 9h | academia", "amanhã às 9h me lembra da academia"), ("20 minutos | x", "daqui a 20 minutos me avisa"),
           ("meio dia | almoço", "me avisa meio dia do almoço"), ("2 horas | remédio", "me lembra do remédio às 14:30")]


# Primeiros parágrafos reais (baixar_wiki.py) e casos difíceis para a limpeza do resumo
WIKI = ["O efeito estufa (português brasileiro) ou efeito de estufa (português europeu) é um processo. Outra frase.",
        "Camberra (pronunciado: [kɐ̃ˈbɛʁɐ]; em inglês:  Canberra) é a capital da Austrália. Com 450 mil habitantes, é grande.",
        "Alberto Santos Dumont (Palmira, 20 de julho de 1873 – Guarujá, 23 de julho de 1932) foi um aeronauta.",
        "O Dr. João (n. 1950) foi médico. Trabalhou em S. Paulo. Morreu em 44 a.C. Deixou obra.", "", "   ", "Sem ponto final",
        "A (B (C) D) E é x. 😀 Emoji conta como um caractere " + "palavra " * 60, "Frase curta. " * 5,
        "Um texto longo, " * 30 + "fim.", "x" * 400, "Rio (em tupi: (y) rio) é ok.\nSegundo parágrafo."]
_artigos = ROOT / "data_wiki" / "artigos.jsonl"
if _artigos.exists():
    WIKI += [json.loads(line)["texto"] for line in open(_artigos, encoding="utf-8")][:3000]


def dur_em(h, ag):
    import datetime
    hh, mm = map(int, ag.split(":"))
    try:
        return tools.parse_duracao(h, datetime.datetime(2026, 10, 17, hh, mm))
    except ValueError:
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = Gen(random.Random(3), split="train")
    exprs = [c["content"].partition(":")[2].strip() for c in eval_cases(40) if c["kind"] == "calc"]
    exprs += [g.rand_expr() for _ in range(300)] + [g.rand_expr_wide() for _ in range(300)]
    exprs += ["2^10", "raiz de 144", "15% de 200", "10/3", "7 // 2", "2,5 vezes 4", "1e3", "100000*3", "-2**2", "0/0",
              "5/0", "sqrt(-1)", "round(2.5)", "round(3.14159, 2)", "2**200", "quanto é 3 x 4?", "√81", "10 % 3", "-7 % 3",
              "1/3", "2/3*3", "0.1+0.2", "123456789*1000", "12345.678*1000", "(1+2)*(3+4)", "pi*2", "abc", "2 3", "", "3 +",
              "log de 1000", "ln(e)", "seno de 0", "2 elevado a 10", "5 ao quadrado", "9 ao cubo", "50 por cento de 80",
              "1000000000000000*10", "0,0001*0,5", "99999*99999", "7 dividido por 2", "8 sobre 2", "1.5e-07*1",
              "metade de 350", "a metade de 7", "o dobro de 21", "o triplo de 15", "um terço de 90", "um quarto de 100",
              "metade de 2,5", "quanto é metade de 1000?", "dobro de 0", "um terço de 10", "o resultado de 185 - 45 * 18",
              "qual o resultado de 7 x 8", "qual é a conta de 3+3", "o resultado de raiz de 144"]
    durs = list(DURATIONS) + [g.rand_duration() for _ in range(200)]
    durs += ["meia hora", "1h30", "2 h e 15", "uma hora e meia", "abc", "0 minutos", "10 dias", "90", "1,5 horas",
             "dez minutos e trinta segundos", "30 seg", "3 min", "duas horas", "8 dias"]

    def calc(e):
        return tools.calcular(e)

    def dur(d):
        try:
            return tools.parse_duracao(d)
        except ValueError:
            return None

    sites = list(tools.SITES) + ["o site do youtube", "youtub", "githb", "exemplo.com.br", "https://x.com", "insta",
                                 "spotify", "netflx", "a pagina do g1", "wikipedia.org", "nada a ver", "mercadolivre",
                                 "google tradutor", "o gmail", "Instagram", "TikTok", "disney +", "site de receitas"]
    fatos = ["meu nome é Matheus", "eu moro em São Paulo", "eu faço jiu jitsu", "eu sou alérgico a camarão",
             "meu cachorro se chama Thor", "eu trabalho com programação", "meu time é o Flamengo", "eu tenho 23 anos"]
    perguntas = ["nome", "moro", "faço", "alérgico", "alergia", "cachorro", "trabalho", "time", "anos", "", "você sabe quem eu sou",
                 "esporte", "programador", "cidade", "qual o nome do meu pet"]
    textos = ["Olá, Mundo!", "  ação  É  ", "e-mail: a_b.c+d#e", "1,5 horas", "Ça va?", "x² ½", "日本", "😀 ok"]
    out = {
        "calcular": [[e, calc(e)] for e in exprs],
        "duracao": [[d, dur(d)] for d in durs],
        "formatar_duracao": [[s, tools.formatar_duracao(s)] for s in [1, 59, 60, 61, 119, 120, 1800, 3599, 3600, 3661, 5400, 7200, 86399]],
        "site_url": [[q, tools._site_url(q)] for q in sites],
        "recordar": [[q, None] for q in perguntas],
        "fatos": fatos,
        "norm": [[t, tools.norm(t)] for t in textos],
        # horário marcado, com "agora" fixo em 17/10/2026 07:00 e 21:30
        "horario": [[f"{ag}|{h}", dur_em(h, ag)] for ag in ("07:00", "21:30") for h in HORARIOS],
        "ancorar": [[f"{arg}|{fala}", tools.ancorar_lembrete(arg, fala)] for arg, fala in ANCORAS],
        "resumo_wiki": [[t, tools.resumo_wiki(t)] for t in WIKI],
        "assunto_wiki": [[a, tools.assunto_wiki(a)] for a in ["o efeito estufa?", "Pelé", " um buraco negro ", "as Cataratas",
                                                              "Osasco", "a", "uma", "?", "Ana Maria Braga!"]],
    }
    tools.MEMORY_DIR = ROOT / "memoria_simulada" / "teste_web"
    tools._save_json("fatos.json", fatos)
    out["recordar"] = [[q, tools.recordar(q)] for q in perguntas]
    path = Path(__file__).with_name("ferramentas.json")
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{path}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
