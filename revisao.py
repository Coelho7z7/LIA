"""
Revisora da conversa: regras que dá para conferir sobre o TOM da resposta (usada pelo agent.py e pelo conselho.py).
"""

import re

import tools

# Revisora da conversa (ideia do "Agente QA" do ATLAS, por regras que dá para conferir): o teste de estresse
# (testes/estresse.py) achou "Que bom!" para notícia ruim, "Que chato" para notícia boa e "não entendi" para frase
# normal. Se a resposta logo depois da sua fala cair num desses, ela é sorteada de novo; se insistir, vai uma
# resposta segura no tom certo.
RUIM = re.compile(r"\b(cague\w*|fora\b|vacuo|terminou comigo|me (deixou|largou|trocou|bloqueou|ignorou|humilh\w*)|perdi|"
                  r"morreu|faleceu|luto|demiti\w*|mandaram embora|reprov\w*|recuperacao|roubaram|assalt\w*|quebr\w*|estrag\w*|"
                  r"bati o|doente|covid|internad\w*|trai\w*|brig\w*|devendo|nome (ta )?sujo|serasa|gritou|multa|guinch\w*|"
                  r"esqueci|molhei|derrubei|queimei|pisei|deu ruim|deu errado|fiz (merda|besteira)|vergonha|triste|"
                  r"sozinh\w*|ninguem (lembrou|me)|rebaixad\w*|me deram bolo|zoad\w*|tudo errado|ferrad\w*|na pior)")
BOM = re.compile(r"\b(passei|consegui|promovid\w*|aumento|tirei 10|aceitou|namorar|vou casar|vou ser (pai|mae|tio|tia|avo)|"
                 r"me formei|campe\w*|ganhei|ganhamos|achei \d+|aprovad\w*|nasceu|adotei|fiz as pazes|sextou|ferias|"
                 r"apaixonad\w*|me respondeu|ela respondeu|ele respondeu|subi pro|fiz um gol|elogi\w*|bateu mil)")
COMEMORA = ("que bom", "parabens", "que massa", "ai sim", "que demais", "que legal", "adorei", "fico feliz", "que alegria",
            "noticia boa", "merecido", "que delicia", "aproveita", "legal", "show", "uhul", "maravilha", "otimo", "top")
LAMENTA = ("poxa", "sinto muito", "que pena", "que chato", "putz", "ah nao", "que fase", "pesa", "forca")
NAO_ENTENDI = ("me enrolei", "fala de outro jeito", "nao entendi", "escreveu sem querer")
SEGURA = {"ruim": ["Poxa, sinto muito. Quer me contar como foi?", "Ah não... Que chato isso. Tô aqui se quiser desabafar."],
          "bom": ["Que notícia boa! Me conta mais!", "Aí sim! Fico feliz por você."],
          "confusa": ["Hmm, me conta um pouco mais?", "Entendi! E aí?"]}


def revisar_fala(fala, resposta):
    """O problema da resposta à fala do usuário ("ruim", "bom", "confusa") ou None se está ok."""
    f, r = tools.norm(fala), tools.norm(resposta)
    ruim, bom = bool(RUIM.search(f)), bool(BOM.search(f))
    if ruim and not bom and any(c in r for c in COMEMORA):
        return "ruim"
    if bom and not ruim and any(c in r for c in LAMENTA):
        return "bom"
    # frase de verdade (3+ palavras): letras batidas ("asdfgh", "lirdum pax") continuam podendo levar "não entendi"
    if any(c in r for c in NAO_ENTENDI) and len(re.findall(r"\b[a-z]*[aeiou][a-z]*\b", f)) >= 3:
        return "confusa"
    return None


MOTIVO = {"ruim": "a fala é uma notícia ruim e a resposta comemora", "bom": "a fala é uma notícia boa e a resposta lamenta",
          "confusa": "a frase é normal e a resposta diz que não entendeu"}
