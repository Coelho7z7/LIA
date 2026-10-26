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
COMEMORA_FORTE = ("que bom", "parabens", "ai sim", "que demais", "que massa", "fico feliz", "que alegria", "noticia boa",
                  "merecido", "uhul", "comemora")
LAMENTA_FORTE = ("poxa", "sinto muito", "que pena", "que chato", "ah nao")
NAO_ENTENDI = ("me enrolei", "fala de outro jeito", "nao entendi", "escreveu sem querer")
SEGURA = {"ruim": ["Poxa, sinto muito. Quer me contar como foi?", "Ah não... Que chato isso. Tô aqui se quiser desabafar."],
          "bom": ["Que notícia boa! Me conta mais!", "Aí sim! Fico feliz por você."],
          "confusa": ["Hmm, me conta um pouco mais?", "Entendi! E aí?"]}


# O tom continua na fala seguinte: "tomei um fora da crush" / "ela disse que só me vê como amigo" -> "Parabéns!"
# (a segunda frase sozinha parece neutra). Só não continua se a pessoa virou o assunto ("mas já superei").
VIROU = re.compile(r"\b(superei|ja passou|to (bem|melhor|de boa|tranquil\w*)|tudo bem agora|ficou tudo bem|deu certo|"
                   r"resolvi|resolveu|pelo menos|ainda bem|mudando de assunto|outra coisa)")


def revisar_fala(fala, resposta, antes=""):
    """O problema da resposta à fala do usuário ("ruim", "bom", "confusa") ou None se está ok.
    `antes` = a fala anterior da pessoa: se a atual é neutra, o tom dela continua valendo."""
    f, r = tools.norm(fala), tools.norm(resposta)
    ruim, bom = bool(RUIM.search(f)), bool(BOM.search(f))
    comemora, lamenta = COMEMORA, LAMENTA
    if antes and not ruim and not bom and not VIROU.search(f):
        a = tools.norm(antes)
        ruim, bom = bool(RUIM.search(a)), bool(BOM.search(a))
        comemora, lamenta = COMEMORA_FORTE, LAMENTA_FORTE  # o assunto pode ter mudado: só barra o que é certeza
    if ruim and not bom and any(c in r for c in comemora):
        return "ruim"
    if bom and not ruim and any(c in r for c in lamenta):
        return "bom"
    # frase de verdade (3+ palavras): letras batidas ("asdfgh", "lirdum pax") continuam podendo levar "não entendi"
    if any(c in r for c in NAO_ENTENDI) and len(re.findall(r"\b[a-z]*[aeiou][a-z]*\b", f)) >= 3:
        return "confusa"
    return None


MOTIVO = {"ruim": "a fala é uma notícia ruim e a resposta comemora", "bom": "a fala é uma notícia boa e a resposta lamenta",
          "confusa": "a frase é normal e a resposta diz que não entendeu"}
