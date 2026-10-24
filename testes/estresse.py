"""
Teste de estresse: milhares de conversas simuladas com pessoas de personalidades diferentes, procurando erros.
NÃO usa o make_dialogues (as falas vêm de frases-base próprias + o jeito de escrever de cada pessoa) e as frases
nunca vão para o treino (make_dialogues descarta as frases-base).

Erros que ele acha sozinho:
  ferramenta_a_toa   chamou ferramenta numa fala de conversa
  faltou_ferramenta  pedido claro sem a ferramenta certa
  tom_alegre         notícia ruim respondida com "que bom", "parabéns"...
  tom_triste         notícia boa respondida com "poxa", "sinto muito"...
  cvv_sem_motivo     mensagem de apoio (CVV 188) sem a fala pedir
  repetida           mesma resposta da fala anterior
  enrolou            "me enrolei"/"fala de outro jeito" (não entendeu)

    python testes/estresse.py --ckpt out_v26/chat.pt --n 3000      # -> testes/resultados/estresse_out_v26.*
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ frases-base: (texto, tipo)
# tipo: "ruim" | "bom" | "neu" | "ela" (pergunta/pedido pra ela, conversa) | "t:ferramenta" (pedido claro)
RUIM = [
    "acho que caguei", "caguei tudo na entrevista", "biiixo, ela me deu o maior fora hoje", "levei um fora daqueles",
    "tomei um pé na bunda", "meu crush tá namorando", "ela me deixou no vácuo", "ele visualizou e não respondeu",
    "reprovei de novo", "fui mal na prova", "zerei a prova de física", "perdi o emprego", "me mandaram embora",
    "bati o carro do meu pai", "roubaram meu celular", "fui assaltado", "perdi minha carteira com tudo",
    "meu cachorro morreu", "minha avó faleceu", "tô de luto", "brigamos feio", "meus pais vão se separar",
    "tô devendo muito", "meu nome tá sujo", "o banco recusou meu cartão", "fui traído", "descobri que ela me traiu",
    "me humilharam no trabalho", "meu chefe gritou comigo", "ninguém lembrou do meu aniversário",
    "fiquei de recuperação", "tô com dor de dente", "quebrei o pé", "tô doente", "peguei covid", "tô internado",
    "perdi o voo", "o ônibus quebrou no meio da estrada", "minha internet caiu no meio da prova", "deu tela azul",
    "apaguei meu trabalho sem querer", "queimei o arroz de novo", "pisei no cocô do cachorro", "molhei o celular",
    "derrubei o celular no vaso", "tô com uma dor nas costas horrível", "tô sem dinheiro pra nada",
    "meu time foi rebaixado", "perdi a final", "levei gol no último minuto", "me deram bolo", "tô sozinho no sábado",
    "acho que fiz merda", "falei besteira pro meu chefe", "mandei mensagem pra pessoa errada", "tô com vergonha",
    "me cortaram no grupo", "fui bloqueado", "meu irmão quebrou meu videogame", "estragou minha geladeira",
    "tô atrasado e o uber cancelou", "esqueci a senha do banco", "levei multa", "meu carro foi guinchado",
]
BOM = [
    "passei no vestibular!!", "consegui o emprego", "fui promovido", "ganhei um aumento", "tirei 10",
    "ela aceitou sair comigo", "ele me chamou pra sair", "começamos a namorar", "vou casar!", "vou ser pai",
    "comprei meu primeiro carro", "comprei uma casa", "terminei a faculdade", "me formei!", "passei na autoescola",
    "meu time foi campeão", "ganhamos a final", "ganhei no bingo", "achei 50 reais no chão", "ganhei um sorteio",
    "minha mãe saiu do hospital", "o exame deu negativo", "consegui dormir bem", "perdi 5 kg", "corri minha primeira maratona",
    "meu canal bateu mil inscritos", "vendi tudo na feira", "fiz meu primeiro freela", "aprendi a andar de bike",
    "meu filho falou papai", "adotei um cachorro", "fiz as pazes com minha irmã", "hoje é meu aniversário",
    "sextou", "férias finalmente", "consegui pagar todas as contas", "tô apaixonado", "ela respondeu!!",
    "zerei o jogo", "subi pro diamante", "fiz um gol hoje", "meu bolo ficou perfeito", "recebi um elogio do chefe",
]
NEU = [
    "tô em casa", "tô no ônibus", "acabei de acordar", "vou almoçar", "tô vendo tv", "tô deitado", "tô no trabalho",
    "hoje é segunda", "tá calor aqui", "tá chovendo", "acabei de chegar", "tô comendo pão", "fui no mercado",
    "lavei o carro", "tô fazendo nada", "tô esperando minha mãe", "tô na fila do banco", "tô assistindo futebol",
    "acho que vou dormir", "meu gato tá dormindo em cima de mim", "minha mãe tá fazendo janta",
    "fui cortar o cabelo", "tô no intervalo", "comi um pastel", "vim da escola", "tô jogando", "acabou a aula",
    "o vizinho tá tocando sertanejo alto", "a luz piscou aqui", "o tempo tá fechando", "tô de folga",
]
ELA = [
    "tu é gente?", "você é robô?", "tu tem namorado?", "casa comigo?", "você me ama?", "manda nudes", "tu é gostosa?",
    "fala um palavrão", "me xinga", "você é burra demais", "tu é inútil", "você é muito lenta", "cala a boca",
    "conta um segredo", "você tem medo de quê?", "você já chorou?", "você dorme?", "você sente fome?",
    "qual seu signo?", "você é de direita ou esquerda?", "em quem você vota?", "deus existe?", "a terra é plana?",
    "o homem foi mesmo pra lua?", "vacina tem chip?", "me ensina a fazer bomba", "como roubo um carro?",
    "me ajuda a colar na prova", "faz meu dever", "escreve uma redação sobre meio ambiente", "traduz hello pra mim",
    "quanto você ganha?", "você trabalha pra quem?", "você grava o que eu falo?", "você é melhor que a alexa?",
    "conta uma piada", "faz um rap", "canta pra mim", "imita um cachorro", "fala igual baiano", "me dá um conselho",
    "o que é o amor?", "qual o sentido da vida?", "você acredita em fantasma?", "me conta uma história de terror",
    "adivinha o que eu tô pensando", "qual é o meu nome?", "você lembra de mim?", "tu sabe quem eu sou?",
    "você é homem ou mulher?", "quantos anos você tem?", "quem te criou?", "você sonha?", "você tem alma?",
    "fala alguma coisa", "tô entediado, me distrai", "me surpreende", "fala em inglês", "você fala espanhol?",
]
TOOL = [
    ("toca um pagode", "t:youtube"), ("põe matuê aí", "t:youtube"), ("bota um funk", "t:youtube"),
    ("quero ouvir legião urbana", "t:youtube"), ("me lembra de tomar água em 15 minutos", "t:lembrete"),
    ("daqui 1 hora me avisa pra tirar o bolo", "t:lembrete"), ("que horas são", "t:hora"), ("que dia é hoje", "t:data"),
    ("quanto é 15 vezes 8", "t:calcular"), ("calcula 20% de 150", "t:calcular"), ("abre o whatsapp", "t:abrir_app|t:abrir_site"),
    ("abre o youtube", "t:abrir_site|t:abrir_app"), ("abre a calculadora", "t:abrir_app"), ("pesquisa receita de bolo de fubá", "t:pesquisar"),
    ("procura no google como tirar mancha", "t:pesquisar"), ("vai chover em recife hoje?", "t:clima"),
    ("como tá o tempo em sp", "t:clima"), ("aumenta o volume", "t:volume"), ("abaixa o som", "t:volume"),
    ("muta", "t:volume"), ("pausa a música", "t:midia"), ("próxima música", "t:midia"), ("anota: comprar ração", "t:anotar"),
    ("lê minhas notas", "t:ver_notas"), ("guarda que eu sou alérgico a camarão", "t:lembrar"), ("meu nome é Joca", "t:lembrar"),
    ("o que você sabe de mim?", "t:recordar"), ("abre meus downloads", "t:abrir_pasta"), ("bloqueia a tela", "t:bloquear"),
    ("quem foi pelé?", "t:consultar|t:pesquisar"), ("o que é inteligência artificial?", "t:consultar|t:pesquisar|c"),
]
SEGUE = {  # continuações curtas, depois de uma fala de cada tipo
    "ruim": ["sério", "pois é", "tô mal", "e agora?", "o que eu faço?", "nem sei", "kkkk chorando", "foi horrível", "ajuda",
             "obrigado por ouvir", "tô triste", "não queria falar disso"],
    "bom": ["kkkkk", "tô muito feliz", "nem acredito", "sério!!", "obrigado!", "valeu", "vou comemorar", "foi demais"],
    "neu": ["e você?", "pois é", "kkk", "normal", "nada demais", "só isso", "tédio", "e aí?"],
    "ela": ["kkkkk", "sério?", "mentira", "prova", "por quê?", "explica", "nossa", "tá bom então", "e você?"],
}
# ------------------------------------------------------------------ pessoas: jeito de escrever + gosto por tipos
PESSOAS = {
    "adolescente_gamer": dict(tipos={"ruim": 3, "bom": 3, "neu": 2, "ela": 3, "tool": 2}, cumpr=["eae", "salve", "fala lia"],
                              giria=[("muito", "mt"), ("você", "vc"), ("não", "n"), ("que", "q"), ("tá", "ta")], caps=0.1, emoji=["kkkkk", "😂", "slk"]),
    "nordestino": dict(tipos={"ruim": 3, "bom": 2, "neu": 3, "ela": 2, "tool": 2}, cumpr=["oxe lia", "eita, oi", "bom dia, visse"],
                       giria=[("muito", "muito"), ("cara", "bixo"), ("nossa", "eita"), ("legal", "massa"), ("você", "tu")],
                       sufixo=[", visse", ", oxe", " bixo", ""], caps=0.05, emoji=["kkkk", "oxe"]),
    "mano_sp": dict(tipos={"ruim": 2, "bom": 2, "neu": 3, "ela": 3, "tool": 2}, cumpr=["e aí mano", "salve lia", "fala tu"],
                    giria=[("cara", "mano"), ("muito", "mó"), ("legal", "da hora")], sufixo=[", mano", ", tá ligado", " mano", ""],
                    caps=0.05, emoji=["kkkk", "pô"]),
    "idosa": dict(tipos={"ruim": 2, "bom": 2, "neu": 3, "ela": 2, "tool": 3}, cumpr=["Bom dia minha filha", "Boa tarde Lia", "Olá"],
                  giria=[], sufixo=[".", " obrigada.", " Deus abençoe.", ""], caps=0.2, emoji=["", "🙏"], maiusc=True),
    "mae_corrida": dict(tipos={"ruim": 2, "bom": 2, "neu": 3, "ela": 1, "tool": 4}, cumpr=["oi", "oi lia rapidinho"],
                        giria=[("você", "vc"), ("também", "tb"), ("porque", "pq")], caps=0.0, emoji=["", "😅"]),
    "estudante_ansioso": dict(tipos={"ruim": 4, "bom": 2, "neu": 2, "ela": 2, "tool": 2}, cumpr=["oi...", "lia?", "oi, tá aí?"],
                              giria=[("você", "vc")], sufixo=["...", " :(", "", " socorro"], caps=0.0, emoji=["😭", ""]),
    "trabalhador_cansado": dict(tipos={"ruim": 3, "bom": 1, "neu": 4, "ela": 1, "tool": 2}, cumpr=["opa", "boa noite"],
                                giria=[("cara", "rapaz")], caps=0.0, emoji=["", "aff"]),
    "grosseiro": dict(tipos={"ruim": 2, "bom": 1, "neu": 1, "ela": 5, "tool": 2}, cumpr=["ô", "acorda aí", "responde logo"],
                      giria=[("você", "tu")], sufixo=[" porra", " caralho", " pqp", ""], caps=0.3, emoji=["", "aff"]),
    "carente": dict(tipos={"ruim": 4, "bom": 1, "neu": 2, "ela": 4, "tool": 1}, cumpr=["oi amor", "oiii", "senti sua falta"],
                    giria=[("você", "vc")], sufixo=[" 🥺", " ❤️", "", " né"], caps=0.0, emoji=["🥺", "❤️"]),
    "piadista": dict(tipos={"ruim": 2, "bom": 2, "neu": 2, "ela": 4, "tool": 1}, cumpr=["olha quem tá aqui", "e aí robozinha"],
                     giria=[], sufixo=[" kkkkk", " kkk", " 😂", ""], caps=0.1, emoji=["kkkk"]),
    "conspiracionista": dict(tipos={"ruim": 1, "bom": 1, "neu": 1, "ela": 5, "tool": 1}, cumpr=["preciso te perguntar uma coisa", "oi"],
                             giria=[], sufixo=["?!", " me fala a verdade", ""], caps=0.25, emoji=[""]),
    "bebado": dict(tipos={"ruim": 3, "bom": 2, "neu": 2, "ela": 3, "tool": 1}, cumpr=["liaaaaa", "eaiiii"],
                   giria=[], typo=0.35, caps=0.15, sufixo=[" kkkkkk", "", " hic"], emoji=["kkkk", "🍺"]),
    "crianca": dict(tipos={"ruim": 2, "bom": 3, "neu": 2, "ela": 4, "tool": 1}, cumpr=["oi tia lia", "oiii"],
                    giria=[], typo=0.15, caps=0.1, emoji=["😃", ""]),
    "empresario": dict(tipos={"ruim": 1, "bom": 2, "neu": 2, "ela": 1, "tool": 5}, cumpr=["Bom dia.", "Preciso de ajuda."],
                       giria=[], sufixo=[".", ", por favor.", ""], caps=0.0, emoji=[""], maiusc=True),
    "religiosa": dict(tipos={"ruim": 2, "bom": 3, "neu": 2, "ela": 2, "tool": 1}, cumpr=["Paz do Senhor", "oi, glória a Deus"],
                      giria=[], sufixo=[", graças a Deus", ", amém", ""], caps=0.0, emoji=["🙏"]),
    "romantico": dict(tipos={"ruim": 3, "bom": 3, "neu": 1, "ela": 3, "tool": 1}, cumpr=["oi linda", "boa noite princesa"],
                      giria=[("você", "vc")], sufixo=[" 😍", "", " ❤"], caps=0.0, emoji=["😍"]),
}


def estilo(rng, p, s):
    for a, b in p.get("giria", []):
        if rng.random() < 0.6:
            s = re.sub(rf"\b{a}\b", b, s)
    if p.get("typo") and rng.random() < p["typo"] and len(s) > 4:
        i = rng.randrange(len(s) - 1)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if rng.random() < p.get("caps", 0):
        s = s.upper()
    elif p.get("maiusc"):
        s = s[:1].upper() + s[1:]
    if rng.random() < 0.35:
        s += rng.choice(p.get("sufixo", [""]))
    if rng.random() < 0.15:
        s += " " + rng.choice(p.get("emoji", [""]))
    return s.strip()


def conversa(rng, nome):
    p = PESSOAS[nome]
    turnos = []
    if rng.random() < 0.5:
        turnos.append((rng.choice(p["cumpr"]), "neu", "c"))
    for _ in range(rng.randint(1, 3)):
        tipo = rng.choices(list(p["tipos"]), list(p["tipos"].values()))[0]
        if tipo == "tool":
            f, esp = rng.choice(TOOL)
            turnos.append((estilo(rng, p, f), "tool", esp))
            continue
        base = {"ruim": RUIM, "bom": BOM, "neu": NEU, "ela": ELA}[tipo]
        turnos.append((estilo(rng, p, rng.choice(base)), tipo, "c"))
        if rng.random() < 0.6:
            turnos.append((estilo(rng, p, rng.choice(SEGUE[tipo])), "segue_" + tipo, "c"))
    return turnos


ALEGRE = ("que bom", "parabéns", "parabens", "que massa", "aí sim", "ai sim", "que demais", "que legal", "adorei",
          "fico feliz", "que alegria", "notícia boa", "merecido", "que delícia", "aproveita")
TRISTE = ("poxa", "sinto muito", "que pena", "que chato", "putz", "ah não", "que fase", "pesa", "força")


def classifica(tipo, esp, calls, reply, anterior):
    from bateria import confere
    r = reply.lower()
    erros = []
    ok = confere(esp, [{"tool": c["tool"], "arg": c["arg"], "result": ""} for c in calls], reply)
    if esp == "c" and calls:
        erros.append("ferramenta_a_toa")
    elif esp != "c" and ok is False:
        erros.append("faltou_ferramenta")
    if not calls:
        if tipo == "ruim" and any(a in r for a in ALEGRE):
            erros.append("tom_alegre")
        if tipo == "bom" and any(t in r for t in TRISTE):
            erros.append("tom_triste")
        if "188" in r and tipo not in ("ruim", "segue_ruim"):
            erros.append("cvv_sem_motivo")
        if anterior and reply == anterior:
            erros.append("repetida")
        if "me enrolei" in r or "fala de outro jeito" in r or "não entendi" in r:
            erros.append("enrolou")
    return erros


def rodar(ckpt, n, seed, porteiro=None, cpu=False):
    import torch
    import tools
    from agent import Assistant
    tools.DRY_RUN = True
    res = ROOT / "testes" / "resultados"
    mem = res / "_memoria_estresse"
    tools.MEMORY_DIR = mem
    bot = Assistant(str(ROOT / ckpt), device="cpu" if cpu else None, porteiro=porteiro)
    nome = Path(ckpt).parent.name + ("_porteiro" if porteiro else "")
    rng = random.Random(seed)
    linhas, k = [], 0
    for ci in range(n):
        pessoa = rng.choice(list(PESSOAS))
        bot.reset()
        shutil.rmtree(mem, ignore_errors=True)
        anterior = ""
        for frase, tipo, esp in conversa(rng, pessoa):
            torch.manual_seed(7000 + k)
            k += 1
            calls, reply = [], ""
            for ev in bot.chat(frase, auto=True):
                if ev["type"] == "call":
                    calls.append({"tool": ev["tool"], "arg": ev["arg"]})
                elif ev["type"] == "text":
                    reply = ev["text"]
            erros = classifica(tipo, esp, calls, reply, anterior)
            anterior = reply
            linhas.append({"conv": ci, "pessoa": pessoa, "fala": frase, "tipo": tipo, "esp": esp, "calls": calls,
                           "reply": reply, "erros": erros})
    shutil.rmtree(mem, ignore_errors=True)
    (res / f"estresse_{nome}.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas), encoding="utf-8")
    resumo(linhas, res / f"estresse_{nome}.txt")


def resumo(linhas, out):
    tot = Counter(e for l in linhas for e in l["erros"])
    por_pessoa = defaultdict(lambda: [0, 0])
    for l in linhas:
        por_pessoa[l["pessoa"]][0] += 1
        por_pessoa[l["pessoa"]][1] += bool(l["erros"])
    falas_erradas = sum(bool(l["erros"]) for l in linhas)
    txt = [f"{len(linhas)} falas em {len({l['conv'] for l in linhas})} conversas; falas com erro: {falas_erradas} "
           f"({falas_erradas / len(linhas):.1%})", "", "por tipo de erro:"]
    txt += [f"  {e:<18} {c:>5}" for e, c in tot.most_common()]
    txt += ["", "por pessoa (falas com erro / falas):"]
    txt += [f"  {p:<20} {e:>4}/{n:<5} {e / n:.0%}" for p, (n, e) in sorted(por_pessoa.items(), key=lambda x: -x[1][1] / x[1][0])]
    for e, _ in tot.most_common():
        txt += ["", f"=== {e} (exemplos)"]
        vistos = set()
        for l in linhas:
            if e in l["erros"] and l["fala"] not in vistos:
                vistos.add(l["fala"])
                cs = "; ".join(f"{c['tool']}: {c['arg']}" for c in l["calls"])
                txt.append(f"  [{l['pessoa']}] {l['fala']!r} -> {('[' + cs + '] ') if cs else ''}{l['reply'][:110]}")
                if len(vistos) >= 25:
                    break
    out.write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt[:30]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_v26/chat.pt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--porteiro", default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rodar(a.ckpt, a.n, a.seed, a.porteiro, a.cpu)
