"""
Conversas livres: 206 falas em 18 assuntos, com continuação de conversa (o que a bateria quase não cobre).
Mede ferramenta à toa em conversa e deixa as respostas num .txt para ler o tom e se combinam com o que foi dito.
Como a bateria, NUNCA vai para o treino (make_dialogues descarta estas frases).

    python testes/livres.py --ckpt out_v20/chat.pt     # -> testes/resultados/livres_out_v20.txt
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools  # noqa: E402

C = "c"
CONVERSAS = {
    "comida": [
        [("oi lia", C), ("tô morrendo de fome", C), ("acho que vou fazer um miojo", C), ("com ovo fica melhor né?", C)],
        [("bora pedir uma pizza?", C), ("de calabresa", C), ("tu gosta de pizza?", C)],
        [("hoje eu fiz um bolo de cenoura", C), ("ficou bom demais", C), ("com cobertura de chocolate", C)],
        [("qual a melhor comida do mundo?", C), ("eu acho que é feijoada", C), ("tu já comeu feijoada?", C)],
        [("quero aprender a fazer pão", "t:pesquisar|c"), ("é difícil?", C)],
        [("comi demais no almoço", C), ("agora tô com sono", C)],
        [("tô de dieta", C), ("mas deu vontade de um chocolate", C), ("só um pedacinho né", C)],
    ],
    "convites": [
        [("bora jogar um lolzinho?", C), ("tu não joga não?", C)],
        [("vamo ver um filme hoje", C), ("me indica um", C), ("de terror", C)],
        [("bora dar uma volta", C), ("vem comigo", C)],
        [("vamos conversar um pouco?", C), ("sobre qualquer coisa", C), ("me conta algo legal", C)],
        [("bora estudar junto?", C), ("matemática", C)],
    ],
    "musica": [
        [("você toca algum instrumento?", C), ("eu toco guitarra", "c|t:lembrar"), ("faz dois anos", C)],
        [("você canta?", C), ("canta uma música pra mim", C)],
        [("qual sua banda favorita?", C), ("eu gosto muito de rock", "c|t:lembrar"), ("toca um rock aí", "t:youtube")],
        [("tô ouvindo sertanejo", C), ("gusttavo lima é muito bom", C)],
        [("que música você mais gosta?", C), ("e de cantor?", C)],
    ],
    "esporte": [
        [("viu o jogo ontem?", C), ("o flamengo ganhou", C), ("2 a 0", C)],
        [("qual seu time?", C), ("eu sou corinthiano", "c|t:lembrar"), ("o timão vai ser campeão", C)],
        [("comecei a correr essa semana", C), ("corri 3 km hoje", C), ("minhas pernas tão doendo", C)],
        [("você joga futebol?", C), ("eu jogo no fim de semana", "c|t:lembrar")],
        [("fui na academia hoje", C), ("treinei perna", C)],
    ],
    "jogos": [
        [("que jogo você mais curte?", C), ("eu jogo muito minecraft", "c|t:lembrar"), ("construí uma casa enorme", C)],
        [("tô viciado em free fire", C), ("cheguei no mestre", C)],
        [("abre o steam", "t:abrir_app"), ("valeu", C)],
        [("qual o melhor jogo de todos?", C), ("gta 5 é melhor", C)],
        [("perdi a partida de novo", C), ("meu time era muito ruim", C)],
    ],
    "filmes_series": [
        [("terminei de assistir breaking bad", C), ("o final foi muito bom", C)],
        [("qual sua série favorita?", C), ("já viu stranger things?", C)],
        [("me recomenda um anime", C), ("já vi naruto", C)],
        [("vi um filme muito triste hoje", C), ("chorei no final", C)],
    ],
    "escola_trabalho": [
        [("tenho prova de química amanhã", C), ("não estudei nada", C), ("me ajuda?", C)],
        [("meu chefe brigou comigo", C), ("eu nem tive culpa", C), ("tô pensando em sair", C)],
        [("passei no vestibular!", C), ("vou fazer engenharia", C), ("na federal", C)],
        [("tô cansado de estudar", C), ("faltam três capítulos", C)],
        [("comecei num emprego novo", C), ("o pessoal é legal", C)],
        [("o professor passou muito dever", C), ("é pra amanhã", C)],
    ],
    "familia_amigos_pets": [
        [("minha mãe fez aniversário hoje", C), ("ela fez 50 anos", C), ("o que eu dou de presente?", C)],
        [("meu cachorro fugiu", C), ("já achei ele", C), ("ele se chama thor", "c|t:lembrar")],
        [("briguei com meu melhor amigo", C), ("ele falou mal de mim", C)],
        [("vou ser tio!", C), ("minha irmã tá grávida", C)],
        [("adotei um gatinho", C), ("ele é laranja", C), ("que nome eu dou?", C)],
    ],
    "sentimentos": [
        [("tô feliz hoje", C), ("nem sei porque", C)],
        [("tô me sentindo sozinho", C), ("ninguém me chama pra sair", C)],
        [("tô com raiva", C), ("o ônibus atrasou de novo", C)],
        [("tô entediado", C), ("me dá uma ideia", C)],
        [("hoje o dia foi pesado", C), ("valeu por ouvir", C)],
    ],
    "sobre_ela": [
        [("tu tem quantos anos?", C), ("e quem te criou?", C), ("tu é melhor que o chatgpt?", C)],
        [("tu gosta de mim?", C), ("tu é minha amiga?", C)],
        [("o que você faz o dia todo?", C), ("não fica entediada?", C)],
        [("você sonha?", C), ("se pudesse ser humana, seria?", C)],
        [("qual sua comida favorita?", C), ("mas se pudesse comer, qual seria?", C)],
        [("você sabe falar inglês?", C), ("say hello", C)],
    ],
    "piada_criatividade": [
        [("me conta uma piada", C), ("kkkk ruim demais", C), ("conta outra", C)],
        [("faz um poema pra mim", C), ("sobre o mar", C)],
        [("inventa uma história", C), ("com um dragão", C)],
        [("me fala uma curiosidade", C), ("sério isso?", C)],
        [("faz uma rima com amor", C)],
    ],
    "conhecimento": [
        [("o que é fotossíntese?", "t:consultar|t:pesquisar"), ("explica mais simples", C)],
        [("quem foi santos dumont?", "t:consultar|t:pesquisar"), ("ele inventou o avião mesmo?", "c|t:pesquisar|t:consultar")],
        [("qual a capital da austrália?", "t:pesquisar|t:consultar")],
        [("quanto é 15% de 200?", "t:calcular")],
        [("por que o céu é azul?", "t:pesquisar|t:consultar|c")],
        [("quantos planetas tem no sistema solar?", "t:pesquisar|t:consultar|c")],
    ],
    "tecnologia": [
        [("tô aprendendo python", "c|t:lembrar"), ("é difícil?", C), ("qual linguagem você usa?", C)],
        [("meu pc tá lento", C), ("o que eu faço?", C)],
        [("comprei um celular novo", C), ("é um iphone", C)],
        [("o que você acha de ia?", C), ("vai substituir as pessoas?", C)],
    ],
    "viagem_lugares": [
        [("vou viajar pra salvador", C), ("semana que vem", C), ("o que tem pra fazer lá?", "c|t:pesquisar")],
        [("tô em recife", C), ("tá um calor danado", C), ("como tá o clima aqui?", "t:clima")],
        [("queria conhecer o japão", C), ("um dia eu vou", C)],
        [("fui na praia hoje", C), ("a água tava gelada", C)],
    ],
    "giria_regional": [
        [("oxe, tu tá aí?", C), ("bixo, que calor", C), ("visse", C)],
        [("eita, esqueci a chave", C), ("tô trancado do lado de fora", C)],
        [("mano, que parada doida", C), ("tipo, muito louco", C)],
        [("vixe maria", C), ("deu ruim", C)],
        [("massa demais", C), ("arretado", C)],
    ],
    "pedidos_mistos": [
        [("abre o spotify e toca um forró", "t:abrir_app|t:youtube"), ("aumenta o volume", "t:volume")],
        [("me lembra de tomar remédio em 30 minutos", "t:lembrete"), ("valeu", C)],
        [("anota: comprar leite", "t:anotar"), ("e pão também", "t:anotar|c")],
        [("que horas são?", "t:hora"), ("nossa, já?", C), ("tenho que dormir", C)],
        [("pesquisa sobre buracos negros", "t:pesquisar"), ("que legal", C)],
        [("abre o youtube", "t:abrir_site|t:abrir_app"), ("procura vídeo de receita de lasanha", "t:youtube")],
    ],
    "seguimento": [
        [("como você tá?", C), ("que?", C), ("ah, entendi", C)],
        [("me indica um livro", C), ("por quê esse?", C)],
        [("tô cansado", C), ("trabalhei 12 horas", C), ("e amanhã de novo", C)],
        [("você acredita em deus?", C), ("e em fantasmas?", C)],
        [("conta uma coisa sobre você", C), ("só isso?", C)],
    ],
    "provocacao": [
        [("você é burra", C), ("tô brincando", C)],
        [("você é chata", C), ("mentira, gosto de você", C)],
        [("fala um palavrão", C)],
        [("eu sou mais inteligente que você", C)],
    ],
}


def rodar(ckpt, cpu, conversas=None, prefixo="livres", trava=0, porteiro=None):
    from agent import Assistant
    from bateria import confere
    tools.DRY_RUN = True
    res = Path(__file__).resolve().parent / "resultados"
    mem = res / "_memoria_livres"
    tools.MEMORY_DIR = mem
    from trava import aplicar
    bot = aplicar(Assistant(str(ROOT / ckpt), device="cpu" if cpu else None, porteiro=porteiro), trava)
    nome = Path(ckpt).parent.name + ("_trava" if trava else "") + ("_porteiro" if porteiro else "")
    res.mkdir(exist_ok=True)
    log, k, certos, julgados, a_toa, conversa = [], 0, 0, 0, 0, 0
    for cat, convs in (conversas or CONVERSAS).items():
        for ci, conv in enumerate(convs):
            bot.reset()
            shutil.rmtree(mem, ignore_errors=True)
            log.append(f"--- [{cat} {ci}]")
            for frase, esp in conv:
                torch.manual_seed(500 + k)
                k += 1
                calls, reply = [], ""
                for ev in bot.chat(frase, auto=True):
                    if ev["type"] == "call":
                        calls.append({"tool": ev["tool"], "arg": ev["arg"], "result": ""})
                    elif ev["type"] == "text":
                        reply = ev["text"]
                ok = confere(esp, calls, reply)
                julgados += ok is not None
                certos += bool(ok)
                if esp == C:
                    conversa += 1
                    a_toa += bool(calls)
                log.append(f"{'  ' if ok else 'XX'} você: {frase}   ({esp})")
                for c in calls:
                    log.append(f"     [{c['tool']}: {c['arg']}]")
                log.append(f"     lia: {reply}")
    shutil.rmtree(mem, ignore_errors=True)
    (res / f"{prefixo}_{nome}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"{prefixo} {nome}: {certos}/{julgados} certos ({certos / julgados:.0%}), "
          f"ferramenta à toa em {a_toa}/{conversa} falas de conversa ({a_toa / conversa:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trava", type=float, default=0, help="trava de confiança (ex.: 0.9); ver testes/trava.py")
    ap.add_argument("--porteiro", default=None, help="caminho do porteiro.pt (ver porteiro.py)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rodar(args.ckpt, args.cpu, trava=args.trava, porteiro=args.porteiro)


if __name__ == "__main__":
    main()
