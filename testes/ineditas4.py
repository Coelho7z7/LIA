"""
Teste LACRADO nº 4 (2026-10-20): mais difícil que o 2 e o 3 (que já passam de 94%): contexto de várias falas,
pedido escondido no meio da frase, frase ambígua e mensagem longa. Nunca olhar os erros para montar treino.

    python testes/ineditas4.py --ckpt out_v26/chat.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

C = "c"
CONVERSAS = {
    "pedido_escondido": [
        [("tô aqui de bobeira, se puder toca um samba antigo", "t:youtube")],
        [("antes que eu esqueça, me lembra de pagar o aluguel em 2 horas", "t:lembrete")],
        [("a calculadora do celular sumiu, quanto dá 345 vezes 12?", "t:calcular")],
        [("minha mãe quer saber se vai chover em belém hoje", "t:clima")],
        [("tô curioso, pesquisa quanto custa uma viagem pro chile", "t:pesquisar")],
        [("cansei desse barulho, deixa o som mais baixo", "t:volume")],
        [("sem querer fechei, abre o chrome de novo", "t:abrir_app|t:abrir_site")],
        [("pra eu não esquecer amanhã: anota que o gás acaba sexta", "t:anotar")],
    ],
    "contexto_longo": [
        [("vou fazer um bolo pro aniversário do meu pai", C), ("ele faz 60", C), ("pesquisa uma receita de bolo de chocolate", "t:pesquisar"),
         ("valeu, vou começar agora", C)],
        [("tô no ônibus indo pro trabalho", C), ("que horas são?", "t:hora"), ("vou chegar atrasado de novo", C),
         ("me lembra de avisar meu chefe em 5 minutos", "t:lembrete")],
        [("meu time joga hoje", C), ("é o grêmio", C), ("acho que vai ganhar", C), ("procura o horário do jogo do grêmio", "t:pesquisar")],
        [("estudando pra prova de história", C), ("quem foi dom pedro segundo?", "t:consultar|t:pesquisar"), ("nossa, que interessante", C),
         ("e a princesa isabel?", "t:consultar|t:pesquisar")],
        [("tô organizando a casa", C), ("põe uma música animada", "t:youtube"), ("mais alto", "t:volume"), ("perfeito", C)],
    ],
    "ambiguas": [
        [("toca o terror kkk", C)],
        [("o clima entre eu e minha namorada tá ótimo", C)],
        [("perdi a noção do tempo jogando", C)],
        [("abre o coração comigo, tá tudo bem?", C)],
        [("anotado, obrigado", C)],
        [("hora de dormir", C)],
        [("pesquisando bastante antes de decidir a faculdade", C)],
        [("lembra de mim amanhã?", C)],
        [("minha nota foi 9,5!", C)],
        [("som na caixa!", "c|t:youtube")],
        [("esse tempo frio me dá sono", C)],
        [("tocando violão aqui", C)],
    ],
    "mensagem_longa": [
        [("cara, hoje eu acordei cedo, fui na academia, depois trabalhei até as 6 e agora tô morto de cansaço", C)],
        [("minha irmã brigou comigo porque eu peguei a roupa dela sem pedir e agora ela não fala comigo", C),
         ("o que eu faço?", C)],
        [("tô pensando em largar a faculdade pra trabalhar mas meus pais não querem e eu não sei o que fazer", C)],
        [("ontem fui num restaurante novo, a comida era boa mas o atendimento foi péssimo, demorou uma hora", C)],
    ],
    "correcoes_e_seguimento": [
        [("abre o word", "t:abrir_app"), ("não, o excel", "t:abrir_app")],
        [("quanto é 30% de 90?", "t:calcular"), ("e de 150?", "t:calcular")],
        [("toca sertanejo", "t:youtube"), ("muda pra rock", "t:youtube")],
        [("como tá o tempo em fortaleza?", "t:clima"), ("e em natal?", "t:clima")],
        [("me chama de Bia", "t:lembrar"), ("como você me chama?", "c|t:recordar")],
        [("pesquisa sobre tubarões", "t:pesquisar"), ("valeu, agora sobre baleias", "t:pesquisar")],
    ],
    "conversa_profunda": [
        [("você acha que eu sou uma pessoa boa?", C)],
        [("às vezes sinto que ninguém liga pra mim", C), ("nem minha família", C)],
        [("o que você pensa sobre a morte?", C)],
        [("você acha que existe alma?", C)],
        [("qual o sentido de trabalhar tanto?", C)],
    ],
    "giria_internet": [
        [("slk, que dia", C)],
        [("tmj lia", C)],
        [("sextou!!", C)],
        [("mds, esqueci a prova", C)],
        [("pprt, tô cansado", C)],
        [("oxi, nem vi", C)],
    ],
}


def main():
    from livres import rodar
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trava", type=float, default=0, help="limite extra (o agent.py já usa 0.9)")
    ap.add_argument("--porteiro", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rodar(args.ckpt, args.cpu, CONVERSAS, "ineditas4", args.trava, args.porteiro)


if __name__ == "__main__":
    main()
