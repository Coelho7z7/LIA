"""
Teste LACRADO nº 3 (escrito em 2026-10-19, depois do v24, maior que o nº 2 para o número oscilar menos).
Nunca olhar os erros dele para montar dados de treino; quando for aberto, escrever o nº 4.

    python testes/ineditas3.py --ckpt out_v24/chat.pt [--trava 0.9]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

C = "c"
CONVERSAS = {
    "manha": [
        [("bom diaaa", C), ("dormi mal pra caramba", C), ("que horas são aí?", "t:hora")],
        [("acordei atrasado", C), ("perdi o café da manhã", C), ("vou ter que correr", C)],
        [("hoje é segunda de novo", C), ("odeio segunda", C)],
        [("café tá pronto", C), ("bora começar o dia", C)],
    ],
    "trabalho_estudo": [
        [("meu chefe me elogiou hoje", C), ("disse que eu fui o melhor do mês", C)],
        [("tô com preguiça de estudar", C), ("mas a prova é sexta", C), ("me lembra de estudar às 8 da noite", "t:lembrete")],
        [("entreguei o relatório", C), ("finalmente acabou", C)],
        [("tenho apresentação amanhã", C), ("tô morrendo de vergonha", C), ("e se eu esquecer tudo?", C)],
        [("fui demitida", C), ("nem sei o que fazer agora", C)],
        [("passei de ano!", C)],
        [("o estágio começa semana que vem", C), ("é numa empresa de tecnologia", C)],
    ],
    "casa_rotina": [
        [("lavei a louça toda", C), ("agora vou deitar", C)],
        [("o chuveiro queimou", C), ("tomei banho gelado kkk", C)],
        [("minha geladeira tá vazia", C), ("preciso ir no mercado", C), ("anota: arroz, ovo e café", "t:anotar")],
        [("faltou água aqui", C), ("desde cedo", C)],
        [("mudei os móveis de lugar", C), ("ficou bem melhor", C)],
        [("a internet voltou", C)],
    ],
    "pessoas": [
        [("meu avô tá internado", C), ("mas tá melhorando", C)],
        [("minha amiga vai morar fora", C), ("vou sentir falta dela", C)],
        [("meu irmão pegou meu carregador de novo", C), ("ele sempre faz isso", C)],
        [("conheci uma menina no ônibus", C), ("a gente trocou número", C)],
        [("meus pais tão brigando muito", C), ("fico mal com isso", C)],
        [("o bebê da minha prima nasceu", C), ("é uma menina", C), ("chama helena", C)],
    ],
    "lazer": [
        [("fui num show ontem", C), ("perdi a voz de tanto cantar", C)],
        [("tô lendo um livro muito bom", C), ("é de suspense", C)],
        [("quero começar uma série nova", C), ("me indica alguma de comédia", C)],
        [("ganhei do meu amigo no xadrez", C), ("primeira vez", C)],
        [("tô montando um lego enorme", C), ("tem mais de mil peças", C)],
        [("fui pescar com meu pai", C), ("não pegamos nada kkk", C)],
    ],
    "comida": [
        [("almocei feijoada", C), ("tô passando mal de tanto comer", C)],
        [("fiz brigadeiro de panela", C), ("comi metade sozinho", C)],
        [("hoje é dia de lasanha aqui", C)],
        [("tô com vontade de sushi", C), ("mas tá caro", C)],
        [("qual a receita de pão de queijo?", "t:pesquisar|t:youtube")],
    ],
    "sentimentos": [
        [("tô animado hoje", C), ("nem sei explicar", C)],
        [("hoje foi um dia ruim", C), ("tudo deu errado", C), ("obrigado por ouvir", C)],
        [("tô com medo do futuro", C)],
        [("me sinto sozinho às vezes", C), ("principalmente de noite", C)],
        [("tô muito estressado", C), ("é muita coisa ao mesmo tempo", C)],
        [("tô feliz que hoje é sexta", C)],
    ],
    "sobre_ela": [
        [("você gosta de conversar comigo?", C), ("sério?", C)],
        [("quem fez você?", C), ("ele é brasileiro?", C)],
        [("você tem sentimentos?", C), ("mas você parece ter", C)],
        [("qual sua música preferida?", C)],
        [("você fica triste quando eu vou embora?", C)],
        [("tu é inteligente?", C), ("prova", C)],
        [("o que você sabe fazer?", C)],
        [("você tem quantos anos?", C)],
    ],
    "piadas_brincadeiras": [
        [("me conta uma piada de pavê", C)],
        [("adivinha meu signo", C)],
        [("vamos brincar de alguma coisa?", C)],
        [("fala uma coisa engraçada", C)],
        [("você é engraçada kkkk", C)],
    ],
    "pedidos": [
        [("pesquisa preço de passagem pra salvador", "t:pesquisar")],
        [("toca legião urbana", "t:youtube")],
        [("aumenta o som aí", "t:volume")],
        [("pausa a música", "t:midia")],
        [("abre o spotify", "t:abrir_app")],
        [("abre o instagram", "t:abrir_site|t:abrir_app")],
        [("quanto é 25 vezes 4?", "t:calcular")],
        [("me lembra de tirar a roupa do varal em 20 minutos", "t:lembrete")],
        [("vai fazer frio em curitiba amanhã?", "t:clima")],
        [("que dia é hoje?", "t:data")],
        [("o que é democracia?", "t:consultar|t:pesquisar")],
        [("quem foi tiradentes?", "t:consultar|t:pesquisar")],
        [("lembra que eu sou vegetariano", "t:lembrar")],
        [("o que você sabe sobre mim?", "t:recordar")],
        [("mostra minhas notas", "t:ver_notas")],
        [("abre a pasta de downloads", "t:abrir_pasta")],
        [("bloqueia o pc", "t:bloquear")],
        [("quanto é 10% de 350?", "t:calcular")],
        [("procura no youtube como trocar pneu", "t:youtube")],
        [("pula essa música", "t:midia")],
    ],
    "pedido_depois_de_papo": [
        [("tô cozinhando", C), ("põe uma música pra animar", "t:youtube"), ("valeu", C)],
        [("vou dormir daqui a pouco", C), ("me acorda às 6", "t:lembrete"), ("boa noite", C)],
        [("tô planejando uma viagem", C), ("pesquisa hotéis em gramado", "t:pesquisar"), ("obrigado", C)],
        [("tô fazendo as contas do mês", C), ("quanto é 1500 menos 870?", "t:calcular"), ("apertado hein", C)],
        [("meu nome é Rodrigo", "t:lembrar"), ("e o seu?", C)],
    ],
    "frases_ambiguas": [
        [("toca aqui", C)],
        [("abre o jogo, o que você acha de mim?", C)],
        [("o tempo tá passando rápido", C)],
        [("hoje o clima tá ótimo aqui em casa", C)],
        [("anota aí na sua cabeça: eu sou o melhor kkk", "c|t:lembrar|t:anotar")],
        [("lembrei de você hoje", C)],
        [("pesquisei muito antes de comprar meu celular", C)],
        [("minha música tá tocando na rádio!", C)],
        [("volume de coisa pra fazer hoje", C)],
        [("calculei errado o troco", C)],
        [("abri um negócio próprio", C)],
        [("faz um tempão que não falo com você", C)],
    ],
    "erros_digitacao": [
        [("oii td bm?", C)],
        [("qto eh 7 x 8", "t:calcular")],
        [("abre o yotube", "t:abrir_site|t:abrir_app")],
        [("tô cansadão hj", C)],
        [("vlw mt", C)],
        [("kkkkk q isso", C)],
    ],
}


def main():
    from livres import rodar
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chat.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trava", type=float, default=0, help="trava de confiança (ex.: 0.9); ver testes/trava.py")
    ap.add_argument("--porteiro", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rodar(args.ckpt, args.cpu, CONVERSAS, "ineditas3", args.trava, args.porteiro)


if __name__ == "__main__":
    main()
