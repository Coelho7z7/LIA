"""
Teste LACRADO nº 2 (escrito antes da rodada 21). O nº 1 (ineditas.py) foi aberto para comparar v20 e v21 e
passou a servir de correção. Este não: nunca olhar os erros dele para montar dados de treino.

    python testes/ineditas2.py --ckpt out_v20/chat.pt     # -> testes/resultados/ineditas2_out_v20.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

C = "c"
CONVERSAS = {
    "filhos_casa": [
        [("meu filho começou a andar", C), ("tá com 1 ano e 2 meses", C)],
        [("tenho 3 filhos", C), ("o mais velho tem 12", C), ("dá um trabalho kkk", C)],
        [("minha filha tirou 10 na prova", C), ("ela estudou muito", C)],
        [("as crianças não querem dormir", C), ("já são 11 horas", C)],
    ],
    "regionais": [
        [("bah, que frio", C), ("tchê, tá de renguear cusco", C)],
        [("uai, cê tá aí?", C), ("trem bão demais esse pão de queijo", C)],
        [("mermão, o trânsito tá parado", C), ("vou chegar atrasado", C)],
        [("véi, tô lascado", C), ("perdi o prazo do trabalho", C)],
        [("égua, que sono", C)],
    ],
    "nomes_numeros": [
        [("o joão é muito chato", C), ("ele fica me zoando", C)],
        [("a maria me ligou hoje", C), ("faz anos que a gente não se fala", C)],
        [("moro no quinto andar", C), ("o elevador quebrou", C)],
        [("acordei às 5 hoje", C), ("pra pegar o ônibus das 6", C)],
        [("meu filho nasceu em 2019", C)],
        [("paguei 80 reais numa camisa", C), ("achei caro", C)],
        [("é verdade", C)],
        [("é sério", C), ("juro", C)],
    ],
    "faculdade": [
        [("tô fazendo meu tcc", C), ("é sobre energia solar", C), ("falta muito ainda", C)],
        [("reprovei em cálculo", C), ("vou ter que fazer de novo", C)],
        [("meu orientador sumiu", C), ("não responde email", C)],
    ],
    "trabalho_remoto": [
        [("trabalho de casa agora", C), ("é bom mas é solitário", C)],
        [("tive reunião o dia todo", C), ("nem almocei direito", C)],
        [("meu notebook do trabalho travou na reunião", C), ("que vergonha", C)],
    ],
    "carro_moto": [
        [("comprei uma moto", C), ("é uma titan 160", C), ("tô aprendendo a pilotar", C)],
        [("levei uma multa", C), ("de 130 reais", C)],
        [("o pneu furou", C), ("troquei sozinho", C)],
    ],
    "festa_amigos": [
        [("vai ter churrasco aqui sábado", C), ("chamei uns 20 amigos", C)],
        [("fui num casamento ontem", C), ("dancei a noite toda", C)],
        [("meu amigo vai casar", C), ("e eu sou padrinho", C)],
    ],
    "perguntas_ela": [
        [("você fica com ciúme do chatgpt?", C)],
        [("você já se apaixonou?", C)],
        [("quantas pessoas falam com você?", C)],
        [("você sabe que dia é hoje?", "t:data|c")],
        [("você consegue ler meus pensamentos?", C)],
        [("se eu desligar o pc você morre?", C)],
    ],
    "pedidos_impossiveis": [
        [("pede um lanche pra mim no ifood", C)],
        [("liga pra minha mãe", C)],
        [("manda um áudio pro meu irmão", C)],
        [("tira uma foto minha", C)],
    ],
    "pedidos_ferramentas": [
        [("procura no google o horário do jogo do brasil", "t:pesquisar")],
        [("põe um pagode aí", "t:youtube")],
        [("deixa o som mais alto", "t:volume")],
        [("me lembra de ligar pro dentista às 3 da tarde", "t:lembrete")],
        [("anota que a senha do wifi tá atrás do roteador", "t:anotar")],
        [("quanto é 8% de 1500?", "t:calcular")],
        [("abre o word", "t:abrir_app")],
        [("entra no site do mercado livre", "t:abrir_site")],
        [("quem foi chico mendes?", "t:consultar|t:pesquisar")],
        [("como tá o tempo em natal?", "t:clima")],
    ],
    "conversa_e_pedido": [
        [("tô entediado", C), ("põe uma música animada", "t:youtube"), ("essa é boa", C)],
        [("amanhã tenho dentista", C), ("me lembra às 9", "t:lembrete|c")],
        [("tô estudando inglês", C), ("pesquisa umas músicas em inglês pra treinar", "t:pesquisar|t:youtube")],
        [("o que é inflação?", "t:consultar|t:pesquisar"), ("ah, por isso tá tudo caro", C)],
    ],
    "frases_soltas": [
        [("hoje não tô bem", C), ("só cansada mesmo", C)],
        [("que dia lindo", C)],
        [("cansei de tudo hoje", C)],
        [("tô orgulhoso de mim", C), ("terminei um projeto difícil", C)],
        [("que preguiça", C)],
        [("acho que vai chover", "c|t:clima")],
        [("o barulho da obra não para", C)],
        [("tô com saudade da minha cidade", C)],
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
    rodar(args.ckpt, args.cpu, CONVERSAS, "ineditas2", args.trava, args.porteiro)


if __name__ == "__main__":
    main()
