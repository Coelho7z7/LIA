"""
Teste LACRADO: conversas inéditas, escritas antes da rodada 20 e nunca usadas para montar dados de treino.
Serve para medir de verdade se a Lia melhora em coisas que ela nunca viu (a bateria e as conversas livres já
guiaram correções). Quando este lote for usado para corrigir algo, ele deixa de ser lacrado e entra outro.

    python testes/ineditas.py --ckpt out_v20/chat.pt     # -> testes/resultados/ineditas_out_v20.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

C = "c"
CONVERSAS = {
    "saude": [
        [("tô com uma dor de cabeça chata", C), ("desde ontem", C), ("tomei um remédio já", C)],
        [("fui no médico hoje", C), ("deu tudo certo nos exames", C)],
        [("acho que tô gripado", C), ("meu nariz não para de escorrer", C), ("me lembra de tomar chá daqui a 1 hora", "t:lembrete")],
        [("dormi só 4 horas", C), ("tô um zumbi hoje", C)],
        [("tô bebendo pouca água", C), ("como eu faço pra lembrar?", "c|t:lembrete")],
    ],
    "dinheiro": [
        [("caiu meu salário", C), ("já tá quase acabando kkk", C), ("como eu economizo?", C)],
        [("quanto dá 1200 dividido por 4?", "t:calcular"), ("é isso que sobra por semana", C)],
        [("tô devendo no cartão", C), ("me ajuda a pensar", C)],
        [("quero juntar dinheiro pra uma moto", C), ("uns 10 mil", C)],
        [("ganhei 50 reais no pix", C), ("de presente da minha vó", C)],
    ],
    "casa": [
        [("tô fazendo faxina", C), ("a cozinha tava um caos", C)],
        [("vou me mudar mês que vem", C), ("pra um apê menor", C), ("tô com medo de não caber tudo", C)],
        [("o vizinho tá com o som alto", C), ("já é meia noite", C)],
        [("acabou a luz aqui", C), ("voltou", C)],
        [("quebrei um copo", C), ("cortei o dedo de leve", C)],
    ],
    "relacionamento": [
        [("tô afim de uma pessoa", C), ("mas sou tímido", C), ("o que eu falo pra ela?", C)],
        [("tenho um encontro hoje", C), ("que roupa eu vou?", C)],
        [("meu namorado esqueceu nosso aniversário", C), ("tô chateada", C)],
        [("ela me respondeu!", C), ("marcamos de sair sábado", C)],
    ],
    "festas_datas": [
        [("tá chegando o são joão", C), ("vou dançar forró a noite toda", C)],
        [("o que você vai fazer no natal?", C), ("eu vou pra casa da minha avó", C)],
        [("amanhã é feriado", C), ("vou dormir até tarde", C)],
        [("quantos dias faltam pro natal?", "t:data|t:calcular|c|t:pesquisar")],
        [("hoje é meu aniversário", C), ("fiz 20 anos", C)],
    ],
    "internet_redes": [
        [("viralizei no tiktok", C), ("mais de 10 mil views", C)],
        [("tu tem instagram?", C), ("me segue lá kkk", C)],
        [("vi um meme muito bom", C), ("de um gato dançando", C)],
        [("minha internet tá horrível hoje", C), ("caiu três vezes", C)],
    ],
    "transporte": [
        [("peguei um uber agora", C), ("o motorista é muito falante", C)],
        [("tirei minha carteira de motorista!", C), ("passei de primeira", C)],
        [("meu carro quebrou na estrada", C), ("tô esperando o guincho", C)],
        [("vou de bike pro trabalho", C), ("são uns 5 km", C)],
    ],
    "aparencia_compras": [
        [("cortei o cabelo", C), ("ficou curtinho", C), ("acho que não gostei", C)],
        [("comprei um tênis novo", C), ("tava na promoção", C)],
        [("quero fazer uma tatuagem", C), ("de uma rosa no braço", C), ("dói muito?", C)],
        [("tu acha que eu devo pintar o cabelo de azul?", C)],
    ],
    "natureza_plantas": [
        [("plantei um pé de tomate", C), ("já tá nascendo", C)],
        [("vi um arco-íris agora", C), ("bem grandão", C)],
        [("tá chovendo muito aqui", C), ("alagou a rua", C)],
        [("minha planta tá morrendo", C), ("as folhas tão amarelas", C), ("o que eu faço?", C)],
    ],
    "pedidos_novos": [
        [("dá um google em como tirar mancha de café", "t:pesquisar")],
        [("me acorda às 7 amanhã", "t:lembrete|c")],
        [("baixa o som um tiquinho", "t:volume")],
        [("bota aquela música do vintage culture", "t:youtube")],
        [("quanto é metade de 350?", "t:calcular")],
        [("abre a calculadora pra mim", "t:abrir_app")],
        [("que dia cai o próximo sábado?", "t:data|c")],
        [("vê se vai chover em campinas", "t:clima")],
        [("guarda aí que meu aniversário é 3 de maio", "t:lembrar")],
        [("o que eu te pedi pra anotar?", "t:ver_notas")],
    ],
    "conversa_profunda": [
        [("às vezes eu acho que tô perdendo tempo na vida", C), ("todo mundo parece estar na frente", C)],
        [("qual o segredo da felicidade?", C), ("mas e se eu não tiver isso?", C)],
        [("você tem medo de ser desligada?", C), ("e se eu apagar você?", C)],
        [("o que você faria se fosse humana por um dia?", C)],
    ],
    "escrita_dificil": [
        [("oi lia td bem cmg e vc", C), ("q bom", C)],
        [("KKKKKKKKKKK MORRI", C)],
        [("vc eh mt legal sabia", C)],
        [("hoje acordei, fui trabalhar, almocei, voltei, fiz janta e agora tô aqui conversando com você", C)],
        [("😂😂😂", C)],
        [("tô sem nada pra fazer, me sugere alguma coisa que não seja ver série", C)],
    ],
    "perguntas_ela_novas": [
        [("você prefere dia ou noite?", C), ("eu prefiro noite", C)],
        [("se você fosse um animal, qual seria?", C)],
        [("qual foi a coisa mais difícil de aprender?", C)],
        [("você lembra da nossa última conversa?", C)],
        [("você é feliz?", C)],
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
    rodar(args.ckpt, args.cpu, CONVERSAS, "ineditas", args.trava, args.porteiro)


if __name__ == "__main__":
    main()
