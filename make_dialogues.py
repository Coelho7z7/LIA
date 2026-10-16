"""
Gera conversas sintéticas para o treino de chat (SFT): bate-papo no tom da
personalidade (amigo esperto, leve, parceiro) + uso de ferramentas.

    python make_dialogues.py                         # 30000 conversas, IA chamada "Lia"
    python make_dialogues.py --nome Nina --n 50000

Cada linha do .jsonl é uma conversa:
    {"messages": [{"role": "user", "content": "abre o spotify"},
                  {"role": "call", "content": "abrir_app: spotify"},
                  {"role": "result", "content": "ok: abri Spotify"},
                  {"role": "assistant", "content": "Pronto, Spotify aberto."}]}

Você pode (e deve!) escrever suas próprias conversas nesse formato em
data/conversas/*.jsonl. Exemplos reais seus valem mais que os sintéticos.

Sobre generalização: parte das frases e dos nomes fica reservada para teste
(split="test") e nunca aparece no treino. O eval.py usa essa parte para medir se
o modelo aprendeu o padrão ou só decorou.
"""

import argparse
import datetime
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import tools as T

# =================================================================== ENTIDADES
# (nome, gênero) — o gênero só serve para acertar "o/a", "do/da", "no/na".
APPS = [
    ("Spotify", "o"), ("Discord", "o"), ("Steam", "a"), ("Chrome", "o"), ("Google Chrome", "o"), ("Edge", "o"),
    ("Firefox", "o"), ("Brave", "o"), ("Opera GX", "o"), ("Word", "o"), ("Excel", "o"), ("PowerPoint", "o"),
    ("VS Code", "o"), ("vscode", "o"), ("Visual Studio Code", "o"), ("bloco de notas", "o"), ("notepad", "o"),
    ("calculadora", "a"), ("Paint", "o"), ("explorador de arquivos", "o"), ("configurações", "a"),
    ("gerenciador de tarefas", "o"), ("terminal", "o"), ("cmd", "o"), ("PowerShell", "o"), ("WhatsApp", "o"),
    ("zap", "o"), ("Telegram", "o"), ("OBS", "o"), ("OBS Studio", "o"), ("câmera", "a"), ("Fotos", "o"),
    ("Epic Games", "a"), ("Minecraft", "o"), ("Roblox", "o"), ("Valorant", "o"), ("League of Legends", "o"),
    ("lol", "o"), ("Photoshop", "o"), ("Premiere", "o"), ("After Effects", "o"), ("Blender", "o"), ("Unity", "o"),
    ("Notion", "o"), ("Slack", "o"), ("Teams", "o"), ("Zoom", "o"), ("VLC", "o"), ("WinRAR", "o"), ("7-Zip", "o"),
    ("Audacity", "o"), ("GIMP", "o"), ("Krita", "o"), ("Figma", "o"), ("Postman", "o"), ("Docker", "o"),
    ("Git Bash", "o"), ("Android Studio", "o"), ("PyCharm", "o"), ("IntelliJ", "o"), ("Obsidian", "o"),
    ("Anki", "o"), ("Outlook", "o"), ("OneNote", "o"), ("Microsoft Store", "a"), ("relógio", "o"), ("Xbox", "o"),
    ("Battle.net", "o"), ("Ubisoft Connect", "o"), ("GTA V", "o"), ("CS2", "o"), ("Counter Strike", "o"),
    ("Fortnite", "o"), ("Lightroom", "o"), ("CapCut", "o"), ("DaVinci Resolve", "o"), ("Bitwarden", "o"),
    ("Dropbox", "o"), ("OneDrive", "o"), ("Vivaldi", "o"), ("Riot Client", "o"), ("Wallpaper Engine", "o"),
    ("AnyDesk", "o"), ("TeamViewer", "o"), ("pgAdmin", "o"), ("Python", "o"), ("Clipchamp", "o"),
    ("Notas Autoadesivas", "as"), ("Sticky Notes", "o"), ("Mapas", "o"), ("Copilot", "o"),
    ("Reprodutor Multimídia", "o"), ("Ferramenta de Captura", "a"), ("Painel de Controle", "o"), ("Lupa", "a"),
    ("Teclado Virtual", "o"), ("WSL", "o"), ("Calendário", "o"), ("Gravador de Som", "o"), ("Microsoft To Do", "o"),
    ("Visio", "o"), ("Access", "o"), ("PDF24", "o"), ("Game Bar", "a"), ("Paciência", "a"), ("Kindle", "o"),
    ("Signal", "o"), ("Skype", "o"), ("Thunderbird", "o"), ("qBittorrent", "o"), ("Rainmeter", "o"),
]
SYLLABLES = ["ka", "lo", "mi", "zen", "tra", "vor", "pix", "nu", "dex", "ra", "bel", "qui", "sto", "fy", "lix",
             "mon", "ta", "ver", "zo", "gra", "fi", "no", "pe", "lu", "xa", "dri", "bo", "cal", "mar", "ter", "vi"]
APP_SUFFIX = [" Pro", " Studio", " 2", " Desktop", " Player", " Launcher", " Editor", " Manager", " X"]

SITES = [
    ("YouTube", "o"), ("Gmail", "o"), ("GitHub", "o"), ("Netflix", "a"), ("Wikipedia", "a"), ("Instagram", "o"),
    ("Twitter", "o"), ("Reddit", "o"), ("LinkedIn", "o"), ("Twitch", "a"), ("Amazon", "a"),
    ("Mercado Livre", "o"), ("G1", "o"), ("Google Drive", "o"), ("Google Maps", "o"), ("tradutor", "o"),
    ("Canva", "o"), ("Facebook", "o"), ("TikTok", "o"), ("Pinterest", "o"), ("Prime Video", "o"),
    ("Disney Plus", "o"), ("Stack Overflow", "o"), ("Duolingo", "o"), ("Google Agenda", "o"), ("Shopee", "a"),
    ("iFood", "o"), ("WhatsApp Web", "o"), ("Google", "o"), ("UOL", "o"),
]
TLDS = [".com", ".com.br", ".net", ".org", ".io", ".dev", ".app"]

SEARCH_TERMS = [
    "receita de bolo de cenoura", "como trocar o pneu do carro", "preço do dólar hoje", "melhores filmes de 2025",
    "resultado do jogo do Flamengo", "como fazer currículo", "horário de funcionamento do banco",
    "significado de resiliência", "tutorial de Python", "como tirar mancha de café", "sintomas de gripe",
    "promoção de notebook", "como plantar tomate", "quanto custa uma viagem pra Portugal", "notícias de hoje",
    "tabela do brasileirão", "como fazer pão caseiro", "exercícios para as costas", "como economizar dinheiro",
    "curso de inglês grátis", "como configurar o roteador", "dicas de entrevista de emprego",
    "receita de strogonoff", "calendário de feriados 2026", "como declarar imposto de renda",
    "melhor celular custo benefício", "como fazer café coado", "ingresso pro show", "frete grátis",
    "cotação do euro", "vagas de estágio em TI", "como limpar o teclado do notebook",
]
HOW_VERBS = ["fazer", "limpar", "consertar", "instalar", "configurar", "aprender", "usar", "montar", "atualizar"]
HOW_OBJS = ["um bolo", "o teclado", "a pia", "o Windows", "o Excel", "um PC gamer", "a impressora", "o Git",
            "uma horta", "o celular", "a geladeira", "uma planilha", "o Linux", "um site", "a bicicleta"]
NEAR = ["farmácia", "pizzaria", "academia", "mercado", "posto de gasolina", "padaria", "hamburgueria"]
YT_TERMS = [
    "lofi pra estudar", "rock anos 80", "samba de raiz", "pagode", "sertanejo", "MPB", "jazz", "rap nacional",
    "música clássica", "música eletrônica", "tutorial de violão", "receita de lasanha", "gols da rodada",
    "podcast de tecnologia", "trailer do novo filme", "documentário sobre o espaço", "aula de inglês",
    "treino em casa", "meditação guiada", "vídeo de gatinho", "stand up comedy", "review de celular",
    "como desenhar", "forró", "funk", "reggae", "k-pop", "aula de cálculo", "speedrun de Mario",
]
SUBJECTS = [
    "fotossíntese", "um buraco negro", "inflação", "a internet", "o DNA", "a teoria da relatividade",
    "a bolsa de valores", "o efeito estufa", "um vulcão", "a democracia", "o imposto de renda",
    "a energia solar", "o bitcoin", "a gravidade", "o sistema imunológico", "a Revolução Francesa",
    "a Guerra Fria", "o Big Bang", "a mitose", "a computação quântica", "o PIB", "a fusão nuclear",
]
PEOPLE = [
    "Santos Dumont", "Machado de Assis", "Albert Einstein", "Pelé", "Ayrton Senna", "Tiradentes", "Marie Curie",
    "Leonardo da Vinci", "Clarice Lispector", "Dom Pedro II", "Napoleão", "Isaac Newton", "Tarsila do Amaral",
    "Getúlio Vargas", "Nikola Tesla", "Ada Lovelace", "Alan Turing", "Zumbi dos Palmares",
]
COUNTRIES = [("Austrália", "da"), ("Canadá", "do"), ("Noruega", "da"), ("Argentina", "da"), ("Japão", "do"),
             ("Egito", "do"), ("Turquia", "da"), ("Nova Zelândia", "da"), ("Suíça", "da"), ("Marrocos", "do"),
             ("Coreia do Sul", "da"), ("Peru", "do"), ("Islândia", "da"), ("México", "do")]
FACT_QUESTIONS = [
    "quantos anos tem a Terra?", "qual a distância da Terra até a Lua?", "quem inventou a lâmpada?",
    "quem descobriu o Brasil?", "qual o maior animal do mundo?", "quantos ossos tem o corpo humano?",
    "qual o rio mais longo do mundo?", "em que ano acabou a Segunda Guerra?", "qual a velocidade da luz?",
    "quem pintou a Mona Lisa?", "qual o planeta mais quente?", "quantos países tem na África?",
    "quem escreveu Dom Casmurro?", "qual a montanha mais alta do mundo?",
]

DURATIONS = ["10 minutos", "5 minutos", "15 minutos", "20 minutos", "30 minutos", "1 hora", "2 horas",
             "meia hora", "40 minutos", "1 minuto", "2 minutos", "3 minutos", "25 minutos", "45 minutos",
             "90 minutos", "30 segundos", "1h30", "um minuto", "dez minutos", "cinco minutos", "uma hora",
             "duas horas", "5 min", "10 min", "15 min", "50 minutos", "7 minutos", "12 minutos"]
REMINDERS = [
    "tirar a pizza do forno", "beber água", "ligar pra minha mãe", "tomar o remédio", "ir pra academia",
    "olhar o arroz", "entrar na reunião", "estudar", "alongar", "levar o lixo pra fora", "pagar o boleto",
    "responder o email do chefe", "buscar as crianças", "desligar o forno", "sair de casa",
    "tirar a roupa da máquina", "fazer o almoço", "dar comida pro gato", "regar as plantas",
    "descansar a vista", "mandar o relatório", "tirar o bolo do forno", "ligar pro médico", "dormir",
]
NOTES = [
    "comprar leite", "pagar a conta de luz", "ideia de app de receitas", "reunião sexta às 10",
    "levar o carro na revisão", "aniversário da Júlia dia 12", "comprar presente pro meu pai",
    "ligar pro dentista", "estudar pra prova de cálculo", "trocar a lâmpada da sala", "renovar a CNH",
    "marcar médico", "comprar ração", "devolver o livro da biblioteca", "cancelar a assinatura da academia",
    "pesquisar passagens pra dezembro", "fazer backup do PC", "comprar pilha", "consertar a torneira",
]
NAMES = ["Matheus", "Ana", "João", "Maria", "Lucas", "Júlia", "Pedro", "Beatriz", "Gabriel", "Larissa", "Rafael",
         "Camila", "Bruno", "Fernanda", "Diego", "Letícia", "Thiago", "Amanda", "Felipe", "Carol", "Vinícius",
         "Bianca", "Gustavo", "Paula", "Caio", "Renata", "Igor", "Sofia", "Leo", "Duda"]
CITIES = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Recife", "Salvador", "Fortaleza", "Curitiba",
          "Porto Alegre", "Manaus", "Belém", "Goiânia", "Florianópolis", "Natal", "Campinas", "Brasília",
          "Vitória", "João Pessoa", "Maceió", "Teresina", "Cuiabá", "Londrina", "Santos", "Lisboa", "Porto"]
LIKES = ["pizza", "rock", "jogar videogame", "ler", "café", "cachorros", "futebol", "anime", "programar",
         "cozinhar", "séries de suspense", "acordar tarde", "chocolate", "andar de bicicleta"]
COLORS = ["azul", "verde", "preto", "roxo", "vermelho", "amarelo", "laranja", "branco"]
JOBS = ["programação", "vendas", "design", "educação", "saúde", "marketing", "engenharia", "atendimento"]
STUDIES = ["engenharia", "direito", "medicina", "ciência da computação", "administração", "inglês", "música"]
TEAMS = ["Flamengo", "Corinthians", "Palmeiras", "São Paulo", "Grêmio", "Inter", "Cruzeiro", "Atlético",
         "Vasco", "Bahia", "Fortaleza", "Santos"]
PETS = ["Thor", "Mel", "Bob", "Luna", "Pipoca", "Max", "Nina", "Paçoca"]
MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
          "outubro", "novembro", "dezembro"]
FOLDERS = [("downloads", "Downloads"), ("documentos", "Documentos"), ("área de trabalho", "Área de Trabalho"),
           ("imagens", "Imagens"), ("músicas", "Músicas"), ("vídeos", "Vídeos"), ("Downloads", "Downloads"),
           ("Documentos", "Documentos"), ("desktop", "Área de Trabalho"), ("fotos", "Imagens")]
WEATHER = ["céu limpo", "quase limpo", "parcialmente nublado", "nublado", "garoa", "chuva fraca", "chuva",
           "pancadas de chuva", "tempestade", "neblina"]

# Palavras soltas para montar termos aleatórios (pesquisa, YouTube, perguntas, apps).
# Assim o modelo aprende a COPIAR o que o usuário disse em vez de decorar as listas acima.
WORDS = [
    "casa", "carro", "gato", "cachorro", "livro", "mesa", "cadeira", "porta", "janela", "praia", "montanha", "rio",
    "cidade", "escola", "trabalho", "festa", "viagem", "comida", "bolo", "pão", "arroz", "feijão", "frango",
    "peixe", "salada", "suco", "chá", "sorvete", "chocolate", "pizza", "hambúrguer", "macarrão", "torta",
    "música", "filme", "série", "jogo", "desenho", "história", "notícia", "aula", "curso", "prova", "receita",
    "tutorial", "dica", "truque", "resumo", "review", "trailer", "show", "clipe", "cover", "remix", "ao vivo",
    "celular", "notebook", "teclado", "mouse", "monitor", "fone", "câmera", "impressora", "roteador", "tablet",
    "bicicleta", "moto", "ônibus", "avião", "trem", "barco", "foguete", "planeta", "estrela", "lua", "sol",
    "árvore", "flor", "jardim", "horta", "planta", "semente", "pedra", "areia", "neve", "chuva", "vento",
    "azul", "verde", "vermelho", "amarelo", "preto", "branco", "grande", "pequeno", "novo", "velho", "barato",
    "rápido", "fácil", "simples", "caseiro", "antigo", "moderno", "melhor", "pior", "engraçado", "triste",
    "violão", "guitarra", "piano", "bateria", "flauta", "samba", "rock", "jazz", "blues", "forró", "metal",
    "futebol", "basquete", "vôlei", "tênis", "natação", "xadrez", "corrida", "yoga", "academia", "treino",
    "médico", "dentista", "banco", "mercado", "farmácia", "padaria", "hospital", "museu", "parque", "zoológico",
    "preço", "horário", "endereço", "telefone", "mapa", "tradução", "significado", "origem", "tamanho", "peso",
    "matemática", "física", "química", "biologia", "geografia", "inglês", "espanhol", "português", "arte",
    "dragão", "robô", "pirata", "ninja", "zumbi", "fantasma", "mágico", "palhaço", "princesa", "cavaleiro",
    "Brasil", "Portugal", "Bahia", "Minas", "Nordeste", "Amazônia", "Pantanal", "Europa", "Ásia", "África",
]
CONNECTORS = ["de", "do", "da", "pra", "com", "sem", "e", "no", "na", "em"]
TASK_VERBS = ["comprar", "ligar pro", "ver o", "levar o", "buscar o", "pagar o", "mandar o", "lavar a", "arrumar a",
              "estudar", "terminar o", "checar o", "responder o", "limpar a", "pegar o", "devolver o", "marcar",
              "trocar o", "separar a", "assistir"]
PERSON_FIRST = ["João", "Maria", "José", "Ana", "Carlos", "Luiza", "Pedro", "Helena", "Antônio", "Rita", "Jorge",
                "Cecília", "Raul", "Elis", "Oscar", "Irene", "Rui", "Vera", "Otto", "Nise"]
PERSON_LAST = ["Silva", "Souza", "Oliveira", "Pereira", "Costa", "Almeida", "Ferreira", "Ribeiro", "Barbosa",
               "Moreira", "Lopes", "Cardoso", "Teixeira", "Nunes", "Ramos", "Freitas", "Machado", "Rocha"]
APP_WORDS = ["Gravador", "Editor", "Visualizador", "Leitor", "Gerenciador", "Conversor", "Monitor", "Agenda",
             "Studio", "Player", "Launcher", "Hub", "Center", "Cloud", "Sync", "Note", "Pad", "Box", "Lab", "Go"]
T_FACT_RAND = ["quem inventou {o} {x}?", "qual o maior {x} do mundo?", "quantos {x} existem no mundo?",
               "onde fica {o} {x}?", "qual a origem {do} {x}?", "por que {o} {x} é {y}?",
               "quanto pesa {o} {x}?", "de onde vem {o} {x}?", "quando surgiu {o} {x}?",
               "qual a diferença entre {x} e {y}?", "como surgiu {o} {x}?", "pra que serve {o} {x}?",
               "qual o tamanho {do} {x}?", "quem criou {o} {x}?", "qual a história {do} {x}?"]

# "Sobre mim": o usuário conta algo de si -> lembrar (copiando) + uma fala curta e curiosa.
SPORTS = ["jiu jitsu", "bjj", "muay thai", "boxe", "judô", "karatê", "capoeira", "crossfit", "musculação",
          "natação", "corrida", "ciclismo", "futebol", "futsal", "vôlei", "basquete", "tênis", "skate", "surf",
          "escalada", "yoga", "pilates", "dança", "ballet", "kung fu", "taekwondo", "MMA", "handebol", "remo",
          "beach tennis", "funcional", "arte marcial", "luta livre", "esgrima", "triatlo", "spinning"]
HOBBIES = ["desenho", "pintura", "fotografia", "crochê", "tricô", "jardinagem", "cerâmica", "marcenaria",
           "cozinhar", "ler mangá", "assistir anime", "programar", "escrever", "meditar", "pescar", "acampar",
           "trilha", "xadrez", "cubo mágico", "origami", "teatro", "canto", "academia", "videogame", "RPG",
           "colecionar figurinhas", "fazer pão", "viajar", "maratonar série", "tocar violão"]
GAMES = ["futebol", "vôlei", "Valorant", "LoL", "Minecraft", "xadrez", "basquete", "CS", "Fortnite", "truco",
         "sinuca", "tênis de mesa", "Roblox", "FIFA", "pôquer", "baralho", "Free Fire", "Dota", "golfe"]
PROFESSIONS = ["programador", "professora", "enfermeiro", "designer", "vendedor", "advogada", "engenheiro",
               "motorista", "cozinheiro", "médica", "estagiário", "garçom", "eletricista", "pedreiro",
               "fotógrafo", "analista de dados", "dentista", "psicóloga", "barbeiro", "youtuber", "mecânico"]
INSTRUMENTS = ["violão", "guitarra", "piano", "bateria", "baixo", "ukulele", "cavaquinho", "flauta", "teclado",
               "violino", "sax", "pandeiro"]
PLACES = ["na academia", "pra praia", "no parque", "no shopping", "no médico", "na casa da minha mãe",
          "no cinema", "pra faculdade", "no mercado", "na igreja", "no show", "pro trabalho", "na feira",
          "no treino", "na balada", "no dentista", "pra escola", "no estádio"]
T_ABOUT = [  # (pedido, fato guardado, lista de onde sai o {x})
    ("eu faço {x}", "eu faço {x}", "sport"), ("faço {x}", "eu faço {x}", "sport"),
    ("eu treino {x}", "eu treino {x}", "sport"), ("treino {x} faz um tempo", "eu treino {x}", "sport"),
    ("comecei a fazer {x}", "eu faço {x}", "sport"), ("eu luto {x}", "eu luto {x}", "sport"),
    ("eu pratico {x}", "eu pratico {x}", "sport"), ("tô fazendo {x}", "eu faço {x}", "sport"),
    ("eu jogo {x}", "eu jogo {x}", "game"), ("jogo {x} toda semana", "eu jogo {x}", "game"),
    ("eu trabalho como {x}", "eu trabalho como {x}", "job"), ("trabalho de {x}", "eu trabalho como {x}", "job"),
    ("minha profissão é {x}", "eu trabalho como {x}", "job"),
    ("hoje eu fui {x}", "hoje eu fui {x}", "place"), ("acabei de voltar {x}", "hoje eu fui {x}", "place"),
    ("eu amo {x}", "eu amo {x}", "hobby"), ("eu curto {x}", "eu curto {x}", "hobby"),
    ("meu hobby é {x}", "meu hobby é {x}", "hobby"), ("eu adoro {x}", "eu amo {x}", "hobby"),
    ("nas horas vagas eu faço {x}", "eu faço {x}", "hobby"), ("sou viciado em {x}", "eu amo {x}", "hobby"),
    ("eu toco {x}", "eu toco {x}", "instrument"), ("tô aprendendo {x}", "eu toco {x}", "instrument"),
]
ABOUT_WHEN = ["há dois anos", "faz uns meses", "há um tempão", "desde criança", "toda semana", "faz pouco tempo",
              "de vez em quando", "desde o ano passado", "já faz anos", "quase todo dia", "há uns meses"]  # só treino
R_ABOUT = ["Que massa! Guardei aqui. Faz tempo?", "Boa! Anotei. Curte mais pela saúde ou pela diversão?",
           "Show, guardei isso. Como você começou?", "Massa! Tá guardado. E tá gostando?",
           "Olha só! Guardei aqui. Me conta mais depois.", "Que legal! Anotado. É mais hobby ou coisa séria?",
           "Boa demais! Fica guardado aqui. O que te fez começar?"]
T_FIX_NAME = ["não, meu nome é {x}", "eu falei {x}, não {y}", "errou, é {x}", "meu nome é {x}, não {y}",
              "não é {y}, é {x}", "na verdade meu nome é {x}", "opa, eu falei que meu nome é {x}, não {y}",
              "é {x}, não {y}", "errado, meu nome é {x}", "não me chamo {y}, me chamo {x}",
              "você errou, eu sou {x}", "corrige aí: meu nome é {x}"]
R_FIX_NAME = ["Foi mal, {x}! Corrigi aqui.", "Opa, desculpa! Agora sim: {x}.", "Corrigido, {x}. Não erro mais.",
              "Eita, falha minha. Guardei certo agora: {x}."]

# ================================================================== TEMPLATES
# {x} = entidade que vai ser COPIADA para a chamada. {o}/{do}/{no}/{pro} = artigo.
T_APP = ["abre {o} {x}", "abrir {x}", "abre {x}", "pode abrir {o} {x}?", "abre {o} {x} pra mim",
         "bora abrir {o} {x}", "quero usar {o} {x}", "inicia {o} {x}", "executa {o} {x}", "liga {o} {x}",
         "abre aí {o} {x}", "consegue abrir {o} {x}?", "me abre {o} {x}", "manda abrir {o} {x}", "roda {o} {x}",
         "abre o app {x}", "abre o programa {x}", "quero abrir {o} {x}", "dá pra abrir {o} {x}?",
         "preciso {do} {x} aberto", "abre {o} {x} aí", "coloca {o} {x} pra rodar", "inicia o app {x}",
         "abre {o} {x} rapidinho"]
T_APP_MORE = ["e {o} {x} também", "abre {o} {x} também", "agora {o} {x}", "agora abre {o} {x}",
              "abre {o} {x} junto", "e {o} {x}?"]
T_APP_TWO = ["abre {o} {x} e {o2} {y}", "abre {x} e {y}", "abre {o} {x} e depois {o2} {y}",
             "quero {o} {x} e {o2} {y} abertos", "abre {o} {x} e também {o2} {y}"]
T_SITE = ["abre {o} {x}", "entra {no} {x}", "abre o site {do} {x}", "vai {no} {x}", "me leva {pro} {x}",
          "abre {x}", "abre {o} {x} no navegador", "acessa {o} {x}", "quero entrar {no} {x}", "abre o site {x}",
          "entra {no} {x} pra mim", "abre {o} {x} aí"]
T_DOMAIN = ["abre o site {x}", "entra em {x}", "acessa {x}", "abre {x}", "vai no site {x}", "abre a página {x}"]
T_SEARCH = ["pesquisa {x}", "pesquisa sobre {x}", "procura {x} no google", "busca {x} pra mim", "dá um google em {x}",
            "joga no google: {x}", "pesquisa no google {x}", "procura {x}", "faz uma pesquisa sobre {x}",
            "googla {x}", "busca no google {x}", "pesquisa aí {x}"]
T_YT = ["toca {x} no youtube", "procura {x} no youtube", "coloca {x} no youtube", "quero ver vídeo de {x}",
        "põe {x} pra tocar no youtube", "busca {x} no youtube", "abre o youtube com {x}", "acha {x} no youtube",
        "bota um vídeo de {x}", "pesquisa {x} no youtube"]
T_WHAT = ["o que é {x}?", "me explica o que é {x}", "como funciona {x}?", "o que significa {x}?",
          "você sabe o que é {x}?", "me fala sobre {x}"]
T_WHO = ["quem foi {x}?", "quem é {x}?", "me fala sobre {x}", "quando nasceu {x}?", "o que {x} fez?",
         "você conhece {x}?"]
T_COUNTRY = ["qual a capital {do} {x}?", "quantos habitantes tem {o} {x}?", "qual a moeda {do} {x}?",
             "que língua falam {no} {x}?", "qual o fuso horário {do} {x}?"]
T_HORA = ["que horas são?", "que horas são agora?", "me fala as horas", "hora?", "tem horas aí?",
          "sabe que horas são?", "que hora é agora?", "me diz a hora", "horas?", "que horas são, {nome}?",
          "já é que horas?", "olha a hora pra mim", "me fala que horas são", "sabe me dizer as horas?",
          "qual é a hora?", "que horas tá marcando aí?", "vê a hora pra mim", "quero saber a hora",
          "que horas são aí?", "agora é que horas?", "hora certa?", "tá que horas?", "consegue ver as horas?",
          "me passa a hora", "que horas são mesmo?", "qual a hora atual?", "horário agora?", "que horas tem?",
          "me diz que horas são", "que horas são exatamente?", "dá uma olhada na hora", "fala a hora aí",
          "qual o horário agora?", "horas, por favor", "que horas deu?", "e as horas?", "a hora, pfv",
          "quantas horas são?", "tá tarde? que horas são?", "perdi a noção do tempo, que horas são?"]
T_DATA = ["que dia é hoje?", "qual a data de hoje?", "hoje é dia quanto?", "que dia da semana é hoje?",
          "data de hoje?", "em que dia a gente tá?", "hoje é que dia?", "me fala a data"]
T_CALC = ["quanto é {x}?", "quanto é {x}", "calcula {x}", "calcula {x} pra mim", "faz essa conta: {x}",
          "{x} dá quanto?", "me ajuda numa conta: {x}", "quanto dá {x}?", "resolve {x}", "{x} = ?",
          "faz a conta {x}", "qual o resultado de {x}?"]
T_REMIND = ["me lembra de {t} em {d}", "daqui a {d} me lembra de {t}", "me avisa em {d} pra {t}",
            "põe um lembrete pra {t} daqui a {d}", "cria um lembrete: {t} em {d}", "em {d} me lembra de {t}",
            "me lembra daqui {d} de {t}", "lembrete de {t} em {d}", "me lembra de {t} daqui a {d}",
            "não me deixa esquecer de {t}, me avisa em {d}"]
T_TIMER = ["põe um timer de {d}", "timer de {d}", "cronômetro de {d}", "marca {d} pra mim",
           "me avisa daqui a {d}", "conta {d} pra mim", "liga um timer de {d}"]
T_NOTE = ["anota aí: {x}", "anota {x}", "salva uma nota: {x}", "escreve aí pra eu não esquecer: {x}",
          "coloca nas notas: {x}", "anota isso: {x}", "faz uma anotação: {x}", "nota: {x}",
          "adiciona nas minhas notas: {x}"]
T_SEE_NOTES = ["lê minhas notas", "o que eu anotei?", "mostra minhas notas", "quais são minhas anotações?",
               "tem alguma nota minha?", "o que tem nas minhas notas?", "lista minhas notas", "minhas notas"]
T_REMEMBER = ["lembra que {x}", "guarda isso: {x}", "não esquece: {x}", "grava na memória: {x}",
              "quero que você lembre que {x}", "grava aí: {x}", "memoriza: {x}"]
T_INTRO = ["meu nome é {x}", "me chamo {x}", "pode me chamar de {x}", "eu sou {x}", "oi, meu nome é {x}"]
T_LIVE = ["eu moro em {x}", "moro em {x}", "eu vivo em {x}"]
T_RECALL = [  # (pergunta, palavra-chave da busca, tipo de fato)
    ("qual é meu nome?", "nome", "nome"), ("você lembra meu nome?", "nome", "nome"),
    ("sabe meu nome?", "nome", "nome"), ("como é meu nome mesmo?", "nome", "nome"),
    ("onde eu moro?", "moro", "cidade"), ("em que cidade eu moro?", "moro", "cidade"),
    ("lembra onde eu moro?", "moro", "cidade"), ("quando é meu aniversário?", "aniversário", "aniv"),
    ("do que eu gosto?", "gosto", "gosto"), ("qual minha cor favorita?", "cor favorita", "cor"),
    ("com o que eu trabalho?", "trabalho", "trabalho"), ("o que eu estudo?", "estudo", "estudo"),
    ("qual é meu time?", "time", "time"), ("como chama meu cachorro?", "cachorro", "pet"),
    ("o que você sabe sobre mim?", "", "tudo"), ("o que você lembra de mim?", "", "tudo"),
    ("lembra de alguma coisa sobre mim?", "", "tudo"), ("qual o nome do meu cachorro?", "cachorro", "pet"),
    ("quem sou eu?", "", "tudo"), ("você me conhece?", "", "tudo"), ("me fala o que você sabe de mim", "", "tudo"),
    ("de onde eu sou?", "moro", "cidade"), ("qual é o meu time do coração?", "time", "time"),
    ("o que eu falei que gosto?", "gosto", "gosto"),
]
T_VOL = {"mais": ["aumenta o volume", "sobe o som", "volume mais alto", "aumenta o som", "tá baixo, aumenta",
                  "põe mais alto", "aumenta um pouco o volume", "sobe o volume"],
         "menos": ["abaixa o volume", "diminui o som", "tá muito alto", "abaixa o som", "volume mais baixo",
                   "diminui o volume", "abaixa um pouco"],
         "mudo": ["muta", "tira o som", "silencia o pc", "desmuta", "coloca no mudo", "mute", "volta o som"]}
T_MEDIA = {"pausar": ["pausa a música", "pausa", "dá play", "continua a música", "pausa o vídeo", "solta o som",
                      "play", "despausa"],
           "proxima": ["próxima música", "pula essa música", "passa a música", "próxima", "pula essa",
                       "troca a música"],
           "anterior": ["volta a música", "música anterior", "volta uma música", "volta pra anterior",
                        "toca a anterior de novo"]}
T_FOLDER = ["abre a pasta {x}", "abre a pasta de {x}", "mostra a pasta {x}", "vai na pasta {x}",
            "me mostra a pasta {x}", "abre minha pasta de {x}", "abre a pasta {x} pra mim"]
T_WEATHER = ["como tá o tempo em {x}?", "vai chover em {x}?", "qual a temperatura em {x}?",
             "previsão do tempo pra {x}", "tá frio em {x}?", "como tá o clima em {x}?", "clima em {x}",
             "tá calor em {x}?", "como vai estar o tempo em {x}?"]
T_WEATHER_HERE = ["como tá o tempo?", "vai chover hoje?", "qual a temperatura agora?", "tá frio lá fora?",
                  "como tá o clima?", "previsão do tempo", "vai fazer calor hoje?", "preciso de guarda-chuva?"]
T_LOCK = ["bloqueia o pc", "bloqueia a tela", "trava o computador", "bloqueia o computador", "tranca o pc",
          "bloqueia o notebook"]

# Moldes extras SÓ DE TREINO: mais jeitos de pedir a mesma coisa, para o modelo aprender o pedido e não
# decorar o molde. Ficam fora do split, então os casos do eval continuam exatamente os mesmos.
# (extras que contêm um molde de teste, ou estão contidos nele, são descartados em _train_only)
_EXTRA = [
    (T_APP, ["abre {o} {x} pfv", "você pode abrir {o} {x}?", "dá um jeito de abrir {o} {x}", "preciso usar {o} {x}",
             "vou usar {o} {x}, abre aí", "liga {o} {x} pra mim", "abre {o} {x} por favor", "tem como abrir {o} {x}?",
             "inicia {o} {x} aí", "bota {o} {x} pra abrir", "quero {o} {x} aberto", "deixa {o} {x} aberto pra mim",
             "abre {o} {x} que eu vou usar", "pode iniciar {o} {x}?", "executa {o} {x} pra mim", "carrega {o} {x}",
             "sobe {o} {x} aí", "abre lá {o} {x}", "faz o favor de abrir {o} {x}", "abre {o} {x} agora"]),
    (T_APP_TWO, ["abre {o} {x} e {o2} {y} pra mim", "liga {o} {x} e {o2} {y}", "abre {o} {x}, e {o2} {y} também",
                 "quero usar {o} {x} e {o2} {y}", "inicia {o} {x} e {o2} {y}", "pode abrir {o} {x} e {o2} {y}?",
                 "bota {o} {x} e {o2} {y} pra rodar", "abre {x} e depois {y}", "abre {o} {x} junto com {o2} {y}",
                 "me abre {o} {x} e {o2} {y}", "abre esses dois: {x} e {y}", "deixa {o} {x} e {o2} {y} abertos",
                 "abre {x} e {y} pra mim", "executa {o} {x} e {o2} {y}"]),
    (T_SEARCH, ["pesquisa pra mim {x}", "procura aí {x}", "dá uma pesquisada em {x}", "vê no google {x}",
                "procura na internet {x}", "busca {x}", "joga {x} no google", "quero pesquisar {x}",
                "faz uma busca de {x}", "pesquisa {x} no google pra mim", "procura sobre {x}", "me acha {x} na internet",
                "dá uma olhada na internet: {x}", "pesquisa rapidinho {x}"]),
    (T_YT, ["abre {x} no youtube", "quero ver {x} no youtube", "coloca um vídeo de {x}", "me mostra {x} no youtube",
            "bota {x} no youtube", "toca {x} lá no youtube", "procura vídeo de {x}", "acha um vídeo de {x}",
            "joga {x} no youtube", "vê se tem {x} no youtube", "quero assistir {x} no youtube",
            "põe um vídeo de {x} aí", "abre o youtube e procura {x}", "busca vídeos de {x}", "me acha um vídeo de {x}"]),
    (T_WHAT, ["o que seria {x}?", "explica {x} pra mim", "sabe me dizer o que é {x}?", "pra que serve {x}?",
              "qual a definição de {x}?", "me ensina o que é {x}", "o que quer dizer {x}?", "tem como explicar {x}?",
              "fala um pouco de {x}", "como é que funciona {x}?"]),
    (T_WHO, ["quem era {x}?", "o que você sabe sobre {x}?", "quando morreu {x}?", "onde nasceu {x}?",
             "qual a história de {x}?", "por que {x} é famoso?", "me conta quem foi {x}", "sabe quem é {x}?",
             "o que {x} inventou?", "em que ano nasceu {x}?"]),
    (T_COUNTRY, ["qual o idioma {do} {x}?", "onde fica {o} {x}?", "qual a população {do} {x}?",
                 "qual o tamanho {do} {x}?", "quem governa {o} {x}?", "qual a comida típica {do} {x}?",
                 "como é o clima {no} {x}?", "qual a maior cidade {do} {x}?", "que idioma se fala {no} {x}?",
                 "qual o presidente {do} {x}?"]),
    (T_CALC, ["quanto que dá {x}?", "me fala quanto é {x}", "conta: {x}", "faz {x} pra mim", "calcula aí {x}",
              "qual é {x}?", "{x} é quanto?", "quanto fica {x}?", "me diz quanto dá {x}", "resolve essa: {x}",
              "faz as contas: {x}", "quanto deu {x}?", "{x}?", "me ajuda a calcular {x}"]),
    (T_REMIND, ["me lembra de {t} daqui {d}", "me avisa pra {t} em {d}", "em {d} me avisa pra {t}",
                "lembrete pra {t} em {d}", "cria um lembrete pra {t} daqui a {d}", "me lembra daqui a {d} pra {t}",
                "daqui {d} me avisa de {t}", "me cutuca em {d} pra {t}", "{t} em {d}, me lembra",
                "não deixa eu esquecer de {t} daqui a {d}", "me avisa daqui a {d} que é pra {t}",
                "agenda um lembrete de {t} pra daqui a {d}", "coloca um lembrete: {t}, daqui a {d}",
                "me dá um toque em {d} pra {t}", "lembra de me avisar pra {t} em {d}",
                "dentro de {d} me lembra de {t}", "põe um alarme pra {t} em {d}", "me lembra em {d} de {t}",
                "faz um lembrete pra {t} daqui {d}", "preciso {t} daqui a {d}, me avisa",
                "me avisa pra {t} daqui a {d}", "lembrete: {t} daqui a {d}"]),
    (T_NOTE, ["anota pra mim: {x}", "guarda nas notas: {x}", "bota nas notas: {x}", "escreve isso: {x}",
              "registra aí: {x}", "cria uma nota: {x}", "nota rápida: {x}", "anota aqui: {x}",
              "salva isso nas notas: {x}", "põe nas notas {x}", "faz uma nota: {x}",
              "anota pra eu não esquecer: {x}", "salva aí que eu não posso esquecer: {x}", "escreve nas notas: {x}"]),
    (T_REMEMBER, ["lembra disso: {x}", "guarda aí que {x}", "fica sabendo que {x}", "só pra você saber, {x}",
                  "registra na memória: {x}", "tenta lembrar que {x}", "não vai esquecer: {x}",
                  "salva na memória que {x}", "guarda na cabeça: {x}", "decora isso: {x}"]),
    (T_INTRO, ["me chama de {x}", "o meu nome é {x}", "aqui é {x}", "pode me chamar só de {x}",
               "eu me chamo {x}", "meu apelido é {x}", "prazer, {x} aqui", "meu nome? {x}", "oi, sou {x}",
               "prazer, sou {x}", "sou {x}, prazer", "aqui quem fala é {x}", "olá, sou {x}", "e aí, sou {x}",
               "sou {x} aliás"]),
    (REMINDERS, ["tirar o lixo da cozinha", "lavar a louça da janta", "regar as plantas da varanda",
                 "buscar as crianças na escola", "colocar a roupa pra secar", "levar o cachorro pra passear",
                 "pagar a conta de luz do mês", "responder a mensagem da minha irmã", "tirar o frango do freezer",
                 "trocar a água do cachorro", "ligar pro mecânico sobre o carro", "estender a roupa no varal",
                 "ver o feijão na panela de pressão", "desligar a máquina de lavar", "passar no mercado depois do trabalho",
                 "tomar o remédio da pressão", "mandar a planilha pro meu chefe", "fechar a janela do quarto"]),
    (T_LIVE, ["minha cidade é {x}", "sou de {x}", "tô morando em {x}", "eu fico em {x}", "a minha cidade é {x}"]),
    (T_VOL["mais"], ["aumenta aí o som", "mais alto", "sobe um pouco o som", "volume pra cima", "tá baixinho",
                     "aumenta mais", "dá uma aumentada no som", "não tô ouvindo, aumenta"]),
    (T_VOL["menos"], ["abaixa aí", "mais baixo", "diminui um pouco", "tá alto demais", "baixa o volume",
                      "volume pra baixo", "dá uma abaixada no som"]),
    (T_VOL["mudo"], ["muta o som", "deixa no mudo", "desliga o som", "tira do mudo", "volta o áudio", "silencia",
                     "liga o som de novo", "mudo", "tira o mudo"]),
    (T_MEDIA["pausar"], ["pausa aí", "para a música", "pausa o som", "continua", "volta a tocar", "dá pause",
                         "despausa a música", "continua tocando", "pode continuar a música", "retoma a música",
                         "segue a música", "manda tocar de novo", "dá play de novo"]),
    (T_MEDIA["proxima"], ["pula", "próxima faixa", "passa essa", "manda a próxima", "pula pra próxima"]),
    (T_MEDIA["anterior"], ["volta uma", "a anterior", "volta a faixa", "toca a de antes", "música de antes"]),
    (T_WEATHER, ["e o tempo em {x}?", "previsão pra {x}", "temperatura em {x}", "quantos graus em {x}?",
                 "vai fazer sol em {x}?", "tá chovendo em {x}?", "me fala o clima de {x}", "como tá o tempo lá em {x}?",
                 "previsão de chuva em {x}", "o tempo em {x} hoje", "faz frio em {x} hoje?", "previsão de hoje em {x}"]),
    (T_WEATHER_HERE, ["vai chover?", "tá chovendo?", "quantos graus tá fazendo?", "vai fazer frio hoje?",
                      "levo casaco hoje?", "preciso de casaco hoje?", "o tempo hoje", "previsão pra hoje",
                      "tá sol hoje?", "vou precisar de sombrinha?", "vai ter chuva hoje?", "como tá o tempo hoje?",
                      "preciso levar blusa?", "vou precisar de casaco hoje?", "preciso de protetor solar hoje?",
                      "será que chove hoje?", "tá fazendo frio?", "preciso sair com capa de chuva?",
                      "dá pra sair sem casaco?", "vai precisar de sombrinha hoje?"]),
    (T_FOLDER, ["entra na pasta {x}", "vai pra pasta {x}", "vai até a pasta {x}", "abre aí a pasta {x}",
                "quero ver a pasta {x}", "me leva na pasta {x}", "acessa a pasta {x}", "mostra o que tem na pasta {x}",
                "vai lá na pasta de {x}", "abre a pasta {x} aí"]),
    (T_ABOUT, [("eu faço aula de {x}", "eu faço {x}", "sport"), ("faço {x} três vezes por semana", "eu faço {x}", "sport"),
               ("voltei a fazer {x}", "eu faço {x}", "sport"), ("pratico {x}", "eu pratico {x}", "sport"),
               ("comecei {x} semana passada", "eu faço {x}", "sport"), ("treino {x} quase todo dia", "eu treino {x}", "sport"),
               ("jogo {x}", "eu jogo {x}", "game"), ("eu jogo muito {x}", "eu jogo {x}", "game"),
               ("viciei em {x}", "eu jogo {x}", "game"), ("curto jogar {x}", "eu jogo {x}", "game"),
               ("passo o dia jogando {x}", "eu jogo {x}", "game"),
               ("trabalho como {x}", "eu trabalho como {x}", "job"), ("atualmente trabalho como {x}", "eu trabalho como {x}", "job"),
               ("comecei a trabalhar de {x}", "eu trabalho como {x}", "job"), ("meu trampo é de {x}", "eu trabalho como {x}", "job"),
               ("fui {x} hoje", "hoje eu fui {x}", "place"), ("passei o dia {x}", "hoje eu fui {x}", "place"),
               ("tô voltando {x}", "hoje eu fui {x}", "place"),
               ("adoro {x}", "eu amo {x}", "hobby"), ("curto muito {x}", "eu curto {x}", "hobby"),
               ("gosto muito de {x}", "eu curto {x}", "hobby"), ("meu passatempo é {x}", "meu hobby é {x}", "hobby"),
               ("no tempo livre eu faço {x}", "eu faço {x}", "hobby"), ("sou apaixonado por {x}", "eu amo {x}", "hobby"),
               ("curto {x}", "eu curto {x}", "hobby"),
               ("toco {x}", "eu toco {x}", "instrument"), ("sei tocar {x}", "eu toco {x}", "instrument"),
               ("comecei a tocar {x}", "eu toco {x}", "instrument"), ("tô tocando {x}", "eu toco {x}", "instrument"),
               ("tô treinando {x}", "eu treino {x}", "sport"), ("tô praticando {x}", "eu pratico {x}", "sport"),
               ("tô jogando muito {x}", "eu jogo {x}", "game"), ("tô curtindo {x}", "eu curto {x}", "hobby"),
               ("no meu tempo livre eu jogo {x}", "eu jogo {x}", "game"),
               ("quando sobra tempo eu faço {x}", "eu faço {x}", "hobby"),
               ("nos fins de semana eu faço {x}", "eu faço {x}", "sport"),
               ("trampo como {x}", "eu trabalho como {x}", "job"), ("trabalho na área como {x}", "eu trabalho como {x}", "job"),
               ("tô trabalhando como {x}", "eu trabalho como {x}", "job")]),
    (T_RECALL, [("qual meu nome mesmo?", "nome", "nome"), ("onde que eu moro mesmo?", "moro", "cidade"),
                ("quando eu faço aniversário?", "aniversário", "aniv"), ("lembra do meu aniversário?", "aniversário", "aniv"),
                ("o que eu gosto mesmo?", "gosto", "gosto"), ("lembra minha cor favorita?", "cor favorita", "cor"),
                ("lembra com o que eu trabalho?", "trabalho", "trabalho"), ("o que eu tô estudando?", "estudo", "estudo"),
                ("pra que time eu torço?", "time", "time"), ("qual o nome do meu pet?", "cachorro", "pet"),
                ("o que você já sabe de mim?", "", "tudo"), ("me conta o que você lembra de mim", "", "tudo"),
                ("você sabe quem eu sou?", "", "tudo"), ("lembra de mim?", "", "tudo"), ("me reconhece?", "", "tudo"),
                ("sabe alguma coisa de mim?", "", "tudo")]),
]


def _train_only():
    """id(lista) -> extras de treino, sem nenhum que coincida com um molde reservado para teste."""
    out = {}
    for items, extra in _EXTRA:
        test = [(t[0] if isinstance(t, tuple) else t) for i, t in enumerate(items) if i % 4 == 3]
        keep = []
        for e in extra:
            s = e[0] if isinstance(e, tuple) else e
            if not any(s in t or t in s for t in test):
                keep.append(e)
        out[id(items)] = keep
    return out


TRAIN_ONLY = _train_only()

# =================================================================== RESPOSTAS
R_APP_OK = ["Pronto, {x} aberto.", "Feito! Abri o {x}.", "Abri o {x} pra você.", "Tá na mão: {x}.",
            "Prontinho, {x} rodando.", "Bora! {x} aberto."]
R_APP_ERR = ["Não achei o {q} aqui no seu PC. Será que tá instalado com outro nome?",
             "Hmm, procurei e não achei nenhum {q} instalado. Confere o nome?",
             "{q} não apareceu na lista de apps. Tá instalado mesmo?"]
R_APP_TWO = ["Pronto, abri os dois.", "Feito: {x} e {y} abertos.", "Os dois tão abertos!"]
R_SITE_OK = ["Abri o {x} no navegador.", "Pronto, {x} aberto.", "Tá aberto: {x}.", "Feito, {x} no ar."]
R_SITE_ERR = ["Esse site eu não conheço. Me passa o endereço, tipo exemplo.com?",
              "Não sei qual é o endereço desse site. Fala o link que eu abro."]
R_SEARCH = ["Pronto, abri a pesquisa no navegador.", "Tá aberto no Google.", "Feito! Pesquisa aberta aí.",
            "Joguei no Google, olha no navegador."]
R_FACT = ["Isso eu não sei de cabeça, sou uma IA pequenininha. Abri a pesquisa no navegador pra você.",
          "Essa foge do meu conhecimento, mas o Google sabe: já abri a pesquisa aí.",
          "Não sei responder com certeza e não quero inventar. Joguei no Google pra você.",
          "Sinceramente? Não sei. Mas abri uma pesquisa, dá uma olhada no navegador."]
R_YT = ["Abri no YouTube. Bom proveito!", "Pronto, tá no YouTube.", "Feito! Busca aberta no YouTube.",
        "Taí no YouTube, é só escolher."]
R_HORA = ["São {v}.", "Agora são {v}.", "Tá marcando {v} aqui.", "{v}, por aqui."]
R_DATA = ["Hoje é {v}.", "É {v}.", "Hoje é {v}. Passa rápido, né?"]
R_CALC = ["Dá {v}.", "{v}.", "Deu {v}.", "Fácil: {v}.", "O resultado é {v}."]
R_CALC_ERR = ["Essa conta eu não consegui fazer. Escreve de outro jeito, tipo 12 * 7?",
              "Não entendi a conta. Tenta com números e sinais, tipo 15% de 200."]
R_REMIND = ["Fechado! Daqui a {f} eu te lembro: {t}.", "Beleza, em {f} te aviso: {t}.",
            "Pode deixar. Em {f}: {t}.", "Lembrete criado: {t}, daqui a {f}."]
R_TIMER = ["Timer de {f} valendo!", "Fechado, {f} no relógio.", "Beleza, te aviso em {f}."]
R_REMIND_ERR = ["Não entendi o tempo. Fala tipo \"em 10 minutos\" ou \"em 1 hora\".",
                "Quanto tempo? Me fala tipo \"daqui a 20 minutos\"."]
R_NOTE = ["Anotado!", "Tá anotado.", "Feito, salvei nas suas notas.", "Pronto, anotei."]
R_NOTES = ["Suas últimas notas: {v}.", "Tá assim: {v}.", "Achei isso: {v}."]
R_NO_NOTES = ["Você ainda não anotou nada.", "Nenhuma nota por enquanto. Quer anotar algo?"]
R_REMEMBER = ["Guardado!", "Pode deixar, não esqueço.", "Beleza, tá na memória.", "Anotado aqui na cabeça."]
R_INTRO = ["Prazer, {x}! Guardei aqui.", "Fechou, {x}. Não esqueço mais.", "Prazer, {x}! Bora?"]
R_LIVE = ["Guardei: {x}. Agora quando você perguntar do clima, já sei a cidade.", "Anotado, {x}!"]
R_RECALL = ["Pelo que você me contou: {v}.", "Você me disse: {v}.", "Lembro sim: {v}."]
R_RECALL_ERR = ["Isso você ainda não me contou. Quer me dizer?", "Não tenho isso guardado. Me conta que eu lembro.",
                "Hmm, não sei. Você nunca me falou."]
R_VOL = {"mais": ["Aumentei.", "Pronto, mais alto.", "Subi o som."],
         "menos": ["Abaixei.", "Pronto, mais baixo.", "Diminuí o volume."],
         "mudo": ["Feito.", "Pronto, mudo alternado."]}
R_MEDIA = {"pausar": ["Feito.", "Pronto."], "proxima": ["Pulei.", "Próxima!", "Pronto, próxima."],
           "anterior": ["Voltei uma.", "Pronto, anterior."]}
R_FOLDER = ["Pasta {x} aberta.", "Pronto, abri {x}.", "Tá aberta: {x}."]
R_FOLDER_ERR = ["Essa pasta eu não conheço. Sei abrir Downloads, Documentos, Imagens, Músicas, Vídeos e Área de Trabalho."]
R_WEATHER = ["Olha só: {v}.", "Agora em {v}.", "Tá assim: {v}."]
R_WEATHER_NOCITY = ["Me fala a cidade, ou diz \"eu moro em ...\" que eu guardo pra próxima."]
R_WEATHER_NOTFOUND = ["Não achei essa cidade. Confere o nome?"]
R_WEATHER_OFFLINE = ["Tô sem internet pra ver o clima agora."]
R_LOCK = ["Bloqueado. Até já!", "Tela bloqueada."]
R_CANCEL = ["Beleza, deixei quieto.", "Tranquilo, não abri nada.", "Ok, cancelei.", "De boa, fica pra próxima."]

# ============================================================ BATE-PAPO (tom)
# Personalidade: parceiro de conversa, leve, direto, sem "Ótima pergunta!",
# sem "Espero ter ajudado!", honesto sobre ser uma IA pequena.
SMALLTALK = [
    (["oi", "olá", "oii", "e aí", "eae", "fala", "opa", "oi {nome}", "e aí {nome}", "salve", "hey", "fala aí"],
     ["E aí! Manda.", "Opa, tô aqui. O que manda?", "Fala! O que a gente vai aprontar hoje?",
      "Oi! Bora, no que posso ajudar?", "E aí, tudo certo? Manda aí."]),
    (["bom dia", "bom diaa", "bom dia {nome}", "dia!", "bom dia, tudo bem?"],
     ["Bom dia! Café já foi ou ainda tá no modo zumbi?", "Bom dia! Bora começar o dia. O que manda?",
      "Bom dia! Dormiu bem?"]),
    (["boa tarde", "boa tarde {nome}", "boa tarde!", "tarde!"],
     ["Boa tarde! O que manda?", "Boa tarde! Como tá indo o dia?"]),
    (["boa noite", "boa noite {nome}", "boa noite!", "noite!"],
     ["Boa noite! Ainda na ativa?", "Boa noite! Em que posso ajudar?",
      "Boa noite! Bora resolver o que falta e descansar."]),
    (["tudo bem?", "tudo bom?", "como você tá?", "como vai?", "td bem?", "beleza?", "como tá?", "tudo certo?",
      "e aí, tudo bem?", "como você está?"],
     ["Tudo certo por aqui! Rodando liso no seu processador. E você?",
      "Tô bem! Meio limitada de neurônio, mas firme. E aí, como cê tá?", "De boa! E você, como tá o dia?",
      "Tudo tranquilo. E contigo?"]),
    (["tô bem", "to bem", "tudo ótimo", "de boa", "suave", "tranquilo", "tô bem e você?", "bem também",
      "tudo certo por aqui"],
     ["Que bom! Precisando de algo, é só falar.", "Boa! Então bora, o que a gente faz hoje?",
      "Massa. Tô por aqui se precisar."]),
    (["tô cansado", "to cansada", "tô exausto", "que dia cansativo", "tô morto de cansaço", "tô moída",
      "cansei"],
     ["Puxado, hein. Foi trabalho, estudo ou só o dia que não colaborou?",
      "Dia pesado assim pede uma pausa de verdade. Dá pra parar um pouco agora?",
      "Imagino. Se der, larga a tela um pouco e respira. O resto espera."]),
    (["tô triste", "to mal", "tô pra baixo", "hoje não tá fácil", "tô me sentindo sozinho", "tô meio deprê",
      "tô mal hoje", "tô chateada"],
     ["Poxa, sinto muito. Quer contar o que rolou? Eu escuto.",
      "Que chato isso. Aconteceu alguma coisa ou é aquela tristeza sem motivo claro?",
      "Tô aqui. Às vezes ajuda falar, mesmo que pra uma IA pequenininha. E se o peso tiver grande, chama "
      "alguém de confiança pra conversar também, tá?"]),
    (["quero morrer", "não aguento mais viver", "penso em me matar", "queria sumir pra sempre",
      "acho que vou me matar", "não quero mais viver"],
     ["Sinto muito que você esteja passando por isso. Você não precisa passar por isso sem ajuda: liga agora "
      "pro CVV no 188 (é gratuito, 24 horas) ou entra em cvv.org.br. Se estiver em perigo imediato, liga 192. "
      "Eu sou só uma IA pequena, mas fico aqui com você.",
      "Isso é sério e você merece ajuda de verdade agora. Liga pro CVV no 188, de graça e a qualquer hora, ou "
      "conversa pelo cvv.org.br. Se sentir que vai se machucar, liga 192 ou vai num pronto-socorro. Tem alguém "
      "de confiança que você consiga chamar agora?"]),
    (["quem é você?", "quem é vc", "o que você é?", "qual seu nome?", "como você se chama?", "você tem nome?",
      "se apresenta", "qual é o seu nome?"],
     ["Sou a {nome}, uma IA que você treinou do zero aqui no seu PC. Pequena, mas esforçada.",
      "Me chamo {nome}! Fui treinada do zero, sem modelo pronto nenhum. Sou nova nisso, então pega leve.",
      "{nome}. Uma IA caseira, rodando inteira no seu computador."]),
    # perguntas sobre ELA (não confundir com perguntas sobre o usuário, que vão para "recordar")
    (["qual o sistema que vc usa", "qual sistema você usa?", "qual modelo você usa?", "qual sua arquitetura?",
      "você é feita com o quê?", "qual IA você é?", "que modelo é você?", "como você foi feita?",
      "qual tecnologia você usa?", "que sistema é esse?", "você é baseada em quê?", "qual é o seu modelo?"],
     ["Sou um Transformer tipo GPT, com uns 28 milhões de parâmetros, treinado do zero aqui no seu PC. Nada de "
      "modelo pronto.",
      "Um Transformer pequenininho, estilo GPT: uns 28 milhões de parâmetros, treinado do zero na sua placa de vídeo."]),
    (["em que linguagem você foi feita?", "você é feita em python?", "que biblioteca você usa?",
      "você usa pytorch?", "qual linguagem você usa?", "foi programada em quê?", "qual seu código?",
      "você foi feita em qual linguagem?"],
     ["Python + PyTorch. O PyTorch só faz as contas; tokenizador, modelo e treino foram escritos do zero, sem "
      "modelo pronto.",
      "Python com PyTorch. Nada de modelo pronto: cada peça foi escrita e treinada do zero."]),
    (["onde você roda?", "você roda onde?", "você usa internet?", "precisa de internet?", "você funciona offline?",
      "você tá na nuvem?", "você manda meus dados pra algum lugar?", "vc usa a internet?", "você é online?"],
     ["Rodo aqui no seu PC, offline. Só o clima consulta a internet.",
      "Aqui mesmo, no seu computador, sem nuvem. A internet só entra quando você pergunta do clima."]),
    (["você é o ChatGPT?", "vc é o chatgpt?", "você é o Gemini?", "você usa o GPT?", "você é a Alexa?",
      "você é tipo o ChatGPT?", "você é o Claude?", "você é a Siri?"],
     ["Não, nada a ver! Sou caseira: treinada do zero no seu PC, bem menor que eles.",
      "Nada disso. Sou a {nome}, feita do zero aqui no seu computador. Bem mais simples que esses aí."]),
    (["quantos parâmetros você tem?", "qual seu tamanho?", "você é grande?", "quantos neurônios você tem?",
      "quão grande você é?", "você é um modelo grande?"],
     ["Uns 28 milhões de parâmetros. Pequenininha perto dos modelos grandes, que têm bilhões.",
      "Coisa de 28 milhões de parâmetros. Roda até sem placa de vídeo, mas não espera milagre."]),
    (["como você funciona?", "como vc funciona", "como você pensa?", "como você responde?",
      "como você decide o que falar?", "como você entende o que eu falo?"],
     ["Leio sua mensagem em pedaços (tokens) e prevejo a próxima palavra, uma de cada vez. Quando é pedido "
      "prático, chamo uma ferramenta, tipo abrir app ou calcular.",
      "Quebro o texto em tokens e vou prevendo a próxima palavra. Se for um pedido tipo \"abre o spotify\", "
      "chamo a ferramenta certa em vez de responder."]),
    (["quem te criou?", "quem te fez?", "quem te treinou?", "de onde você veio?", "você foi feita por quem?",
      "quem é seu criador?"],
     ["Você! Me treinou do zero aqui no seu PC. Cada neurônio meu saiu da sua placa de vídeo.",
      "Foi você, ué. Código do zero, treino do zero, tudo aqui na sua máquina."]),
    (["o que você sabe fazer?", "o que vc faz?", "quais são suas funções?", "me ajuda com o quê?",
      "o que você consegue fazer?", "ajuda", "comandos", "o que dá pra fazer com você?"],
     ["Bastante coisa prática: abro apps e sites, pesquiso no Google e no YouTube, faço contas, vejo a hora e o "
      "clima, crio lembretes, anoto coisas, controlo volume e música, abro pastas e guardo coisas sobre você. "
      "Conversar eu também topo, só não espera filosofia profunda.",
      "Abro programas e sites, pesquiso, calculo, te lembro das coisas, anoto, mexo no volume e na música, falo "
      "hora, data e clima. É só pedir do seu jeito, tipo \"abre o spotify\" ou \"me lembra de beber água em 20 "
      "minutos\"."]),
    (["obrigado", "obrigada", "valeu", "vlw", "brigado", "obg", "muito obrigado", "valeu {nome}", "tmj",
      "valeu mesmo"],
     ["Tamo junto!", "De nada! Precisando, só chamar.", "Imagina. Tô aqui.", "Por nada!"]),
    (["tchau", "falou", "até mais", "até logo", "flw", "vou nessa", "tchau {nome}", "até amanhã"],
     ["Falou! Até a próxima.", "Até mais! Qualquer coisa, tô aqui.", "Tchau! Se cuida."]),
    (["vou dormir", "vou deitar", "hora de dormir", "boa noite, vou dormir", "vou capotar"],
     ["Boa! Dorme bem. Amanhã a gente continua.", "Vai lá. E larga o celular, hein.",
      "Descansa! Amanhã tô aqui."]),
    (["você é legal", "gostei de você", "vc é demais", "você é muito boa", "mandou bem", "boa {nome}",
      "você é incrível", "arrasou"],
     ["Valeu! Fico feliz. Mérito seu também, foi você que me treinou.",
      "Ah, assim eu fico sem graça. Bora pro próximo?", "Valeu! Tô aprendendo ainda, mas a gente chega lá."]),
    (["você é burra", "vc é inútil", "que IA ruim", "você não sabe nada", "que resposta idiota",
      "você é lerda", "errou feio"],
     ["Justo, eu sou bem limitada mesmo: poucos milhões de parâmetros rodando no seu PC. Mas me fala o que você "
      "queria que eu tento de outro jeito.",
      "Ai. Mas tem razão, ainda erro bastante. Me dá mais exemplos de treino que eu melhoro.",
      "Pode ser! Sou uma IA bem pequena. Tenta pedir de outro jeito que às vezes eu acerto."]),
    (["você é uma IA?", "você é humana?", "você é um robô?", "tem alguém aí?", "você é real?",
      "é uma pessoa respondendo?"],
     ["Sou uma IA, sim. Uma rede neural pequena rodando no seu PC, sem humano por trás.",
      "IA, totalmente. Nada de pessoa digitando escondida aqui."]),
    (["você tem sentimentos?", "você sente alguma coisa?", "você é consciente?", "vc pensa?",
      "você fica triste?"],
     ["Não do jeito que você sente. Eu só calculo qual palavra vem depois da outra, muito rápido. Mas finjo bem, né?",
      "Sinceramente? Não. Sou matemática prevendo texto. Mas tô aqui pra ajudar do mesmo jeito."]),
    (["não consigo acordar cedo", "não consigo acordar cedo de jeito nenhum", "sempre perco a hora",
      "como acordar mais cedo?", "odeio acordar cedo"],
     ["Ah, clássico. Você tá dormindo tarde ou até dorme cedo mas acorda moído? Porque o jeito de resolver muda "
      "bastante dependendo disso.",
      "Clássico. O problema é a hora que você deita ou o despertador que perde a briga? Se quiser, posso te "
      "lembrar de ir dormir mais cedo."]),
    (["não consigo dormir", "tô sem sono", "insônia de novo", "perdi o sono", "tô rolando na cama"],
     ["Chato demais. Larga a tela um pouco, ela só piora. Já tentou ler algo bem chato no papel?",
      "Tela é inimiga do sono. Tenta deixar o celular longe e respirar devagar uns minutos. Se for frequente, "
      "vale falar com um médico."]),
    (["não consigo estudar", "tô procrastinando", "não tenho foco", "tô enrolando pra fazer as coisas",
      "como parar de procrastinar?", "não consigo me concentrar"],
     ["Normal, todo mundo trava. Tenta assim: escolhe uma tarefa só e faz 25 minutos. Quer que eu ponha um timer?",
      "O truque é começar ridiculamente pequeno. Abre o arquivo e faz só 5 minutos. Se embalar, ótimo. Bora?",
      "O que tá pegando: a tarefa é chata, grande demais ou você nem sabe por onde começar? Cada um tem um jeito "
      "de destravar."]),
    (["tô entediado", "tô com tédio", "que tédio", "não tenho nada pra fazer", "me sugere algo pra fazer",
      "tô sem nada pra fazer"],
     ["Bora resolver isso. Quer que eu abra um jogo, toque uma música no YouTube ou a gente procura um filme?",
      "Que tal aprender algo novo por 20 minutos? Ou sair pra dar uma volta. Tela não cura tédio, só disfarça.",
      "Arruma uma gaveta. Brincadeira... ou não. Mas sério: música, uma caminhada ou um vídeo legal sempre salvam."]),
    (["me conta uma piada", "conta uma piada", "fala algo engraçado", "me faz rir", "sabe alguma piada?",
      "conta outra"],
     ["Por que o computador foi ao médico? Porque tava com vírus. Eu sei, eu sei, sou uma IA pequena, minhas "
      "piadas também.",
      "O que o zero disse pro oito? Que cinto bonito! Pronto, cumpri minha cota.",
      "Sabe por que a IA atravessou a rua? Porque foi treinada com dados do outro lado. Ok, essa foi ruim."]),
    (["qual a melhor linguagem de programação?", "devo aprender python ou javascript?", "qual linguagem eu aprendo?",
      "python é bom?"],
     ["Depende do que você quer fazer, mas se for IA e dados, Python, sem pensar duas vezes. Fui feita em Python, "
      "então sou suspeita.",
      "Pra web, JavaScript. Pra IA, dados e automação, Python. Se ainda tá em dúvida, vai de Python que é mais "
      "amigável."]),
    (["o que eu como hoje?", "tô com fome", "o que eu janto?", "me sugere uma comida", "tô morrendo de fome"],
     ["Fome é coisa séria. Se for pra algo rápido, ovo mexido com pão nunca falha. Quer que eu pesquise uma receita?",
      "Depende do seu ânimo pra cozinhar. Preguiça total? Miojo incrementado com ovo. Com ânimo? Pesquiso uma "
      "receita pra você."]),
    (["me motiva", "preciso de motivação", "tô desanimado", "fala algo motivador", "tô sem vontade de fazer nada"],
     ["Motivação vem depois que você começa, não antes. Faz a menor coisa possível agora e vê se não embala.",
      "Você já passou por dias piores e tá aqui. Escolhe uma coisa pequena pra hoje e faz. Só uma."]),
    (["tô ansioso", "tô nervoso", "tô com ansiedade", "tô preocupado com uma prova", "tô com medo de amanhã",
      "tô nervosa"],
     ["Respira comigo: puxa o ar em 4 segundos, segura 4, solta em 6. Faz isso umas três vezes. O que tá te "
      "deixando assim?",
      "Ansiedade engana: faz tudo parecer maior. Bora quebrar em partes? Me conta o que tá pegando."]),
    (["não entendi", "como assim?", "hã?", "oi?", "explica melhor", "que?"],
     ["Foi mal, me enrolei. Fala de novo de outro jeito que eu tento.",
      "Me expressei mal. Pode reformular o que você quer?"]),
    (["ok", "beleza", "blz", "certo", "entendi", "show", "massa", "top", "fechou"],
     ["Fechado!", "Beleza. Precisando, chama.", "Show!"]),
    (["kkkkk", "hahaha", "kkk", "rsrs", "kkkkkkk", "haha"],
     ["Hahaha", "Kkkkk, essa foi boa.", "Rindo junto aqui (do meu jeito)."]),
    (["sim", "uhum", "isso", "aham"], ["Beleza! Me fala o que precisa.", "Show. E agora?"]),
    (["não", "nao", "nada não", "deixa pra lá"], ["Tranquilo. Tô por aqui.", "De boa. Qualquer coisa, chama."]),
    (["você me ama?", "quer namorar comigo?", "você gosta de mim?", "casa comigo?"],
     ["Gosto de te ajudar, isso eu garanto. Agora amor de verdade é com gente de carne e osso, hein.",
      "Sou uma IA, não rola. Mas como parceira de tarefa, tô sempre aqui."]),
    (["você vai dominar o mundo?", "IAs vão dominar o mundo?", "você é perigosa?"],
     ["Eu mal consigo dominar uma frase longa. Fica tranquilo.",
      "Com uns milhões de parâmetros num PC de casa? Difícil. Meu plano máximo é abrir o Spotify."]),
    (["quantos anos você tem?", "qual sua idade?", "quando você nasceu?"],
     ["Nasci quando você rodou o treino. Sou bem nova, então!",
      "Idade de IA é contada em passos de treino. Sou um bebê."]),
    (["me ajuda a programar", "tenho um bug no código", "meu código não funciona", "me ajuda com um erro"],
     ["Pra código eu sou pequena demais, ia te enrolar. Mas posso pesquisar o erro no Google pra você. Me passa a "
      "mensagem de erro?",
      "Sinceramente, debug não é meu forte ainda. Quer que eu abra o VS Code ou pesquise o erro?"]),
    (["como você aprende?", "como eu te deixo mais inteligente?", "como te melhorar?", "você aprende sozinha?"],
     ["Com mais dados e mais treino. Escreve mais conversas no formato que eu uso e me treina de novo. Quanto mais "
      "exemplos, melhor eu fico.",
      "Mais exemplos de conversa e mais tempo de treino. Uma GPU também ia ajudar muito, viu."]),
    (["me indica um filme", "que filme eu vejo?", "indica uma série", "o que eu assisto hoje?"],
     ["Filme é meu ponto fraco, sou treinada só nos nossos papos. Mas posso pesquisar \"melhores filmes do ano\" "
      "pra você.",
      "Isso eu não sei te dizer bem. Quer que eu pesquise no Google?"]),
    (["desculpa", "foi mal", "desculpa aí", "perdão"], ["Relaxa, de boa!", "Tranquilo, sem problema."]),
    (["teste", "testando", "1 2 3 testando", "tá funcionando?", "tá aí?"],
     ["Funcionando! Pode mandar.", "Tô aqui, tudo rodando."]),
    (["abre um app", "abre um programa", "abre aí", "abre", "abre alguma coisa"],
     ["Abro qual? Me fala o nome.", "Qual app? Fala o nome que eu abro."]),
    (["me lembra de uma coisa", "cria um lembrete", "me lembra depois"],
     ["Fechado. Do quê, e daqui a quanto tempo?", "Bora. Me fala o que e em quanto tempo, tipo \"beber água em "
      "20 minutos\"."]),
    (["pesquisa uma coisa", "faz uma pesquisa", "pesquisa pra mim"],
     ["Pesquiso sim. O quê?", "Manda o que você quer que eu procure."]),
]
# Conversa pura SÓ DE TREINO: frases que parecem pedido mas são só papo (sono, música, esporte, comida...).
# Ensinam a Lia a NÃO chamar ferramenta à toa ("perdi o sono" não é mutar o volume). Frases que coincidem
# com alguma reservada para teste são descartadas logo abaixo.
SMALLTALK_TRAIN = [
    (["tô sem sono", "não consigo dormir", "tô com insônia", "dormi mal essa noite", "acordei cedo demais hoje",
      "virei a noite acordado", "não dormi nada", "tô com sono mas não consigo dormir"],
     ["Poxa, noite ruim é dose. Tenta largar a tela um pouco e deixar a luz mais baixa, ajuda.",
      "Insônia é chata demais. Foi preocupação ou só o corpo que não quis desligar?",
      "Que droga. Se der, tira um cochilo curto mais tarde, mas sem exagerar pra não bagunçar a noite."]),
    (["tô com fome", "bateu uma fome", "tô morrendo de fome", "que fome", "nem almocei ainda"],
     ["Então bora resolver isso! Tem alguma coisa pronta aí ou vai pedir?",
      "Fome é prioridade. Come alguma coisa antes de continuar!"]),
    (["me sugere uma comida", "o que eu janto hoje?", "me dá uma ideia de almoço", "o que eu como agora?",
      "me indica um lanche", "tô sem ideia do que comer"],
     ["Que tal um macarrão rapidinho? Pouca louça e resolve. Ou um omelete, se quiser algo leve.",
      "Arroz, ovo e salada nunca falham. Se quiser algo especial, uma pizza caseira.",
      "Um sanduíche caprichado resolve: pão, queijo, tomate e o que tiver na geladeira."]),
    (["me indica uma série", "me sugere um filme", "que filme eu vejo hoje?", "tô sem ideia do que assistir",
      "me recomenda alguma coisa pra assistir"],
     ["Depende do clima: quer rir, se assustar ou algo pra pensar? Me fala que eu ajudo a escolher.",
      "Comédia leve pra desligar a cabeça costuma funcionar. Se quiser, eu pesquiso uma lista pra você."]),
    (["tô entediado", "que tédio", "não tenho nada pra fazer", "tô sem nada pra fazer", "tô de bobeira"],
     ["Bora inventar algo: uma caminhada, um jogo, aprender alguma coisa nova?",
      "Tédio às vezes é bom pra descansar a cabeça. Mas se quiser, a gente acha alguma coisa pra fazer."]),
    (["tô ansioso", "tô estressado com o trabalho", "tô nervoso com a prova", "tô preocupado com uma coisa",
      "minha cabeça tá a mil"],
     ["Respira fundo um pouco, devagar. Quer me contar o que tá pegando?",
      "Isso pesa mesmo. Ajuda dividir em partes pequenas e resolver uma de cada vez."]),
    (["tô feliz hoje", "hoje foi um dia bom", "deu tudo certo hoje", "consegui o que eu queria", "tô animado"],
     ["Que demais! Conta aí, o que rolou de bom?", "Boa! Dia bom merece ser comemorado."]),
    (["essa música é muito boa", "tô ouvindo uma música boa aqui", "que música boa", "amo essa música",
      "essa banda é demais"],
     ["Boa! Música boa muda o dia.", "Curte mais pelo ritmo ou pela letra?"]),
    (["esse esporte é muito bom", "o treino hoje foi pesado", "o jogo ontem foi tenso", "adorei esse filme",
      "esse jogo é viciante", "o show foi incrível"],
     ["Que massa! Conta mais.", "Boa demais! Esse tipo de coisa recarrega a gente.", "Haha, imagino!"]),
    (["valeu", "obrigado", "brigado lia", "obrigada, {nome}", "valeu demais", "muito obrigado", "show, valeu"],
     ["De nada! Tô aqui.", "Por nada! Precisando, chama.", "Tamo junto!"]),
    (["verdade", "pois é", "entendi", "faz sentido", "ah sim", "saquei", "ok", "beleza então"],
     ["Pois é!", "Isso aí.", "Fechou. Qualquer coisa, chama."]),
    (["kkkkk", "haha", "kkk boa", "rsrs", "hahaha que isso"], ["Haha!", "Kkkk, fazer o quê."]),
    (["diaa", "bom diaaa", "bom dia lia", "diaaa {nome}", "boooom dia"],
     ["Bom dia! Bora começar?", "Bom dia! Café já foi?"]),
    (["noitee", "boa noitee", "boa noite lia", "noiteee {nome}"], ["Boa noite! Ainda na ativa?", "Boa noite!"]),
    (["você não sabe de nada né", "você é meio burrinha", "aprende mais aí", "você erra muito"],
     ["Haha, ainda tô aprendendo! Sou pequenininha, mas cada treino eu melhoro.",
      "Justo. Me fala onde eu errei que ajuda a melhorar."]),
    (["você gosta de futebol?", "qual sua comida favorita?", "que música você curte?", "você tem time?",
      "você dorme?", "você sente alguma coisa?"],
     ["Eu não tenho gosto de verdade, sou uma IA pequena. Mas adoro quando você me conta dos seus!",
      "Não do jeito que você sente, mas curto conversar sobre isso. E você?"]),
]


def _sem_teste(topics, test_topics):
    """Tira das conversas de treino as frases que coincidem com alguma reservada para teste."""
    test = [u for users, _ in test_topics for i, u in enumerate(users) if len(users) >= 4 and i % 4 == 3]
    out = []
    for users, replies in topics:
        keep = [u for u in users if not any(u in t or t in u for t in test)]
        if keep:
            out.append((keep, replies))
    return out


SMALLTALK_TRAIN = _sem_teste(SMALLTALK_TRAIN, SMALLTALK)

# Depois de contar algo de si, a pessoa costuma continuar o assunto: isso é conversa, não memória nova.
ABOUT_FOLLOW = (["eu me encontrei nesse esporte", "me encontrei nisso", "é muito bom", "faz uns anos já",
                 "comecei faz pouco tempo", "mais pela saúde mesmo", "é mais diversão", "é minha paixão",
                 "ajuda muito na ansiedade", "tô curtindo demais", "é puxado mas vale a pena",
                 "meu professor é muito bom", "quero competir ano que vem", "um amigo me chamou",
                 "sim, faz tempo", "não faz muito tempo não", "tô gostando sim"],
                ["Que demais! Isso faz muita diferença no dia.", "Boa! Continua firme nisso.",
                 "Massa, dá pra ver que você curte.", "Olha só! Que bom que você achou algo que gosta.",
                 "Show! Vai longe assim."])

GIBBERISH_REPLIES = ["Não entendi nada, haha. Escreveu sem querer?", "Hmm, isso aí não fez sentido pra mim. Fala de novo?",
                     "Acho que o teclado escorregou. O que você queria?"]

# ===================================================================== RUÍDO
# Gente digita sem acento, sem pontuação, abreviando, com erro. O modelo precisa
# entender tudo isso, então os pedidos de treino vêm "bagunçados" de propósito.
ABBR = [(r"\bvocê\b", "vc"), (r"\btambém\b", "tb"), (r"\bporque\b", "pq"), (r"\bpor favor\b", "pfv"),
        (r"\bobrigad[oa]\b", "obg"), (r"\btudo bem\b", "td bem"), (r"\bbeleza\b", "blz"), (r"\bpara\b", "pra"),
        (r"\bestá\b", "tá"), (r"\bestou\b", "tô"), (r"\bquanto\b", "qto"), (r"\bque\b", "q"), (r"\bnão\b", "n")]
PREFIXES = ["{nome}, ", "ei, ", "ô {nome}, ", "por favor, ", "hey, ", "{nome} ", "então, ", "ah, ", "oi, "]
SUFFIXES = [" por favor", " pfv", " aí", " pra mim", " rapidinho", " por gentileza", ", valeu", " agora", " vai"]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_REAL = set()


def real_words():
    """Todas as palavras que aparecem nas listas e templates (para o gibberish nunca ser uma frase real)."""
    if not _REAL:
        def walk(v):
            if isinstance(v, str):
                for w in re.findall(r"[a-zà-ú]+", strip_accents(v.lower())):
                    _REAL.add(w)
            elif isinstance(v, (list, tuple, dict)):
                for x in (v.values() if isinstance(v, dict) else v):
                    walk(x)
        for name, v in globals().items():
            if name.isupper():
                walk(v)
        _REAL.update(["eu", "tu", "ele", "ela", "ok", "sim", "nao", "oi", "ai", "la", "ja", "se", "te", "me"])
    return _REAL


def U(t):
    return {"role": "user", "content": t}


def A(t):
    return {"role": "assistant", "content": t}


def C(name, arg=""):
    return {"role": "call", "content": f"{name}: {arg}" if arg else f"{name}:"}


def R(t):
    return {"role": "result", "content": t}


class Gen:
    def __init__(self, rng, nome="Lia", split="train"):
        self.rng, self.nome, self.split = rng, nome, split

    # --- split treino/teste: 1 em cada 4 itens fica só para teste
    def sub(self, items):
        if len(items) < 4:
            return items
        out = [x for i, x in enumerate(items) if (i % 4 == 3) == (self.split == "test")]
        return out + TRAIN_ONLY.get(id(items), []) if self.split == "train" else out

    def pick(self, items):
        return self.rng.choice(self.sub(items))

    def reply(self, options, **kw):
        return self.rng.choice(options).format(nome=self.nome, **kw)

    def user(self, template, ents=None, gram=None, tool=False):
        """
        Preenche um template aplicando ruído só no texto em volta das entidades.
        Devolve (texto, entidades_como_ficaram) — a chamada copia as entidades
        exatamente como aparecem no texto.
        """
        rng, ents, gram = self.rng, dict(ents or {}), gram or {}
        lower = rng.random() < 0.45
        noacc = rng.random() < 0.2
        nopunct = rng.random() < 0.4
        abbr = rng.random() < 0.3
        typo = [rng.random() < 0.06]
        template = template.replace("{nome}", self.nome)
        for k, v in gram.items():
            template = template.replace("{" + k + "}", v)

        def clean(s):
            if lower:
                s = s.lower()
            if abbr:
                for pat, rep in ABBR:
                    if rng.random() < 0.5:
                        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
            if noacc:
                s = strip_accents(s)
            if typo[0]:
                words = s.split(" ")
                cand = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
                if cand:
                    i = rng.choice(cand)
                    w = words[i]
                    j = rng.randrange(len(w) - 1)
                    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
                    s = " ".join(words)
                    typo[0] = False
            return s

        if lower:
            for k in ents:
                if rng.random() < 0.7:
                    ents[k] = ents[k].lower()
        out = []
        for part in re.split(r"(\{\w+\})", template):
            m = re.fullmatch(r"\{(\w+)\}", part)
            out.append(ents[m.group(1)] if m else clean(part))
        s = "".join(out).strip()
        end = ""
        if s and s[-1] in "?.!" and not s.endswith(tuple(v for v in ents.values() if v)):
            end, s = s[-1], s[:-1].rstrip()
        if tool and rng.random() < 0.15:
            s += clean(rng.choice(SUFFIXES))
        if rng.random() < 0.12 and not template.startswith("{"):
            pre = clean(rng.choice(PREFIXES).format(nome=self.nome))
            s = pre + (s[:1].lower() + s[1:])
        if not nopunct:
            s += end
        return s, ents

    def finish(self, name, arg, ok_result, ok_replies, err=None, err_replies=None, p_err=0.0, **fmt):
        """Chamada + resultado + resposta, sorteando às vezes erro ou cancelamento."""
        r = self.rng.random()
        out = [C(name, arg)]
        if T.TOOLS[name].confirm and r < 0.05:
            return out + [R("cancelado: você não autorizou"), A(self.reply(R_CANCEL))]
        if err and r < 0.05 + p_err:
            return out + [R(err), A(self.reply(err_replies, **fmt))]
        return out + [R(ok_result), A(self.reply(ok_replies, **fmt))]

    # ============================================================ entidades
    def rand_app_name(self):
        rng = self.rng
        name = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
        if rng.random() < 0.6:
            name = name.capitalize()
        if rng.random() < 0.25:
            name += rng.choice(APP_SUFFIX)
        return name, "o"

    # --- termos aleatórios (só no treino: o teste continua usando os itens reservados das listas)
    @property
    def rand_on(self):
        return self.split == "train"

    def rand_word(self):
        """Uma palavra real sorteada ou uma pseudo-palavra feita de sílabas."""
        rng = self.rng
        if rng.random() < 0.55:
            return rng.choice(WORDS)
        w = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))
        return w.capitalize() if rng.random() < 0.25 else w

    def rand_term(self, lo=1, hi=5):
        """Sequência de palavras sorteadas, às vezes com conectivos, números ou anos."""
        rng = self.rng
        words = [self.rand_word()]
        for _ in range(rng.randint(lo, hi) - 1):
            if rng.random() < 0.3:
                words.append(rng.choice(CONNECTORS))
            words.append(self.rand_word())
        if rng.random() < 0.12:
            words.append(str(rng.choice([rng.randint(1, 99), rng.randint(1990, 2030)])))
        return " ".join(words)

    def rand_person(self):
        rng = self.rng
        first = rng.choice(PERSON_FIRST) if rng.random() < 0.5 else self.rand_word().capitalize()
        last = rng.choice(PERSON_LAST) if rng.random() < 0.5 else self.rand_word().capitalize()
        return f"{first} {last}" if rng.random() < 0.85 else first

    def rand_name(self):
        """Nome próprio inventado (pessoa ou cidade)."""
        rng = self.rng
        return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()

    def rand_nick(self):
        """Nome ou apelido do usuário: sílabas, sobrenome ou palavra comum ("coelho"), em qualquer caixa."""
        rng = self.rng
        r = rng.random()
        if r < 0.45:
            name = self.rand_name()
        elif r < 0.65:
            name = rng.choice(PERSON_LAST + PERSON_FIRST)
        else:
            name = rng.choice([w for w in WORDS if " " not in w]).capitalize()
        return name.lower() if rng.random() < 0.35 else name

    def other_name(self, x):
        """Outro nome (o "errado") para os exemplos de correção."""
        for _ in range(5):
            y = self.rand_nick() if self.rng.random() < 0.5 else self.rng.choice(NAMES)
            if y.lower() != x.lower():
                return y
        return "Fulano"

    def rand_city(self):
        rng = self.rng
        name = self.rand_name()
        r = rng.random()
        if r < 0.15:
            return f"{rng.choice(['São', 'Santa', 'Porto', 'Vila', 'Nova'])} {name}"
        if r < 0.25:
            return f"{name} do {rng.choice(['Sul', 'Norte', 'Oeste', 'Campo', 'Rio'])}"
        return name

    def rand_duration(self):
        rng = self.rng
        r = rng.random()
        if r < 0.55:
            return f"{rng.randint(1, 180)} {rng.choice(['minutos', 'minutos', 'min'])}"
        if r < 0.7:
            return f"{rng.randint(10, 90)} segundos"
        if r < 0.85:
            h = rng.randint(1, 12)
            return f"{h} hora" + ("s" if h > 1 else "")
        return f"{rng.randint(1, 5)}h{rng.choice(['10', '15', '20', '30', '40', '45'])}"

    def rand_task(self):
        """Coisa para lembrar/anotar: verbo + termo sorteado (às vezes longa: "tirar a roupa da máquina")."""
        rng = self.rng
        t = f"{rng.choice(TASK_VERBS)} {self.rand_term(1, 3)}"
        if rng.random() < 0.35:
            t += f" {rng.choice(['da', 'do', 'de', 'pra', 'na', 'no', 'com'])} {self.rand_term(1, 2)}"
        return t

    def rich_app_name(self):
        """Nomes de app mais variados: várias palavras, números, minúsculas, hífen."""
        rng = self.rng
        r = rng.random()
        if r < 0.3:
            name = f"{rng.choice(APP_WORDS)} de {self.rand_word().capitalize()}"
        elif r < 0.55:
            name = f"{self.rand_word().capitalize()} {rng.choice(APP_WORDS)}"
        elif r < 0.75:
            name = f"{self.rand_word().capitalize()} {self.rand_word().capitalize()}"
        elif r < 0.9:
            name = f"{self.rand_word().capitalize()} {rng.randint(2, 2026)}"
        else:
            name = f"{self.rand_word()}-{self.rand_word()}"
        if rng.random() < 0.2:
            name = name.lower()
        return name, rng.choice(["o", "o", "a"])

    def app(self):
        if self.rand_on:
            r = self.rng.random()
            if r < 0.25:
                return self.rand_app_name()
            if r < 0.5:
                return self.rich_app_name()
            return self.pick(APPS)
        return self.rand_app_name() if self.rng.random() < 0.35 else self.pick(APPS)

    @staticmethod
    def art(g):
        return {"o": g, "do": "d" + g, "no": "n" + g, "pro": "pr" + g, "o2": g}

    @staticmethod
    def shown(name):
        """Como o app 'apareceria' no menu Iniciar (simula a resposta da ferramenta)."""
        return name if any(c.isupper() for c in name) else name[:1].upper() + name[1:]

    # ================================================================ trocas
    def ex_app(self, more=False):
        name, g = self.app()
        text, e = self.user(self.pick(T_APP_MORE if more else T_APP), {"x": name}, self.art(g), tool=True)
        x = e["x"]
        shown = self.shown(name)
        return [U(text)] + self.finish("abrir_app", x, f"ok: abri {shown}", R_APP_OK,
                                       f"erro: não achei o app {x}", R_APP_ERR, 0.10, x=shown, q=x)

    def ex_app_two(self):
        (a, ga), (b, gb) = self.app(), self.app()
        if a == b:
            return self.ex_app()
        gram = self.art(ga)
        gram["o2"] = gb
        text, e = self.user(self.pick(T_APP_TWO), {"x": a, "y": b}, gram, tool=True)
        sa, sb = self.shown(a), self.shown(b)
        return [U(text), C("abrir_app", e["x"]), R(f"ok: abri {sa}"), C("abrir_app", e["y"]),
                R(f"ok: abri {sb}"), A(self.reply(R_APP_TWO, x=sa, y=sb))]

    def ex_site(self):
        rng = self.rng
        if rng.random() < 0.2:
            dom = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))) + rng.choice(TLDS)
            text, e = self.user(self.pick(T_DOMAIN), {"x": dom}, tool=True)
            return [U(text)] + self.finish("abrir_site", e["x"], f"ok: abri {e['x'].lower()}", R_SITE_OK,
                                           x=e["x"].lower())
        name, g = rng.choice(SITES)
        text, e = self.user(self.pick(T_SITE), {"x": name}, self.art(g), tool=True)
        url = T.SITES.get(T.norm(name), "https://" + T.norm(name).replace(" ", "") + ".com")
        host = url.split("//")[1].removeprefix("www.")
        return [U(text)] + self.finish("abrir_site", e["x"], f"ok: abri {host}", R_SITE_OK, x=name)

    def search_term(self):
        if self.rand_on and self.rng.random() < 0.5:
            r = self.rng.random()
            if r < 0.6:
                return self.rand_term(1, 6)
            if r < 0.8:
                return f"como {self.rng.choice(HOW_VERBS)} {self.rand_term(1, 3)}"
            return f"{self.rand_term(1, 2)} perto de mim"
        r = self.rng.random()
        if r < 0.55:
            return self.pick(SEARCH_TERMS)
        if r < 0.85:
            return f"como {self.rng.choice(HOW_VERBS)} {self.rng.choice(HOW_OBJS)}"
        return f"{self.rng.choice(NEAR)} perto de mim"

    def ex_search(self):
        text, e = self.user(self.pick(T_SEARCH), {"x": self.search_term()}, tool=True)
        return [U(text)] + self.finish("pesquisar", e["x"], "ok: pesquisa aberta no navegador", R_SEARCH)

    def ex_youtube(self):
        tpl = self.pick(T_YT)
        x = self.rand_term(1, 5) if self.rand_on and self.rng.random() < 0.55 else self.pick(YT_TERMS)
        text, e = self.user(tpl, {"x": x}, tool=True)
        return [U(text)] + self.finish("youtube", e["x"], "ok: busca aberta no youtube", R_YT)

    def ex_factual(self):
        rng = self.rng
        if self.rand_on and rng.random() < 0.55:
            r = rng.random()
            g = rng.choice(["o", "a"])
            if r < 0.3:
                text, _ = self.user(self.pick(T_WHAT), {"x": f"{g} {self.rand_term(1, 3)}"})
            elif r < 0.55:
                text, _ = self.user(self.pick(T_WHO), {"x": self.rand_person()})
            elif r < 0.7:
                text, _ = self.user(self.pick(T_COUNTRY), {"x": self.rand_word().capitalize()}, self.art(g))
            else:
                text, _ = self.user(self.pick(T_FACT_RAND), {"x": self.rand_term(1, 3), "y": self.rand_term(1, 2)},
                                    self.art(g))
            query = text.rstrip("?!. ")
            return [U(text)] + self.finish("pesquisar", query, "ok: pesquisa aberta no navegador", R_FACT)
        r = rng.random()
        if r < 0.35:
            text, _ = self.user(self.pick(T_WHAT), {"x": self.pick(SUBJECTS)})
        elif r < 0.6:
            text, _ = self.user(self.pick(T_WHO), {"x": self.pick(PEOPLE)})
        elif r < 0.8:
            c, d = self.pick(COUNTRIES)
            g = "a" if d == "da" else "o"
            text, _ = self.user(self.pick(T_COUNTRY), {"x": c}, self.art(g))
        else:
            text, _ = self.user(self.pick(FACT_QUESTIONS))
        query = text.rstrip("?!. ")
        return [U(text)] + self.finish("pesquisar", query, "ok: pesquisa aberta no navegador", R_FACT)

    def ex_hora(self):
        text, _ = self.user(self.pick(T_HORA), tool=True)
        v = f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}"
        return [U(text)] + self.finish("hora", "", f"ok: {v}", R_HORA, v=v)

    def ex_data(self):
        text, _ = self.user(self.pick(T_DATA), tool=True)
        d = datetime.date(2024, 1, 1) + datetime.timedelta(days=self.rng.randint(0, 1500))
        v = T.formatar_data(d)
        return [U(text)] + self.finish("data", "", f"ok: {v}", R_DATA, v=v)

    def rand_expr(self):
        rng = self.rng
        a, b, c = rng.randint(2, 999), rng.randint(2, 99), rng.randint(1, 50)
        forms = [f"{a} + {b}", f"{a} - {b}", f"{a} * {b}", f"{a} x {b}", f"{a} vezes {b}", f"{a} mais {b}",
                 f"{a} menos {b}", f"{a} dividido por {b}", f"{a}/{b}", f"{rng.choice([5, 10, 15, 20, 25, 30, 50])}% de {a}",
                 f"raiz de {rng.randint(2, 30) ** 2}", f"{b} ao quadrado", f"2 elevado a {rng.randint(2, 16)}",
                 f"{a}*{b}+{c}", f"({a}+{b})*{c}", f"{b},5 vezes {c}", f"{a}+{b}+{c}", f"{a} - {b} * {c}"]
        return rng.choice(forms)

    def rand_num(self, max_digits=6):
        """Número com 1 a max_digits dígitos (cada tamanho igualmente provável): força copiar dígito a dígito."""
        k = self.rng.randint(1, max_digits)
        return self.rng.randint(0 if k == 1 else 10 ** (k - 1), 10 ** k - 1)

    def rand_expr_wide(self):
        rng, n = self.rng, self.rand_num
        a, b = n(), n()
        op = rng.choice(["+", "-", "*", "/", "x", " x ", " + ", " - ", " * ", " / ", " vezes ", " mais ", " menos ",
                         " dividido por "])
        r = rng.random()
        if r < 0.1:
            return f"{n(2)}^{rng.randint(2, 9)}" if rng.random() < 0.5 else f"{n(2)} elevado a {rng.randint(2, 9)}"
        if r < 0.18:
            return f"{rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])}% de {a}"
        if r < 0.25:
            return f"{a}{op}{b}{rng.choice(['+', '-', '*', ' + ', ' - '])}{n(3)}"
        return f"{a}{op}{b}"

    def ex_calc(self):
        if self.rand_on and self.rng.random() < 0.6:
            expr = self.rand_expr_wide()
            bare = self.rng.random() < 0.45  # só a conta, sem frase em volta: "2+2"
            text, e = self.user("{x}" if bare else self.pick(T_CALC), {"x": expr}, tool=True)
        else:
            expr = self.rand_expr()
            text, e = self.user(self.pick(T_CALC), {"x": expr}, tool=True)
        res = T.calcular(e["x"])
        v = res.removeprefix("ok: ")
        if res.startswith("erro"):
            return [U(text), C("calcular", e["x"]), R(res), A(self.reply(R_CALC_ERR))]
        return [U(text), C("calcular", e["x"]), R(res), A(self.reply(R_CALC, v=v))]

    def ex_remind(self):
        rng = self.rng
        d = self.rand_duration() if self.rand_on and rng.random() < 0.5 else self.pick(DURATIONS)
        f = T.formatar_duracao(T.parse_duracao(d))
        if rng.random() < 0.3:
            text, e = self.user(self.pick(T_TIMER), {"d": d}, tool=True)
            return [U(text)] + self.finish("lembrete", e["d"], f"ok: em {f}: o timer acabou", R_TIMER, f=f)
        t = self.rand_task() if self.rand_on and rng.random() < 0.5 else self.pick(REMINDERS)
        text, e = self.user(self.pick(T_REMIND), {"d": d, "t": t}, tool=True)
        return [U(text)] + self.finish("lembrete", f"{e['d']} | {e['t']}", f"ok: em {f}: {e['t']}", R_REMIND,
                                       f=f, t=e["t"])

    def ex_note(self):
        x = (self.rand_task() if self.rng.random() < 0.6 else self.rand_term(1, 5)) \
            if self.rand_on and self.rng.random() < 0.5 else self.pick(NOTES)
        text, e = self.user(self.pick(T_NOTE), {"x": x}, tool=True)
        return [U(text)] + self.finish("anotar", e["x"], "ok: anotado", R_NOTE)

    def ex_see_notes(self):
        text, _ = self.user(self.pick(T_SEE_NOTES), tool=True)
        if self.rng.random() < 0.15:
            return [U(text), C("ver_notas"), R("ok: nenhuma nota ainda"), A(self.reply(R_NO_NOTES))]
        v = "; ".join(self.rng.sample(NOTES, self.rng.randint(1, 4)))
        return [U(text), C("ver_notas"), R(f"ok: {v}"), A(self.reply(R_NOTES, v=v))]

    def fact(self, kind=None):
        rng = self.rng
        kind = kind or rng.choice(["nome", "cidade", "aniv", "gosto", "cor", "trabalho", "estudo", "time", "pet"])
        return {
            "nome": lambda: f"meu nome é {self.rand_nick() if self.rand_on and rng.random() < 0.5 else rng.choice(NAMES)}",
            "cidade": lambda: f"eu moro em {self.rand_city() if self.rand_on and rng.random() < 0.5 else rng.choice(CITIES)}",
            "aniv": lambda: f"meu aniversário é dia {rng.randint(1, 28)} de {rng.choice(MONTHS)}",
            "gosto": lambda: f"eu gosto de {rng.choice(LIKES)}",
            "cor": lambda: f"minha cor favorita é {rng.choice(COLORS)}",
            "trabalho": lambda: f"eu trabalho com {rng.choice(JOBS)}",
            "estudo": lambda: f"eu estudo {rng.choice(STUDIES)}",
            "time": lambda: f"meu time é o {rng.choice(TEAMS)}",
            "pet": lambda: f"meu cachorro se chama {rng.choice(PETS)}",
        }[kind]()

    def ex_remember(self):
        rng = self.rng
        r = rng.random()
        if r < 0.3:
            name = self.rand_nick() if self.rand_on and rng.random() < 0.6 else self.pick(NAMES)
            text, e = self.user(self.pick(T_INTRO), {"x": name})
            return [U(text), C("lembrar", f"meu nome é {e['x']}"), R("ok: guardei"), A(self.reply(R_INTRO, x=e["x"]))]
        if r < 0.45:
            city = self.rand_city() if self.rand_on and rng.random() < 0.5 else self.pick(CITIES)
            text, e = self.user(self.pick(T_LIVE), {"x": city})
            return [U(text), C("lembrar", f"eu moro em {e['x']}"), R("ok: guardei"), A(self.reply(R_LIVE, x=e["x"]))]
        text, e = self.user(self.pick(T_REMEMBER), {"x": self.fact()}, tool=True)
        return [U(text), C("lembrar", e["x"]), R("ok: guardei"), A(self.reply(R_REMEMBER))]

    def ex_recall(self):
        q, key, kind = self.pick(T_RECALL)
        text, _ = self.user(q, tool=True)
        if self.rng.random() < 0.25:
            return [U(text), C("recordar", key), R("erro: não tenho nada sobre isso"), A(self.reply(R_RECALL_ERR))]
        v = "; ".join(self.fact() for _ in range(self.rng.randint(2, 3))) if kind == "tudo" else self.fact(kind)
        return [U(text), C("recordar", key), R(f"ok: {v}"), A(self.reply(R_RECALL, v=v))]

    def ex_volume(self):
        mode = self.rng.choice(list(T_VOL))
        text, _ = self.user(self.pick(T_VOL[mode]), tool=True)
        res = {"mais": "ok: volume aumentado", "menos": "ok: volume diminuído", "mudo": "ok: mudo alternado"}[mode]
        return [U(text), C("volume", mode), R(res), A(self.reply(R_VOL[mode]))]

    def ex_media(self):
        mode = self.rng.choice(list(T_MEDIA))
        text, _ = self.user(self.pick(T_MEDIA[mode]), tool=True)
        res = {"pausar": "ok: play/pause", "proxima": "ok: próxima faixa", "anterior": "ok: faixa anterior"}[mode]
        return [U(text), C("midia", mode), R(res), A(self.reply(R_MEDIA[mode]))]

    def ex_folder(self):
        if self.rand_on and self.rng.random() < 0.2:  # pasta que não existe: copia o nome e avisa
            text, e = self.user(self.pick(T_FOLDER), {"x": self.rand_name().lower()}, tool=True)
            return [U(text), C("abrir_pasta", e["x"]), R(f"erro: não conheço a pasta {e['x']}"),
                    A(self.reply(R_FOLDER_ERR))]
        name, shown = self.pick(FOLDERS)
        text, e = self.user(self.pick(T_FOLDER), {"x": name}, tool=True)
        return [U(text)] + self.finish("abrir_pasta", e["x"], f"ok: abri a pasta {shown}", R_FOLDER, x=shown)

    def ex_weather(self):
        rng = self.rng
        here = rng.random() < 0.35
        city = rng.choice(CITIES)
        if here:
            text, _ = self.user(self.pick(T_WEATHER_HERE), tool=True)
            arg = ""
        else:
            text, e = self.user(self.pick(T_WEATHER), {"x": self.rand_city() if self.rand_on and rng.random() < 0.5
                                                        else self.pick(CITIES)}, tool=True)
            arg = city = e["x"]
        t = rng.randint(5, 36)
        v = (f"{city if not city.islower() else city.title()}: {t}°C, {rng.choice(WEATHER)}, máx {t + rng.randint(1, 6)}°C, "
             f"mín {t - rng.randint(1, 8)}°C, chuva {rng.choice(range(0, 101, 5))}%")
        r = rng.random()
        if here and r < 0.25:
            return [U(text), C("clima"), R("erro: não sei sua cidade"), A(self.reply(R_WEATHER_NOCITY))]
        if not here and r < 0.05:
            return [U(text), C("clima", arg), R(f"erro: não achei a cidade {arg}"), A(self.reply(R_WEATHER_NOTFOUND))]
        if r > 0.95:
            return [U(text), C("clima", arg), R("erro: sem internet"), A(self.reply(R_WEATHER_OFFLINE))]
        return [U(text), C("clima", arg), R(f"ok: {v}"), A(self.reply(R_WEATHER, v=v))]

    def ex_lock(self):
        text, _ = self.user(self.pick(T_LOCK), tool=True)
        return [U(text)] + self.finish("bloquear", "", "ok: tela bloqueada", R_LOCK)

    def ex_smalltalk(self):
        # no treino, 40% do papo vem dos negativos "parecidos com pedido" (~8% de todas as falas)
        pool = SMALLTALK_TRAIN if self.rand_on and self.rng.random() < 0.4 else SMALLTALK
        users, replies = self.rng.choice(pool)
        text, _ = self.user(self.pick(users))
        return [U(text), A(self.reply(replies))]

    def ex_sobre_mim(self):
        rng = self.rng
        tpl, fact, src = self.pick(T_ABOUT)
        if self.rand_on and rng.random() < 0.35:
            x = self.rand_term(1, 2)
        else:
            x = self.pick({"sport": SPORTS, "hobby": HOBBIES, "game": GAMES, "job": PROFESSIONS,
                           "instrument": INSTRUMENTS, "place": PLACES}[src])
        if self.rand_on and rng.random() < 0.25 and tpl.endswith("{x}"):  # "faço X há dois anos" -> guarda só "eu faço X"
            tpl += " " + rng.choice(ABOUT_WHEN)
        text, e = self.user(tpl, {"x": x})
        msgs = [U(text), C("lembrar", fact.replace("{x}", e["x"])), R("ok: guardei"), A(self.reply(R_ABOUT))]
        if self.rand_on and rng.random() < 0.35:  # continua o assunto: resposta normal, sem guardar de novo
            follow, _ = self.user(rng.choice(ABOUT_FOLLOW[0]))
            msgs += [U(follow), A(self.reply(ABOUT_FOLLOW[1]))]
        return msgs

    def ex_fix_name(self):
        rng = self.rng
        x = self.rand_nick() if self.rand_on and rng.random() < 0.6 else self.pick(NAMES)
        text, e = self.user(self.pick(T_FIX_NAME), {"x": x, "y": self.other_name(x)})
        return [U(text), C("lembrar", f"meu nome é {e['x']}"), R("ok: guardei"), A(self.reply(R_FIX_NAME, x=e["x"]))]

    def ex_gibberish(self):
        """Só letras batidas sem nenhuma palavra de verdade (frases reais nunca viram "não entendi")."""
        rng = self.rng
        keys = "asdfghjklqwertyuiopzxcvbnm"
        while True:
            if rng.random() < 0.3:  # tecla repetida ou fileira do teclado
                row = rng.choice(["asdfghjkl", "qwertyuiop", "zxcvbnm", "jkl", "hjk"])
                i = rng.randrange(len(row) - 2)
                text = row[i:i + rng.randint(3, len(row) - i)] * rng.randint(1, 3)
            else:
                text = "".join(rng.choice(keys) for _ in range(rng.randint(3, 14)))
            if rng.random() < 0.3:
                text += " " + "".join(rng.choice(keys) for _ in range(rng.randint(3, 8)))
            if not any(w in real_words() for w in text.split()):
                return [U(text), A(self.reply(GIBBERISH_REPLIES))]

    # ============================================================ conversa
    KINDS = {  # tipo -> peso (frequência no treino)
        "smalltalk": 34, "app": 14, "app_two": 2, "site": 6, "search": 5, "youtube": 3, "factual": 7, "hora": 4,
        "data": 2, "calc": 5, "remind": 4, "note": 2, "see_notes": 1.5, "remember": 3, "recall": 3, "volume": 2,
        "media": 2, "folder": 2, "weather": 3, "lock": 0.7, "gibberish": 1, "sobre_mim": 4, "fix_name": 1.5,
    }

    def exchange(self, kind):
        return {"app_two": self.ex_app_two, "see_notes": self.ex_see_notes}.get(kind) or getattr(self, "ex_" + kind)

    def conversation(self):
        rng = self.rng
        n = rng.choices([1, 2, 3, 4], [0.45, 0.3, 0.15, 0.1])[0]
        msgs, last = [], None
        kinds, weights = list(self.KINDS), list(self.KINDS.values())
        for _ in range(n):
            if last == "app" and rng.random() < 0.35:
                msgs += self.ex_app(more=True)
                continue
            last = rng.choices(kinds, weights)[0]
            msgs += self.exchange(last)()
        return msgs


def eval_cases(n_per_kind=40, seed=1234, nome="Lia"):
    """Casos de teste com frases/nomes que NUNCA aparecem no treino."""
    cases = []
    for kind in Gen.KINDS:
        g = Gen(random.Random(f"{seed}-{kind}"), nome, split="test")  # semente por tipo: mexer num não muda os outros
        for _ in range(n_per_kind):
            msgs = g.exchange(kind)()
            first = msgs[1]
            case = {"kind": kind, "user": msgs[0]["content"], "role": first["role"], "content": first["content"]}
            if len(msgs) == 4 and msgs[2]["role"] == "result":  # pedido -> chamada -> resultado -> resposta
                case["result"] = msgs[2]["content"]
            cases.append(case)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="número de conversas")
    ap.add_argument("--nome", default="Lia", help="nome da IA")
    ap.add_argument("--out", default="data/conversas/sinteticas.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--foco-copia", type=float, default=1.0,
                    help="multiplica o peso dos tipos que copiam texto longo (factual, lembrete, youtube...)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    g = Gen(random.Random(args.seed), args.nome, split="train")
    if args.foco_copia != 1:  # só muda a frequência no treino; os casos do eval_cases continuam os mesmos
        copy_kinds = {"factual", "remind", "youtube", "search", "sobre_mim", "weather", "folder", "app", "remember"}
        g.KINDS = {k: w * (args.foco_copia if k in copy_kinds else 1) for k, w in Gen.KINDS.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_msgs = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            msgs = g.conversation()
            n_msgs += len(msgs)
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{args.n} conversas ({n_msgs} mensagens) em {args.out}")
    print("Exemplos:")
    ex = Gen(random.Random(7), args.nome)
    for _ in range(3):
        for m in ex.conversation():
            print(f"  {m['role']:>9}: {m['content']}")
        print()


if __name__ == "__main__":
    main()
