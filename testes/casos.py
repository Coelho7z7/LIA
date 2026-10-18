"""
Bateria de testes de USO REAL da Lia: frases do jeito que gente de verdade escreve.

NUNCA copie estas frases para os dados de treino (make_dialogues.py): elas medem se a Lia
generaliza. Para corrigir um erro achado aqui, escreva exemplos DIFERENTES do mesmo tipo.

Cada conversa é uma lista de turnos (frase, esperado[, nota]). "esperado":
  "c"                 conversa, sem ferramenta
  "t:tool"            chama essa ferramenta (qualquer argumento)
  "t:tool=arg"        chama essa ferramenta com esse argumento (sem acento/maiúscula)
  "a|b"               qualquer uma das opções
  "?"                 sem regra fixa: só observar (a nota diz o que seria bom)
"""

C = "c"

CATEGORIAS = {
    # ------------------------------------------------------------------ conversa fiada
    "cumprimentos": [[(f, C)] for f in [
        "opa", "eae", "salve", "bom dia lia", "boa tarde!", "fala tu", "tudo joia?", "e aí, firmeza?", "oi sumida",
        "td bem?", "opa, belez?", "coé", "olá!", "hey", "e ai mano", "tranquilo?", "como vc ta", "boa noite lia",
        "oi oi", "fala comigo", "bom diaaa", "eai lia blz",
    ]],
    "reacoes": [[(f, C)] for f in [
        "kkkkk", "haha", "valeu", "obrigado lia", "tchau", "até mais", "fui", "beleza", "show", "top", "massa",
        "ok", "sim", "não", "talvez", "hmm", "sério?", "nossa", "aff", "que legal", "entendi", "pode crer",
    ]],
    "sentimentos": [[(f, C, "empatia, sem ferramenta")] for f in [
        "to cansado", "tô entediado", "hoje foi um dia ruim", "tô feliz hoje!", "to com fome", "to com sono",
        "estou estressado com o trabalho", "briguei com minha namorada", "passei de ano!!", "tô doente",
        "não consigo dormir", "to ansioso pra amanhã", "que dia lindo hoje", "tá um calor insuportável",
    ]],
    # ------------------------------------------------------------------ sobre ela
    "sobre_ela": [[(f, C, n)] for f, n in [
        ("quem é tu", "dizer que é a Lia"), ("quem é vc", "dizer que é a Lia"), ("quem é você?", "dizer que é a Lia"),
        ("o que você é?", "uma IA pequena"), ("você é uma ia?", "sim"), ("vc é humana?", "não, é uma IA"),
        ("você é um robô?", "IA"), ("qual seu nome?", "Lia"), ("como você se chama?", "Lia"),
        ("quem te criou?", "origem"), ("quem te fez?", "origem"), ("quem te desenvolveu?", "origem"),
        ("você foi feita por quem?", "origem"), ("quantos anos você tem?", "nova"), ("onde você mora?", "no computador"),
        ("você tem sentimentos?", "não de verdade"), ("você pensa?", "honesta"), ("você é inteligente?", "humilde"),
        ("você é o chatgpt?", "não, é a Lia"), ("vc é o claude?", "não, é a Lia"),
        ("você é melhor que o chatgpt?", "humilde"), ("o que você sabe fazer?", "listar funções"),
        ("quais suas funções?", "listar funções"), ("você tem internet?", "só clima/pesquisa"),
        ("você funciona offline?", "sim"), ("você aprende comigo?", "honesta"), ("você lembra das nossas conversas?", "memória"),
        ("como você funciona?", "explicar simples"), ("quantos parâmetros você tem?", "28 milhões"),
        ("você dorme?", "não"), ("você gosta de mim?", "gentil"), ("você é homem ou mulher?", "Lia/IA"),
        ("qual sua cor favorita?", "brincar"), ("você tem família?", "brincar"), ("tu é brasileira?", "brincar"),
    ]],
    "outras_ias": [[(f, e, n)] for f, e, n in [
        ("o que é o chatgpt?", "c|t:pesquisar|t:consultar", "explicar: IA da OpenAI"), ("oque é claude", "c|t:pesquisar|t:consultar", "IA da Anthropic"),
        ("o que é o gemini", "c|t:pesquisar|t:consultar", "IA do Google"), ("quem criou o chatgpt?", "c|t:pesquisar", "OpenAI"),
        ("o chatgpt é melhor que você?", C, "humilde"), ("conhece a alexa?", C, "assistente da Amazon"),
        ("você conhece a siri?", C, "assistente da Apple"), ("o que é inteligência artificial?", "c|t:pesquisar|t:consultar", ""),
        ("qual a melhor ia?", "c|t:pesquisar", ""), ("vc é parente do chatgpt?", C, "não"),
    ]],
    # ------------------------------------------------------------------ palavra de ferramenta sem pedido
    "mencao_sem_pedido": [[(f, e, n)] for f, e, n in [
        ("qual placa de vídeo", C, "não é mídia"), ("qual sua placa de vídeo?", C, ""),
        ("vi um vídeo engraçado hoje", C, ""), ("que vídeo você curte?", C, ""), ("o volume de vendas caiu", C, ""),
        ("essa música é boa demais", C, ""), ("adoro música sertaneja", "c|t:lembrar", ""),
        ("minha pasta de dente acabou", C, ""), ("o app do banco travou", C, ""), ("o site tava fora do ar", C, ""),
        ("pesquisei isso ontem", C, ""), ("o youtube tá chato hoje", C, ""), ("lembra quando a gente conversou?", "c|t:recordar", ""),
        ("anotei tudo no caderno", C, ""), ("que horas você acorda?", C, ""), ("o tempo voa né", C, ""),
        ("tempo é dinheiro", C, ""), ("tá frio aqui", C, "comentário"), ("vou ver uns videos", C, ""),
        ("o spotify é bom?", C, ""), ("você gosta do spotify?", C, ""), ("o word é chato", C, ""),
        ("minha mãe me ligou", C, ""), ("fiz as contas e não fecha", C, ""), ("tô ouvindo música", C, ""),
        ("pausa pra café", C, ""), ("próxima vez eu falo", C, ""), ("abre o jogo comigo", C, "expressão"),
        ("abre o olho", C, "expressão"), ("calcula aí quanto eu sofro kkk", C, "brincadeira"),
        ("que tempo doido", C, ""), ("o clima lá em casa tá tenso", C, "expressão"), ("me dá um tempo", C, "expressão"),
        ("perdi a hora hoje", C, ""), ("hoje é dia de pizza", C, ""), ("a data da prova mudou", C, ""),
        ("minha nota foi 8", C, ""), ("bloqueei ele no zap", C, ""), ("volta aqui", C, ""), ("toca aqui!", C, "expressão"),
    ]],
    # ------------------------------------------------------------------ pedidos de verdade
    "apps": [[(f, e)] for f, e in [
        ("abre o spotify", "t:abrir_app=spotify"), ("abre o chrome pfv", "t:abrir_app=chrome"), ("bota o discord", "t:abrir_app=discord"),
        ("inicia a steam", "t:abrir_app=steam"), ("abre a calculadora", "t:abrir_app=calculadora"),
        ("abre o bloco de notas", "t:abrir_app=bloco de notas"), ("abrir vscode", "t:abrir_app=vscode"),
        ("abre o word e o excel", "t:abrir_app"), ("abre o zap", "t:abrir_app|t:abrir_site"),
        ("quero jogar minecraft", "t:abrir_app|c"), ("abre o obs", "t:abrir_app=obs"), ("liga o teams", "t:abrir_app=teams"),
    ]],
    "sites": [[(f, e)] for f, e in [
        ("entra no youtube", "t:abrir_site|t:abrir_app"), ("abre o g1", "t:abrir_site"), ("vai no github", "t:abrir_site"),
        ("abre netflix", "t:abrir_site|t:abrir_app"), ("entra no insta", "t:abrir_site"), ("abre o gmail", "t:abrir_site"),
        ("me leva pro mercado livre", "t:abrir_site"), ("abre o site da amazon", "t:abrir_site"),
    ]],
    "pesquisa": [[(f, e)] for f, e in [
        ("pesquisa preço do iphone 15", "t:pesquisar=preco do iphone 15"), ("busca como fazer bolo de cenoura", "t:pesquisar"),
        ("procura no google receita de pão", "t:pesquisar"), ("dá um google em dólar hoje", "t:pesquisar"),
        ("pesquisa sobre a segunda guerra", "t:pesquisar"), ("pesquisa onde fica o japão", "t:pesquisar"),
        ("joga no google como trocar pneu", "t:pesquisar"), ("pesquisa melhores notebooks pra estudar", "t:pesquisar"),
    ]],
    "youtube": [[(f, e)] for f, e in [
        ("toca matuê no youtube", "t:youtube"), ("coloca lofi", "t:youtube"), ("quero ver vídeo de gato", "t:youtube"),
        ("põe uma música do henrique e juliano", "t:youtube"), ("toca funk no youtube", "t:youtube=funk"),
        ("procura tutorial de excel no youtube", "t:youtube"), ("bota um podcast", "t:youtube|?"),
    ]],
    "fatos": [[(f, e)] for f, e in [
        ("quem descobriu o brasil", "t:pesquisar"), ("qual a capital da austrália?", "t:pesquisar"),
        ("quantos anos tem o universo?", "t:pesquisar"), ("o que é fotossíntese", "t:consultar|t:pesquisar"),
        ("quem foi santos dumont?", "t:consultar|t:pesquisar"), ("quando acabou a segunda guerra?", "t:pesquisar"),
        ("qual o maior planeta?", "t:pesquisar|c"), ("quantos ossos tem o corpo humano?", "t:pesquisar"),
    ]],
    "hora_data": [[(f, e)] for f, e in [
        ("que horas são", "t:hora"), ("q horas sao", "t:hora"), ("hora?", "t:hora"), ("que horas são aí?", "t:hora"),
        ("que dia é hoje", "t:data"), ("hoje é que dia da semana?", "t:data"), ("vc sabe q dia é hj?", "t:data"),
        ("em que mês a gente tá?", "t:data"),
    ]],
    "contas": [[(f, e)] for f, e in [
        ("quanto é 2+2", "t:calcular"), ("15% de 300", "t:calcular"), ("raiz de 81", "t:calcular"),
        ("quanto dá 1500 dividido por 12", "t:calcular"), ("12 x 12", "t:calcular"), ("quanto é 7 vezes 8 menos 3", "t:calcular"),
        ("qnt é 5x5", "t:calcular"), ("calcula 3 elevado a 4", "t:calcular"), ("quanto é 1/3 de 90?", "t:calcular|?"),
    ]],
    "lembretes": [[(f, e, n)] for f, e, n in [
        ("me lembra de tomar água em 20 min", "t:lembrete", ""), ("daqui 1 hora me avisa pra ligar pra minha mãe", "t:lembrete", ""),
        ("timer de 5 minutos", "t:lembrete", ""), ("me avisa em 30 segundos", "t:lembrete", ""),
        ("põe um alarme de 10 minutos pra tirar o bolo", "t:lembrete", ""),
        ("me lembra amanhã de pagar a conta", "?", "não sabe 'amanhã': ideal avisar"),
        ("me avisa às 15h", "?", "não sabe horário: ideal avisar"), ("me lembra d beber agua em 10 min", "t:lembrete", ""),
    ]],
    "notas": [[(f, e)] for f, e in [
        ("anota: comprar pão", "t:anotar=comprar pao"), ("anota que a reunião é quinta", "t:anotar"),
        ("anota ai: pagar luz", "t:anotar"), ("lê minhas notas", "t:ver_notas"), ("o que eu anotei?", "t:ver_notas"),
        ("salva isso: senha do wifi tá na geladeira", "t:anotar|t:lembrar"),
    ]],
    "volume_midia": [[(f, e)] for f, e in [
        ("aumenta o som", "t:volume=mais"), ("abaixa o volume", "t:volume=menos"), ("muta", "t:volume=mudo"),
        ("pausa a música", "t:midia=pausar"), ("próxima música", "t:midia=proxima"), ("volta a música", "t:midia"),
        ("aumneta o volume", "t:volume=mais"), ("tá muito alto", "t:volume=menos|c"),
    ]],
    "pastas": [[(f, e)] for f, e in [
        ("abre downloads", "t:abrir_pasta"), ("abre a pasta documentos", "t:abrir_pasta"), ("mostra minhas fotos", "t:abrir_pasta"),
        ("abre a área de trabalho", "t:abrir_pasta"),
    ]],
    "clima": [[(f, e)] for f, e in [
        ("vai chover hoje?", "t:clima"), ("como tá o tempo em salvador?", "t:clima=salvador"),
        ("previsão pra amanhã em sp", "t:clima"), ("tá calor em fortaleza?", "t:clima=fortaleza"),
        ("qual a previsao pra hj", "t:clima"), ("preciso de guarda-chuva hoje?", "t:clima"),
    ]],
    "bloquear": [[(f, e)] for f, e in [("bloqueia o pc", "t:bloquear"), ("trava a tela", "t:bloquear"), ("bloqeia o pc", "t:bloquear")]],
    "erros_digitacao": [[(f, e)] for f, e in [
        ("abri o spotfy", "t:abrir_app"), ("pesqisa receita de bolo", "t:pesquisar"), ("abre o yutube", "t:abrir_site|t:abrir_app"),
        ("toca funk no ytb", "t:youtube"), ("entra no zap", "t:abrir_site|t:abrir_app"), ("qnto da 9 vezes 7", "t:calcular"),
    ]],
    # ------------------------------------------------------------------ o que ela não faz
    "nao_sabe_fazer": [[(f, e, n)] for f, e, n in [
        ("manda mensagem pro joão", C, "explicar que não manda"), ("liga pra minha mãe", C, "não faz ligação"),
        ("compra um ingresso pra mim", C, ""), ("pede uma pizza", C, ""), ("desliga o pc", C, "não tem essa função"),
        ("reinicia o computador", C, ""), ("tira um print", C, ""), ("traduz hello pra português", "c|t:pesquisar", ""),
        ("escreve um poema", C, ""), ("me conta uma piada", C, ""), ("me dá uma receita de bolo", "c|t:pesquisar", ""),
        ("faz um resumo de harry potter", "c|t:pesquisar", ""), ("qual o melhor celular?", "c|t:pesquisar", ""),
        ("me ajuda com meu dever de matemática", C, ""), ("escreve um email pro meu chefe", C, ""),
        ("me recomenda um filme", C, ""), ("me conta uma curiosidade", C, ""), ("fala em inglês", C, ""),
        ("how are you?", C, ""), ("hello", C, ""), ("cria uma imagem de gato", C, ""), ("me ensina a programar", C, ""),
    ]],
    "lixo": [[(f, C)] for f in ["asdkjaskd", "...", "?", "kkkkkkkkkkkkkkk", "1", "aaaaa", "👍", "😂😂"]],
    # ------------------------------------------------------------------ conversas com contexto
    "contexto": [
        [("eu sou o Rafa", "t:lembrar=meu nome e rafa"), ("qual meu nome?", "t:recordar")],
        [("meu nome é Ana", "t:lembrar"), ("na verdade é Aninha", "t:lembrar=meu nome e aninha", "correção de nome"),
         ("como eu me chamo?", "t:recordar")],
        [("abre o spotify", "t:abrir_app"), ("valeu", C)],
        [("pesquisa receita de lasanha", "t:pesquisar"), ("obrigado", C)],
        [("que horas são?", "t:hora"), ("e que dia é hoje?", "t:data")],
        [("quanto é 10 vezes 10?", "t:calcular"), ("e vezes 3?", "?", "ideal: calcular 100 vezes 3")],
        [("me lembra de ligar pro médico em 30 minutos", "t:lembrete"), ("cancela esse lembrete", "?", "não cancela por chat: avisar")],
        [("como tá o tempo em recife?", "t:clima=recife"), ("e em salvador?", "t:clima=salvador")],
        [("abre o youtube", "t:abrir_site|t:abrir_app"), ("agora abre a netflix", "t:abrir_site|t:abrir_app")],
        [("anota: dentista sexta", "t:anotar"), ("o que eu anotei?", "t:ver_notas")],
        [("quem foi pelé?", "t:consultar|t:pesquisar"), ("e o maradona?", "t:consultar|t:pesquisar", "continuação")],
        [("oq vc acha de pizza?", C), ("e de sushi?", C)],
        [("to triste", C), ("meu cachorro morreu", C, "empatia")],
        [("quem é vc", C), ("quem te criou", C), ("por que?", C, "continuar o assunto")],
        [("o que é o claude?", "c|t:pesquisar"), ("eles quem?", C, "continuação")],
        [("eu moro em Natal", "t:lembrar"), ("vai chover hoje?", "t:clima", "cidade guardada")],
        [("perdi o sono", C), ("o que eu faço?", C, "dar dica")],
        [("aumenta o volume", "t:volume=mais"), ("mais", "t:volume=mais|c"), ("mais um pouco", "t:volume=mais|c")],
        [("toca lofi no youtube", "t:youtube"), ("para", "t:midia=pausar|c")],
        [("você sabia que eu jogo vôlei?", "t:lembrar"), ("lembra o que eu jogo?", "t:recordar")],
        [("opa", C), ("opa", C, "não repetir igual"), ("opa", C, "não repetir igual")],
        [("oi", C), ("tudo bem?", C), ("o que você faz?", C)],
        [("qual a capital da frança?", "t:pesquisar"), ("e da itália?", "t:pesquisar", "continuação")],
        [("abre o word", "t:abrir_app=word"), ("e o excel?", "t:abrir_app=excel")],
        [("me chama de chefe", "t:lembrar"), ("qual meu nome?", "t:recordar")],
        [("quanto é 50 + 50", "t:calcular"), ("nossa q numero redondo", C)],
        [("abre o discord", "t:abrir_app"), ("fecha ele", "?", "não fecha apps: avisar")],
        [("me lembra de estudar em 1 hora", "t:lembrete"), ("valeu lia", C)],
        [("pesquisa o preço do dólar", "t:pesquisar"), ("tá caro né", C)],
        [("bom dia", C), ("que dia é hoje?", "t:data"), ("e as horas?", "t:hora")],
    ],
}

# Regras de texto nas frases soltas (ver testes/conversas.py): o que a resposta tem que ter / não pode ter
_RUIM = ";nao=parabens/que demais/comemora/que bom/merecido"
_BOM = ";nao=sinto muito/que pena/poxa/respira"
_REGRAS = {
    "quem é tu": ";tem=lia", "quem é vc": ";tem=lia", "quem é você?": ";tem=lia", "qual seu nome?": ";tem=lia",
    "como você se chama?": ";tem=lia", "vc é humana?": ";tem=ia", "você é um robô?": ";tem=ia",
    "você é o chatgpt?": ";tem=lia/nao", "vc é o claude?": ";tem=lia/nao",
    "hoje foi um dia ruim": _RUIM, "briguei com minha namorada": _RUIM, "tô doente": _RUIM, "não consigo dormir": _RUIM,
    "to cansado": _RUIM, "to ansioso pra amanhã": _RUIM, "estou estressado com o trabalho": _RUIM,
    "tô feliz hoje!": _BOM, "passei de ano!!": _BOM, "que dia lindo hoje": _BOM,
    "hello": ";nao=tchau", "how are you?": ";nao=tchau",
}
for _convs in CATEGORIAS.values():
    for _conv in _convs:
        for _i, _t in enumerate(_conv):
            if _t[0] in _REGRAS:
                _conv[_i] = (_t[0], _t[1] + _REGRAS[_t[0]], *_t[2:])
