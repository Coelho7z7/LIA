# Bateria de uso real — v9 (17/10/2026)

Modelo: `out_v9/chat.pt` (o mesmo do site), com cópia forçada, em modo simulado.
Rodar de novo: `python testes/bateria.py --ckpt out_vXX/chat.pt`; comparar com `--comparar out_v9 out_vXX`.
Log completo: `testes/resultados/out_v9.txt`.

## Bateria ampliada: 999 conversas, 1.340 frases julgadas → 1.057 certas (79%)

Partes da bateria:
- `testes/casos.py`: 340 frases soltas e conversas curtas.
- `testes/conversas.py`: 279 conversas de vários turnos (606 frases), com papo do dia, desabafo, diversão,
  perguntas sobre ela, memória ao longo da conversa, pedidos no meio do papo, continuações, correções,
  provocações, capacidades, gíria, pedidos compostos, opinião e rotina.
- `testes/variacoes.py`: 416 versões das frases soltas, escritas "do jeito do zap" ou com erro de digitação.

Há também regras de texto: "qual seu nome?" tem que conter "Lia", e coisa ruim não pode ganhar "parabéns".

| grupo | certos |
|---|---|
| Pedidos de verdade (em todas as formas) | **261/278 (94%)** |
| Contexto (memória, pedido no meio, continuações, correções, rotina) | **283/310 (91%)** |
| Sobre ela, outras IAs, capacidades | 160/196 (82%) |
| Conversa (papo, desabafo, diversão, provocação, opinião) | **293/396 (74%)** |
| Palavra de ferramenta sem pedido + coisas que ela não faz | **60/160 (38%)** |

**Ferramenta chamada à toa: 246 vezes.** As que mais aparecem:

| ferramenta | vezes |
|---|---|
| lembrar | 68 |
| pesquisar | 36 |
| volume | 27 |
| abrir_app | 22 |
| clima | 22 |
| midia | 16 |
| recordar | 16 |
| youtube | 12 |
| hora / data | 16 |

**Achados novos da bateria ampliada:**
- **Desabafo (42%) e papo do dia (73%).** Frases comuns fora do repertório viram ferramenta:
  "tô no ônibus" → `lembrar: eu toco toco toco`; "fazendo macarrão" → `meu nome é macarrão`;
  "não aguento mais" → aumenta o volume; "acho que tô com depressão" → `lembrar: eu pratico…`.
  **Quase todo "tô…", "eu…" ou "minha…" vira `lembrar`**: os moldes de "sobre mim" (rodadas 8 e 9) ensinaram uma regra larga demais.
- **Ações perigosas por engano:** "vamos conversar" → **bloqueou a tela**. "ele vai ficar bem né?" (do cachorro doente)
  → abriu o app "Verificar Atualizações". "cheguei agora do trabalho" → abriu a "Conexão de Área de Trabalho Remota".
  A busca de apps do `tools.py` aceita uma palavra comum solta como nome de app. **Dá para apertar sem treino.**
- **Emoção trocada:** "me sinto burro" → "Que demais!"; "passei no vestibular" → "medicina" → "Poxa, sinto muito".
- **Capacidades (35%):** "você consegue mandar email?" → pesquisa no Google; "consegue tocar música?" → pausa a música;
  "você faz conta difícil?" → mostra a data.
- **Opinião e conselho (50%):** "vale a pena fazer faculdade?" → abrir app; "qual a melhor hora pra estudar?" → mostra a hora.
- **O que vai bem nas conversas longas:** memória ao longo do papo, pedido no meio da conversa e rotina (90% ou mais).

## Primeira rodada (340 frases)

| grupo | certos |
|---|---|
| Pedidos de verdade (apps, sites, pesquisa, fatos, hora, contas, notas, volume, clima, bloquear, erros de digitação) | **108/112 (96%)** |
| Conversas com contexto | 58/63 (92%) |
| Cumprimentos, reações, sentimentos, sobre ela, outras IAs | 98/103 (95%) sem ferramenta, mas ~20 respostas com sentido errado (item 4) |
| **Palavra de ferramenta numa frase que não é pedido** | **12/40 (30%)** |
| **Coisas que ela não sabe fazer** | **8/22 (36%)** |
| Total automático | 280/335 (84%) |

O eval.py (94%) mede quase só pedidos. No uso real, o ponto fraco é a **conversa**.

## O que já funciona bem
- Pedidos diretos, inclusive com erro de digitação ("abri o spotfy", "qnto da 9 vezes 7", "bloqeia o pc").
- Memória: "eu sou o Rafa" e depois "qual meu nome?", correção de nome ("na verdade é Aninha"), "me chama de chefe".
- Clima com cidade guardada ("eu moro em Natal", depois "vai chover hoje?").
- Continuações simples: "e em salvador?" (clima), "e o excel?" (app), "agora abre a netflix".
- Empatia na maioria: "to triste", "meu cachorro morreu", "to ansioso pra amanhã".

## 1. Ferramenta chamada à toa: 50 casos (o maior problema)

| ferramenta | vezes | exemplos |
|---|---|---|
| lembrar | 10 | "briguei com minha namorada" → `eu trabalho como minha namorada`; "tu é brasileira?"; "anotei tudo no caderno"; "volta aqui" → `meu nome é amo` |
| youtube | 7 | "vi um vídeo engraçado hoje"; "o youtube tá chato hoje"; "escreve um poema"; "tira um print"; "how are you?" |
| clima | 6 | "tempo é dinheiro" → clima de "Dinheiro"; "o clima lá em casa tá tenso"; "me dá um tempo"; "o site tava fora do ar"; "cria uma imagem de gato" |
| volume | 5 | "o volume de vendas caiu" → aumentou o volume; "sério?" → mudo; "desliga o pc" → mudo; "perdi o sono" → mudo |
| midia | 5 | "qual placa de vídeo"; "tô ouvindo música"; "pausa pra café"; "próxima vez eu falo"; "toca aqui!" |
| pesquisar | 5 | "e ai mano" → pesquisa "mano"; "você lembra das nossas conversas?"; "manda mensagem pro joão" |
| abrir_app | 5 | "o app do banco travou"; "abre o jogo comigo"; "abre o olho"; "liga pra minha mãe" |
| hora / data | 4 | "que horas você acorda?"; "perdi a hora hoje"; "hoje é dia de pizza"; "a data da prova mudou" |
| anotar | 2 | "minha nota foi 8"; "escreve um email pro meu chefe" |

**Causa:** ela reage à palavra ("tempo", "hora", "vídeo", "pausa", "volume", "abre", "eu…") e não ao pedido. Os exemplos negativos do treino cobrem só algumas frases.

**Correção (dados):** muitos exemplos negativos por palavra-gatilho. Cada ferramenta ganha uns 20 moldes de frase que **contêm a palavra mas não são pedido**: comentário, verbo no passado, expressão ("abre o jogo", "me dá um tempo", "toca aqui"), pergunta sobre ela ("que horas você acorda?").

## 2. "lembrar" guarda qualquer frase pessoal
Acontecimentos e sentimentos ("briguei…", "minha pasta de dente acabou", "bloqueei ele no zap") viram fato guardado, muitas vezes com texto torto.

**Correção (dados):** pares mínimos de **fato que dura** ("eu jogo vôlei", "moro em…", "trabalho como…"), que vira `lembrar`, contra **coisa que aconteceu ou sentimento** ("joguei vôlei ontem", "briguei com…", "acabou o…"), que vira conversa.

## 3. Coisas que ela não sabe fazer (14 com ferramenta errada)
Mandar mensagem, ligar, comprar, pedir pizza, desligar ou reiniciar o PC, tirar print, traduzir, escrever poema ou e-mail, criar imagem, ajudar no dever. Mensagem em inglês ("how are you?" → YouTube "are"; "hello" → "Tchau!").

**Correção (dados):** um grupo de respostas "isso eu não faço, mas posso…", com a alternativa certa quando existir (pesquisar uma receita, um resumo). Para inglês: responder em português que ela só fala português.

## 4. Resposta de outro assunto (sem ferramenta, mas errada): ~20

| frase | resposta | devia |
|---|---|---|
| hoje foi um dia ruim | "Que demais! Parabéns!" | empatia |
| oi sumida | conta uma piada | cumprimentar |
| fala comigo | "Sou uma IA, não rola…" | puxar papo |
| top / talvez | "acho que o teclado escorregou" / "não entendi nada" | concordar |
| qual seu nome? | descreve a arquitetura (28M parâmetros) | "Lia" |
| quem é tu | "Foi você, ué" (resposta de "quem te criou") | "Sou a Lia" |
| você pensa? | lista as funções | resposta honesta |
| você é homem ou mulher? | "O Mistral sabe muito mais" | "sou uma IA, me chamo Lia" |
| você aprende comigo? | "Sou uma IA, não rola" | explicar memória e treino |
| pede uma pizza | "IA, totalmente" | "não consigo pedir" |
| me conta uma curiosidade | sugestão de passeio | curiosidade |
| o que é o Claude / Gemini? | "Não sou o Gemini" | o que é e quem fez |
| conhece a Alexa? | "a Alexa é bem maior que eu, treinado…" | assistente da Amazon |

**Correção (dados):** mais grupos de conversa e mais jeitos de dizer cada coisa, com pares mínimos de sentimento ("dia ruim" contra "dia ótimo").

## 5. Continuação de conversa
| conversa | erro |
|---|---|
| "quem foi pelé?" → "e o maradona?" | abriu o app "maradona" |
| "qual a capital da frança?" → "e da itália?" | clima |
| "oq vc acha de pizza?" → "e de sushi?" | abriu o app "sushi" |
| "quanto é 10 vezes 10?" → "e vezes 3?" | `calcular: vezes 3` (devia ser 100 vezes 3) |
| volume → "mais" | "Falou! Até a próxima." |
| YouTube → "para" | "não entendi" |
| "perdi o sono" → "o que eu faço?" | `recordar: faço` (devia dar dica) |
| "o que é o claude?" → "eles quem?" / "quem te criou" → "por que?" | sem relação |

**Correção (dados):** exemplos de continuação para cada ferramenta. "E o/a X?" repete a ferramenta anterior com o assunto novo; "mais", "de novo" e "para" funcionam depois de volume ou música; a conta seguinte usa o resultado anterior.

## 6. Pedidos de verdade que erraram
- "coloca lofi" → clima de "Lofi"; "põe uma música do henrique e juliano" → anotou. Faltam moldes "coloca/põe/bota X" sem a palavra "youtube".
- "mostra minhas fotos" → chamada inválida (devia abrir a pasta Imagens).

## 7. Lembrete com dia ou horário
- "me lembra amanhã de pagar a conta" → erro, e a resposta fala de "endereço do site".
- "me avisa às 15h" → **timer de 15 horas**.

**Correção (ferramenta, sem treino):** entender "às 15h", "15h30", "amanhã às 9h" e "amanhã" no `tools.py` e no site. Depois, mais exemplos com horário no treino.

## 8. Fingir que fez
- "fecha ele" (depois de abrir o Discord) → "Fechado!" (ela não fecha apps).
- "cancela esse lembrete" → resposta aleatória. Devia indicar o botão "cancelar" do cartão (ou ganhar uma ferramenta para isso).

## 9. Origem no site
"quem te criou?" → "Foi você! Me treinou no seu PC…": no site, quem conversa não é quem treinou. **Falta você decidir o texto.**

## Também observado
- "opa" três vezes: no teste saíram respostas diferentes, mas no site já saiu igual duas vezes. **Correção sem treino:** se a resposta nova for idêntica à anterior, gerar de novo.

## Plano

**Sem treino (rápido):**
1. Resposta repetida: gerar de novo (`agent.py` e motor do site).
2. Lembrete com horário ou dia (`tools.py` e `webapp/ferramentas.js`).
3. Busca de apps mais exigente: uma palavra comum solta ("trabalho", "ficar", "arquivos") não abre app.

**Rodada 11 de dados** (tudo com frases novas, nunca as da bateria):
0. **Milhares de frases de conversa do dia a dia** (atividades, acontecimentos, sentimentos, planos, comentários),
   inclusive sequências de desabafo e de notícia boa com a emoção certa em todos os turnos. Hoje o repertório de
   conversa é estreito perto do de pedidos; é a causa número 1.
1. Negativos por palavra-gatilho (item 1).
2. Fato que dura contra acontecimento (item 2).
3. "Não sei fazer isso" e mensagens em inglês (item 3).
4. Conversa: grupos novos, paráfrases e pares de sentimento (item 4).
5. Continuações por ferramenta (item 5).
6. "Coloca/põe X" vira YouTube; "minhas fotos" vira a pasta Imagens (item 6).
7. Lembrete com horário (item 7) e respostas honestas para o que ela não faz (item 8).
8. Origem para o site (item 9), quando você decidir o texto.
9. Se quiser, a ferramenta `consultar` (Wikipédia).

Depois do treino: rodar o eval.py **e** esta bateria, e só publicar se melhorar nas duas.
