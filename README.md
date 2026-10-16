# mini-GPT: uma IA pessoal feita do zero

Uma assistente que conversa no tom de um amigo esperto e **faz coisas no seu PC**:
abre apps, sites e pastas, pesquisa, calcula, cria lembretes, anota, controla volume e
música, vê o clima e guarda coisas sobre você.

Tudo foi feito do zero: tokenizador, Transformer, treino, uso de ferramentas e
interface. Nenhum modelo pronto, nenhum peso pré-treinado de terceiros. O PyTorch só
fornece tensores e gradientes. Roda inteira na sua máquina (só o clima consulta a
internet).

## Começo rápido

```powershell
$py = "$env:USERPROFILE\.venvs\mini-gpt\Scripts\python.exe"

& $py web.py          # interface no navegador (http://127.0.0.1:7860)
& $py chat.py         # ou no terminal
```

Para treinar de novo do começo (gera dados, tokeniza, treina e avalia):

```powershell
& $py treinar.py --nome Lia --minutos-chat 90
```

Em outro computador: `python -m venv .venv`, `pip install -r requirements.txt` e baixe o
modelo pronto (não precisa treinar):

```powershell
gh release download treino-v7 --repo Coelho7z7/mini-gpt --dir out
```

## O que ela sabe fazer

| Pedido (do seu jeito) | Ferramenta | Pede confirmação? |
|---|---|---|
| "abre o spotify", "bora abrir o vscode" | `abrir_app`: procura nos apps do menu Iniciar | sim |
| "entra no youtube", "abre o site exemplo.com" | `abrir_site` | sim |
| "pesquisa receita de bolo", "quem foi Santos Dumont?" | `pesquisar` (Google) | sim |
| "toca lofi no youtube" | `youtube` | sim |
| "abre a pasta de downloads" | `abrir_pasta` | sim |
| "bloqueia o pc" | `bloquear` | sim |
| "que horas são?", "que dia é hoje?" | `hora`, `data` | não |
| "quanto é 15% de 200?", "raiz de 144" | `calcular` (seguro, sem `eval`) | não |
| "me lembra de tirar a pizza em 10 minutos", "timer de meia hora" | `lembrete` (notificação do Windows com "Adiar" e "Ok") | não |
| "anota aí: comprar leite", "lê minhas notas" | `anotar`, `ver_notas` | não |
| "meu nome é Ana", "qual é meu nome?" | `lembrar`, `recordar` | não |
| "aumenta o volume", "próxima música" | `volume`, `midia` | não |
| "como tá o tempo em Recife?" | `clima` (Open-Meteo, sem chave) | não |

Perguntas de conhecimento geral ela **não inventa**: admite que é pequena e abre uma
pesquisa. Para conversa, segue o tom do prompt de personalidade: leve, direta,
parceira, sem "Ótima pergunta!" e sem "Espero ter ajudado!".

## Como funciona

```
você digita ──► tokenizador BPE ──► Transformer (KV cache) ──► 1º token decide:
                                                               │
                         ┌─────────── texto ◄──────────────────┤
                         │                                     └─► <|call|> abrir_app: spotify
                         ▼                                               │
                    resposta em                          valida ► confirma? ► executa (tools.py)
                    streaming                                            │
                         ▲                                   <|result|> ok: abri Spotify
                         └────── o modelo comenta o resultado ◄──────────┘
```

### Arquivos

| Arquivo | O que faz |
|---|---|
| `tokenizer.py` | BPE em nível de bytes + tokens especiais de conversa (`<\|user\|>`, `<\|call\|>`...) |
| `model.py` | Transformer estilo Llama: RMSNorm, RoPE, SwiGLU, KV cache, amostragem top-k/top-p |
| `tools.py` | As ferramentas, a busca de apps instalados e a calculadora segura |
| `make_dialogues.py` | Gera milhares de conversas de treino no tom da personalidade |
| `prepare.py` | Treina o tokenizador e transforma textos e conversas em tokens |
| `train.py` | Treino em duas etapas: pré-treino (seus textos) e chat (SFT) |
| `agent.py` | O "cérebro": histórico, KV cache entre turnos, chamada de ferramentas |
| `chat.py` / `web.py` | Interface no terminal / no navegador (`web/index.html`) |
| `eval.py` | Mede acerto em pedidos que o modelo nunca viu |
| `treinar.py` | Roda tudo em sequência |
| `test_ia.py` | Testes da mecânica (não precisa de modelo treinado) |
| `generate.py` | Continua texto com o modelo base (pré-treino) |

### Detalhes que fazem diferença

- **A decisão de usar ferramenta é do modelo.** O código só valida: ferramentas
  inventadas são rejeitadas, e não existe ferramenta de "rodar comando". Na pior das
  hipóteses ela abre o app errado, e é por isso que ações que abrem coisas pedem
  confirmação (desligue com `/auto` ou no Ajustes da interface web).
- **Ruído de propósito.** As conversas de treino vêm sem acento, sem pontuação, com
  "vc", "pfv" e erros de digitação, porque é assim que a gente escreve.
- **Nomes inventados.** Um terço dos pedidos de app usa nomes que não existem
  ("abre o Pixdex Studio"), então ela aprende a *copiar* o nome que você disser em
  vez de decorar uma lista.
- **Teste honesto.** Uma em cada quatro frases e nomes fica fora do treino e só é
  usada pelo `eval.py`. Se o número lá for alto, ela generalizou de verdade.
- **Treino só nas falas dela.** A perda ignora o que o usuário e as ferramentas
  dizem, então ela não aprende a "falar pelo usuário".
- **Segurança da interface web.** Só escuta em `127.0.0.1` e exige um token
  aleatório por sessão, então um site aberto no navegador não consegue mandá-la abrir
  coisas. O que você digita nunca vira token especial, ou seja, digitar `<|call|>`
  não finge uma chamada de ferramenta.

## Treino na GPU (a receita do modelo atual)

O modelo em `out/` (release `treino-v7`) tem ~28 milhões de parâmetros e foi treinado
do zero numa RTX 3050 6 GB em ~3h40. Para refazer (em pastas separadas, sem mexer no
`out/`):

```powershell
.venv\Scripts\python make_dialogues.py --n 1200000 --foco-copia 2 --out data_v8/conversas/sinteticas.jsonl
.venv\Scripts\python prepare.py --conversas data_v8/conversas --textos data_v8/textos --out out_v8 --bpe-dropout 0.1
.venv\Scripts\python train.py --stage sft --data out_v8 --out out_v8 --preset medium --batch-size 32 --max-iters 22000 --save-last
.venv\Scripts\python eval.py --ckpt out_v8/chat.pt
```

(`make_dialogues.py` em vários processos com `--seed` diferentes e `--out` diferentes
na mesma pasta gera bem mais rápido; o `prepare.py` lê todos os `.jsonl` dela.)

Evolução no `eval.py` (mesmos 920 casos nunca vistos):

| modelo | rota | ferramenta | argumento | fidelidade |
|---|---|---|---|---|
| small, 60k conversas, CPU (`treino-v3`) | 95% | 86% | 65% | 96% |
| medium, 400k conversas, 1h GPU | 96% | 87% | 67% | 96% |
| + BPE-dropout + moldes só de treino, 600k, 2h | 99% | 94% | 87% | 99% |
| + 1,2M conversas, 22 mil passos | 99% | 97% | 89% | 99% |
| + conversa pura (negativos), `treino-v7` | 99% | 97% | 90% | 99% |

O que mais fez diferença:

- **BPE-dropout** (`--bpe-dropout`). Um BPE normal cria pedaços "intermediários" que
  quase nunca aparecem nos dados: ~18% do vocabulário ficava sem treino, e palavras
  novas caem justamente neles ("pagode" virava "pagAdmin"). Pulando fusões ao acaso no
  treino, o modelo vê e aprende todos os pedaços. Copiar tokens raros foi de 3% para 86%.
- **Moldes extras só de treino** (`TRAIN_ONLY` em `make_dialogues.py`): mais jeitos de
  pedir a mesma coisa, sem tocar nos casos do teste (o `eval.py` continua idêntico).
- **Conversa pura** (`SMALLTALK_TRAIN`): frases que parecem pedido mas são papo
  ("perdi o sono", "essa música é boa"). Sem elas, a IA mutava o volume quando você
  dizia que perdeu o sono.
- Mais dados ajudam, mas sozinhos quase nada: o ganho veio de *quais* dados.

## Deixando ela melhor

0. **Marque os erros dela.** Na interface, passe o mouse numa resposta e clique em
   **errou?**: a conversa vai para `data/feedback/erros.jsonl` (fora do git e fora do
   treino). Depois corrija a resposta e mova o exemplo certo para `data/conversas/`.

1. **Escreva conversas suas** em `data/conversas/minhas.jsonl` (uma por linha):

   ```json
   {"messages": [{"role": "user", "content": "e aí, bora codar?"}, {"role": "assistant", "content": "Bora! Abro o VS Code?"}]}
   {"messages": [{"role": "user", "content": "abre meu jogo"}, {"role": "call", "content": "abrir_app: Valorant"}, {"role": "result", "content": "ok: abri VALORANT"}, {"role": "assistant", "content": "Valorant aberto. Boa partida!"}]}
   ```

   Papéis: `user`, `assistant`, `call` (`ferramenta: argumento`) e `result` (`ok: ...`
   ou `erro: ...`). Exemplos seus valem mais que os sintéticos.
2. **Dê textos para ela ler** em `data/textos/*.txt` (anotações, livros em domínio
   público, conversas exportadas). O `treinar.py` faz o pré-treino com eles antes do
   chat, e ela ganha vocabulário e fluência.
3. **Mais tempo de treino** (`--minutos-chat 180`) ou **modelo maior** (`--preset
   medium`, de preferência com GPU NVIDIA).
4. **Nova ferramenta:** escreva uma função com `@tool("nome", "descrição")` em
   `tools.py`, adicione exemplos de pedido em `make_dialogues.py` e treine de novo.
5. **Outro nome/personalidade:** `treinar.py --nome Nina --regerar`. As respostas
   ficam em `SMALLTALK` e nas listas `R_*` de `make_dialogues.py`. As frases foram
   escritas com a IA no feminino ("a Lia"); se trocar para um nome masculino, ajuste.

## Limites (sendo sincero)

- Com ~28 milhões de parâmetros treinados numa placa de vídeo de entrada, ela é ótima em **pedidos
  práticos** e **conversa curta** no tom da personalidade. Não espere raciocínio,
  conhecimento geral ou textos longos coerentes: para isso seriam bilhões de
  parâmetros e muitos dados.
- O bate-papo é em grande parte "decorado" das conversas de treino. Ela recombina,
  mas não cria ideias novas.
- Lembretes vivem enquanto o chat/web estiver aberto.
- Feito para Windows (menu Iniciar, teclas de mídia, pastas do sistema). Em Linux/Mac
  a conversa funciona, mas várias ferramentas ficam limitadas.

## Lendo o treino

- `loss`: o quanto o modelo se surpreende com a resposta certa. Chute aleatório ≈
  `ln(4096)` ≈ 8,3. Abaixo de 1 já é bom para chat.
- `train` caindo e `val` subindo = está decorando. Pare, use mais dados ou aumente
  o `--dropout`.
- `out/log_sft.csv` guarda as perdas para você plotar.
