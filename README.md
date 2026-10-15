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

Em outro computador: `python -m venv .venv`, `pip install -r requirements.txt`.

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
| "me lembra de tirar a pizza em 10 minutos", "timer de meia hora" | `lembrete` (alerta na tela) | não |
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

## Deixando ela melhor

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

- Com ~7 milhões de parâmetros treinados numa CPU, ela é ótima em **pedidos
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
