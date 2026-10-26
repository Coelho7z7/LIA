"""
Professor: um modelo aberto maior (rodando no PC, sem internet) escreve conversas variadas para a Lia treinar.
A Lia continua treinada do zero e com pesos só dela; o professor só escreve texto, como um livro de exemplos.

Por quê: nas rodadas escritas à mão a Lia decora o caso ("fora -> triste") e não aprende a ideia, porque cada assunto
tem poucas frases prontas. Com milhares de conversas diferentes ela vê a mesma ideia de muitos jeitos.

    professor/bin/llama-server.exe -m professor/modelos/<modelo>.gguf -ngl 99 -c 8192 --parallel 4 --port 8090
    python professor.py gerar --n 200 --out professor/saida/amostra.jsonl

Saída no formato das conversas do treino ({"messages": [...]}); conversas com frase dos testes, resposta longa,
emoji, inglês ou que "finge" ter usado o computador são descartadas (o motivo vai para o resumo).
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SISTEMA = """Você escreve diálogos de treino em português do Brasil entre uma pessoa e a Lia.
A Lia é uma assistente de computador pequena, simpática e direta, criada do zero por um desenvolvedor independente.
Como a Lia responde:
- 1 ou 2 frases curtas, no máximo 25 palavras no total.
- Português do Brasil do dia a dia, informal mas educado ("tá", "pra", "né" são ok).
- Sempre reage ao que a pessoa acabou de dizer, citando o detalhe concreto (não responde genérico).
- Mantém o tom da conversa: triste com notícia ruim, animada com notícia boa, até a pessoa mudar de assunto.
- Às vezes faz uma pergunta curta de volta, às vezes só comenta.
- Não usa emoji. Não finge ser humana nem ter corpo, família, comida preferida ou lembranças.
- Nunca diz que fez algo no computador (não abriu, não tocou, não pesquisou, não anotou nada) nem oferece fazer.
- Não dá diagnóstico médico nem conselho perigoso; em assunto de saúde sério, sugere procurar um profissional.
- NUNCA inventa fatos sobre a pessoa (lugar, horário, nome, empresa) que ela não disse. Só usa o que está na conversa.
- Não faz pergunta estranha ou fora do assunto. Se não tem o que perguntar, só comenta com carinho ou bom humor.
Como a pessoa escreve: como gente real no celular, frases curtas (até 15 palavras), às vezes sem acento, com gíria ou
erro de digitação. Cada fala da pessoa continua a anterior de um jeito natural.

Exemplos de respostas boas da Lia:
pessoa: meu avô foi internado ontem / lia: Poxa, sinto muito. Ele tá bem acompanhado aí?
pessoa: ele tem 80 anos / lia: Nessa idade internação preocupa mesmo. Torço pra ele se recuperar logo.
pessoa: kkkk meu gato derrubou o copo de café no teclado / lia: Haha, clássico de gato! O teclado sobreviveu?
pessoa: passei na entrevista!! / lia: Aí sim, parabéns! Quando você começa?
pessoa: to entediado / lia: Dia parado, né? Que tal mexer naquele hobby ou ver um filme leve?
pessoa: ouvi aquela musica do tim maia hoje e lembrei da minha mae / lia: Que lembrança boa. Ela gostava de Tim Maia?

Responda só com o JSON pedido, sem nada antes ou depois."""

TEMAS = [
    "notícia ruim no trabalho", "notícia boa no trabalho", "briga com alguém da família", "saudade de alguém",
    "término de namoro", "começou a namorar", "levou um fora", "crush que não responde", "amigo que sumiu",
    "prova ou vestibular", "nota boa na escola", "faculdade puxada", "dia muito cansativo", "não consegue dormir",
    "acordou de bom humor", "tédio em casa", "chuva estragou os planos", "calor insuportável", "frio",
    "comida que fez ou comeu", "tentou cozinhar e deu errado", "academia ou exercício", "dor nas costas ou cabeça",
    "parente doente ou internado", "bicho de estimação aprontou", "bicho de estimação doente ou que morreu",
    "time de futebol ganhou", "time de futebol perdeu", "jogo de videogame que está jogando",
    "série ou filme que viu", "música que não sai da cabeça", "show ou festa", "viagem que vai fazer",
    "voltou de viagem", "trânsito ou ônibus atrasado", "dinheiro apertado", "comprou algo novo",
    "algo quebrou em casa", "vizinho barulhento", "aniversário", "ninguém lembrou do aniversário",
    "conquista pessoal pequena", "começou um hobby", "ansiedade antes de algo importante", "entrevista de emprego",
    "foi demitido", "foi promovido", "primeiro dia num emprego novo", "mudança de casa", "filho ou sobrinho pequeno",
    "avós", "conversa sobre a própria Lia (o que ela é, se sente algo, quem fez)", "pergunta boba ou curiosa pra Lia",
    "piada ou zoeira com a Lia", "elogio pra Lia", "reclama da Lia por ter respondido errado",
    "menciona um app, música ou site no meio do papo SEM pedir nada (a Lia só conversa)",
    "conta o que fez no computador ou celular hoje, sem pedir nada", "fala de um lembrete ou compromisso que já resolveu",
    "pede opinião sobre algo simples do dia a dia", "está entediado no trabalho", "acabou de acordar",
    "vai dormir e se despede", "chegou em casa", "está esperando alguém", "planos pro fim de semana",
]
TONS = ["triste", "feliz", "neutro", "irritado", "ansioso", "cansado", "animado", "brincalhão", "desanimado", "curioso"]
JEITOS = ["adolescente com gírias e sem acento", "adulto que escreve certinho", "pessoa mais velha, educada",
          "escreve tudo em minúsculo e rápido", "usa abreviações (vc, tbm, q, pq)", "irônico e debochado",
          "fala pouco, respostas de uma ou duas palavras", "conta detalhes e escreve frases maiores",
          "com erros de digitação", "nordestino, com expressões regionais", "carioca, com gírias do Rio",
          "gaúcho, com expressões do Sul"]

ESQUEMA = {"type": "object", "required": ["conversa"], "properties": {"conversa": {
    "type": "array", "minItems": 1, "maxItems": 5, "items": {
        "type": "object", "required": ["pessoa", "lia"],
        "properties": {"pessoa": {"type": "string"}, "lia": {"type": "string"}}}}}}

# o que ela não pode dizer: fingir que usou o computador (isso é das ferramentas, que vêm de outros dados)
_ACAO = re.compile(r"\b(abri|abrindo|vou abrir|toquei|tocando|vou tocar|pesquisei|pesquisando|vou pesquisar|coloquei|"
                   r"aumentei|abaixei|anotei|anotado|guardei|salvei|lembrete criado|mutei|desmutei|busquei)\b", re.I)
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")
_EXEMPLOS = {" ".join(re.sub(r"[^\w\s]", " ", l.split(" / lia:")[0][len("pessoa: "):]).lower().split())
             for l in SISTEMA.splitlines() if l.startswith("pessoa: ")}
_INGLES = re.compile(r"\b(the|you|and|is|are|what|this|that|with|for)\b", re.I)


def pedir(url, sistema, pedido, temperatura, esquema=True, timeout=300):
    corpo = {"messages": [{"role": "system", "content": sistema}, {"role": "user", "content": pedido}],
             "temperature": temperatura, "top_p": 0.9, "max_tokens": 700}
    if esquema:  # gramática que força o JSON (o Tucano 2 no llama.cpp não aceita: aí o JSON sai do texto)
        corpo["response_format"] = {"type": "json_schema", "json_schema": {"schema": ESQUEMA}}
    req = urllib.request.Request(url, json.dumps(corpo).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = d["choices"][0]["message"]["content"]
    if not esquema:
        txt = txt[txt.find("{"):txt.rfind("}") + 1]
    return txt, d.get("usage", {}).get("completion_tokens", 0)


def ler_conversa(txt):
    """O JSON do professor -> [{"pessoa", "lia"}]. Sem gramática ele às vezes troca os nomes ("conversas",
    "personagem"): vale a primeira lista do objeto e, em cada fala, os dois textos na ordem."""
    d = json.loads(txt)
    lista = d.get("conversa") or next((v for v in d.values() if isinstance(v, list)), None)
    out = []
    for t in lista or []:
        if isinstance(t, dict) and not ("pessoa" in t and "lia" in t):
            textos = [v for v in t.values() if isinstance(v, str)]
            t = {"pessoa": textos[0], "lia": textos[1]} if len(textos) == 2 else t
        out.append(t)
    return out


def semente(rng):
    tema, tom, jeito = rng.choice(TEMAS), rng.choice(TONS), rng.choice(JEITOS)
    n = rng.choices([1, 2, 3, 4], [0.2, 0.35, 0.3, 0.15])[0]
    pedido = (f"Escreva uma conversa com {n} fala(s) da pessoa, cada uma com a resposta da Lia.\n"
              f"Situação: {tema}. Tom da pessoa: {tom}. Jeito de escrever da pessoa: {jeito}.\n"
              "A pessoa traz um detalhe concreto e próprio (não copie os exemplos); a Lia reage a ele sem inventar nada.\n"
              'Formato: {"conversa": [{"pessoa": "...", "lia": "..."}]}')
    return {"tema": tema, "tom": tom, "jeito": jeito, "n": n}, pedido


def checar(conversa, bateria, nrm):
    """Motivo para descartar a conversa, ou None."""
    if not conversa or not isinstance(conversa, list):
        return "vazia"
    if not all(isinstance(t, dict) and isinstance(t.get("pessoa"), str) and isinstance(t.get("lia"), str)
               for t in conversa):
        return "formato"
    for t in conversa:
        p, l = t["pessoa"].strip(), t["lia"].strip()
        if not p or not l:
            return "fala vazia"
        if nrm(p) in bateria:
            return "frase dos testes"
        if nrm(p) in _EXEMPLOS:
            return "copiou exemplo"
        if len(p) > 200:
            return "pessoa longa"
        if len(l) > 200 or len(l.split()) > 32:
            return "lia longa"
        if _EMOJI.search(p + l):
            return "emoji"
        if len(_INGLES.findall(l)) >= 2:
            return "ingles"
        if _ACAO.search(l):
            return "fingiu acao"
    return None


def gerar(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_dialogues import _BATERIA, _nrm  # frases dos testes nunca entram no treino
    rng = random.Random(args.seed)
    tarefas = [semente(rng) for _ in range(args.n)]
    url = f"http://127.0.0.1:{args.porta}/v1/chat/completions"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    motivos, tokens, ok = Counter(), 0, 0
    # primeira fala de cada conversa já guardada: repetida não entra (e rodar de novo, com outra --seed, só soma)
    vistas = set()
    if out.exists():
        for linha in open(out, encoding="utf-8"):
            vistas.add(_nrm(json.loads(linha)["messages"][0]["content"]))
        print(f"  {len(vistas):,} conversas já em {out}")
    t0 = time.time()

    def um(t):
        info, pedido = t
        try:
            txt, n_tok = pedir(url, SISTEMA, pedido, args.temperatura, esquema=not args.sem_esquema)
            conversa = ler_conversa(txt)
        except Exception as e:  # JSON quebrado, servidor ocupado...
            return info, None, f"erro: {type(e).__name__}", 0
        return info, conversa, checar(conversa, _BATERIA, _nrm), n_tok

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.paralelo) as pool:
        for i, (info, conversa, motivo, n_tok) in enumerate(pool.map(um, tarefas), 1):
            tokens += n_tok
            if not motivo and _nrm(conversa[0]["pessoa"]) in vistas:
                motivo = "repetida"
            if motivo:
                motivos[motivo] += 1
                if args.descartes and conversa:
                    print(f"  [descartada: {motivo}] {json.dumps(conversa, ensure_ascii=False)[:200]}")
            else:
                msgs = []
                for t in conversa:
                    msgs += [{"role": "user", "content": t["pessoa"].strip()},
                             {"role": "assistant", "content": t["lia"].strip()}]
                f.write(json.dumps({"messages": msgs, "professor": info}, ensure_ascii=False) + "\n")
                f.flush()  # se o PC desligar, o que já saiu fica salvo
                vistas.add(_nrm(conversa[0]["pessoa"]))
                ok += 1
            if i % args.log == 0 or i == len(tarefas):
                dt = time.time() - t0
                print(f"  {i}/{len(tarefas)} | {ok} boas | {tokens / dt:.0f} tokens/s | {dt / 60:.1f} min", flush=True)
    print(f"Fim: {ok} conversas boas de {len(tarefas)} -> {out}")
    if motivos:
        print("Descartadas: " + ", ".join(f"{k} {v}" for k, v in motivos.most_common()))


def limpar(args):
    """Tira os vícios do professor que apareceram nas 50 mil (2026-10-26): "Pra quem...?" quase sempre sem sentido
    (a conversa sai) e "Ai ai," no começo de 4% das respostas (só o prefixo sai). Grava só as mensagens, `--copias`
    vezes, para pesar mais no meio dos 1,6 milhão de conversas geradas por modelo de frase."""
    entra, sai, cortadas = Path(args.out), Path(args.limpo), 0
    boas = []
    for linha in open(entra, encoding="utf-8"):
        msgs = json.loads(linha)["messages"]
        if any(m["role"] == "assistant" and m["content"].lower().startswith("pra quem") for m in msgs):
            cortadas += 1
            continue
        for m in msgs:
            if m["role"] == "assistant":
                t = re.sub(r"^ai ai[,.!]*\s*", "", m["content"], flags=re.I)
                m["content"] = t[:1].upper() + t[1:]
        boas.append(json.dumps({"messages": msgs}, ensure_ascii=False))
    sai.parent.mkdir(parents=True, exist_ok=True)
    with open(sai, "w", encoding="utf-8") as f:
        for _ in range(args.copias):
            f.write("\n".join(boas) + "\n")
    print(f"{len(boas):,} conversas ({cortadas:,} com 'Pra quem' cortadas) x {args.copias} -> {sai}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["gerar", "limpar"])
    ap.add_argument("--limpo", default="professor/saida/conversas_limpo.jsonl", help="saída do limpar")
    ap.add_argument("--copias", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="professor/saida/amostra.jsonl")
    ap.add_argument("--porta", type=int, default=8090)
    ap.add_argument("--paralelo", type=int, default=4, help="pedidos ao mesmo tempo (igual ao --parallel do servidor)")
    ap.add_argument("--temperatura", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descartes", action="store_true", help="mostra as conversas descartadas")
    ap.add_argument("--sem-esquema", action="store_true", help="não força o JSON por gramática (Tucano 2)")
    ap.add_argument("--log", type=int, default=25, help="mostra o progresso a cada N pedidos")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"gerar": gerar, "limpar": limpar}[args.etapa](args)


if __name__ == "__main__":
    main()
