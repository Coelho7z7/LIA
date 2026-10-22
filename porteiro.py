"""
Porteiro: modelo pequeno e separado que decide, a cada fala do usuário, se a Lia usa ferramenta ou conversa.

Por quê: no modelo único, corrigir o tom da conversa mexia na decisão das ferramentas (e vice-versa). Com o
porteiro, cada parte é treinada sozinha e em minutos. Ele é um GPT minúsculo (preset "tiny") com o mesmo
tokenizador; olha só os últimos JANELA tokens do contexto e, no último, prevê <|call|> (ferramenta) ou
<|end|> (conversa). Assim o motor do site roda ele sem código novo.

    python porteiro.py dados --conversas data_v29/conversas --out out_porteiro      # monta os exemplos
    python porteiro.py treinar --out out_porteiro                                     # treina (minutos)
    python porteiro.py testar --out out_porteiro "abre o spotify" "tá chovendo"        # p(ferramenta)
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

JANELA = 96  # tokens de contexto que o porteiro olha (a decisão quase sempre está na última fala)
CFG = dict(n_layer=4, n_head=4, n_embd=192, block_size=JANELA)


def janela(ids):
    return ids[-JANELA:]


def montar(conversas, tok, max_exemplos, seed):
    """Cada fala do usuário vira um exemplo: (contexto até <|assistant|>, 1 se a próxima é chamada)."""
    rng = random.Random(seed)
    arquivos = sorted(Path(conversas).rglob("*.jsonl"))
    exemplos = []
    for f in arquivos:
        for line in f.read_text(encoding="utf-8").splitlines():
            msgs = json.loads(line)["messages"]
            ids = [tok.bos]
            for i, m in enumerate(msgs):
                kw = {"dropout": 0.1, "rng": rng, "seg": {}} if m["role"] == "user" else {}
                t, _ = tok.encode_turn(m["role"], m["content"], **kw)
                ids += t
                if m["role"] == "user" and i + 1 < len(msgs):
                    exemplos.append((janela(ids + [tok.assistant]), int(msgs[i + 1]["role"] == "call")))
        if len(exemplos) >= max_exemplos:
            break
    rng.shuffle(exemplos)
    return exemplos[:max_exemplos]


def cmd_dados(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(a.tokenizer)
    ex = montar(a.conversas, tok, a.max, a.seed)
    n_val = min(20000, len(ex) // 20)
    torch.save({"treino": ex[n_val:], "val": ex[:n_val], "tokenizer": tok.to_dict()}, out / "exemplos.pt")
    pos = sum(y for _, y in ex)
    print(f"{len(ex)} exemplos ({pos / len(ex):.1%} ferramenta), {n_val} de validação -> {out / 'exemplos.pt'}")


def lotes(exemplos, tam, rng):
    """Agrupa por tamanho (sem preenchimento: o modelo não tem máscara de padding)."""
    por = defaultdict(list)
    for ids, y in exemplos:
        por[len(ids)].append((ids, y))
    chaves = list(por)
    pesos = [len(por[k]) for k in chaves]
    while True:
        k = rng.choices(chaves, pesos)[0]
        b = rng.sample(por[k], min(tam, len(por[k])))
        yield torch.tensor([i for i, _ in b]), torch.tensor([y for _, y in b])


def perda(model, x, y, tok, dev):
    x, y = x.to(dev), y.to(dev)
    alvo = torch.full_like(x, -100)
    alvo[:, -1] = torch.where(y == 1, torch.tensor(tok.call, device=dev), torch.tensor(tok.end, device=dev))
    logits, loss = model(x, alvo)
    pred = (logits[:, tok.call] > logits[:, tok.end]).long()
    return loss, (pred == y).float().mean().item()


def cmd_treinar(a):
    out = Path(a.out)
    d = torch.load(out / "exemplos.pt", weights_only=False)
    tok = BPETokenizer.from_dict(d["tokenizer"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg = GPTConfig(vocab_size=tok.vocab_size, dropout=0.1, **CFG)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    rng = random.Random(a.seed)
    tr, va = lotes(d["treino"], a.batch, rng), lotes(d["val"], 512, random.Random(1))
    print(f"porteiro: {model.num_params() / 1e6:.2f}M parâmetros, {len(d['treino'])} exemplos, {dev}")
    for it in range(1, a.iters + 1):
        lr = a.lr * min(1, it / 200) * (0.1 + 0.9 * (1 - it / a.iters))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        loss, acc = perda(model, *next(tr), tok, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 500 == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                accs = [perda(model, *next(va), tok, dev)[1] for _ in range(20)]
            print(f"passo {it}: perda {loss.item():.4f} | acerto val {sum(accs) / len(accs):.4f}", flush=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "tokenizer": d["tokenizer"], "janela": JANELA,
                "iter": a.iters, "best_val": 0.0, "stage": "porteiro"}, out / "porteiro.pt")
    print("salvo em", out / "porteiro.pt")


class Porteiro:
    """Uso no agente: p = porteiro.p_ferramenta(contexto_ids_terminando_em_assistant)."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.tok = BPETokenizer.from_dict(ck["tokenizer"])
        self.model = GPT(GPTConfig(**ck["config"])).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device, self.janela = device, ck["janela"]

    @torch.no_grad()
    def p_ferramenta(self, ids):
        x = torch.tensor([ids[-self.janela:]], device=self.device)
        logits, _ = self.model(x)
        two = torch.stack([logits[0, -1, self.tok.call], logits[0, -1, self.tok.end]]).float()
        return torch.softmax(two, 0)[0].item()


def cmd_testar(a):
    p = Porteiro(Path(a.out) / "porteiro.pt")
    tok = p.tok
    for f in a.frases:
        ids, _ = tok.encode_turn("user", f)
        print(f"{p.p_ferramenta([tok.bos] + ids + [tok.assistant]):.3f}  {f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dados")
    d.add_argument("--conversas", required=True)
    d.add_argument("--tokenizer", default="out_v15/tokenizer.json")
    d.add_argument("--out", default="out_porteiro")
    d.add_argument("--max", type=int, default=3_000_000)
    d.add_argument("--seed", type=int, default=7)
    t = sub.add_parser("treinar")
    t.add_argument("--out", default="out_porteiro")
    t.add_argument("--iters", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("testar")
    s.add_argument("--out", default="out_porteiro")
    s.add_argument("frases", nargs="+")
    a = ap.parse_args()
    {"dados": cmd_dados, "treinar": cmd_treinar, "testar": cmd_testar}[a.cmd](a)


if __name__ == "__main__":
    main()
