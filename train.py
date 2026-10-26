"""
Treina o modelo em duas etapas (como os LLMs de verdade):

  1. Pré-treino: aprende a língua lendo seus textos (data/textos/). Opcional.
       python train.py --stage pretrain --preset small --max-minutes 60
     -> out/base.pt

  2. Chat (SFT): aprende a conversar e usar ferramentas com as conversas.
     Se out/base.pt existir, parte dele; senão começa do zero.
       python train.py --stage sft --preset small --max-minutes 45
     -> out/chat.pt

Ctrl+C a qualquer momento: o melhor checkpoint já está salvo.
    python train.py --stage sft --resume     # continua de onde parou
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=192, block_size=256),    # ~2.6M parâmetros, minutos na CPU
    "small": dict(n_layer=6, n_head=6, n_embd=288, block_size=512),   # ~7M, bom para CPU
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=512),  # ~27M, melhor com GPU
    "large": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),  # ~88M, só com GPU
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft"], default="sft")
    ap.add_argument("--data", default="out")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--init", default=None, help="checkpoint inicial (padrão no sft: out/base.pt se existir)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1, help="acumula N batches antes de cada passo")
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--max-minutes", type=float, default=0, help="para após N minutos (0 = sem limite)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--compile", action="store_true", help="torch.compile (precisa de compilador C++)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-last", action="store_true",
                    help="grava o checkpoint do fim mesmo se o val não melhorou (ex.: dados com BPE-dropout)")
    ap.add_argument("--ema", type=float, default=0,
                    help="média móvel dos pesos (ex.: 0.999): avalia e grava a média, que generaliza melhor")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 na GPU acelera muito; na CPU (sem suporte nativo) fica em float32.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    tok = BPETokenizer.load(data / "tokenizer.json")
    ckpt_path = out / ("base.pt" if args.stage == "pretrain" else "chat.pt")

    # ------------------------------------------------------------------ dados
    if args.stage == "pretrain":
        if not (data / "pretrain_train.bin").exists():
            raise SystemExit("Sem dados de pré-treino. Coloque .txt em data/textos/ e rode prepare.py.")
        splits = {s: {"ids": np.memmap(data / f"pretrain_{s}.bin", dtype=np.uint16, mode="r")} for s in ("train", "val")}
    else:
        splits = {s: {"ids": np.memmap(data / f"sft_{s}.bin", dtype=np.uint16, mode="r"),
                      "mask": np.memmap(data / f"sft_{s}_mask.bin", dtype=np.uint8, mode="r"),
                      "starts": np.load(data / f"sft_{s}_starts.npy")} for s in ("train", "val")}

    # ----------------------------------------------------------------- modelo
    init_path = args.init
    if init_path is None and args.stage == "sft" and (out / "base.pt").exists():
        init_path = str(out / "base.pt")
    if init_path == "none":
        init_path = None
    start_iter, best_val, ckpt = 0, float("inf"), None
    retomar_path = out / f"retomar_{ckpt_path.name}"
    if args.resume:
        de = retomar_path if retomar_path.exists() and retomar_path.stat().st_mtime > ckpt_path.stat().st_mtime \
            else ckpt_path
        ckpt = torch.load(de, map_location=device, weights_only=False)
        print(f"Retomando de {de}")
        cfg = GPTConfig(**ckpt["config"])
    elif init_path:
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        cfg = GPTConfig(**ckpt["config"])
        if cfg.vocab_size != tok.vocab_size:
            raise SystemExit(f"{init_path} usa vocab {cfg.vocab_size}, mas o tokenizador tem {tok.vocab_size}. "
                             "Refaça o pré-treino depois de rodar prepare.py.")
        cfg.dropout = args.dropout
        print(f"Partindo de {init_path} (pré-treinado)")
    else:
        cfg = GPTConfig(vocab_size=tok.vocab_size, dropout=args.dropout, **PRESETS[args.preset])
    model = GPT(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    # EMA: cópia dos pesos que anda devagar atrás do treino (média dos últimos ~1/(1-ema) passos).
    # O checkpoint guarda a média em "model" (o que o agente usa) e os pesos crus em "raw_params".
    params = list(model.parameters())
    ema = [p.detach().clone() for p in params] if args.ema else None
    if ema is not None and args.resume and "raw_params" in ckpt:
        with torch.no_grad():
            for p, r in zip(params, ckpt["raw_params"]):
                p.copy_(r)
    lr = args.lr or (3e-4 if (init_path and args.stage == "sft") else 1e-3)
    min_lr = lr * args.min_lr_frac

    T = cfg.block_size
    n_train = len(splits["train"]["ids"])
    tokens_per_iter = args.batch_size * args.grad_accum * T
    print(f"Modelo: {model.num_params() / 1e6:.2f}M parâmetros | {cfg} | device={device}")
    print(f"Etapa {args.stage}: {n_train:,} tokens de treino, {tokens_per_iter:,} tokens/passo, "
          f"~{args.max_iters * tokens_per_iter / n_train:.1f} épocas em {args.max_iters} passos | lr {lr:g}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device == "cuda")
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
        print(f"Retomando da iteração {start_iter} (melhor val loss {best_val:.4f})")
    train_model = torch.compile(model) if args.compile else model

    def get_batch(split):
        d = splits[split]
        ids = d["ids"]
        if args.stage == "sft":
            # Cada exemplo começa no início de uma conversa e leva T tokens (pode
            # emendar na conversa seguinte, igual a um texto corrido).
            st = d["starts"][np.random.randint(0, len(d["starts"]), args.batch_size)]
            st = np.minimum(st, len(ids) - T - 1)
        else:
            st = np.random.randint(0, len(ids) - T - 1, args.batch_size)
        x = torch.from_numpy(np.stack([ids[s:s + T] for s in st]).astype(np.int64))
        y = torch.from_numpy(np.stack([ids[s + 1:s + 1 + T] for s in st]).astype(np.int64))
        if args.stage == "sft":
            m = torch.from_numpy(np.stack([d["mask"][s + 1:s + 1 + T] for s in st]).astype(bool))
            y[~m] = -100  # só as falas do assistente contam na perda
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        res = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_iters):
                with autocast:
                    losses.append(model(*get_batch(split))[1].item())
            res[split] = sum(losses) / len(losses)
        model.train()
        return res

    @torch.no_grad()
    def show_samples():
        model.eval()
        if args.stage == "pretrain":
            ids = list(model.generate([tok.bos], 80, temperature=0.8))
            print("  amostra:", repr(tok.decode(ids))[:250])
        else:
            for q in ("oi, tudo bem?", "abre o spotify pra mim", "quanto é 12 vezes 8?", "me conta uma piada"):
                ctx = [tok.bos] + tok.encode_turn("user", q)[0] + [tok.assistant]
                ids = list(model.generate(ctx, 60, temperature=0, stop_ids=(tok.end,)))
                print(f"  {q!r:28} -> {tok.decode(ids)!r}")
        model.train()

    def lr_at(it):
        # Warmup linear e depois cosseno até min_lr. O "progresso" é o que acabar
        # primeiro, passos ou minutos, então parar pelo relógio também termina
        # com a taxa de aprendizado baixa (o que deixa o modelo final melhor).
        if it < args.warmup:
            return lr * (it + 1) / args.warmup
        ratio = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        if args.max_minutes:
            ratio = max(ratio, (time.time() - t_start) / (args.max_minutes * 60))
        ratio = min(1.0, ratio)
        return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)

    @torch.no_grad()
    def swap_ema():
        """Troca pesos crus <-> média (chamar de novo desfaz)."""
        for p, e in zip(params, ema):
            tmp = p.detach().clone()
            p.copy_(e)
            e.copy_(tmp)

    def save(it, path=ckpt_path):
        # com EMA, é chamado com a média no modelo (e os pesos crus guardados em `ema`)
        extra = {"raw_params": [e.detach().clone() for e in ema], "ema": args.ema} if ema is not None else {}
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg),
                    "tokenizer": tok.to_dict(), "stage": args.stage, "iter": it, "best_val": best_val,
                    "args": vars(args), **extra}, path)

    log_path = out / f"log_{args.stage}.csv"
    if not args.resume or not log_path.exists():
        log_path.write_text("iter,train_loss,val_loss,minutes\n")

    t_start = t_last = time.time()
    steps_since = 0
    model.train()
    it = start_iter
    try:
        for it in range(start_iter, args.max_iters + 1):
            last = it == args.max_iters or (args.max_minutes and time.time() - t_start > args.max_minutes * 60)
            if it % args.eval_interval == 0 or last:
                losses = estimate_loss()
                minutes = (time.time() - t_start) / 60
                raw_info = ""
                if ema is not None:  # daqui até o fim do bloco o modelo fica com a média
                    raw_info = f" | cru: val {losses['val']:.4f}"
                    swap_ema()
                    losses = estimate_loss()
                print(f"\n[iter {it}] train {losses['train']:.4f} | val {losses['val']:.4f}{raw_info} | {minutes:.1f} min")
                with open(log_path, "a") as f:
                    f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f},{minutes:.2f}\n")
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save(it)
                    print(f"  checkpoint salvo em {ckpt_path}")
                elif last and args.save_last:
                    save(it)
                    print(f"  checkpoint do fim salvo em {ckpt_path} (val não melhorou)")
                elif args.save_last:  # ponto de retomada: parar no meio não volta ao último "melhor"
                    save(it, retomar_path)
                show_samples()
                print()
                t_last, steps_since = time.time(), 0
                if last:
                    break
                if ema is not None:
                    swap_ema()  # volta aos pesos crus para seguir treinando

            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            for _ in range(args.grad_accum):
                x, y = get_batch("train")
                with autocast:
                    _, loss = train_model(x, y)
                (loss / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita passos gigantes
            opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    d = min(args.ema, (1 + it) / (10 + it))  # começo: média curta, senão fica presa no início
                    torch._foreach_mul_(ema, d)
                    torch._foreach_add_(ema, [p.detach() for p in params], alpha=1 - d)

            steps_since += 1
            if it % args.log_interval == 0:
                dt = time.time() - t_last
                print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr_at(it):.2e} | "
                      f"{steps_since * tokens_per_iter / dt:,.0f} tok/s")
                t_last, steps_since = time.time(), 0
    except KeyboardInterrupt:
        print("\nInterrompido.")

    print(f"Fim na iteração {it}. Melhor val loss: {best_val:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
