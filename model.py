"""
Transformer decoder-only (arquitetura GPT) escrito do zero, com as melhorias que
os modelos atuais usam (estilo Llama):

  - RMSNorm no lugar de LayerNorm: mais simples e rápida.
  - RoPE (rotary position embeddings): a posição entra girando os vetores Q e K,
    então a atenção enxerga a distância RELATIVA entre tokens. Sem tabela de
    posição aprendida.
  - SwiGLU no MLP: uma "porta" multiplicativa que aprende melhor que GELU puro.
  - KV cache: na geração, as chaves/valores de tokens antigos ficam guardados,
    então cada token novo custa O(1) passes em vez de reprocessar o contexto todo.
  - Sem bias nas camadas lineares e embeddings amarrados (weight tying).

Fluxo de um forward:
  ids (B, T) -> embedding (B, T, C)
    -> N blocos [RMSNorm -> atenção causal com RoPE -> +residual,
                 RMSNorm -> SwiGLU                  -> +residual]
    -> RMSNorm -> projeção no vocabulário = logits (B, T, V)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512  # contexto máximo em tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 288
    dropout: float = 0.0
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Divide cada vetor pela raiz da média dos quadrados e aplica um ganho aprendido."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(max_len, head_dim, theta):
    """cos/sin de cada (posição, frequência). Frequências altas = detalhes locais."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Gira pares de dimensões de x por um ângulo proporcional à posição."""
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Guarda K e V já calculados de todas as camadas para gerar token a token."""

    def __init__(self, cfg, batch_size=1, device="cpu", dtype=torch.float32):
        head_dim = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch_size, cfg.n_head, cfg.block_size, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = cfg.block_size
        self.pos = 0  # quantos tokens já estão no cache

    def update(self, layer, k, v):
        end = self.pos + k.size(2)
        self.k[layer, :, :, self.pos:end] = k
        self.v[layer, :, :, self.pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def reset(self):
        self.pos = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "RoPE precisa de head_dim par"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, layer=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # softmax(Q·Kᵀ/√d + máscara causal)·V. O PyTorch faz isso num kernel otimizado;
        # a versão "na mão" seria:
        #   att = (q @ k.transpose(-2, -1)) / sqrt(d); att[futuro] = -inf
        #   y = softmax(att) @ v
        if cache is None:
            # Sem dropout DENTRO da atenção: na CPU ele força o caminho lento do PyTorch
            # (2,3x mais devagar medido num Ryzen 5). O dropout residual já regulariza.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            past = cache.pos
            k, v = cache.update(layer, k, v)
            if T == 1:
                mask = None  # um único token novo pode olhar tudo que já passou
            else:
                # token i do pedaço novo (posição past+i) enxerga as posições 0..past+i
                mask = torch.arange(k.size(2), device=x.device)[None, :] <= (
                    past + torch.arange(T, device=x.device)[:, None])
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    """MLP com porta: proj(silu(gate(x)) * up(x))."""

    def __init__(self, cfg):
        super().__init__()
        hidden = 64 * math.ceil(8 * cfg.n_embd / 3 / 64)  # ~2/3 de 4x, mesmo custo do MLP clássico
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, layer)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Projeções que somam no fluxo residual começam menores, para a soma de
        # muitas camadas não explodir no início do treino.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def new_cache(self, batch_size=1):
        p = self.tok_emb.weight
        return KVCache(self.cfg, batch_size, p.device, p.dtype)

    def forward(self, idx, targets=None, cache=None):
        """
        Treino: passe `targets` (posições com -100 são ignoradas na perda).
        Inferência: sem targets, devolve só os logits da última posição.
        """
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.block_size, f"{start + T} tokens > block_size {self.cfg.block_size}"
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        if cache is not None:
            cache.pos += T
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None
        # Só calcula o vocabulário inteiro onde há alvo (no chat, as falas do usuário
        # são ignoradas): economiza a camada mais cara por token.
        valid = targets != -100
        logits = self.lm_head(x[valid])
        if logits.numel() == 0:
            return logits, x.sum() * 0.0
        return logits, F.cross_entropy(logits, targets[valid])

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, stop_ids=()):
        """Continua uma lista de ids, devolvendo cada token novo assim que sai."""
        device = self.tok_emb.weight.device
        ids = list(ids)[-self.cfg.block_size:]
        cache = self.new_cache()
        logits, _ = self(torch.tensor([ids], device=device), cache=cache)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[0, -1], temperature, top_k, top_p)
            if nxt in stop_ids:
                return
            yield nxt
            ids.append(nxt)
            if cache.pos >= self.cfg.block_size:
                # Contexto cheio: recomeça o cache com a metade mais recente.
                keep = ids[-(self.cfg.block_size // 2):]
                cache.reset()
                logits, _ = self(torch.tensor([keep], device=device), cache=cache)
            else:
                logits, _ = self(torch.tensor([[nxt]], device=device), cache=cache)


def apply_repetition_penalty(logits, recent_ids, penalty):
    """Deixa menos provável repetir tokens que acabaram de aparecer."""
    if penalty == 1.0 or not recent_ids:
        return logits
    idx = torch.tensor(sorted(set(recent_ids)), device=logits.device)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def sample_next(logits, temperature=0.8, top_k=0, top_p=1.0):
    """
    Escolhe o próximo token a partir dos logits (V,).
      temperature: <1 deixa mais conservador, >1 mais criativo, 0 = sempre o mais provável.
      top_k: só considera os k tokens mais prováveis.
      top_p: só considera o menor grupo de tokens que soma probabilidade p (nucleus).
    """
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        sp[(sp.cumsum(0) - sp) > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))
