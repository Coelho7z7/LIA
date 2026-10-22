// O mesmo Transformer do model.py (estilo Llama), só a parte de inferência, em JavaScript puro:
// RMSNorm, atenção causal com RoPE (metades, não intercalado), SwiGLU, embeddings amarrados e KV cache.
// Todos os pesos ficam num único Float32Array; cada tensor é um deslocamento dentro dele.

// f16 -> f32 por tabela (65536 valores), bem mais rápido que converter bit a bit
let HALF = null;
function halfTable() {
  if (HALF) return HALF;
  HALF = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    HALF[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return HALF;
}

// manifest = modelo.json; buffer = os pedaços pesos.NN.bin colados (ArrayBuffer)
export function carregarPesos(manifest, buffer) {
  let total = 0;
  const pos = {};
  for (const t of manifest.tensores) {
    pos[t.name] = total;
    total += t.shape.reduce((a, b) => a * b, 1);
  }
  const W = new Float32Array(total);
  const half = halfTable();
  for (const t of manifest.tensores) {
    const n = t.shape.reduce((a, b) => a * b, 1), o = pos[t.name];
    if (t.dtype === "f32") {
      W.set(new Float32Array(buffer, t.offset, n), o);
    } else if (t.dtype === "f16") {
      const src = new Uint16Array(buffer, t.offset, n);
      for (let i = 0; i < n; i++) W[o + i] = half[src[i]];
    } else {  // i8 por linha: valor = q * escala da linha
      const [rows, cols] = t.shape, q = new Int8Array(buffer, t.offset, n), sc = new Float32Array(buffer, t.scales, rows);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) W[o + r * cols + c] = q[r * cols + c] * sc[r];
    }
  }
  return { W, pos };
}

// Y (T x rows) = X (T x cols) · Wᵀ, com W (rows x cols) começando em `off`.
// A linha do peso fica no cache enquanto passa por todos os tokens.
function matmul(X, T, W, off, rows, cols, Y) {
  for (let r = 0; r < rows; r++) {
    const b = off + r * cols;
    let t = 0;
    // Leitura do contexto (vários tokens): 4 tokens por passada, cada peso lido uma vez serve aos 4.
    // A soma de cada token segue na mesma ordem do caso de 1 token, então o resultado é idêntico.
    for (; t + 4 <= T; t += 4) {
      const x0 = t * cols, x1 = x0 + cols, x2 = x1 + cols, x3 = x2 + cols;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      let c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
      for (let i = 0; i < cols; i += 4) {
        const w0 = W[b + i], w1 = W[b + i + 1], w2 = W[b + i + 2], w3 = W[b + i + 3];
        a0 += w0 * X[x0 + i]; a1 += w1 * X[x0 + i + 1]; a2 += w2 * X[x0 + i + 2]; a3 += w3 * X[x0 + i + 3];
        b0 += w0 * X[x1 + i]; b1 += w1 * X[x1 + i + 1]; b2 += w2 * X[x1 + i + 2]; b3 += w3 * X[x1 + i + 3];
        c0 += w0 * X[x2 + i]; c1 += w1 * X[x2 + i + 1]; c2 += w2 * X[x2 + i + 2]; c3 += w3 * X[x2 + i + 3];
        d0 += w0 * X[x3 + i]; d1 += w1 * X[x3 + i + 1]; d2 += w2 * X[x3 + i + 2]; d3 += w3 * X[x3 + i + 3];
      }
      Y[t * rows + r] = a0 + a1 + a2 + a3;
      Y[(t + 1) * rows + r] = b0 + b1 + b2 + b3;
      Y[(t + 2) * rows + r] = c0 + c1 + c2 + c3;
      Y[(t + 3) * rows + r] = d0 + d1 + d2 + d3;
    }
    for (; t < T; t++) {
      const xb = t * cols;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < cols; i += 4) {
        s0 += W[b + i] * X[xb + i];
        s1 += W[b + i + 1] * X[xb + i + 1];
        s2 += W[b + i + 2] * X[xb + i + 2];
        s3 += W[b + i + 3] * X[xb + i + 3];
      }
      Y[t * rows + r] = s0 + s1 + s2 + s3;
    }
  }
}

function rmsnorm(X, T, C, W, off, Y, eps = 1e-6) {
  for (let t = 0; t < T; t++) {
    const b = t * C;
    let ss = 0;
    for (let i = 0; i < C; i++) ss += X[b + i] * X[b + i];
    const k = 1 / Math.sqrt(ss / C + eps);
    for (let i = 0; i < C; i++) Y[b + i] = X[b + i] * k * W[off + i];
  }
}

export class Modelo {
  constructor(manifest, buffer) {
    const c = manifest.config;
    this.cfg = c;
    this.V = c.vocab_size; this.C = c.n_embd; this.L = c.n_layer; this.H = c.n_head;
    this.D = c.n_embd / c.n_head; this.Tmax = c.block_size;
    this.F = 64 * Math.ceil(8 * c.n_embd / 3 / 64);  // largura do SwiGLU, igual ao Python
    const { W, pos } = carregarPesos(manifest, buffer);
    this.W = W; this.pos = pos;
    // tabelas do RoPE: cos/sin de cada (posição, frequência)
    const half = this.D / 2;
    this.cos = new Float32Array(this.Tmax * half);
    this.sin = new Float32Array(this.Tmax * half);
    for (let p = 0; p < this.Tmax; p++) {
      for (let i = 0; i < half; i++) {
        const inv = 1 / c.rope_theta ** ((2 * i) / this.D);
        this.cos[p * half + i] = Math.cos(p * inv);
        this.sin[p * half + i] = Math.sin(p * inv);
      }
    }
  }

  novoCache() {
    const n = this.L * this.H * this.Tmax * this.D;
    return { pos: 0, k: new Float32Array(n), v: new Float32Array(n) };
  }

  // Processa `ids` a partir de cache.pos e devolve os logits (V) do último token.
  forward(ids, cache) {
    const { C, H, D, F, V, W, pos: P } = this;
    const T = ids.length, start = cache.pos, half = D / 2, scale = 1 / Math.sqrt(D);
    if (start + T > this.Tmax) throw new Error(`${start + T} tokens > block_size ${this.Tmax}`);
    const x = new Float32Array(T * C), xn = new Float32Array(T * C);
    const qkv = new Float32Array(T * 3 * C), att = new Float32Array(T * C), tmp = new Float32Array(T * C);
    const g = new Float32Array(T * F), u = new Float32Array(T * F);
    const scores = new Float32Array(start + T);
    const emb = P["tok_emb.weight"];
    for (let t = 0; t < T; t++) x.set(W.subarray(emb + ids[t] * C, emb + (ids[t] + 1) * C), t * C);

    for (let l = 0; l < this.L; l++) {
      const pre = `blocks.${l}.`;
      rmsnorm(x, T, C, W, P[pre + "norm1.weight"], xn);
      matmul(xn, T, W, P[pre + "attn.qkv.weight"], 3 * C, C, qkv);
      // RoPE em q e k (no lugar) e guarda k, v no cache
      for (let t = 0; t < T; t++) {
        const p = start + t, cb = p * half;
        for (let h = 0; h < H; h++) {
          for (const part of [0, C]) {  // q e depois k
            const b = t * 3 * C + part + h * D;
            for (let i = 0; i < half; i++) {
              const x1 = qkv[b + i], x2 = qkv[b + i + half], co = this.cos[cb + i], si = this.sin[cb + i];
              qkv[b + i] = x1 * co - x2 * si;
              qkv[b + i + half] = x1 * si + x2 * co;
            }
          }
          const cOff = ((l * H + h) * this.Tmax + p) * D, kb = t * 3 * C + C + h * D, vb = kb + C;
          for (let i = 0; i < D; i++) { cache.k[cOff + i] = qkv[kb + i]; cache.v[cOff + i] = qkv[vb + i]; }
        }
      }
      // atenção: o token t enxerga as posições 0..start+t
      for (let t = 0; t < T; t++) {
        const n = start + t + 1;
        for (let h = 0; h < H; h++) {
          const qb = t * 3 * C + h * D, base = (l * H + h) * this.Tmax * D;
          let mx = -Infinity;
          for (let j = 0; j < n; j++) {
            const kb = base + j * D;
            let s = 0;
            for (let i = 0; i < D; i++) s += qkv[qb + i] * cache.k[kb + i];
            s *= scale;
            scores[j] = s;
            if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j < n; j++) { const e = Math.exp(scores[j] - mx); scores[j] = e; sum += e; }
          const ob = t * C + h * D;
          for (let i = 0; i < D; i++) att[ob + i] = 0;
          for (let j = 0; j < n; j++) {
            const w = scores[j] / sum, vb = base + j * D;
            for (let i = 0; i < D; i++) att[ob + i] += w * cache.v[vb + i];
          }
        }
      }
      matmul(att, T, W, P[pre + "attn.proj.weight"], C, C, tmp);
      for (let i = 0; i < T * C; i++) x[i] += tmp[i];
      // SwiGLU: proj(silu(gate(x)) * up(x))
      rmsnorm(x, T, C, W, P[pre + "norm2.weight"], xn);
      matmul(xn, T, W, P[pre + "mlp.gate.weight"], F, C, g);
      matmul(xn, T, W, P[pre + "mlp.up.weight"], F, C, u);
      for (let i = 0; i < T * F; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      matmul(g, T, W, P[pre + "mlp.proj.weight"], C, F, tmp);
      for (let i = 0; i < T * C; i++) x[i] += tmp[i];
    }
    cache.pos += T;
    // norma final e vocabulário só no último token (lm_head = embedding)
    const last = x.subarray((T - 1) * C), ln = new Float32Array(C), logits = new Float32Array(V);
    rmsnorm(last, 1, C, W, P["norm.weight"], ln);
    matmul(ln, 1, W, emb, V, C, logits);
    return logits;
  }
}
