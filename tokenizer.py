"""
Tokenizador BPE (Byte Pair Encoding) em nível de bytes, implementado do zero.

Ideia:
  1. O vocabulário começa com os 256 valores possíveis de um byte. Qualquer texto
     UTF-8 pode ser representado com eles, então nunca há "palavra desconhecida".
  2. Repetidamente encontramos o par de tokens vizinhos mais frequente e o fundimos
     num token novo (ex.: b"q" + b"ue" -> b"que"). Cada fusão vira uma regra (merge).
  3. Para codificar texto novo, aplicamos as regras na mesma ordem em que foram
     aprendidas.

Antes de tudo, o texto é quebrado em pedaços (palavras, números, pontuação, espaços)
por uma expressão regular, e as fusões nunca atravessam a fronteira entre pedaços.
Isso evita tokens estranhos como "a, e" e deixa o treino muito mais rápido, porque
basta contar cada pedaço único uma vez (com sua frequência).

Tokens especiais marcam a estrutura de uma conversa:

  <|bos|> <|user|>abre o spotify<|end|>
          <|assistant|><|call|>abrir_app: spotify<|end|>
          <|result|>ok: abri Spotify<|end|>
          <|assistant|>Pronto, Spotify aberto.<|end|>

Eles só são inseridos pelo código (encode_turn); texto digitado nunca vira token
especial, então ninguém consegue "fingir" uma chamada de ferramenta digitando "<|call|>".
"""

import json
import re
from collections import Counter, defaultdict

# Pedaços: palavra (com espaço opcional na frente), até 3 dígitos, pontuação, espaços.
# [^\W\d_] = qualquer letra Unicode (inclui acentos).
SPLIT_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+")

SPECIAL_TOKENS = ["<|bos|>", "<|user|>", "<|assistant|>", "<|call|>", "<|result|>", "<|end|>"]


def merge(ids, pair, new_id):
    """Substitui toda ocorrência de `pair` em `ids` por `new_id`."""
    out = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None, special=None):
        self.merges = {}  # (id_a, id_b) -> novo_id, na ordem em que foram aprendidos
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special = {}  # "<|end|>" -> id
        self._cache = {}
        for a, b in merges or []:
            self._add_merge(a, b)
        self._set_special(SPECIAL_TOKENS if special is None else special)

    def _add_merge(self, a, b):
        new_id = 256 + len(self.merges)
        self.merges[(a, b)] = new_id
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def _set_special(self, names):
        # Os especiais ficam sempre no fim do vocabulário, depois dos merges.
        for i in self.special.values():
            self.vocab.pop(i, None)
        base = 256 + len(self.merges)
        self.special = {name: base + k for k, name in enumerate(names)}
        for name, i in self.special.items():
            self.vocab[i] = name.encode("utf-8")
        s = self.special
        self.bos, self.user, self.assistant = s.get("<|bos|>"), s.get("<|user|>"), s.get("<|assistant|>")
        self.call, self.result, self.end = s.get("<|call|>"), s.get("<|result|>"), s.get("<|end|>")

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ treino
    def train(self, text, vocab_size, verbose=True):
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges >= 0, "vocab_size pequeno demais"
        counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        # Contagem de pares + índice "par -> palavras onde ele aparece".
        # Com o índice, cada fusão só reprocessa as palavras afetadas.
        pair_counts = Counter()
        where = defaultdict(set)
        for i, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pair_counts[p] += freqs[i]
                where[p].add(i)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}
        for k in range(n_merges):
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[pair] < 2:
                break
            new_id = 256 + k
            for i in where.pop(pair):
                ids, f = words[i], freqs[i]
                if len(ids) < 2:
                    continue
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                ids = merge(ids, pair, new_id)
                words[i] = ids
                for p in zip(ids, ids[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
            self._add_merge(*pair)
            if verbose and (k + 1) % 500 == 0:
                tok = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {k + 1}/{n_merges}: {tok!r}")
        self._set_special(SPECIAL_TOKENS)
        self._cache = {}

    # --------------------------------------------------------------- encode
    def _encode_chunk(self, data):
        ids = list(data)
        while len(ids) >= 2:
            # O par com a regra aprendida mais cedo é aplicado primeiro.
            pair = min(set(zip(ids, ids[1:])), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        """Texto comum -> ids. Nunca produz tokens especiais."""
        out = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 200_000:
                    self._cache[chunk] = ids
            out.extend(ids)
        return out

    def encode_turn(self, role, text):
        """
        Uma fala da conversa -> (ids, mask). mask=1 marca os tokens que o modelo
        deve aprender a gerar: só as falas do assistente (texto ou chamada de
        ferramenta). O que o usuário diz e o que a ferramenta devolve é contexto.
        """
        body = self.encode(text)
        if role == "user":
            ids = [self.user] + body + [self.end]
            return ids, [0] * len(ids)
        if role == "result":
            ids = [self.result] + body + [self.end]
            return ids, [0] * len(ids)
        if role == "assistant":
            ids = [self.assistant] + body + [self.end]
            return ids, [0] + [1] * (len(ids) - 1)
        if role == "call":
            ids = [self.assistant, self.call] + body + [self.end]
            return ids, [0] + [1] * (len(ids) - 1)
        raise ValueError(f"papel desconhecido: {role}")

    def encode_chat(self, messages):
        """Lista de {"role", "content"} -> (ids, mask) começando com <|bos|>."""
        ids, mask = [self.bos], [0]
        for m in messages:
            i, k = self.encode_turn(m["role"], m["content"])
            ids += i
            mask += k
        return ids, mask

    # --------------------------------------------------------------- decode
    def decode_bytes(self, ids):
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids, skip_special=False):
        if skip_special:
            ids = [i for i in ids if i < 256 + len(self.merges)]
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- persistência
    def to_dict(self):
        return {"merges": [[a, b] for (a, b) in self.merges], "special": list(self.special)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"], d.get("special"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
