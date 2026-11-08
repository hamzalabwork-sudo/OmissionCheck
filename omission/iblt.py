"""Linear invertible Bloom lookup table (IBLT) over 256-bit keys with signed counts.

A sketch is linear: the cell-wise sum of the sketches of two multisets is the sketch of their signed union. Each
cell holds a count (added modulo 2^64), a key sum and a 128-bit check sum (both XOR). An item with weight +1 or -1
lands in one cell of each of K equal partitions, so the K cells are always distinct. The difference of two sketches
decodes by peeling as long as the symmetric difference is below roughly m / 1.222 items for K = 3 (Goodrich and
Mitzenmacher 2011). Keys are rows of four uint64 words.

The cell hashes h_1..h_K and the check function g are keyed by an epoch seed that is drawn only after every bank has
committed to its books (protocol.epoch_seed), so no bank can choose keys that collide under them. The fast keyed mixer
below stands in for keyed BLAKE2b in the simulation; bench.py times the deployed primitive. seed=b"" gives the fixed,
public functions and the 64-bit check of the naive design, kept to demonstrate the attack they allow.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

K = 3
WORDS = 4
CHECK = 2            # check words: 128 bits (the naive design uses one)
LEGACY = b""         # seed of the fixed, public hash functions of the naive design
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(x: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 finaliser: a fast, well-mixed 64-bit hash (vectorised)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64((0x9E3779B97F4A7C15 * (seed + 1)) & 0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


@lru_cache(maxsize=4096)
def _salts(seed: bytes) -> np.ndarray:
    """Eight 64-bit salts of an epoch seed: cell hash, K partitions, CHECK check words (zero for the legacy seed)."""
    if not seed:
        return np.zeros(8, np.uint64)
    return np.frombuffer(hashlib.blake2b(seed, person=b"iblt-salts", digest_size=64).digest(), np.uint64).copy()


def checksum(keys: np.ndarray, seed: bytes = LEGACY) -> np.ndarray:
    """Check value g(key), shape (n, CHECK), independent of the cell indices. Legacy: 64 bits (second word zero)."""
    s = _salts(seed)
    out = np.zeros((len(keys), CHECK), dtype=np.uint64)
    for c in range(1 if not seed else CHECK):
        h = np.full(len(keys), s[4 + c], dtype=np.uint64)
        for w in range(WORDS):
            h = _mix(h ^ keys[:, w], 101 + w + 10 * c)
        out[:, c] = h
    return out


def cells(keys: np.ndarray, m: int, seed: bytes = LEGACY) -> np.ndarray:
    """Cell index of each key in each of the K partitions, shape (n, K)."""
    part = m // K
    s = _salts(seed)
    h = np.full(len(keys), s[0], dtype=np.uint64)
    for w in range(WORDS):
        h = _mix(h ^ keys[:, w], 7 + w)
    out = np.empty((len(keys), K), dtype=np.int64)
    for j in range(K):
        out[:, j] = j * part + (_mix(h ^ s[1 + j], 31 + j) % np.uint64(part)).astype(np.int64)
    return out


@dataclass
class Sketch:
    count: np.ndarray    # (m,) uint64, two's-complement signed counts
    keysum: np.ndarray   # (m, WORDS) uint64, XOR
    chk: np.ndarray      # (m, CHECK) uint64, XOR
    seed: bytes = LEGACY

    @classmethod
    def empty(cls, m: int, seed: bytes = LEGACY) -> "Sketch":
        if m % K:
            raise ValueError(f"m must be a multiple of {K}")
        return cls(np.zeros(m, np.uint64), np.zeros((m, WORDS), np.uint64), np.zeros((m, CHECK), np.uint64), seed)

    @property
    def m(self) -> int:
        return len(self.count)

    def insert(self, keys: np.ndarray, signs: np.ndarray) -> "Sketch":
        """Add each key with weight sign (+1 or -1)."""
        if not len(keys):
            return self
        idx = cells(keys, self.m, self.seed)
        chk = checksum(keys, self.seed)
        w = signs.astype(np.int64).astype(np.uint64)  # -1 -> 2^64 - 1
        for j in range(K):
            np.add.at(self.count, idx[:, j], w)
            for c in range(CHECK):
                np.bitwise_xor.at(self.chk[:, c], idx[:, j], chk[:, c])
            for c in range(WORDS):
                np.bitwise_xor.at(self.keysum[:, c], idx[:, j], keys[:, c])
        return self

    def __add__(self, other: "Sketch") -> "Sketch":
        with np.errstate(over="ignore"):
            return Sketch(self.count + other.count, self.keysum ^ other.keysum, self.chk ^ other.chk, self.seed)

    def __neg__(self) -> "Sketch":
        """Additive inverse: negated counts; the XOR fields are their own inverse."""
        with np.errstate(over="ignore"):
            return Sketch(np.uint64(0) - self.count, self.keysum.copy(), self.chk.copy(), self.seed)

    def mask(self, rng: np.random.Generator, sign: int) -> "Sketch":
        """Add (sign=+1) or remove (sign=-1) a pseudo-random mask drawn from rng; equal draws cancel exactly."""
        c = rng.integers(0, 2**64, self.m, dtype=np.uint64, endpoint=False)
        ks = rng.integers(0, 2**64, (self.m, WORDS), dtype=np.uint64, endpoint=False)
        ch = rng.integers(0, 2**64, (self.m, CHECK), dtype=np.uint64, endpoint=False)
        with np.errstate(over="ignore"):
            self.count = self.count + c if sign > 0 else self.count - c
        self.keysum ^= ks
        self.chk ^= ch
        return self

    def is_zero(self) -> bool:
        return not (self.count.any() or self.keysum.any() or self.chk.any())

    def nbytes(self) -> int:
        return self.m * 64     # two 256-bit words per cell, as stored and sent on chain

    def decode(self) -> tuple[bool, np.ndarray, np.ndarray]:
        """Peel the sketch. Returns (complete, keys, signs). complete=False leaves some items undecoded."""
        s = Sketch(self.count.copy(), self.keysum.copy(), self.chk.copy(), self.seed)
        signed = s.count.view(np.int64)
        keys, signs = [], []
        queue = list(np.flatnonzero((signed == 1) | (signed == -1)))
        while queue:
            i = queue.pop()
            if signed[i] not in (1, -1):
                continue
            key = s.keysum[i:i + 1].copy()
            ck = checksum(key, s.seed)[0]
            if not np.array_equal(ck, s.chk[i]):
                continue
            sg = int(signed[i])
            keys.append(key[0])
            signs.append(sg)
            for j in cells(key, s.m, s.seed)[0]:
                signed[j] -= sg
                s.keysum[j] ^= key[0]
                s.chk[j] ^= ck
                if signed[j] in (1, -1):
                    queue.append(j)
        complete = s.is_zero()
        k = np.array(keys, dtype=np.uint64).reshape(-1, WORDS)
        return complete, k, np.array(signs, dtype=np.int64)


def threshold(k: int = K) -> float:
    """Asymptotic peeling threshold c_k: decoding succeeds w.h.p. when m > c_k * d (c_3 = 1.222)."""
    return {3: 1.222, 4: 1.295, 5: 1.425}[k]
