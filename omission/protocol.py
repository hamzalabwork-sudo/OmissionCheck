"""Counterparty co-commitment with masked sketches, the baselines, and the attacks.

Books: one row per (bank, side) of a covered transfer. The payer's row carries sign +1, the payee's row sign -1. In
each epoch every consortium bank publishes S_b = sketch(its rows) + sum of pairwise masks shared with its kappa mask
neighbours (+ for the lower identifier, - for the higher). Summing the published sketches cancels every mask and every
transfer that both witnesses recorded identically, so the aggregate is the sketch of the discrepancies alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .iblt import K, WORDS, Sketch, cells, checksum

CELL_BYTES = 8 + 8 * WORDS + 8
TAG_BYTES = 32
TOTAL_BYTES = 24


@dataclass
class Books:
    bank: np.ndarray        # int64
    cpty: np.ndarray        # int64 counterparty
    epoch: np.ndarray       # int32
    keys: np.ndarray        # (n, 4) uint64
    sign: np.ndarray        # int8, +1 payer side, -1 payee side
    amount: np.ndarray      # float64
    tx: np.ndarray          # int64 index of the covered transfer, -1 if fabricated

    def take(self, sel) -> "Books":
        return Books(*(getattr(self, f)[sel] for f in self.__dataclass_fields__))

    def __len__(self):
        return len(self.bank)

    @staticmethod
    def concat(parts: list["Books"]) -> "Books":
        return Books(*(np.concatenate([getattr(p, f) for p in parts]) for f in Books.__dataclass_fields__))


def honest_books(sel: pd.DataFrame, keys: np.ndarray) -> Books:
    n = len(sel)
    payer, payee = sel["payer"].to_numpy(np.int64), sel["payee"].to_numpy(np.int64)
    ep = sel["epoch"].to_numpy(np.int32)
    amt = sel["amount"].to_numpy(float)
    tx = np.arange(n, dtype=np.int64)
    return Books(np.concatenate([payer, payee]), np.concatenate([payee, payer]), np.concatenate([ep, ep]),
                 np.concatenate([keys, keys]), np.concatenate([np.ones(n, np.int8), -np.ones(n, np.int8)]),
                 np.concatenate([amt, amt]), np.concatenate([tx, tx]))


# ----------------------------------------------------------------------------------------------- attacks
@dataclass
class Attack:
    name: str
    books: Books
    attacked_tx: set = field(default_factory=set)   # covered transfers whose books were tampered with
    attackers: set = field(default_factory=set)


def omit(b: Books, drop: np.ndarray, name: str, attackers=()) -> Attack:
    return Attack(name, b.take(~drop), set(b.tx[drop].tolist()), set(attackers))


def random_omission(b: Books, p: float, rng) -> Attack:
    """Every bank independently loses each of its rows with probability p (systematic or careless omission)."""
    return omit(b, rng.random(len(b)) < p, f"random p={p:g}")


def targeted_laundering(b: Books, laundering: np.ndarray, bank: int) -> Attack:
    """An insider at one bank deletes every laundering leg the bank took part in."""
    return omit(b, (b.bank == bank) & laundering[b.tx], f"insider deletes laundering legs at bank {bank}", [bank])


def collusion(b: Books, laundering: np.ndarray, banks: set[int]) -> Attack:
    """A coalition deletes the laundering legs among its members from both sides."""
    both = np.isin(b.bank, list(banks)) & np.isin(b.cpty, list(banks))
    return omit(b, both & laundering[b.tx], f"coalition of {len(banks)} deletes shared laundering legs", banks)


def substitution(b: Books, bank: int, per_epoch: int, rng) -> Attack:
    """A bank replaces some of its rows by fabricated rows with the same counterparty, side and amount: per-pair counts
    and sums still balance."""
    mine = np.flatnonzero(b.bank == bank)
    ep = b.epoch[mine]
    pick = []
    for e in np.unique(ep):
        rows = mine[ep == e]
        pick.extend(rng.choice(rows, min(per_epoch, len(rows)), replace=False).tolist())
    pick = np.array(pick, dtype=np.int64)
    drop = np.zeros(len(b), bool)
    drop[pick] = True
    fake = b.take(pick)
    payer = np.where(fake.sign > 0, fake.bank, fake.cpty)
    payee = np.where(fake.sign > 0, fake.cpty, fake.bank)
    fake.keys = data.fabricate_keys(payer, payee, f"sub-{bank}")
    fake.tx = -np.ones(len(pick), np.int64)
    out = Books.concat([b.take(~drop), fake])
    return Attack(f"substitution at bank {bank}", out, set(b.tx[pick].tolist()), {bank})


def alteration(b: Books, bank: int, per_epoch: int, rng, factor: float = 0.99) -> Attack:
    """A bank under-reports the amount of some of its rows; the tag of an altered transfer no longer matches."""
    a = substitution(b, bank, per_epoch, rng)
    fake = a.books.tx < 0
    a.books.amount[fake] *= factor
    a.name = f"alteration at bank {bank}"
    return a


# ---------------------------------------------------------------------------------------------- protocol
def mask_pairs(n: int, kappa: int) -> np.ndarray:
    """Harary-style mask graph: bank i shares a mask with banks i+1 .. i+kappa/2 (mod n). Returns (P, 2) positions."""
    half = max(1, min(kappa // 2, (n - 1) // 2))
    i = np.repeat(np.arange(n), half)
    j = (i + np.tile(np.arange(1, half + 1), n)) % n
    return np.stack([np.minimum(i, j), np.maximum(i, j)], 1)


def _pair_rng(lo: int, hi: int, epoch: int) -> np.random.Generator:
    # stands in for a PRF keyed by the pair key (ChaCha20 in the deployed protocol, timed in bench.py)
    seed = hashlib.blake2b(f"mask|{lo}|{hi}|{epoch}".encode(), key=data.MASTER, digest_size=16).digest()
    return np.random.Generator(np.random.Philox(key=np.frombuffer(seed, np.uint64)))


def bank_sketches(b: Books, banks: np.ndarray, m: int) -> Sketch:
    """Unmasked sketches of every bank, stacked: arrays of shape (len(banks) * m, ...)."""
    pos = np.searchsorted(banks, b.bank)
    flat = Sketch(np.zeros(len(banks) * m, np.uint64), np.zeros((len(banks) * m, WORDS), np.uint64),
                  np.zeros(len(banks) * m, np.uint64))
    if len(b):
        idx = cells(b.keys, m) + (pos * m)[:, None]
        chk = checksum(b.keys)
        w = b.sign.astype(np.int64).astype(np.uint64)
        for j in range(K):
            np.add.at(flat.count, idx[:, j], w)
            np.bitwise_xor.at(flat.chk, idx[:, j], chk)
            for c in range(WORDS):
                np.bitwise_xor.at(flat.keysum[:, c], idx[:, j], b.keys[:, c])
    return flat


def apply_masks(flat: Sketch, banks: np.ndarray, m: int, epoch: int, kappa: int) -> None:
    for lo, hi in mask_pairs(len(banks), kappa):
        r = slice(lo * m, lo * m + m), slice(hi * m, hi * m + m)
        rng = _pair_rng(int(banks[lo]), int(banks[hi]), epoch)
        mk = Sketch.empty(m).mask(rng, +1)
        with np.errstate(over="ignore"):
            flat.count[r[0]] += mk.count
            flat.count[r[1]] -= mk.count
        flat.keysum[r[0]] ^= mk.keysum
        flat.keysum[r[1]] ^= mk.keysum
        flat.chk[r[0]] ^= mk.chk
        flat.chk[r[1]] ^= mk.chk


def aggregate(flat: Sketch, n: int, m: int) -> Sketch:
    with np.errstate(over="ignore"):
        return Sketch(flat.count.reshape(n, m).sum(0, dtype=np.uint64),
                      np.bitwise_xor.reduce(flat.keysum.reshape(n, m, WORDS), axis=0),
                      np.bitwise_xor.reduce(flat.chk.reshape(n, m), axis=0))


def truth(b: Books) -> dict[tuple, int]:
    """Net weight of every key that does not cancel: what a perfect reconciler must report."""
    if not len(b):
        return {}
    k = pd.DataFrame(b.keys.view(np.int64), columns=list("abcd"))
    k["s"] = b.sign.astype(np.int64)
    g = k.groupby(list("abcd"), sort=False)["s"].sum()
    g = g[g != 0]
    return {tuple(np.array(i, np.int64).view(np.uint64).tolist()): int(v) for i, v in g.items()}


@dataclass
class EpochResult:
    epoch: int
    rows: int
    residual: int           # true number of non-cancelling keys
    flagged: bool           # aggregate non-zero
    complete: bool          # peeling emptied the aggregate
    found: int              # decoded keys that are true discrepancies with the right sign
    spurious: int           # decoded keys that are not
    t_build_s: float
    t_decode_s: float


def run_epoch(b: Books, banks: np.ndarray, m: int, epoch: int, kappa: int | None) -> tuple[EpochResult, Sketch, Sketch]:
    import time
    t0 = time.perf_counter()
    flat = bank_sketches(b, banks, m)
    if kappa:
        apply_masks(flat, banks, m, epoch, kappa)
    agg = aggregate(flat, len(banks), m)
    t1 = time.perf_counter()
    complete, keys, signs = agg.decode()
    t2 = time.perf_counter()
    tr = truth(b)
    dec = {tuple(k.tolist()): int(s) for k, s in zip(keys, signs)}
    found = sum(1 for k, s in dec.items() if tr.get(k) == s)
    return (EpochResult(epoch, len(b), len(tr), not agg.is_zero(), complete, found, len(dec) - found,
                        t1 - t0, t2 - t1), agg, flat)


# --------------------------------------------------------------------------------------------- baselines
def baseline_totals(b: Books) -> set[tuple[int, int, int]]:
    """Per-pair totals: both banks publish the count and amount (cents) of their transfers per ordered pair and epoch.
    Returns the (payer, payee, epoch) cells whose two sides disagree."""
    payer = np.where(b.sign > 0, b.bank, b.cpty)
    payee = np.where(b.sign > 0, b.cpty, b.bank)
    df = pd.DataFrame({"payer": payer, "payee": payee, "epoch": b.epoch, "s": b.sign.astype(np.int64),
                       "c": np.round(b.amount * 100).astype(np.int64) * b.sign.astype(np.int64)})
    g = df.groupby(["payer", "payee", "epoch"]).agg(n=("s", "sum"), v=("c", "sum"))
    bad = g[(g["n"] != 0) | (g["v"] != 0)]
    return set(map(tuple, bad.index.to_list()))


def totals_bytes(b: Books) -> int:
    return TOTAL_BYTES * len(pd.DataFrame({"a": b.bank, "c": b.cpty, "e": b.epoch}).drop_duplicates())
