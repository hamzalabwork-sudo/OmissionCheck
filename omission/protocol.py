"""Counterparty co-commitment with masked sketches, the baselines, and the attacks.

Books: one row per (bank, side) of a covered transfer. The payer's row carries sign +1, the payee's row sign -1.
An epoch runs in three phases, each closed by a barrier on the ledger:
  1. books   every bank commits to the Merkle root of its epoch book (salted leaves);
  2. seed    the epoch seed is drawn from randomness that appears only after the last book commitment (the hash of the
             block that completed phase 1); it keys the sketch hash functions and the mask-graph positions;
  3. sketch  every bank commits to the hash of S_b = sketch(its rows) + its pairwise masks, and reveals S_b only once
             every sketch hash is on the ledger.
Summing the revealed sketches cancels every mask and every transfer that both witnesses recorded identically, so the
aggregate is the sketch of the discrepancies alone. A bank can neither pick its entries after seeing the hash
functions (they did not exist when it committed) nor pick its sketch after seeing the others' (it committed first).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .iblt import CHECK, K, LEGACY, WORDS, Sketch, cells, checksum

CELL_BYTES = 64          # two 256-bit words per cell, as stored and sent on chain
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


# ------------------------------------------------------------------------------------------ commitments
def epoch_seed(epoch: int, attempt: int = 0, beacon: bytes = b"simulation") -> bytes:
    """The epoch seed. On the ledger it is keccak(hash of the block that completed the book commitments || epoch),
    and a re-publication (attempt > 0) draws a fresh one; here a keyed hash stands in for that block hash."""
    return hashlib.blake2b(f"seed|{epoch}|{attempt}".encode(), key=beacon, digest_size=32).digest()


def _h(prefix: bytes, data_: bytes) -> bytes:
    return hashlib.sha256(prefix + data_).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    """RFC 6962 Merkle tree hash: leaves and nodes are domain-separated (0x00 / 0x01), odd nodes are promoted, not
    duplicated, so two different leaf lists never share a root."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    level = [_h(b"\x00", x) for x in leaves]
    while len(level) > 1:
        nxt = [_h(b"\x01", level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def book_leaves(b: Books, salt_key: bytes) -> list[bytes]:
    """Salted leaves of a bank's epoch book: H(salt_i || key || sign). Salts come from a key the bank holds and
    discloses to its auditor, so the root reveals nothing about low-entropy entries."""
    out = []
    for i, (k, s) in enumerate(zip(b.keys, b.sign.tolist())):
        salt = hashlib.blake2b(i.to_bytes(8, "big"), key=salt_key, digest_size=16).digest()
        out.append(salt + k.astype(">u8").tobytes() + int(s).to_bytes(1, "big", signed=True))
    return out


# ---------------------------------------------------------------------------------------------- protocol
def mask_pairs(n: int, kappa: int, seed: bytes | None = None) -> np.ndarray:
    """Harary-style mask graph over positions: position i shares a mask with positions i+1 .. i+kappa/2 (mod n).
    With a seed, banks are assigned to positions by a fresh random permutation each epoch, so nobody can choose its
    mask neighbours. Returns (P, 2) bank indices, smaller first."""
    half = max(1, min(kappa // 2, (n - 1) // 2))
    i = np.repeat(np.arange(n), half)
    j = (i + np.tile(np.arange(1, half + 1), n)) % n
    if seed:
        perm = np.random.Generator(np.random.Philox(key=np.frombuffer(
            hashlib.blake2b(seed, person=b"mask-graph", digest_size=16).digest(), np.uint64))).permutation(n)
        i, j = perm[i], perm[j]
    return np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], 1), axis=0)


def _pair_rng(lo: int, hi: int, epoch: int, m: int, attempt: int) -> np.random.Generator:
    # stands in for ChaCha20 keyed by the pair's mask sub-key with nonce (epoch, m, attempt) (timed in bench.py): a
    # re-publication at another size or attempt never reuses a mask stream
    k = data.subkey(data.pair_key(lo, hi), "mask")
    seed = hashlib.blake2b(f"mask|{epoch}|{m}|{attempt}".encode(), key=k, digest_size=16).digest()
    return np.random.Generator(np.random.Philox(key=np.frombuffer(seed, np.uint64)))


def bank_sketches(b: Books, banks: np.ndarray, m: int, seed: bytes = LEGACY) -> Sketch:
    """Unmasked sketches of every bank, stacked: arrays of shape (len(banks) * m, ...)."""
    pos = np.searchsorted(banks, b.bank)
    flat = Sketch(np.zeros(len(banks) * m, np.uint64), np.zeros((len(banks) * m, WORDS), np.uint64),
                  np.zeros((len(banks) * m, CHECK), np.uint64), seed)
    if len(b):
        idx = cells(b.keys, m, seed) + (pos * m)[:, None]
        chk = checksum(b.keys, seed)
        w = b.sign.astype(np.int64).astype(np.uint64)
        for j in range(K):
            np.add.at(flat.count, idx[:, j], w)
            for c in range(CHECK):
                np.bitwise_xor.at(flat.chk[:, c], idx[:, j], chk[:, c])
            for c in range(WORDS):
                np.bitwise_xor.at(flat.keysum[:, c], idx[:, j], b.keys[:, c])
    return flat


def masks(banks: np.ndarray, m: int, epoch: int, kappa: int, seed: bytes | None, attempt: int = 0):
    """Yield (lo, hi, mask) for every edge of the epoch's mask graph; lo adds the mask, hi subtracts it."""
    for lo, hi in mask_pairs(len(banks), kappa, seed):
        rng = _pair_rng(int(banks[lo]), int(banks[hi]), epoch, m, attempt)
        yield int(lo), int(hi), Sketch.empty(m).mask(rng, +1)


def _add_mask(flat: Sketch, pos: int, m: int, mk: Sketch, sign: int) -> None:
    r = slice(pos * m, pos * m + m)
    with np.errstate(over="ignore"):
        flat.count[r] = flat.count[r] + mk.count if sign > 0 else flat.count[r] - mk.count
    flat.keysum[r] ^= mk.keysum
    flat.chk[r] ^= mk.chk


def apply_masks(flat: Sketch, banks: np.ndarray, m: int, epoch: int, kappa: int, seed: bytes | None = None,
                attempt: int = 0) -> None:
    for lo, hi, mk in masks(banks, m, epoch, kappa, seed, attempt):
        _add_mask(flat, lo, m, mk, +1)
        _add_mask(flat, hi, m, mk, -1)


def aggregate(flat: Sketch, n: int, m: int, present: np.ndarray | None = None) -> Sketch:
    """Sum of the published sketches (of the banks in `present`, default all)."""
    sel = np.ones(n, bool) if present is None else present
    with np.errstate(over="ignore"):
        return Sketch(flat.count.reshape(n, m)[sel].sum(0, dtype=np.uint64),
                      np.bitwise_xor.reduce(flat.keysum.reshape(n, m, WORDS)[sel], axis=0),
                      np.bitwise_xor.reduce(flat.chk.reshape(n, m, CHECK)[sel], axis=0), flat.seed)


def recover_dropout(flat: Sketch, banks: np.ndarray, m: int, epoch: int, kappa: int, seed: bytes, absent: set[int],
                    attempt: int = 0) -> Sketch:
    """Aggregate an epoch in which the banks at positions `absent` revealed no sketch: each present mask neighbour of
    an absent bank reveals the mask they share, which the verifier removes. The absent bank's transfers then appear
    as one-sided records of its counterparties, attributed to its absence."""
    present = np.array([i not in absent for i in range(len(banks))])
    agg = aggregate(flat, len(banks), m, present)
    for lo, hi, mk in masks(banks, m, epoch, kappa, seed, attempt):
        if (lo in absent) == (hi in absent):
            continue
        # the present side added +mk (if lo) or -mk (if hi); the revealed mask cancels it
        agg = agg + (-mk if hi in absent else mk)
    return agg


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


def run_epoch(b: Books, banks: np.ndarray, m: int, epoch: int, kappa: int | None, seed: bytes | None = None,
              attempt: int = 0) -> tuple[EpochResult, Sketch, Sketch]:
    """One epoch of the protocol. seed=None draws the epoch seed (the protocol); LEGACY uses the fixed functions."""
    import time
    if seed is None:
        seed = epoch_seed(epoch, attempt)
    t0 = time.perf_counter()
    flat = bank_sketches(b, banks, m, seed)
    if kappa:
        apply_masks(flat, banks, m, epoch, kappa, seed, attempt)
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
