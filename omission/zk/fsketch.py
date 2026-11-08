"""Field version of the masked sketch, the form a zero-knowledge proof can bind to a book commitment.

The deployed sketch (omission.iblt) adds counts modulo 2^64 and XORs key and check sums. Here every cell is a triple
of BN254 field elements and all three are added: count += w, keysum += w*key, chk += w*h(s, key), with w = +1 for the
payer and -1 for the payee. The sketch is still linear, so pairwise masks (+M for one side of a pair, -M for the
other) cancel in the sum, and the sum of all banks' sketches is the field sketch of the discrepancies. A pure cell has
count = +-1 and chk = count * h(s, keysum * count), and peeling recovers each one-sided record as before. A key is
one field element of 248 bits laid out like the deployed key: payer bank (32 bits) || payee bank (32 bits) || the
first 184 bits of the canonical transfer hash. h(s, key) is a keyed Poseidon hash whose low bits give the K = 3 cell
positions (one per partition of m/3 cells) and whose full value is the check.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .field import P, commit, poseidon, stream

K = 3
DOM_CELLS = 0x53_4B_45_54     # commitment domains: cells, pair keys, challenges
DOM_PAIR = 0x50_41_49_52
KEY_BITS = 248
BANK_BITS, TAG_BITS = 32, 184


def make_key(payer: int, payee: int, tag: int) -> int:
    return (payer << (BANK_BITS + TAG_BITS)) | (payee << TAG_BITS) | (tag & ((1 << TAG_BITS) - 1))


def key_banks(key: int) -> tuple[int, int]:
    return key >> (BANK_BITS + TAG_BITS), (key >> TAG_BITS) & ((1 << BANK_BITS) - 1)


def weight(key: int, bank: int) -> int:
    """+1 if the bank is the key's payer, -1 if its payee, 0 otherwise (or for a transfer within one bank)."""
    p, q = key_banks(key)
    return int(p == bank) - int(q == bank)


def part_bits(m: int) -> int:
    part = m // K
    if m % K or part & (part - 1):
        raise ValueError("m must be 3 * 2^b so that a partition index is b bits of the hash")
    return part.bit_length() - 1


def entry_hash(seed: int, key: int) -> int:
    return poseidon(seed, key)


def positions(h: int, m: int) -> list[int]:
    b = part_bits(m)
    return [j * (m // K) + ((h >> (b * j)) & ((1 << b) - 1)) for j in range(K)]


@dataclass
class FSketch:
    count: list[int]
    keysum: list[int]
    chk: list[int]

    @classmethod
    def empty(cls, m: int) -> "FSketch":
        return cls([0] * m, [0] * m, [0] * m)

    @property
    def m(self) -> int:
        return len(self.count)

    def cells(self) -> list[int]:
        """Flat vector of 3m field elements in the order the circuit commits to: counts, key sums, check sums."""
        return self.count + self.keysum + self.chk

    def __add__(self, o: "FSketch") -> "FSketch":
        return FSketch([(a + b) % P for a, b in zip(self.count, o.count)],
                       [(a + b) % P for a, b in zip(self.keysum, o.keysum)],
                       [(a + b) % P for a, b in zip(self.chk, o.chk)])

    def insert(self, key: int, w: int, seed: int) -> None:
        h = entry_hash(seed, key)
        for c in positions(h, self.m):
            self.count[c] = (self.count[c] + w) % P
            self.keysum[c] = (self.keysum[c] + w * key) % P
            self.chk[c] = (self.chk[c] + w * h) % P


def sketch(entries: list[tuple[int, int]], seed: int, m: int) -> FSketch:
    s = FSketch.empty(m)
    for key, w in entries:
        if w:
            s.insert(key, w, seed)
    return s


# ---------------------------------------------------------------------------------------------- book commitment
def salts(salt_key: bytes, n: int) -> list[int]:
    return [int.from_bytes(hashlib.blake2b(i.to_bytes(8, "big"), key=salt_key, digest_size=31).digest(), "big")
            for i in range(n)]


def leaf(salt: int, key: int, w: int) -> int:
    return poseidon(salt, key, w % P)


def pad(entries: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Books padded to the circuit size n (a power of 4) with zero-weight entries."""
    if len(entries) > n:
        raise ValueError("books larger than the circuit")
    return list(entries) + [(0, 0)] * (n - len(entries))


def book_root(entries: list[tuple[int, int]], salt_list: list[int]) -> int:
    """4-ary Poseidon Merkle root over salted leaves of the padded books."""
    level = [leaf(s, k, w) for s, (k, w) in zip(salt_list, entries)]
    if len(level) & (len(level) - 1) or (len(level).bit_length() - 1) % 2:
        raise ValueError("n must be a power of 4")
    while len(level) > 1:
        level = [poseidon(*level[i:i + 4]) for i in range(0, len(level), 4)]
    return level[0]


# --------------------------------------------------------------------------------------------------- masks
def pair_key(secret: bytes, a: int, b: int) -> int:
    lo, hi = sorted((a, b))
    return int.from_bytes(hashlib.blake2b(b"%d|%d" % (lo, hi), key=secret, digest_size=31).digest(), "big")


def pair_commitment(k: int) -> int:
    return poseidon(k, DOM_PAIR)


def mask_nonce(epoch: int, m: int, attempt: int) -> int:
    return (epoch << 40) | (m << 8) | attempt


def neighbours(i: int, n_banks: int, kappa: int) -> list[tuple[int, int]]:
    """Mask slots of bank i on a ring: (neighbour, sign); kappa/2 neighbours ahead (+1), then kappa/2 behind (-1)."""
    d = range(1, kappa // 2 + 1)
    return [((i + x) % n_banks, +1) for x in d] + [((i - x) % n_banks, -1) for x in d]


def mask_vector(keys: list[int], signs: list[int], epoch: int, m: int, attempt: int) -> list[int]:
    out = [0] * (3 * m)
    nonce = mask_nonce(epoch, m, attempt)
    for k, sg in zip(keys, signs):
        for c, v in enumerate(stream(k, nonce, 3 * m)):
            out[c] = (out[c] + sg * v) % P
    return out


def masked(s: FSketch, keys: list[int], signs: list[int], epoch: int, m: int, attempt: int) -> FSketch:
    mv = mask_vector(keys, signs, epoch, m, attempt)
    cells = [(a + b) % P for a, b in zip(s.cells(), mv)]
    return FSketch(cells[:m], cells[m:2 * m], cells[2 * m:])


def cell_commitment(cells: list[int]) -> int:
    return commit(cells, DOM_CELLS)


# ------------------------------------------------------------------------------------------------- decoding
def decode(s: FSketch, seed: int) -> tuple[list[tuple[int, int]], bool]:
    """Peel the field sketch: returns the (key, weight) records and whether the sketch emptied completely."""
    s = FSketch(list(s.count), list(s.keysum), list(s.chk))
    out: list[tuple[int, int]] = []
    progress = True
    while progress:
        progress = False
        for c in range(s.m):
            cnt = s.count[c]
            if cnt not in (1, P - 1):
                continue
            w = 1 if cnt == 1 else -1
            key = (s.keysum[c] * w) % P
            if key >> KEY_BITS or (s.chk[c] - w * entry_hash(seed, key)) % P:
                continue
            out.append((key, w))
            s.insert(key, -w, seed)
            progress = True
    empty = not any(s.count) and not any(s.keysum) and not any(s.chk)
    return out, empty
