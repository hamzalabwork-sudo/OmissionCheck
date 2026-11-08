"""Tests of the field sketch and the binding circuit model: python -m omission.zk.test_zk"""
from __future__ import annotations

import random

from . import circuit as Z
from . import fsketch as F
from .field import P

M = 192


def _books(n_banks: int, n_tx: int, rng: random.Random):
    """Transfers between random bank pairs; books[b] lists (key, w): +1 as payer, -1 as payee."""
    books = {b: [] for b in range(n_banks)}
    for _ in range(n_tx):
        a, b = rng.sample(range(n_banks), 2)
        key = F.make_key(a, b, rng.getrandbits(F.TAG_BITS))
        books[a].append((key, +1))
        books[b].append((key, -1))
    return books


def test_masked_sum_decodes() -> None:
    rng = random.Random(7)
    n_banks, kappa, seed, epoch = 8, 4, rng.getrandbits(250), 3
    books = _books(n_banks, 120, rng)
    gone = books[2][:3]                          # bank 2 deletes three records
    books[2] = books[2][3:]
    fake = (F.make_key(5, 1, rng.getrandbits(F.TAG_BITS)), +1)    # bank 5 fabricates a payment to bank 1
    books[5].append(fake)
    secret = b"pair-key-test"
    total = F.FSketch.empty(M)
    for b in range(n_banks):
        slots = F.neighbours(b, n_banks, kappa)
        keys = [F.pair_key(secret, b, o) for o, _ in slots]
        total = total + F.masked(F.sketch(books[b], seed, M), keys, [s for _, s in slots], epoch, M, 0)
    got, empty = F.decode(total, seed)
    want = {(k, -w) for k, w in gone} | {fake}
    assert empty and set(got) == want, (got, want)


BANK = 3


def _entry(rng: random.Random, bank: int = BANK) -> tuple[int, int]:
    other = rng.choice([x for x in range(20) if x != bank])
    pair = (bank, other) if rng.random() < .5 else (other, bank)
    key = F.make_key(*pair, rng.getrandbits(F.TAG_BITS))
    return key, F.weight(key, bank)


def _bank(n: int, rng: random.Random, kappa: int):
    entries = F.pad([_entry(rng) for _ in range(n - 5)], n)
    salts = F.salts(b"bank-salt", n)
    keys = [rng.getrandbits(250) for _ in range(kappa)]
    signs = [+1] * (kappa // 2) + [-1] * (kappa // 2)
    return entries, salts, keys, signs


def test_circuit(n: int = 64, kappa: int = 4) -> None:
    rng = random.Random(11)
    seed, epoch = rng.getrandbits(250), 9
    entries, salts, keys, signs = _bank(n, rng, kappa)
    st, wt = Z.honest(BANK, entries, salts, seed, M, epoch, 0, keys, signs)
    res = Z.check(st, wt, n, M, kappa)
    assert res.ok, res.failures
    assert res.counts == Z.count(n, M, kappa), (res.counts, Z.count(n, M, kappa))

    def fails(st2, wt2, why):
        r = Z.check(st2, wt2, n, M, kappa)
        assert not r.ok and any(why in f for f in r.failures), (why, r.failures)

    # self-cancellation: the bank removes one of its own committed entries from the published cells
    k, w = entries[0]
    unmasked = F.sketch(entries, seed, M)
    unmasked.insert(k, -w, seed)
    cells = F.masked(unmasked, keys, signs, epoch, M, 0).cells()
    fails(Z.Statement(BANK, st.root, seed, F.cell_commitment(cells), st.nonce, st.pair_commitments),
          Z.Witness(entries, salts, cells, keys), "not the masked sketch")
    # a sketch of other books than the committed ones
    other = list(entries)
    other[1] = _entry(rng)
    fails(st, Z.Witness(other, salts, wt.cells, keys), "book root")
    # a "mask" that is not the committed pair key's stream
    fails(st, Z.Witness(entries, salts, wt.cells, [keys[0] + 1] + keys[1:]), "pair key")
    # an aliased bit decomposition of an entry hash (h + p < 2^254), which would move the entry to other cells
    bits = None
    for i, (kk, _) in enumerate(entries):
        h = F.entry_hash(seed, kk)
        if h + P < 1 << Z.HBITS:
            bits = [[(F.entry_hash(seed, e[0]) >> j) & 1 for j in range(Z.HBITS)] for e in entries]
            bits[i] = [((h + P) >> j) & 1 for j in range(Z.HBITS)]
            break
    assert bits is not None
    fails(st, Z.Witness(entries, salts, wt.cells, keys, bits), "aliased")
    # framing: a committed entry naming two other banks, inserted with weight +1
    framed = list(entries)
    framed[-1] = (F.make_key(7, 8, rng.getrandbits(F.TAG_BITS)), 1)
    salt_root = F.book_root(framed, salts)
    cells = F.masked(F.sketch(framed, seed, M), keys, signs, epoch, M, 0).cells()
    fails(Z.Statement(BANK, salt_root, seed, F.cell_commitment(cells), st.nonce, st.pair_commitments),
          Z.Witness(framed, salts, cells, keys), "weight")
    # a flipped sign: the bank books its own payment as received
    flipped = list(entries)
    flipped[0] = (entries[0][0], -entries[0][1])
    cells = F.masked(F.sketch(flipped, seed, M), keys, signs, epoch, M, 0).cells()
    fails(Z.Statement(BANK, F.book_root(flipped, salts), seed, F.cell_commitment(cells), st.nonce,
                      st.pair_commitments), Z.Witness(flipped, salts, cells, keys), "weight")


if __name__ == "__main__":
    test_masked_sum_decodes()
    print("masked field sketches sum and decode: ok")
    test_circuit()
    print("circuit model: honest witness satisfies, six tampered witnesses fail: ok")
