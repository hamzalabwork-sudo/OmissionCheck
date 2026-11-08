"""X9: adaptive attacks on the omission check, against the naive design and the protocol.

  decoy       an insider deletes a laundering record and adds fabricated entries chosen so that they cancel it in
              the sketch: decoys that land in the record's K cells and whose key and check sums, with the right mix
              of payer and payee roles, equal the record's. Against the fixed public hash functions and 64-bit check
              of the naive design this hides the omission completely. In the protocol the books are
              committed before the epoch seed exists, so the attacker must fix its decoys first; the same books are
              then checked under fresh seeds.
  lastmover   the bank that omitted publishes last and sets its sketch to minus the sum of the others' (first
              version: sketches are public when it moves). With commit-then-reveal it must commit before any sketch
              is revealed.
  dropout     banks that commit but never reveal: their mask neighbours reveal the shared masks, and the aggregate
              of the present banks is recovered exactly.

    python -m omission.attacks decoy lastmover dropout
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from finverity.config import RESULTS

from . import data, protocol as P
from .iblt import CHECK, K, LEGACY, WORDS, Sketch, cells, checksum

OUT = RESULTS / "paper2"
N_MEMBERS = 1000
M = 192
KAPPA = 16


def _setup(n_members=N_MEMBERS):
    sel, keys = data.covered_keys(n_members)
    banks = np.array(sorted(data.members(n_members)), dtype=np.int64)
    books = P.honest_books(sel, keys)
    return sel, keys, banks, books


def _target_epoch(sel, books):
    """The epoch with the most covered laundering legs, its books, and the record the insider deletes: the payer
    side of a laundering leg there (the insider bank is the payer)."""
    L = sel["laundering"].to_numpy()
    e = int(pd.Series(sel["epoch"].to_numpy()[L]).value_counts().index[0])
    b = books.take(books.epoch == e)
    row = int(np.flatnonzero(L[b.tx] & (b.sign > 0))[0])
    return e, b, row


# --------------------------------------------------------------------------------------------- GF(2) algebra
def _bits(words: np.ndarray) -> np.ndarray:
    """(n, w) uint64 -> (n, 64 w) bits."""
    return np.unpackbits(words.astype(">u8").view(np.uint8), axis=1)


def _solve_gf2(A: np.ndarray, t: np.ndarray):
    """All solutions of A s = t over GF(2): (particular solution, null-space basis) or None if inconsistent."""
    A = A.copy().astype(np.uint8)
    t = t.copy().astype(np.uint8)
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        p = np.flatnonzero(A[r:, c])
        if not len(p):
            continue
        p = r + p[0]
        A[[r, p]], t[[r, p]] = A[[p, r]], t[[p, r]]
        hit = np.flatnonzero(A[:, c])
        hit = hit[hit != r]
        A[hit] ^= A[r]
        t[hit] ^= t[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    if t[r:].any():
        return None
    x = np.zeros(cols, np.uint8)
    x[piv] = t[:len(piv)]
    free = [c for c in range(cols) if c not in set(piv)]
    basis = []
    for f in free:
        v = np.zeros(cols, np.uint8)
        v[f] = 1
        v[piv] = A[:len(piv), f]
        basis.append(v)
    return x, np.array(basis, np.uint8)


# ----------------------------------------------------------------------------------------------- decoy attack
def grind(target: np.ndarray, word0: list[int], m: int, need: int, seed: bytes, rng, chunk=1 << 22):
    """Random keys (word 0 from word0, words 1-3 free) whose K cells equal the target's, under the hash functions
    of `seed`. Returns the keys and the number of hash evaluations."""
    want = cells(target[None, :], m, seed)[0]
    found, tried = [], 0
    while sum(len(f) for f in found) < need:
        k = rng.integers(0, 2**64, (chunk, WORDS), dtype=np.uint64)
        k[:, 0] = np.asarray(word0, np.uint64)[rng.integers(0, len(word0), chunk)]
        c = cells(k, m, seed)
        found.append(k[(c == want).all(1)])
        tried += chunk
    return np.concatenate(found)[:need], tried


def decoys(target: np.ndarray, cand: np.ndarray, payer_word0: int, seed: bytes, rng, samples=200_000):
    """Choose a subset of candidate keys whose sketch equals +target's: XOR of keys = target, XOR of checks =
    check(target) (a GF(2) system), and #payer-role - #payee-role = +1 (an integer constraint met by sampling the
    null space). Returns (keys, signs) or None."""
    chk_words = 1 if seed == LEGACY else CHECK
    A = np.concatenate([_bits(cand), _bits(checksum(cand, seed)[:, :chk_words])], 1).T
    t = np.concatenate([_bits(target[None, :]), _bits(checksum(target[None, :], seed)[:, :chk_words])], 1)[0]
    sol = _solve_gf2(A, t)
    if sol is None:
        return None
    x, basis = sol
    role = np.where(cand[:, 0] == np.uint64(payer_word0), 1, -1)
    for _ in range(samples):
        s = x ^ (rng.integers(0, 2, len(basis), dtype=np.uint8) @ basis % 2).astype(np.uint8) if len(basis) else x
        if int(role[s == 1].sum()) == 1:
            sel = s == 1
            return cand[sel], role[sel]
    return None


def x9_decoy(trials_new: int = 1000, overprovision: int = 40, masked: int = 20):
    sel, keys, banks, books = _setup()
    e, b, row = _target_epoch(sel, books)
    A, B = int(b.bank[row]), int(b.cpty[row])
    target = b.keys[row]
    rng = np.random.default_rng(99)
    t0 = time.time()
    # the attacker knows the naive design's hash functions, so it grinds against them before the epoch
    w_pay, w_rec = (A << 32) | B, (B << 32) | A
    need = 256 + 64 + overprovision
    cand, tried = grind(target, [w_pay, w_rec], M, need, LEGACY, rng)
    t_grind = time.time() - t0
    pick = decoys(target, cand, w_pay, LEGACY, rng)
    assert pick is not None
    dk, ds = pick
    fake = P.Books(np.full(len(dk), A, np.int64), np.full(len(dk), B, np.int64), np.full(len(dk), e, np.int32),
                   dk, ds.astype(np.int8), np.zeros(len(dk)), -np.ones(len(dk), np.int64))
    attacked = P.Books.concat([b.take(np.arange(len(b)) != row), fake])
    res = dict(epoch=e, rows=len(b), insider=A, counterparty=B, m=M, candidates=len(cand), hash_evaluations=tried,
               grind_s=t_grind, decoys=len(dk), decoys_payer=int((ds > 0).sum()), decoys_payee=int((ds < 0).sum()))
    # naive design: fixed functions, 64-bit check, masks on the fixed Harary graph
    r, agg, _ = P.run_epoch(attacked, banks, M, e, KAPPA, LEGACY)
    res.update(old_flagged=r.flagged, old_residual=r.residual, old_found=r.found, old_aggregate_zero=agg.is_zero())
    # the protocol: the books (with the decoys) are committed, then the seed is drawn
    out = []
    for t in range(trials_new):
        s = P.epoch_seed(e, 0, beacon=f"x9-{t}".encode())
        # masks cancel exactly in the sum (X2 checks every honest epoch), so only the first trials apply them
        kappa = KAPPA if t < masked else None
        r, agg, _ = P.run_epoch(attacked, banks, M, e, kappa, s)
        found_target = r.complete or _decoded(agg, target)
        att, m_t = 0, M
        while not r.complete and att < 6:    # escalation: re-publish the same committed books at twice the size
            att += 1
            m_t *= 2
            r2, _, _ = P.run_epoch(attacked, banks, m_t, e, kappa, P.epoch_seed(e, att, beacon=f"x9-{t}".encode()),
                                   attempt=att)
            r = r2
        out.append(dict(trial=t, masked=kappa is not None, flagged=not agg.is_zero(), complete_m=bool(att == 0), target_decoded=found_target,
                        attempts=att, m_final=m_t, complete_final=r.complete, found_final=r.found,
                        residual=r.residual))
    df = pd.DataFrame(out)
    res.update(new_trials=trials_new, new_flagged=float(df.flagged.mean()), new_complete_first=float(df.complete_m.mean()),
               new_target_decoded_first=float(df.target_decoded.mean()), new_complete_final=float(df.complete_final.mean()),
               new_m_final_max=int(df.m_final.max()), new_attempts_max=int(df.attempts.max()),
               new_residual=int(df.residual.iloc[0]))
    df.to_csv(OUT / "x9_decoy_trials.csv", index=False)
    (OUT / "x9_decoy.json").write_text(json.dumps(res, indent=1, default=lambda v: v.item() if hasattr(v, "item") else v))
    print(json.dumps(res, indent=1, default=str))


def _decoded(agg: Sketch, target: np.ndarray) -> bool:
    _, k, _ = agg.decode()
    return bool(len(k) and (k == target).all(1).any())


# ------------------------------------------------------------------------------------------ last-mover attack
def x9_lastmover(epochs: int = 50):
    """In each of the busiest epochs the bank with the most rows omits one of them and publishes last."""
    sel, keys, banks, books = _setup()
    busy = pd.Series(books.epoch).value_counts().index[:epochs]
    rows = []
    for e in busy:
        b = books.take(books.epoch == e)
        A = int(pd.Series(b.bank).value_counts().index[0])
        row = int(np.flatnonzero(b.bank == A)[0])
        attacked = b.take(np.arange(len(b)) != row)
        pos = int(np.searchsorted(banks, A))
        for version, seed in (("first", LEGACY), ("revised", P.epoch_seed(int(e)))):
            flat = P.bank_sketches(attacked, banks, M, seed)
            P.apply_masks(flat, banks, M, int(e), KAPPA, None if seed == LEGACY else seed)
            n = len(banks)
            others = np.ones(n, bool)
            others[pos] = False
            rest = P.aggregate(flat, n, M, others)
            if version == "first":
                # sketches are public as they are published: the last mover sends minus the others' sum
                agg = rest + (-rest)
            else:
                # commit-then-reveal: A commits before any sketch is revealed, so its sketch is the one it built
                own = Sketch(flat.count[pos * M:(pos + 1) * M].copy(), flat.keysum[pos * M:(pos + 1) * M].copy(),
                             flat.chk[pos * M:(pos + 1) * M].copy(), seed)
                agg = rest + own
            complete, k, s = agg.decode()
            tr = P.truth(attacked)
            dec = {tuple(x.tolist()): int(y) for x, y in zip(k, s)}
            hit = sum(1 for kk, v in dec.items() if tr.get(kk) == v)
            # the omitted record is decoded with the sign that names the payer or payee side that lacks it
            rows.append(dict(epoch=int(e), version=version, bank=A, flagged=not agg.is_zero(), complete=complete,
                             residual=len(tr), found=hit))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x9_lastmover.csv", index=False)
    print(df.groupby("version")[["flagged", "complete", "found", "residual"]].mean())


# ---------------------------------------------------------------------------------------------------- dropout
def x9_dropout(epochs: int = 100, max_absent: int = 3):
    sel, keys, banks, books = _setup()
    rng = np.random.default_rng(5)
    ep = pd.Series(books.epoch).value_counts().index[:epochs * 3]
    rows = []
    n = len(banks)
    for e in rng.choice(ep, epochs, replace=False):
        b = books.take(books.epoch == e)
        active = np.unique(b.bank)
        k = int(rng.integers(1, max_absent + 1))
        gone = rng.choice(active, k, replace=False)
        absent = set(np.searchsorted(banks, gone).tolist())
        seed = P.epoch_seed(int(e))
        flat = P.bank_sketches(b, banks, M, seed)
        P.apply_masks(flat, banks, M, int(e), KAPPA, seed)
        rec = P.recover_dropout(flat, banks, M, int(e), KAPPA, seed, absent)
        present = np.array([i not in absent for i in range(n)])
        expect = P.aggregate(P.bank_sketches(b, banks, M, seed), n, M, present)   # unmasked sum of present banks
        exact = rec.is_zero() == expect.is_zero() and (rec.count == expect.count).all() and \
            (rec.keysum == expect.keysum).all() and (rec.chk == expect.chk).all()
        left = b.take(~np.isin(b.bank, gone))
        tr = P.truth(left)
        complete, kk, ss = rec.decode()
        dec = {tuple(x.tolist()): int(y) for x, y in zip(kk, ss)}
        found = sum(1 for q, v in dec.items() if tr.get(q) == v)
        # privacy after the mask reveal: present banks' remaining present mask neighbours
        pairs = P.mask_pairs(n, KAPPA, seed)
        deg = np.zeros(n, int)
        for lo, hi in pairs:
            if lo not in absent and hi not in absent:
                deg[lo] += 1
                deg[hi] += 1
        mask_edges = int(sum((lo in absent) != (hi in absent) for lo, hi in pairs))
        rows.append(dict(epoch=int(e), absent=k, rows=len(b), exact=bool(exact), residual=len(tr),
                         complete=complete, found=found, revealed_masks=mask_edges,
                         min_present_neighbours=int(deg[present].min())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x9_dropout.csv", index=False)
    print(df.describe().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="+", choices=["decoy", "lastmover", "dropout"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in a.exp:
        t = time.time()
        {"decoy": x9_decoy, "lastmover": x9_lastmover, "dropout": x9_dropout}[e]()
        print(f"{e} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
