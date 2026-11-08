"""R1CS model of the sketch-binding circuit: exact multiplicative-constraint counts and a relation checker.

Statement (public): the bank's identifier b, book root R, epoch seed s, cell commitment C, mask nonce (epoch, m,
attempt), and the commitments Q_1..Q_kappa to the bank's pair keys in slot order. Witness: the n padded book entries (key, w) with their salts, the
3m published (masked) cells, the kappa pair keys and the bit decompositions of the entry hashes.

The circuit enforces
  1. the salted leaves hash to R (4-ary Poseidon tree), and each entry's weight is fixed by its key:
     w = [payer(key) = b] - [payee(key) = b], so an entry that does not name b contributes nothing (no framing) and
     the sign cannot be flipped (no self-cancellation through a wrong sign);
  2. h_i = Poseidon(s, key_i), decomposed into 254 bits with circomlib's alias check (so the low bits, which give the
     cell positions, are unique);
  3. the cells hash to C, and Q_b = Poseidon(K_b, DOM_PAIR) for every mask slot;
  4. with challenges r, alpha = Poseidon(C, R, s, 1 | 2) drawn from the committed values,
         sum_c r^c (u_c + alpha v_c + alpha^2 z_c) = sum_i w_i (1 + alpha key_i + alpha^2 h_i) sum_j r^{pos_ij},
     where (u, v, z) are the published cells minus the masks sum_b sign_b PRF(K_b, nonce). By Schwartz-Zippel the
     unmasked cells then equal the sketch of the committed books except with probability (m + 1) / p.

Poseidon permutations are evaluated natively and counted at 3 constraints per S-box (circomlib's structure); every
other gadget is evaluated constraint by constraint on the prover's values, so a tampered witness fails a check.
Counts are multiplicative constraints; linear ones are removed by the compiler's simplification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dfield

from . import fsketch as F
from .field import P, commit_perms, constraints, poseidon, stream, stream_perms

DOM_R, DOM_A = 1, 2
HBITS = 254


@dataclass
class Statement:
    bank: int
    root: int
    seed: int
    cells_commitment: int
    nonce: int
    pair_commitments: list[int]

    def public_inputs(self) -> list[int]:
        return [self.bank, self.root, self.seed, self.cells_commitment, self.nonce, *self.pair_commitments]


@dataclass
class Witness:
    entries: list[tuple[int, int]]
    salts: list[int]
    cells: list[int]
    pair_keys: list[int]
    bits: list[list[int]] | None = None     # prover-supplied hash decompositions (None: canonical)


@dataclass
class Result:
    ok: bool
    failures: list[str]
    counts: Counter = dfield(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def comp_constant(bits: list[int], ct: int) -> tuple[int, list[int]]:
    """circomlib CompConstant(ct): returns (out, check bits) where out = 1 iff the 254-bit input exceeds ct."""
    s, a, b, e = 0, 1, (1 << 128) - 1, 1
    for i in range(127):
        cl, cm = (ct >> (2 * i)) & 1, (ct >> (2 * i + 1)) & 1
        sl, sm = bits[2 * i], bits[2 * i + 1]
        if cm == 0 and cl == 0:
            part = -b * sm * sl + b * sm + b * sl
        elif cm == 0 and cl == 1:
            part = a * sm * sl - a * sl + b * sm - a * sm + a
        elif cm == 1 and cl == 0:
            part = b * sm * sl - a * sm + a
        else:
            part = -a * sm * sl + a
        s = (s + part) % P
        b, a, e = b - e, a + e, 2 * e
    nb = [(s >> k) & 1 for k in range(135)]
    return nb[127], nb if sum(x << k for k, x in enumerate(nb)) == s else []


def check(st: Statement, wt: Witness, n: int, m: int, kappa: int, attempt_signs: list[int] | None = None) -> Result:
    res = Result(True, [])
    cnt = res.counts

    def fail(msg: str) -> None:
        res.ok = False
        if len(res.failures) < 8:
            res.failures.append(msg)

    def pos(t: int, what: str) -> None:
        cnt[what] += constraints(t)

    if len(wt.entries) != n or len(wt.salts) != n or len(wt.cells) != 3 * m or len(wt.pair_keys) != kappa:
        raise ValueError("witness does not match the circuit size")
    signs = attempt_signs or [+1] * (kappa // 2) + [-1] * (kappa // 2)
    bshift = F.part_bits(m)
    if m != 192:
        raise ValueError("partition offsets r^(64 j) are wired for m = 192")

    # 1. leaves and 4-ary Merkle root
    level = []
    for (k, w), salt in zip(wt.entries, wt.salts):
        level.append(poseidon(salt, k, w % P)); pos(4, "leaves")
        # key bits (248 < 254, so the decomposition is unique), two IsZero gadgets, w linear in their outputs
        cnt["key bits"] += F.KEY_BITS
        cnt["weights"] += 4
        if k >> F.KEY_BITS:
            fail("key wider than 248 bits")
        if w % P != F.weight(k, st.bank) % P:
            fail("weight does not follow from the key and the bank")
    while len(level) > 1:
        level = [poseidon(*level[i:i + 4]) for i in range(0, len(level), 4)]
        cnt["merkle"] += constraints(5) * len(level)
    if level[0] != st.root:
        fail("leaves do not hash to the book root")

    # 3a. cell commitment and challenges
    if F.cell_commitment(wt.cells) != st.cells_commitment:
        fail("cells do not match the cell commitment")
    cnt["cell commitment"] += commit_perms(3 * m) * constraints(17)
    r = poseidon(st.cells_commitment, st.root, st.seed, DOM_R)
    al = poseidon(st.cells_commitment, st.root, st.seed, DOM_A)
    cnt["challenges"] += 2 * constraints(5)
    r2 = [r]                                    # r^(2^k), k = 0..7
    for _ in range(7):
        r2.append(r2[-1] * r2[-1] % P)
    al2 = al * al % P
    cnt["challenges"] += 7 + 1

    # 2. entries: hash, strict bits, positions, contribution
    rhs = 0
    for i, (k, w) in enumerate(wt.entries):
        h = poseidon(st.seed, k); pos(3, "entry hash")
        bits = wt.bits[i] if wt.bits else [(h >> j) & 1 for j in range(HBITS)]
        cnt["hash bits"] += HBITS
        if any(b not in (0, 1) for b in bits) or sum(b << j for j, b in enumerate(bits)) % P != h:
            fail(f"bad decomposition of entry {i}")
        out, nb = comp_constant(bits, P - 1)
        cnt["alias check"] += 127 + 135
        if out != 0 or not nb:
            fail(f"aliased decomposition of entry {i}")
        sp = 0
        for j in range(F.K):
            acc = (1, r2[6], r2[7])[j]            # r^(64 j) for m = 192: r^0, r^64, r^128
            for b in range(bshift):
                bit = bits[bshift * j + b]
                t = bit * (r2[b] - 1) % P
                acc = acc * (1 + t) % P
                cnt["positions"] += 1 if (j == 0 and b == 0) else 2
            sp += acc
        q = (1 + al * k + al2 * h) * sp % P
        rhs = (rhs + w * q) % P
        cnt["contribution"] += 4

    # 3b. masks
    mask = [0] * (3 * m)
    for kb, qb, sg in zip(wt.pair_keys, st.pair_commitments, signs):
        if F.pair_commitment(kb) != qb:
            fail("pair key does not match its commitment")
        pos(3, "masks")
        for c, v in enumerate(stream(kb, st.nonce, 3 * m)):
            mask[c] = (mask[c] + sg * v) % P
        cnt["masks"] += stream_perms(3 * m) * constraints(17)

    # 4. random evaluation of the unmasked cells
    lhs, rc = 0, 1
    for c in range(m):
        if c >= 2:
            cnt["evaluation"] += 1                # r^c = r^(c-1) * r
        u = (wt.cells[c] - mask[c]) % P
        v = (wt.cells[m + c] - mask[m + c]) % P
        z = (wt.cells[2 * m + c] - mask[2 * m + c]) % P
        lhs = (lhs + rc * (u + al * v + al2 * z)) % P
        cnt["evaluation"] += 3
        rc = rc * r % P
    if lhs != rhs:
        fail("published cells are not the masked sketch of the committed books")
    return res


def honest(bank: int, entries: list[tuple[int, int]], salt_list: list[int], seed: int, m: int, epoch: int, attempt: int,
           pair_keys: list[int], signs: list[int]) -> tuple[Statement, Witness]:
    s = F.masked(F.sketch(entries, seed, m), pair_keys, signs, epoch, m, attempt)
    cells = s.cells()
    st = Statement(bank, F.book_root(entries, salt_list), seed, F.cell_commitment(cells), F.mask_nonce(epoch, m, attempt),
                   [F.pair_commitment(k) for k in pair_keys])
    return st, Witness(entries, salt_list, cells, pair_keys)


def count(n: int, m: int, kappa: int) -> Counter:
    """Constraint counts by component, from the gadget structure alone (no witness)."""
    c: Counter = Counter()
    c["leaves"] = n * constraints(4)
    c["key bits"] = n * F.KEY_BITS
    c["weights"] = 4 * n
    nodes, x = 0, n
    while x > 1:
        x //= 4
        nodes += x
    c["merkle"] = nodes * constraints(5)
    c["cell commitment"] = commit_perms(3 * m) * constraints(17)
    c["challenges"] = 2 * constraints(5) + 8
    c["entry hash"] = n * constraints(3)
    c["hash bits"] = n * HBITS
    c["alias check"] = n * 262
    b = F.part_bits(m)
    c["positions"] = n * (F.K * b * 2 - 1)
    c["contribution"] = 4 * n
    c["masks"] = kappa * (constraints(3) + stream_perms(3 * m) * constraints(17))
    c["evaluation"] = 3 * m + (m - 2)
    return c
