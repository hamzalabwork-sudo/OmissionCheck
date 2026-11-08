"""Arithmetic over the BN254 scalar field and a Poseidon permutation for the sketch-binding circuit.

The permutation is circomlib's Poseidon (x^5 S-box, 8 full rounds, the partial-round counts of circomlib for widths
2..17, capacity in state[0]) with circomlib's round constants and MDS matrices, so its outputs equal those of the
circuit's Poseidon templates: 3 multiplicative constraints per S-box. The constants are read from circomlibjs
(npm install circomlibjs@0.1.7), at $POSEIDON_CONSTANTS or under omission/zk/node_modules.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617   # BN254 scalar field
RF = 8
RP = {2: 56, 3: 57, 4: 56, 5: 60, 6: 60, 7: 63, 8: 64, 9: 63, 10: 60, 11: 66, 12: 60, 13: 65, 14: 70, 15: 60,
      16: 64, 17: 68}
RATE = 16            # widest permutation: 16 rate elements and one capacity element


def inv(x: int) -> int:
    return pow(x % P, P - 2, P)


CONSTANTS = Path(os.environ.get("POSEIDON_CONSTANTS", Path(__file__).parent / "node_modules" / "circomlibjs" / "src"
                                / "poseidon_constants.json"))


@lru_cache(maxsize=None)
def _constants() -> dict:
    if not CONSTANTS.exists():
        raise FileNotFoundError(f"circomlib Poseidon constants not found at {CONSTANTS}; npm install circomlibjs@0.1.7 "
                                "in omission/zk or set POSEIDON_CONSTANTS")
    return json.loads(CONSTANTS.read_text())


@lru_cache(maxsize=None)
def _params(t: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    c = _constants()
    ark = tuple(int(x, 16) for x in c["C"][t - 2])
    mds = tuple(tuple(int(x, 16) for x in row) for row in c["M"][t - 2])
    assert len(ark) == (RF + RP[t]) * t and len(mds) == t
    return ark, mds


def permute(state: list[int]) -> list[int]:
    t = len(state)
    ark, mds = _params(t)
    s = [x % P for x in state]
    half = RF // 2
    for r in range(RF + RP[t]):
        s = [(x + ark[r * t + i]) % P for i, x in enumerate(s)]
        if r < half or r >= half + RP[t]:
            s = [pow(x, 5, P) for x in s]
        else:
            s[0] = pow(s[0], 5, P)
        s = [sum(mds[i][j] * s[j] for j in range(t)) % P for i in range(t)]
    return s


def constraints(t: int) -> int:
    """Multiplicative constraints of one permutation of width t: x^2, x^4, x^5 per S-box."""
    return 3 * (RF * t + RP[t])


def poseidon(*xs: int) -> int:
    """Hash of 1..16 field elements (circomlib layout: capacity first, output state[0])."""
    return permute([0, *xs])[0]


def commit(elems: list[int], domain: int) -> int:
    """Commitment to a fixed-length vector: a chain of width-17 permutations absorbing 15 elements each."""
    h = domain % P
    for i in range(0, len(elems), RATE - 1):
        chunk = list(elems[i:i + RATE - 1])
        h = poseidon(h, *(chunk + [0] * (RATE - 1 - len(chunk))))
    return h


def commit_perms(length: int) -> int:
    return -(-length // (RATE - 1))


def stream(key: int, nonce: int, length: int) -> list[int]:
    """Pseudorandom field elements: a sponge keyed in its rate, squeezed 16 rate elements per permutation; the
    capacity element is never output, so outputs do not reveal the key by inverting the permutation."""
    s = permute([0, key % P, nonce % P] + [0] * (RATE - 2))
    out: list[int] = []
    while True:
        out.extend(s[1:])
        if len(out) >= length:
            return out[:length]
        s = permute(s)


def stream_perms(length: int) -> int:
    return -(-length // RATE)
