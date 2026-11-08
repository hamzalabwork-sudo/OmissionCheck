"""X7: per-bank computation with the deployed primitives (X25519, HKDF, BLAKE2b, ChaCha20, SHA-256, Keccak) on one
core.

    python -m omission.bench
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Protocol.DH import key_agreement
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
from Crypto.PublicKey import ECC

from finverity.config import RESULTS

from eth_utils import keccak

from . import protocol as P
from .iblt import CHECK, WORDS, Sketch

OUT = RESULTS / "paper2"


def _t(f, reps):
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def main():
    rng = np.random.default_rng(11)
    res = {}
    a, b = ECC.generate(curve="Curve25519"), ECC.generate(curve="Curve25519")
    kdf = lambda x: HKDF(x, 32, b"", SHA256)
    res["x25519_pair_key_s"] = _t(lambda: key_agreement(static_priv=a, static_pub=b.public_key(), kdf=kdf), 200)
    key = bytes(32)
    canon = b"2022/09/01 00:20|3208|8000F4580|1|8000F5340|0.01|US Dollar|0.01|US Dollar|Cheque|0"
    res["tag_blake2b_s"] = _t(lambda: hashlib.blake2b(canon, key=key, digest_size=24).digest(), 200_000)
    res["subkeys_hkdf_s"] = _t(lambda: HKDF(key, 32, b"", SHA256, num_keys=3,
                                            context=b"omission/mac|mask|pair-id"), 20_000)
    # cell index and check of one key under the epoch seed: keyed BLAKE2b over the 32-byte key (the deployed
    # stand-in for iblt.cells / iblt.checksum)
    k32 = bytes(range(32))
    res["cell_hash_blake2b_s"] = _t(lambda: hashlib.blake2b(k32, key=key, digest_size=40).digest(), 200_000)
    for n in (1_000, 10_000):
        leaves = [bytes(16) + i.to_bytes(32, "big") + bytes([1]) for i in range(n)]
        res[f"book_root_n{n}_s"] = _t(lambda: P.merkle_root(leaves), 5)
    sk = bytes(192 * 64)
    res["sketch_commit_keccak_m192_s"] = _t(lambda: keccak(sk), 2000)
    for m in (96, 192, 384, 768):
        nbytes = m * 8 * (1 + WORDS + CHECK)
        res[f"mask_chacha20_m{m}_s"] = _t(
            lambda: np.frombuffer(ChaCha20.new(key=key, nonce=bytes(12)).encrypt(bytes(nbytes)), np.uint64), 2000)
    for n in (1_000, 10_000, 100_000):
        keys = rng.integers(0, 2**64, (n, WORDS), dtype=np.uint64)
        signs = np.where(rng.random(n) < 0.5, 1, -1)
        res[f"sketch_build_n{n}_m192_s"] = _t(lambda: Sketch.empty(192).insert(keys, signs), 20)
    for d in (10, 50, 150):
        m = 3 * int(np.ceil(1.5 * d / 3))
        keys = rng.integers(0, 2**64, (d, WORDS), dtype=np.uint64)
        s = Sketch.empty(m).insert(keys, np.ones(d, int))
        res[f"decode_d{d}_m{m}_s"] = _t(lambda: s.decode(), 50)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "x7_bench.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:32s} {v * 1e6:12.1f} us")


if __name__ == "__main__":
    main()
