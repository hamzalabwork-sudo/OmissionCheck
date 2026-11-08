"""Compile, set up, prove and verify the sketch-binding circuit with circom and snarkjs (Groth16 over BN254).

For each book size n the circuit is compiled, the witness is computed by circom's witness generator from inputs built
by the Python reference (omission.zk.circuit.honest), so a mismatch between the two fails witness generation, and a
proof is generated and verified. Wall-clock time and peak resident memory of each step are measured on the process
tree. The powers of tau are a local single-contributor setup: fine for timing, not a trusted setup.

    python -m omission.zk.prove --tools <dir with bin/circom(.exe) and node_modules/{snarkjs,circomlib}> --ptau <file>
        [--sizes 64 256 1024]

Writes paper2/zk/prove_results.json; the build directory (r1cs, wasm, zkey, proofs) is --work (default: <tools>/build).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import subprocess
import time
from pathlib import Path

import psutil

from finverity.config import ROOT

from . import circuit as Z
from . import fsketch as F
from .field import P

OUT = ROOT / "paper2" / "zk"
CIRCUIT = Path(__file__).parent / "circuits" / "sketch_binding.circom"
M, LAMBDA = 192, 16
NODE_MEM = "--max-old-space-size=12000"


def run(cmd: list[str], cwd: Path) -> dict:
    """Run cmd, sampling the resident memory of its process tree; returns seconds, peak MB and stdout."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc, peak = psutil.Process(p.pid), 0
    out: list[str] = []
    while p.poll() is None:
        try:
            rss = sum(x.memory_info().rss for x in [proc, *proc.children(recursive=True)])
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.05)
    out.append(p.stdout.read())
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{''.join(out)[-3000:]}")
    return {"s": time.perf_counter() - t0, "peak_mb": peak / 2**20, "out": "".join(out)}


def inputs(n: int, rng: random.Random) -> tuple[dict, Z.Statement]:
    """Honest statement and witness for a bank with n - 1 transfers, padded to n (as in omission.zk.run)."""
    bank = 3
    entries = []
    for _ in range(n - 1):
        other = rng.choice([x for x in range(1000) if x != bank])
        pair = (bank, other) if rng.random() < .5 else (other, bank)
        key = F.make_key(*pair, rng.getrandbits(F.TAG_BITS))
        entries.append((key, F.weight(key, bank)))
    entries = F.pad(entries, n)
    salts = F.salts(b"bench", n)
    keys = [rng.getrandbits(250) for _ in range(LAMBDA)]
    signs = [+1] * (LAMBDA // 2) + [-1] * (LAMBDA // 2)
    st, wt = Z.honest(bank, entries, salts, rng.getrandbits(250), M, 1, 0, keys, signs)
    s = lambda x: str(x % P)
    return {"bank": s(st.bank), "root": s(st.root), "seed": s(st.seed), "cellsCommitment": s(st.cells_commitment),
            "nonce": s(st.nonce), "pairCommitment": [s(x) for x in st.pair_commitments],
            "key": [s(k) for k, _ in wt.entries], "w": [s(w) for _, w in wt.entries], "salt": [s(x) for x in wt.salts],
            "cells": [s(x) for x in wt.cells], "pairKey": [s(x) for x in wt.pair_keys]}, st


def prove_size(n: int, tools: Path, ptau: Path, work: Path, rng: random.Random) -> dict:
    d = work / f"n{n}"
    d.mkdir(parents=True, exist_ok=True)
    name = f"sb{n}"
    src = re.sub(r"SketchBinding\(\d+, \d+\);", f"SketchBinding({n}, {LAMBDA});", CIRCUIT.read_text())
    (d / f"{name}.circom").write_text(src)
    circom = str(next((tools / "bin").glob("circom*")))
    snarkjs = ["node", NODE_MEM, str(tools / "node_modules" / "snarkjs" / "build" / "cli.cjs")]
    res: dict = {"n": n, "m": M, "lambda": LAMBDA}

    if not (d / f"{name}.r1cs").exists():
        # --O2 substitutes every linear constraint, so the R1CS holds the non-linear constraints only
        res["compile"] = run([circom, f"{name}.circom", "--O2", "--r1cs", "--wasm", "-l", str(tools / "node_modules"),
                              "-o", "."], d)
    info = run([*snarkjs, "r1cs", "info", f"{name}.r1cs"], d)["out"]
    num = lambda label: int(re.search(rf"# of {label}: (\d+)", info).group(1))
    res.update(constraints=num("Constraints"), wires=num("Wires"), public_inputs=num("Public Inputs"),
               private_inputs=num("Private Inputs"), model_constraints=sum(Z.count(n, M, LAMBDA).values()))

    inp, st = inputs(n, rng)
    (d / "input.json").write_text(json.dumps(inp))
    res["witness"] = run(["node", NODE_MEM, f"{name}_js/generate_witness.js", f"{name}_js/{name}.wasm", "input.json",
                          "witness.wtns"], d)

    if not (d / f"{name}.zkey").exists():
        res["setup"] = run([*snarkjs, "groth16", "setup", f"{name}.r1cs", str(ptau), f"{name}_0.zkey"], d)
        res["contribute"] = run([*snarkjs, "zkey", "contribute", f"{name}_0.zkey", f"{name}.zkey", "--name=local",
                                 f"-e={secrets.token_hex(32)}"], d)
        os.remove(d / f"{name}_0.zkey")
        run([*snarkjs, "zkey", "export", "verificationkey", f"{name}.zkey", "vkey.json"], d)
    res["prove"] = run([*snarkjs, "groth16", "prove", f"{name}.zkey", "witness.wtns", "proof.json", "public.json"], d)
    res["verify"] = run([*snarkjs, "groth16", "verify", "vkey.json", "public.json", "proof.json"], d)
    if "OK" not in res["verify"]["out"]:
        raise RuntimeError(f"proof for n = {n} did not verify:\n{res['verify']['out']}")
    pub = json.loads((d / "public.json").read_text())
    if [int(x) for x in pub] != [x % P for x in st.public_inputs()]:
        raise RuntimeError("public signals differ from the statement")
    res["zkey_mb"] = (d / f"{name}.zkey").stat().st_size / 2**20
    res["proof_json_bytes"] = (d / "proof.json").stat().st_size
    for step in ("compile", "witness", "setup", "contribute", "prove", "verify"):
        if step in res:
            res[step].pop("out", None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--ptau", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024])
    a = ap.parse_args()
    work = a.work or a.tools / "build"
    path = OUT / "prove_results.json"
    done = {r["n"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    rng = random.Random(2026)
    for n in a.sizes:
        r = prove_size(n, a.tools, a.ptau.resolve(), work, rng)
        done[n] = r
        path.write_text(json.dumps([done[k] for k in sorted(done)], indent=1))
        print(json.dumps(r))


if __name__ == "__main__":
    main()
