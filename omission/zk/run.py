"""Constraint counts, relation checks and macros of the sketch-binding circuit: python -m omission.zk.run

Writes paper2/zk/constraints.csv, paper2/zk/zk_results.json and paper2/zk/zk_macros.tex. The verifier gas comes from
paper2/zk/verifier_gas.json (python -m omission.zk.gas, under the ledger lock).
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pandas as pd

from finverity.config import ROOT
from omission import data as D

from . import circuit as Z
from . import fsketch as F
from .field import P

OUT = ROOT / "paper2" / "zk"
M, KAPPA = 192, 16
SIZES = [64, 256, 1024, 4096, 16384]
CHECKED = [64, 256, 1024]


def per_bank_entries(n_members: int = 1000) -> pd.Series:
    """Book entries of each member bank in each epoch (payer side plus payee side of covered transfers)."""
    t = D.transfers()
    t = t[D.covered(t, D.members(n_members))]
    e = pd.concat([t[["epoch", "payer"]].rename(columns={"payer": "bank"}),
                   t[["epoch", "payee"]].rename(columns={"payee": "bank"})])
    return e.groupby(["epoch", "bank"]).size()


def check_size(n: int, rng: random.Random) -> dict:
    bank = 3
    entries = []
    for _ in range(n - 1):
        other = rng.choice([x for x in range(1000) if x != bank])
        pair = (bank, other) if rng.random() < .5 else (other, bank)
        key = F.make_key(*pair, rng.getrandbits(F.TAG_BITS))
        entries.append((key, F.weight(key, bank)))
    entries = F.pad(entries, n)
    salts = F.salts(b"bench", n)
    keys = [rng.getrandbits(250) for _ in range(KAPPA)]
    signs = [+1] * (KAPPA // 2) + [-1] * (KAPPA // 2)
    seed = rng.getrandbits(250)
    st, wt = Z.honest(bank, entries, salts, seed, M, 1, 0, keys, signs)
    t0 = time.perf_counter()
    res = Z.check(st, wt, n, M, KAPPA)
    dt = time.perf_counter() - t0
    if not res.ok or res.counts != Z.count(n, M, KAPPA):
        raise AssertionError((n, res.failures))
    return {"n": n, "checked": True, "check_s": dt, "public_inputs": len(st.public_inputs())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    rows = []
    for n in SIZES:
        for kappa in (0, KAPPA):
            c = Z.count(n, M, kappa)
            rows.append({"n": n, "kappa": kappa, "total": sum(c.values()), **c})
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(OUT / "constraints.csv", index=False)
    checks = [check_size(n, rng) for n in CHECKED]
    print(df[["n", "kappa", "total"]].to_string(index=False))
    print(checks)

    big = Z.count(16384, M, 0)
    small = Z.count(4096, M, 0)
    per_entry = (sum(big.values()) - sum(small.values())) / (16384 - 4096)
    entry_parts = {k: (big[k] - small[k]) / (16384 - 4096) for k in big}
    c64 = Z.count(64, M, 0)
    fixed0 = c64["cell commitment"] + c64["challenges"] + c64["evaluation"]   # independent of n and kappa
    mask_slot = (sum(Z.count(64, M, KAPPA).values()) - sum(Z.count(64, M, 0).values())) / KAPPA

    g = per_bank_entries()
    share = {s: float((g <= s).mean()) for s in SIZES}
    top_max = int(g.max())
    top_pad = 4 ** math.ceil(math.log(top_max, 4))
    gas = json.loads((OUT / "verifier_gas.json").read_text())
    gas = {r["n_public"]: r for r in gas}
    res = {
        "m": M, "kappa": KAPPA, "per_entry": per_entry, "entry_parts": entry_parts, "fixed_no_mask": fixed0,
        "mask_per_slot": mask_slot, "totals": {f"{r['n']}_{r['kappa']}": r["total"] for r in rows},
        "checked": checks, "entries_per_bank_epoch": {"median": float(g.median()), "p99": float(g.quantile(.99)),
                                                      "p999": float(g.quantile(.999)), "max": top_max,
                                                      "bank_epochs": int(len(g))},
        "share_within": share, "top_pad": top_pad, "soundness_log2": math.log2((M + 1) / P), "gas": gas,
    }
    (OUT / "zk_results.json").write_text(json.dumps(res, indent=1, default=float))

    tot = {(r["n"], r["kappa"]): r["total"] for r in rows}
    hashbits = entry_parts["hash bits"] + entry_parts["alias check"]
    mac = {
        "ZkM": M, "ZkLambda": KAPPA,
        "ZkConsPerEntry": f"{per_entry:,.0f}", "ZkConsFixed": f"{fixed0:,.0f}", "ZkConsMaskSlot": f"{mask_slot:,.0f}",
        "ZkConsMasks": f"{mask_slot * KAPPA:,.0f}", "ZkBitsShare": f"{100 * hashbits / per_entry:.0f}",
        "ZkConsSmall": f"{tot[(64, KAPPA)]:,}", "ZkConsMid": f"{tot[(256, KAPPA)]:,}",
        "ZkConsLarge": f"{tot[(1024, KAPPA)]:,}", "ZkConsHuge": f"{tot[(4096, KAPPA)]:,}",
        "ZkConsTop": f"{tot[(16384, KAPPA)]:,}",
        "ZkShareSmall": f"{100 * share[64]:.1f}", "ZkShareMid": f"{100 * share[256]:.1f}",
        "ZkShareLarge": f"{100 * share[1024]:.1f}", "ZkAboveHuge": f"{int((g > 4096).sum())}", "ZkBankEpochs": f"{len(g):,}",
        "ZkEntriesMedian": f"{g.median():.0f}", "ZkEntriesPninetynine": f"{g.quantile(.99):.0f}",
        "ZkEntriesMax": f"{top_max:,}", "ZkTopPad": f"{top_pad:,}",
        "ZkSoundBits": f"{-math.log2((M + 1) / P):.0f}",
        "ZkPublicInputs": 5 + KAPPA,
        "ZkVerifyGas": f"{gas[5 + KAPPA]['verify_exec_gas']:,}", "ZkVerifyTxGas": f"{gas[5 + KAPPA]['check_tx_gas']:,}",
        "ZkVerifyGasNoMask": f"{gas[5]['verify_exec_gas']:,}", "ZkPostGas": f"{gas[5 + KAPPA]['post_tx_gas']:,}",
        "ZkDeployGas": f"{gas[5 + KAPPA]['deploy_gas']:,}", "ZkProofBytes": 256,
        "ZkCheckedSizes": ", ".join(str(n) for n in CHECKED),
    }
    # measured with circom and snarkjs (python -m omission.zk.prove) and on the Clique network with a real proof
    # (python -m omission.zk.gas --real ...)
    proved = OUT / "prove_results.json"
    if proved.exists():
        pr = {r["n"]: r for r in json.loads(proved.read_text())}
        for n, tag in ((64, "Small"), (256, "Mid"), (1024, "Large")):
            if n in pr:
                r = pr[n]
                mac |= {f"ZkCompiled{tag}": f"{r['constraints']:,}", f"ZkProve{tag}": f"{r['prove']['s']:.0f}",
                        f"ZkProveMem{tag}": f"{r['prove']['peak_mb'] / 1024:.1f}",
                        f"ZkVerifySec{tag}": f"{r['verify']['s']:.1f}",
                        f"ZkSetupMin{tag}": f"{(r['setup']['s'] + r['contribute']['s']) / 60:.0f}"}
        dev = max(abs(r["constraints"] - r["model_constraints"]) / r["model_constraints"] for r in pr.values())
        mac["ZkModelDevPct"] = f"{100 * dev:.1f}"
    real = OUT / "verifier_gas_real.json"
    if real.exists():
        rg = json.loads(real.read_text())[0]
        mac |= {"ZkRealVerifyTxGas": f"{rg['valid_check_tx_gas']:,}", "ZkRealDeployGas": f"{rg['deploy_gas']:,}"}
    lines = ["% generated by python -m omission.zk.run; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (OUT / "zk_macros.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
