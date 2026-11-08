"""Paper 2 tables, macros and figures from results/paper2 (never retype a number by hand).

    python -m omission.report      -> paper2/generated/*.tex, paper2/figures/*.pdf
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finverity.config import ETH_USD as ETH_USD_RATE, MAINNET_GAS_GWEI as MAINNET_GWEI, ROOT

from .iblt import threshold
from .run import DEDUCTED_WIRES as DEDUCTED, RETURNED

RES = ROOT / "results" / "paper2"
GEN = ROOT / "paper2" / "generated"
FIG = ROOT / "paper2" / "figures"
MACROS: dict[str, str] = {}


def M(name: str, value) -> None:
    assert not any(ch.isdigit() for ch in name), name
    MACROS[name] = str(value)


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}\\%"


def num(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def sci(x) -> str:
    e = int(np.floor(np.log10(x)))
    return f"${x / 10 ** e:.1f}\\times 10^{{{e}}}$"


def dec1(x) -> str:
    return f"{x:,.1f}".replace(",", "{,}")


def kb(x) -> str:
    return f"{x / 1024:.1f}"


# tables too wide for one column of the two-column journal layout span both columns (table*)
WIDE = {"t_detection", "t_chain", "t_aml", "t_spill", "t_benign", "t_scale", "t_amlstreams", "t_amlcase",
        "t_amlpna", "t_amlthr"}
CELL = 64          # bytes per sketch cell: two 256-bit words, as stored and sent on chain


def _tex(name: str, body: str) -> None:
    # never let a table run past the text block: shrink only tables wider than \linewidth
    body = body.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}")
    body = body.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
    if name in WIDE:
        body = body.replace(r"\begin{table}", r"\begin{table*}").replace(r"\end{table}", r"\end{table*}")
    (GEN / f"{name}.tex").write_text(body, encoding="utf-8")


# ----------------------------------------------------------------------------------------------------- X1
def t_coverage():
    df = pd.read_csv(RES / "x1_coverage.csv")
    rows = []
    for r in df.itertuples():
        label = "all" if r.members == df.members.max() else num(r.members)
        rows.append(f"{label} & {pct(r.transfers)} & {pct(r.laundering)} & {pct(r.laundering_value)} & "
                    f"{pct(r.covered_of_member)} \\\\")
    _tex("t_coverage", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Two-witness coverage of IBM AML HI-Small when the $N$ busiest banks form the consortium. A transfer is "
        r"covered when its payer bank and its payee bank are distinct members. Value is in US dollars at the exchange "
        r"rates implied by the dataset's own cross-currency transfers.}\label{tab:coverage}",
        r"\begin{tabular}{rrrrr}\toprule",
        r"Members $N$ & Transfers & Laundering legs & Laundering value & Member transfers covered \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    a = df.iloc[-1]
    M("TransfersTotal", num(5078345))
    M("InterbankAll", pct(a.transfers))
    M("LaunderingInterbankAll", pct(a.laundering))
    M("IntraBank", pct(a.intra_bank))
    M("LaunderingIntra", pct(a.laundering_intra))
    for n, w in ((10, "Ten"), (20, "Twenty"), (30, "Thirty"), (50, "Fifty"), (100, "Hundred"), (500, "FiveHundred"),
                 (1000, "Thousand"), (2000, "TwoThousand")):
        r = df[df.members == n].iloc[0]
        M(f"Cov{w}", pct(r.transfers))
        M(f"CovL{w}", pct(r.laundering))
        M(f"CovV{w}", pct(r.laundering_value))
    M("BanksAll", num(df.members.max()))


# ----------------------------------------------------------------------------------------------------- X2
SC_LABEL = [("honest", "No tampering"), ("random p=0.0001", r"Random omission, $p=10^{-4}$"),
            ("random p=0.001", r"Random omission, $p=10^{-3}$"), ("random p=0.01", r"Random omission, $p=10^{-2}$"),
            ("insider deletes", "Insider deletes laundering legs"), ("substitution", "Substitution (same amount)"),
            ("alteration", r"Alteration ($-1\%$ amount)"), ("coalition", "Two colluding banks")]


def _label(s: str) -> str:
    for k, v in SC_LABEL:
        if s.startswith(k):
            return v
    return s


def t_detection():
    df = pd.read_csv(RES / "x2_scenarios.csv")
    cfg = json.loads((RES / "x2_config.json").read_text())
    ins = df[df.scenario.str.startswith("insider")]
    rest = df[~df.scenario.str.startswith("insider")]
    agg = ins.agg({"attacked_transfers": "sum", "attacked_epochs": "sum", "ours_epoch_detection": "mean",
                   "ours_item_recall": "mean", "ours_complete_share": "min", "totals_tx_detection": "mean",
                   "ours_false_epochs": "sum", "totals_false_cells": "sum"})
    agg["scenario"] = f"insider deletes laundering legs ({len(ins)} banks)"
    df2 = pd.concat([rest.iloc[:4], pd.DataFrame([agg]), rest.iloc[4:]], ignore_index=True)

    def f(x):
        return "--" if pd.isna(x) else pct(x)
    rows = []
    for r in df2.itertuples():
        rows.append(f"{_label(r.scenario)} & {num(r.attacked_transfers)} & {f(r.totals_tx_detection)} & "
                    f"{f(r.ours_epoch_detection)} & {f(r.ours_complete_share)} & {f(r.ours_item_recall)} & "
                    f"{int(r.ours_false_epochs)} \\\\")
    _tex("t_detection", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        rf"\caption{{Detection on IBM AML HI-Small (publish or accumulate mode; the sum is the same in every mode): "
        rf"consortium of the {num(cfg['members'])} busiest banks, {num(cfg['covered_transfers'])} covered transfers in "
        rf"hourly epochs, sketches of $m={cfg['m']}$ cells, a fresh seed per epoch. Per-pair totals compare the count "
        r"and amount each side reports per ordered bank pair and epoch (detection of tampered transfers). Flagged: "
        r"tampered epochs whose sum is non-zero. Decoded: tampered epochs whose sum peels completely. Items: "
        r"non-cancelling keys returned with the correct sign. False: untampered epochs flagged. Integrity-only "
        r"anchoring detects none of these attacks by construction and is not shown; -- marks an undefined rate.}"
        r"\label{tab:detection}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r" & Tampered & Per-pair & \multicolumn{4}{c}{This protocol} \\\cmidrule(lr){4-7}",
        r"Scenario & transfers & totals & Flagged & Decoded & Items & False \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    h = df[df.scenario == "honest"].iloc[0]
    M("Epochs", num(h.epochs))
    M("CoveredTransfers", num(cfg["covered_transfers"]))
    M("Members", num(cfg["members"]))
    M("Cells", cfg["m"])
    M("Kappa", cfg["kappa"])
    M("HonestFalse", int(h.ours_false_epochs))
    n_mem, m = cfg["members"], cfg["m"]
    M("CellBytes", CELL)
    M("BytesBank", kb(m * CELL))
    M("BytesOurs", f"{h.ours_bytes_epoch / 2**20:.1f}")
    # commit mode: a book root and a sketch hash per bank, and the posted sum (calldata)
    M("BytesOursChain", kb(n_mem * 64 + m * CELL))
    M("BreakEven", num(m * CELL / 32))
    M("EntriesPerBank", f"{2 * cfg['covered_transfers'] / (n_mem * h.epochs):.0f}")
    M("ChainRatioTags", f"{h.tags_bytes_epoch / (n_mem * 64 + m * CELL):.0f}")
    M("DecodeCapacity", f"{m / threshold():.0f}")
    # a bank is unmasked when all lambda (= kappa in the config) of its mask neighbours are corrupt: about f^lambda per
    # bank and epoch for a corrupt fraction f; per year at N = 1,000 hourly epochs
    for f, w in ((0.1, "Tenth"), (0.25, "Quarter"), (0.5, "Half")):
        p = f ** cfg["kappa"]
        M(f"Iso{w}", sci(p))
        M(f"Iso{w}Year", sci(p * n_mem * 8760))
    M("BytesTotals", kb(h.totals_bytes_epoch))
    M("BytesTags", kb(h.tags_bytes_epoch))
    M("PairsRevealed", num(h.pairs_revealed_epoch))
    M("BuildMedianMs", f"{1e3 * h.build_s_median:.0f}")
    sub = df[df.scenario.str.startswith("substitution")].iloc[0]
    M("SubTotals", pct(sub.totals_tx_detection))
    M("SubOurs", pct(sub.ours_item_recall))
    M("InsiderBanks", len(ins))
    M("InsiderLegs", num(ins.attacked_transfers.sum()))
    M("InsiderRecall", pct((ins.ours_item_recall * ins.residual_items).sum() / ins.residual_items.sum()))
    rp = df[df.scenario == "random p=0.01"].iloc[0]
    M("RandHighComplete", pct(rp.ours_complete_share))
    M("RandHighRecall", pct(rp.ours_item_recall))
    M("RandHighDetect", pct(rp.ours_epoch_detection))
    M("RandHighResidual", num(rp.residual_items))
    co = df[df.scenario.str.startswith("coalition")].iloc[0]
    M("CoalitionLegs", num(co.attacked_transfers))
    M("CoalitionDetect", "0\\%" if co.residual_items == 0 else pct(co.ours_epoch_detection))


# ----------------------------------------------------------------------------------------------------- X3
def _wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def f_sizing():
    df = pd.read_csv(RES / "x3_sizing.csv")
    g = df.groupby(["d", "ratio"])["complete"].agg(["sum", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (d, h), mk in zip(g.groupby("d"), "osD^"):
        p = h["sum"] / h["count"]
        lo, hi = _wilson(h["sum"].to_numpy(), h["count"].to_numpy())
        ax.errorbar(h.ratio, 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], marker=mk, ms=3.5, lw=1.2,
                    capsize=2, elinewidth=0.8, label=f"d = {d}")
    ax.axvline(threshold(), color="0.5", ls=":", lw=1)
    ax.annotate("c3 = 1.222", xy=(threshold(), 50), xytext=(1.8, 25), fontsize=8, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    ax.set_xlabel("Sketch cells per discrepancy, m/d")
    ax.set_ylabel("Epochs fully decoded (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F2_sizing.pdf")
    fig.savefig(FIG / "F2_sizing.png", dpi=300)
    plt.close(fig)
    M("SizingDetect", pct(df.flagged.mean()))
    r15 = df[(df.ratio == 1.5) & (df.d >= 100)].complete.mean()
    M("SizingCompleteOneFiveLarge", pct(r15))
    r3 = df[(df.ratio == 3.0)].complete.mean()
    M("SizingCompleteThree", pct(r3))
    r2 = df[(df.ratio == 2.0)].complete.mean()
    M("SizingCompleteTwo", pct(r2))
    small = df[(df.ratio == 1.5) & (df.d == 10)].complete.mean()
    M("SizingCompleteOneFiveTen", pct(small))
    M("SizingTrials", int(df.trial.max() + 1))


# ----------------------------------------------------------------------------------------------------- X4
def f_collusion():
    df = pd.read_csv(RES / "x4_collusion.csv")
    ex = pd.read_csv(RES / "x4_exact.csv")
    total = ex.optimum.iloc[-1] / ex.share_covered.iloc[-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(ex.c, 100 * ex.share_covered, marker="o", ms=3.5, lw=1.2, label="optimum (MILP)")
    ax.plot(df.c, 100 * df.legs_share_covered, marker="", ls="--", lw=1.2, label="greedy")
    ax.set_xlabel("Colluding member banks c")
    ax.set_ylabel("Laundering legs hidden (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F3_collusion.pdf")
    fig.savefig(FIG / "F3_collusion.png", dpi=300)
    plt.close(fig)
    for c, w in ((2, "Two"), (10, "Ten"), (30, "Thirty")):
        r = df[df.c == c].iloc[0]
        M(f"Collude{w}", pct(r.legs_share_covered))
        M(f"ColludeAll{w}", pct(r.legs_share_all))
        e = ex[ex.c == c].iloc[0]
        M(f"ColludeOpt{w}", pct(e.share_covered))
        M(f"ColludeOptLegs{w}", num(e.optimum))
        M(f"ColludeGreedyLegs{w}", num(r.legs))
    M("ColludeOptAllOptimal", "all" if ex.optimal.all() else "not all")
    M("ColludeCoveredLegs", num(total))
    M("ColludeOptMaxS", f"{ex.seconds.max():.0f}")


# ----------------------------------------------------------------------------------------------------- X9
def t_attacks():
    """Adaptive attacks against a naive design and the protocol."""
    d = json.loads((RES / "x9_decoy.json").read_text())
    M("DecoyCandidates", num(d["candidates"]))
    M("DecoyHashes", f"{d['hash_evaluations'] / 1e6:.0f}")
    M("DecoyGrindS", f"{d['grind_s']:.0f}")
    M("DecoyCount", num(d["decoys"]))
    M("DecoyPayer", num(d["decoys_payer"]))
    M("DecoyPayee", num(d["decoys_payee"]))
    M("DecoyEpochRows", num(d["rows"]))
    M("DecoyOldHidden", "yes" if (not d["old_flagged"]) and d["old_aggregate_zero"] else "no")
    M("DecoyOldResidual", num(d["old_residual"]))
    M("DecoyTrials", num(d["new_trials"]))
    M("DecoyNewFlagged", pct(d["new_flagged"]))
    M("DecoyNewCompleteFirst", pct(d["new_complete_first"]))
    M("DecoyNewTargetFirst", pct(d["new_target_decoded_first"]))
    M("DecoyNewCompleteFinal", pct(d["new_complete_final"]))
    M("DecoyNewMFinal", num(d["new_m_final_max"]))
    M("DecoyNewAttempts", d["new_attempts_max"])
    lm = pd.read_csv(RES / "x9_lastmover.csv")
    g = lm.groupby("version")
    M("LastEpochs", lm.epoch.nunique())
    for v, key in (("first", "Old"), ("revised", "New")):
        h = lm[lm.version == v]
        M(f"Last{key}Flagged", pct(h.flagged.mean(), 0))
        M(f"Last{key}Found", pct((h.found == h.residual).mean(), 0))
    dr = pd.read_csv(RES / "x9_dropout.csv")
    M("DropEpochs", len(dr))
    M("DropExact", pct(dr.exact.mean(), 0))
    M("DropMaxAbsent", int(dr.absent.max()))
    M("DropResidualMedian", num(dr.residual.median()))
    M("DropResidualMax", num(dr.residual.max()))
    M("DropComplete", pct(dr.complete.mean(), 0))
    M("DropFound", pct(dr.found.sum() / dr.residual.sum()))
    M("DropMasksMedian", num(dr.revealed_masks.median()))
    M("DropMinNeighbours", int(dr.min_present_neighbours.min()))
    rows = [
        f"Decoy cancellation (one insider) & hidden: sum $=0$ & flagged in {pct(d['new_flagged'])}, decoded after "
        f"$\\le${d['new_attempts_max']} re-publication{'s' if d['new_attempts_max'] != 1 else ''} \\\\",
        f"Last-mover cancellation & flagged in {MACROS['LastOldFlagged']} & flagged in {MACROS['LastNewFlagged']}, "
        f"exact in {MACROS['LastNewFound']} \\\\",
        f"Dropout of 1--{MACROS['DropMaxAbsent']} banks & epoch undecodable & sum recovered exactly in "
        f"{MACROS['DropExact']} \\\\"]
    _tex("t_attacks", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Adaptive attacks and dropouts on IBM AML HI-Small ($N=1{,}000$, $m=192$, $\lambda=16$) under the "
        r"naive design (fixed public hash functions, 64-bit check, publication in any order) and the protocol "
        r"(books committed before the epoch seed, 128-bit check, commit-then-reveal, mask recovery).}\label{tab:attacks}",
        r"\begin{tabular}{L{0.30\linewidth}L{0.22\linewidth}L{0.36\linewidth}}\toprule",
        r"Attack & Naive design & Protocol \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


# ----------------------------------------------------------------------------------------------------- X5
def t_benign():
    df = pd.read_csv(RES / "x5_benign.csv")
    base = df[~df.attack]
    mix = df[df.attack].set_index(["lag_min", "close_min"])
    rows = []
    for r in base.itertuples():
        if r.canon == "canonical" and r.lag_min == 1.0:
            continue
        name = "Naive" if r.canon == "naive" else "Canonical"
        close = "--" if r.canon == "naive" else f"{r.close_min:g}"
        dec = "--" if pd.isna(getattr(r, "complete_share", np.nan)) else pct(r.complete_share, 0)
        key = (r.lag_min, r.close_min)
        prec = pct(mix.loc[key].alert_precision) if r.canon == "canonical" and key in mix.index else "--"
        rows.append(f"{name} & {r.lag_min:g} & {close} & {pct(r.false_rate, 3)} & {dec1(r.residual_mean)} & "
                    f"{num(r.residual_p99)} & {pct(r.epochs_flagged, 0)} & {dec} & {prec} \\\\")
    _tex("t_benign", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Benign discrepancies over all covered transfers of the 1,000-bank consortium. "
        r"Naive canon: each witness tags the amount it booked and files the transfer under its own booking hour; "
        r"returns are reversed by the payer. Canonical: $\mathrm{canon}(x)$ holds the end-to-end reference, the "
        r"settlement amount and currency and the value date, and an epoch is closed $\delta$ minutes after its end; "
        r"a payee booking later than that leaves a one-sided record. Payee booking lag is exponential with mean "
        rf"$\mu$; deductions reach the credited amount of {pct(DEDUCTED)} of wires; {pct(RETURNED, 1)} of "
        rf"transfers are returned (assumed rates). Decoded: epochs whose sum peels completely at $m=192$. "
        r"Precision: share of decoded discrepancies that are an insider's deletions when the bank with the most "
        r"laundering legs also deletes them ($\mu=5$ only).}"
        r"\label{tab:benign}",
        r"\begin{tabular}{lrrrrrrrr}\toprule",
        r"Canon & $\mu$ (min) & $\delta$ (min) & False rate & Mean $|D^e|$ & p99 $|D^e|$ & Flagged & Decoded & "
        r"Precision \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    nv = base[(base.canon == "naive") & (base.lag_min == 5)].iloc[0]
    M("BenignNaiveRate", pct(nv.false_rate, 1))
    M("BenignNaiveMean", num(nv.residual_mean))
    for close, w in ((0, "Zero"), (15, "Fifteen"), (60, "Sixty")):
        c = base[(base.canon == "canonical") & (base.lag_min == 5) & (base.close_min == close)].iloc[0]
        M(f"BenignCanon{w}Rate", pct(c.false_rate, 3))
        M(f"BenignCanon{w}Mean", f"{c.residual_mean:.1f}")
        M(f"BenignCanon{w}Decoded", pct(c.complete_share, 0))
        M(f"BenignCanon{w}Flagged", pct(c.epochs_flagged, 0))
        M(f"BenignCanon{w}Per", num(1 / c.false_rate) if c.false_rate > 0 else "none")
    slow = base[(base.canon == "canonical") & (base.lag_min == 30)]
    ok = slow[slow.complete_share >= 0.99]
    M("BenignSlowClose", f"{ok.close_min.min():g}" if len(ok) else "more than 120")
    s120 = slow[slow.close_min == slow.close_min.max()].iloc[0]
    M("BenignSlowRate", pct(s120.false_rate, 2))
    M("BenignSlowDecoded", pct(s120.complete_share, 0))
    for close, w in ((0, "Zero"), (15, ""), (30, "Thirty"), (60, "Sixty")):
        if (5.0, close) not in mix.index:
            continue
        a = mix.loc[(5.0, close)]
        M(f"BenignMix{w}Precision", pct(a.alert_precision))
        M(f"BenignMix{w}Recall", pct(a.malicious_recall))
        M(f"BenignMix{w}Malicious", num(a.malicious_total))
        M(f"BenignMix{w}Decoded", num(a.decoded))
        M(f"BenignMix{w}Complete", pct(a.complete_share, 0))
    M("BenignWireShare", pct(base.wire_share.iloc[0]))
    raw = pd.read_csv(ROOT / "data" / "raw" / "ibm_aml" / "HI-Small_Trans.csv",
                      usecols=["From Bank", "To Bank", "Payment Currency", "Receiving Currency"])
    fx = raw["Payment Currency"] != raw["Receiving Currency"]
    M("FxTransfers", num(fx.sum()))
    M("FxInterbank", num((fx & (raw["From Bank"] != raw["To Bank"])).sum()))


# ---------------------------------------------------------------------------------------------------- X12
def t_scale():
    """State, gas and chain growth per epoch at consortium sizes N, from X12 (data) and the measured gas of X6."""
    sc = pd.read_csv(RES / "x12_scale.csv")
    tx = _chain_tx()
    if tx is None:
        return
    gwei, usd = MAINNET_GWEI, ETH_USD_RATE

    def g(phase, mode=None):
        h = tx[(tx.phase == phase) & ((tx["mode"] == mode) if mode else True)]
        return h.gas.median(), h.calldata_bytes.median(), h.calldata_zero.median()

    books, _, _ = g("books")
    sketch, _, _ = g("sketch")
    pub, pub_b, pub_z = g("reveal", "publish")
    acc_later = tx[(tx["mode"] == "accumulate") & (tx.phase == "reveal")].gas.quantile(0.5)
    agg, agg_b, agg_z = g("aggregate", "commit")
    floor = lambda b, z: 21000 + 10 * (z + 4 * (b - z))
    pub7623 = max(pub, floor(pub_b, pub_z))
    M("GasPublishSevenSixTwoThree", num(pub7623))
    M("GasPublishFloorPerByte", "40")
    rows = []
    for r in sc.itertuples():
        m = r.m_needed
        k = m / 192
        sc_ = lambda g: 21000 + (g - 21000) * k     # the transaction base cost does not grow with m
        per_bank = {"publish": books + sketch + sc_(pub7623), "commit": books + sketch,
                    "accumulate": books + sketch + sc_(acc_later)}
        per_epoch = {md: r.members * v + (sc_(agg) if md == "commit" else 0) for md, v in per_bank.items()}
        cd = {"publish": r.members * m * CELL, "commit": m * CELL, "accumulate": r.members * m * CELL}
        fee_year = lambda md: per_epoch[md] * 8760 * gwei * 1e-9 * usd
        rows.append(f"{num(r.members)} & {r.entries_bank_epoch:.0f} & {r.residual_p99:.0f} & {num(m)} & "
                    f"{kb(m * CELL)} & {per_epoch['publish'] / 1e6:.1f} & {per_epoch['commit'] / 1e6:.2f} & "
                    f"{cd['publish'] * 8760 / 2**30:.1f} & \\${fee_year('publish') / 1e6:.2f}M & "
                    f"\\${fee_year('commit') / 1e6:.2f}M \\\\")
        if r.members == 1000:
            M("ScaleThousandM", num(m))
            M("ScaleThousandPubGasM", f"{per_epoch['publish'] / 1e6:.0f}")
            M("ScaleThousandComGasM", f"{per_epoch['commit'] / 1e6:.1f}")
            M("ScaleThousandPubBlocks", f"{per_epoch['publish'] / 30e6:.0f}")
            M("ScaleThousandGrowth", f"{cd['publish'] * 8760 / 2**30:.0f}")
            M("ScaleThousandFeePub", f"{fee_year('publish') / 1e6:.1f}")
            M("ScaleThousandFeeCom", f"{fee_year('commit') / 1e6:.1f}")
            M("ScaleThousandEntries", f"{r.entries_bank_epoch:.0f}")
            M("ScaleThousandResidual", f"{r.residual_p99:.0f}")
            M("ScaleMaskEdges", num(r.mask_edges))
    _tex("t_scale", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Cost per hourly epoch at consortium size $N$. Entries: mean covered entries per bank and epoch. "
        r"$|D^e|$: 99th-percentile benign residual (canonical canon, $\mu=5$, $\delta=15$\,min); $m$ is sized to "
        r"$1.5\,|D^e|$. Gas per epoch for all banks, from the measured gas of each phase on the ledgers "
        r"(book commitment, sketch commitment and, in publish mode, the reveal repriced under EIP-7623), scaled with "
        r"$m$. Growth: publish-mode calldata per year. Fees per year at mainnet-equivalent prices (" + MACROS["ChainGwei"]
        + r"\,gwei, \$" + MACROS["ChainEthUsd"] + r" per ether); on a fee-free permissioned ledger the cost is block "
        r"space only.}\label{tab:scale}",
        r"\begin{tabular}{rrrrrrrrrr}\toprule",
        r" & & & & Sketch & \multicolumn{2}{c}{Gas/epoch ($10^6$)} & Growth & \multicolumn{2}{c}{Fee per year} \\"
        r"\cmidrule(lr){6-7}\cmidrule(lr){9-10}",
        r"$N$ & Entries & $|D^e|$ & $m$ & (KB) & Publish & Commit & (GB/yr) & Publish & Commit \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


# ----------------------------------------------------------------------------------------------------- X6
def _chain_tx():
    d = RES / "chain"
    files = sorted(d.glob("*_tx.csv"))
    if not files:
        return None
    tx = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return tx if "phase" in tx else None


LEDGERS = [("besu", "Besu QBFT", "Besu"), ("poa2_local", "Clique", "Emu"), ("ethpos", "PoS devnet", "Pos"),
           ("purechain", "PureChain", "Pure")]


def t_chain():
    d = RES / "chain"
    tx = _chain_tx()
    if tx is None:
        return
    ep = pd.concat(map(pd.read_csv, sorted(d.glob("*_epochs.csv"))), ignore_index=True)
    meta = [json.loads(p.read_text()) for p in sorted(d.glob("*_meta.json"))]
    ref = meta[0]
    usd = lambda gas: gas * ref["mainnet_gas_gwei"] * 1e-9 * ref["eth_usd"]
    floor = lambda r: 21000 + 10 * (r.calldata_zero + 4 * (r.calldata_bytes - r.calldata_zero))
    tx = tx.assign(gas7623=np.maximum(tx.gas, tx.apply(floor, axis=1)))
    ledgers = [x for x in LEDGERS if (tx.backend == x[0]).any()]
    books, sketch = tx[tx.phase == "books"].gas.median(), tx[tx.phase == "sketch"].gas.median()
    M("ChainGasBooks", num(books))
    M("ChainGasSketch", num(sketch))
    M("ChainGasCommitReveal", num(books + sketch))
    rows = []
    for mode in ("accumulate", "publish", "commit"):
        g = tx[tx["mode"] == mode]
        if g.empty:
            continue
        if mode == "commit":
            fin = g[g.phase == "aggregate"]
            last = fin.gas.median()
            label_last = f"{num(last)} (sum)"
            per_bank = books + sketch
        else:
            rv = g[g.phase == "reveal"].sort_values("gas")
            first = rv.groupby(["backend", "contract_epoch"]).gas.max().median() if mode == "accumulate" else None
            # accumulate: median of the reveals after each epoch's first (which pays the zero-to-nonzero stores)
            first_ = rv.groupby(["backend", "contract_epoch"]).gas.transform("max")
            last = rv[rv.gas < first_].gas.median() if mode == "accumulate" else rv.gas.median()
            label_last = (f"{num(first)} / {num(last)}" if mode == "accumulate" else num(last))
            per_bank = books + sketch + last
        g7623 = g[g.phase == "reveal"].gas7623.median() if mode == "publish" else None
        fee = usd(per_bank) if mode != "publish" else usd(books + sketch + g7623)
        lat = " & ".join(f"{g[g.backend == b].latency_s.median():.2f}" if (g.backend == b).any() else "--"
                         for b, _, _ in ledgers)
        e = ep[ep["mode"] == mode]
        exact = f"{int(((e.found == e.residual) & (e.spurious == 0)).sum())}/{len(e)}"
        rows.append(f"{mode} & {label_last} & {num(per_bank)} & \\${fee:.2f} & {lat} & {exact} \\\\")
        key = {"accumulate": "Acc", "publish": "Pub", "commit": "Com"}[mode]
        M(f"ChainGas{key}Bank", num(per_bank))
        M(f"ChainFee{key}", f"\\${fee:.2f}")
        if mode == "accumulate":
            M("ChainGasAccFirst", num(first))
            M("ChainGasAccLater", num(last))
        elif mode == "publish":
            M("ChainGasPubReveal", num(last))
            M("ChainGasPubRevealSevenSix", num(g7623))
            M("ChainPubCalldata", num(g[g.phase == "reveal"].calldata_bytes.median()))
            M("ChainPubCalldataZero", num(g[g.phase == "reveal"].calldata_zero.median()))
        else:
            M("ChainGasAggregate", num(last))
            M("ChainAggregateZero", pct((fin.calldata_zero / fin.calldata_bytes).median(), 0))
    M("ChainGwei", f"{ref['mainnet_gas_gwei']:g}")
    M("ChainEthUsd", num(ref["eth_usd"]))
    M("ChainRuns", ep.groupby(["backend", "mode"]).ngroups)
    M("ChainEpochs", len(ep))
    ok = (ep.found == ep.residual) & (ep.spurious == 0)
    M("ChainExactCount", int(ok.sum()))
    M("ChainExact", pct(ok.mean(), 0))
    M("ChainSeedMatch", pct(ep.seed_matches.mean(), 0))
    dro = ep[ep.dropout]
    if len(dro):
        M("ChainDropoutEpochs", len(dro))
        M("ChainDropoutExact", f"{int(((dro.found == dro.residual) & (dro.spurious == 0)).sum())}/{len(dro)}")
    lrows = []
    for b, name, key in ledgers:
        mt = [x for x in meta if x["backend"] == b]
        h = tx[tx.backend == b]
        M(f"ChainLat{key}", f"{h.latency_s.median():.1f}")
        seed_w = np.median([x["phase_s"]["seed_settle"] for x in mt])
        final_w = np.median([x["phase_s"]["final_settle"] for x in mt])
        total = np.median([x["phase_s"]["total"] / x["epochs"] for x in mt])
        rule = "finalized" if mt[0]["finalized"] else (f"{mt[0]['confirm_depth']} deep" if mt[0]["confirm_depth"]
                                                         else "head (instant finality)")
        M(f"ChainSeedWait{key}", f"{seed_w:.0f}")
        M(f"ChainFinalWait{key}", f"{final_w:.0f}")
        M(f"ChainEpochTime{key}", f"{total:.0f}")
        lrows.append(f"{name} & {rule} & {h.latency_s.median():.1f} & {seed_w:.0f} & {final_w:.0f} & "
                     f"{np.median([x['phase_s']['total'] for x in mt]):.0f} & {mt[0]['epochs']} \\\\")
        if b == "ethpos":
            gp = np.median([x["gas_price_wei"] for x in mt])
            M("ChainPosGwei", f"{gp / 1e9:.2f}")
            # fee cap a client checks before accepting a transaction: gas limit x gas price
            M("ChainPosAccCapEth", f"{9_300_000 * gp / 1e18:.3f}")
    M("ChainDepth", max(x["confirm_depth"] for x in meta))
    M("ChainWindow", ref["window"])
    hr = sorted((d / "headread").glob("purechain_accumulate_*_epochs.csv"))
    if hr:   # an earlier PureChain run read at the chain head instead of k confirmations deep
        h = pd.read_csv(hr[0])
        M("ChainHeadReadExact", pct(((h.found == h.residual) & (h.spurious == 0)).mean(), 0))
        M("ChainHeadReadEpochs", len(h))
    short = " & ".join(n for _, n, _ in ledgers)
    _tex("t_chain", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{The protocol on the ledgers: 16 banks, $m=192$, injected omissions and fabrications. "
        r"Every bank sends a book commitment (" + num(books) + r" gas) and a sketch commitment (" + num(sketch) +
        r" gas) per epoch; the last column of gas is the reveal (accumulate: first / later in an epoch) or, in commit "
        r"mode, the posted sum. Per bank: all of a bank's transactions in an epoch. Fee: per bank and epoch at "
        r"mainnet-equivalent prices, publish mode repriced under EIP-7623. Latency: median from sending to receipt "
        r"(s). Exact: epochs whose sum read from the ledger decoded to exactly the injected discrepancies.}"
        r"\label{tab:chain}",
        r"\begin{tabular}{lrrr" + "r" * len(ledgers) + "r}\\toprule",
        r" & \multicolumn{2}{c}{Gas} & & \multicolumn{" + str(len(ledgers)) + r"}{c}{Latency (s)} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-" + str(4 + len(ledgers)) + "}",
        r"Mode & Reveal or sum & Per bank & Fee & " + short + r" & Exact \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    _tex("t_ledgers", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Phase timing per ledger (medians over runs). Read rule: the block at which the seed block and the "
        r"sum are read. Seed wait: from the last book commitment until the seed block is settled. Final wait: from "
        r"the last transaction until it is settled. Run: wall time of a run of all its epochs, whose phases run "
        r"together.}\label{tab:ledgers}",
        r"\begin{tabular}{llrrrrr}\toprule",
        r"Ledger & Read rule & Latency (s) & Seed wait (s) & Final wait (s) & Run (s) & Epochs \\\midrule",
        *lrows, r"\bottomrule\end{tabular}\end{table}"]))


# ----------------------------------------------------------------------------------------------------- X7
def t_bench():
    p = RES / "x7_bench.json"
    if not p.exists():
        return
    b = json.loads(p.read_text())
    M("BenchDH", f"{1e6 * b['x25519_pair_key_s']:.0f}")
    M("BenchTag", f"{1e6 * b['tag_blake2b_s']:.1f}")
    M("BenchMask", f"{1e6 * b['mask_chacha20_m192_s']:.0f}")
    M("BenchBuildTenK", f"{1e3 * b['sketch_build_n10000_m192_s']:.1f}")
    M("BenchBuildHundredK", f"{1e3 * b['sketch_build_n100000_m192_s']:.0f}")
    k = next(k for k in b if k.startswith("decode_d150"))
    M("BenchDecode", f"{1e3 * b[k]:.0f}")
    M("BenchHkdf", f"{1e6 * b['subkeys_hkdf_s']:.0f}")
    M("BenchCellHash", f"{1e6 * b['cell_hash_blake2b_s']:.1f}")
    M("BenchRootTenK", f"{1e3 * b['book_root_n10000_s']:.0f}")
    M("BenchCommitHash", f"{1e6 * b['sketch_commit_keccak_m192_s']:.0f}")


# ----------------------------------------------------------------------------------------------------- X8
AML_SETTING = {"global": "Pooled monitor", "views": "Per-bank monitors"}
AML_COLS = [("standard", "none"), ("dropedge", "none"), ("omission", "none"),
            ("standard", "protocol restore"), ("standard", "protocol restore + alert")]
WORD = {1: "One", 5: "Five", 10: "Ten", 20: "Twenty"}


def _aml_label(s: str) -> str:
    if s == "clean":
        return "No attack"
    kind, k, *b = s.split()
    k = k.split("=")[1]
    if kind == "insider":
        return f"Insiders at {k} bank{'s' if k != '1' else ''}"
    return f"{kind.capitalize()}, 5 banks, $\\beta={b[0].split('=')[1]}$"


def _ms(v: pd.Series, d=1) -> str:
    """mean, with the standard deviation over seeds when there is more than one"""
    v = v.dropna()
    if v.empty:
        return "--"
    s = f"{100 * v.mean():.{d}f}"
    return s + (f"\\,$\\pm$\\,{100 * v.std():.{d}f}" if len(v) > 1 else "")


def _ci(lo: pd.Series, hi: pd.Series) -> str:
    """envelope of the per-seed paired bootstrap intervals (percentage points)"""
    if lo.dropna().empty:
        return "--"
    return f"[{100 * lo.min():+.1f}, {100 * hi.max():+.1f}]"


AML_MODEL = {"gin": "Multi-GIN+EU", "pna": "Multi-PNA+EU"}
VT = ["tau", "f1_vt", "precision_vt", "recall_vt"]
AML_KEY = ["setting", "model", "variant", "seed", "scenario", "defence"]


def _aml_frame() -> pd.DataFrame | None:
    """Every attack row (x8b), with the validation-threshold columns of monitors attacked before those existed taken
    from their insider-only re-run (x8t)."""
    def done(tag):   # finished runs only (an interrupted attack leaves NAME.partial.csv)
        return [p for p in sorted((RES / "aml").glob(f"{tag}_*.csv"))
                if not p.stem.endswith("_s99") and not p.stem.endswith(".partial")]
    files = done("x8b")
    if not files:
        return None
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    extra = done("x8t")
    if extra:
        vt = pd.concat(map(pd.read_csv, extra), ignore_index=True)[AML_KEY + VT]
        df = df.merge(vt, on=AML_KEY, how="left", suffixes=("", "_t"))
        for c in VT:
            if c + "_t" in df:
                df[c] = df[c].fillna(df[c + "_t"])
                df = df.drop(columns=c + "_t")
    for c in VT:
        if c not in df:
            df[c] = np.nan
    return df


def t_aml():
    df = _aml_frame()
    if df is None:
        return
    df = df[df.model == "gin"]
    std = df[df.variant == "standard"]
    M("AmlSeeds", std.groupby("setting").seed.nunique().min())
    M("AmlSeedsMax", std.groupby("setting").seed.nunique().max())
    hard = df[df.variant != "standard"]
    M("AmlHardSeeds", hard.groupby(["setting", "variant"]).seed.nunique().min() if len(hard) else 0)
    M("AmlHardSeedsMax", hard.groupby(["setting", "variant"]).seed.nunique().max() if len(hard) else 0)
    order = [s for s in dict.fromkeys(df.scenario) if not s.endswith("b=0.25")]
    rows, srows = [], []
    for st, label in AML_SETTING.items():
        d = df[df.setting == st]
        if d.empty:
            continue
        P = "Pooled" if st == "global" else "Views"
        s = d[d.variant == "standard"]
        pos = s[(s.scenario == "clean") & (s.defence == "none")].laundering_visible.iloc[0]
        M(f"Aml{P}Positives", num(pos))
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for sc in order:
            g = d[d.scenario == sc]
            gs = g[g.variant == "standard"]
            nd = gs[gs.defence == "none"].deleted_transfers.mean()

            def cell(v, f, col="f1"):
                return _ms(g[(g.variant == v) & (g.defence == f)][col])
            union = cell("standard", "union of books") if st == "global" and sc != "clean" else "--"
            rest = (cell("standard", "protocol restore"), cell("standard", "protocol restore", "pr_auc")) \
                if sc != "clean" else ("--", "--")
            share = pct(nd / pos, 0) if nd else "--"
            rows.append(f"{_aml_label(sc)} & {num(nd)} & {share} & {cell('standard', 'none')} & "
                        f"{cell('standard', 'none', 'pr_auc')} & {cell('dropedge', 'none')} & "
                        f"{cell('omission', 'none')} & {union} & {rest[0]} & {rest[1]} \\\\")
        # streams under the twenty-bank insider attack, standard model
        g = s[s.scenario == "insider k=20"]
        srows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}, insiders at 20 banks}}}} \\\\")
        for f, name in (("none", "No defence"), ("union of books", "Union of books"),
                        ("protocol restore", "Restore (model only)"), ("protocol alert", "Alerts only"),
                        ("protocol restore + alert", "Restore + alerts")):
            h = g[g.defence == f]
            if h.empty:
                continue
            srows.append(f"{name} & {_ms(h.f1)} & {_ms(h.f1_visible)} & {_ms(h.pr_auc)} & {_ms(h.pr_auc_visible)} & "
                         f"{_ci(h.d_pr_auc_lo, h.d_pr_auc_hi)} & {_ms(h.precision)} & "
                         f"{_ms(h.recall_deleted, 0)} \\\\")
        c = s[(s.scenario == "clean") & (s.defence == "none")]
        M(f"Aml{P}CleanF", _ms(c.f1))
        M(f"Aml{P}CleanPr", _ms(c.pr_auc))
        M(f"Aml{P}CleanFLo", f"{100 * c.f1_lo.min():.1f}")
        M(f"Aml{P}CleanFHi", f"{100 * c.f1_hi.max():.1f}")
        for k, w in WORD.items():
            g = s[s.scenario == f"insider k={k}"]
            M(f"Aml{P}Ins{w}Deleted", num(g[g.defence == "none"].deleted_transfers.mean()))
            M(f"Aml{P}Ins{w}DelClean", _ms(g[g.defence == "none"].recall_deleted_clean, 0))
            for f, tag in (("none", ""), ("union of books", "Union"), ("protocol restore", "Restore"),
                           ("protocol alert", "AlertOnly"), ("protocol restore + alert", "Alert")):
                h = g[g.defence == f]
                if h.empty:
                    continue
                M(f"Aml{P}Ins{w}{tag}F", _ms(h.f1))
                M(f"Aml{P}Ins{w}{tag}Pr", _ms(h.pr_auc))
                M(f"Aml{P}Ins{w}{tag}FVis", _ms(h.f1_visible))
                M(f"Aml{P}Ins{w}{tag}PrVis", _ms(h.pr_auc_visible))
                M(f"Aml{P}Ins{w}{tag}Del", _ms(h.recall_deleted, 0))
                M(f"Aml{P}Ins{w}{tag}Prec", _ms(h.precision))
                M(f"Aml{P}Ins{w}{tag}DPrCi", _ci(h.d_pr_auc_lo, h.d_pr_auc_hi))
                M(f"Aml{P}Ins{w}{tag}DFCi", _ci(h.d_f1_lo, h.d_f1_hi))
        for who in ("aware", "random"):   # five banks, half of their laundering records
            g = s[s.scenario == f"{who} k=5 b=0.5"]
            for f, tag in (("none", ""), ("protocol restore", "Restore"), ("protocol restore + alert", "Alert")):
                M(f"Aml{P}{who.capitalize()}Half{tag}F", _ms(g[g.defence == f].f1))
        for v in ("dropedge", "omission"):
            g = d[(d.variant == v) & (d.defence == "none")]
            M(f"Aml{P}{v.capitalize()}CleanF", _ms(g[g.scenario == "clean"].f1))
            M(f"Aml{P}{v.capitalize()}InsTwentyF", _ms(g[g.scenario == "insider k=20"].f1))
        for v, V in (("standard", ""), ("dropedge", "Dropedge"), ("omission", "Omission")):
            g = d[(d.variant == v) & (d.defence == "none")].pivot_table(index="seed", columns="scenario", values="f1")
            if "insider k=20" in g:
                M(f"Aml{P}{V}LossF", _ms(g["clean"] - g["insider k=20"]))
        pr = s.pivot_table(index="seed", columns=["scenario", "defence"], values="pr_auc")
        clean_pr, hit = pr[("clean", "none")], pr[("insider k=20", "none")]
        M(f"Aml{P}PrClean", f"{clean_pr.mean():.2f}")
        M(f"Aml{P}PrTwenty", f"{hit.mean():.2f}")
        M(f"Aml{P}PrTwentyRestore", f"{pr[('insider k=20', 'protocol restore')].mean():.2f}")
        M(f"Aml{P}PrDrop", pct((1 - hit / clean_pr).mean(), 0))
    r = df[df.defence == "protocol restore"]
    M("AmlRestoredWrong", num(r.restored_wrong.sum()))
    M("AmlRestoredShare", pct(r.restored_records.sum() / r.deleted_records.sum()))
    _tex("t_aml", "\n".join([
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\caption{Insider deletion and the Multi-GIN+EU monitor on the HI-Small test period (mean $\pm$ standard "
        r"deviation over seeds, \%). Deleted: laundering transfers whose record was deleted, and their share of the "
        r"test positives. A transfer missing from the monitor's books scores 0 and counts as a false negative. "
        r"Union of books: a pooled monitor that reads both witnesses' books, so a transfer stays while either record "
        r"does. Restore: the records the protocol decodes are put back (the honest witness supplies the fields). "
        f"Seeds: standard {MACROS['AmlSeeds']}--{MACROS['AmlSeedsMax']}, hardened "
        f"{MACROS['AmlHardSeeds']}--{MACROS['AmlHardSeedsMax']} per configuration.}}\\label{{tab:aml}}",
        r"\begin{tabular}{lrrrrrrrrr}\toprule",
        r" & \multicolumn{2}{c}{Deleted} & \multicolumn{2}{c}{Standard} & DropEdge & Omission- & Union & "
        r"\multicolumn{2}{c}{Restore} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){9-10}",
        r"Attack & $n$ & share & F1 & PR-AUC & F1 & aware F1 & F1 & F1 & PR-AUC \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    _tex("t_amlstreams", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{The monitor's model output and the protocol's alerts as separate streams (standard model, "
        r"insiders at the 20 banks with the most laundering records; \%). Full: all test transfers; visible: the "
        r"transfers still in the monitor's books, or raised as alerts. $\Delta$PR-AUC: paired Poisson bootstrap "
        r"interval of the difference to the clean books (200 replicates per seed; envelope over seeds, points). "
        r"Precision and deleted recall are on the full set. Every deleted record is laundering here, so alert "
        r"precision is 100\% by construction; Experiment II of the manuscript measures it with benign "
        r"discrepancies.}"
        r"\label{tab:amlstreams}",
        r"\begin{tabular}{lrrrrrrr}\toprule",
        r" & \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{PR-AUC} & & & Deleted \\"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Stream & full & visible & full & visible & $\Delta$PR-AUC & Precision & recall \\\midrule",
        *srows, r"\bottomrule\end{tabular}\end{table}"]))


CASE = [("clean", "none", "f1"), ("insider k=20", "none", "f1"), ("insider k=20", "union of books", "f1"),
        ("insider k=20", "protocol restore", "f1"), ("clean", "none", "pr_auc"), ("insider k=20", "none", "pr_auc"),
        ("insider k=20", "protocol restore", "pr_auc")]


def _cell(g: pd.DataFrame, sc: str, f: str, col: str, d=1) -> str:
    return _ms(g[(g.scenario == sc) & (g.defence == f)][col], d)


def t_amlcase():
    """Main-text case study: both detectors and both monitor settings under the twenty-bank insider attack."""
    df = _aml_frame()
    if df is None:
        return
    s = df[df.variant == "standard"]
    rows = []
    for mdl, mlabel in AML_MODEL.items():
        g_m = s[s.model == mdl]
        M(f"Aml{mdl.capitalize()}Seeds", g_m.groupby("setting").seed.nunique().min() if len(g_m) else 0)
        M(f"Aml{mdl.capitalize()}SeedsMax", g_m.groupby("setting").seed.nunique().max() if len(g_m) else 0)
        for st, slabel in AML_SETTING.items():
            g = g_m[g_m.setting == st]
            P = ("Pooled" if st == "global" else "Views")
            cells = [_cell(g, sc, f, col) for sc, f, col in CASE]
            if st != "global":
                cells[2] = "--"
            rows.append(f"{mlabel} & {slabel} & {g.seed.nunique() if len(g) else '--'} & "
                        + " & ".join(cells) + r" \\")
            if mdl != "pna" or g.empty:
                continue
            pre = f"AmlPna{P}"
            M(f"{pre}CleanF", _cell(g, "clean", "none", "f1"))
            M(f"{pre}CleanPr", _cell(g, "clean", "none", "pr_auc"))
            for f, tag in (("none", ""), ("union of books", "Union"), ("protocol restore", "Restore")):
                h = g[(g.scenario == "insider k=20") & (g.defence == f)]
                if h.empty:
                    continue
                M(f"{pre}InsTwenty{tag}F", _ms(h.f1))
                M(f"{pre}InsTwenty{tag}Pr", _ms(h.pr_auc))
                M(f"{pre}InsTwenty{tag}PrVis", _ms(h.pr_auc_visible))
                M(f"{pre}InsTwenty{tag}Del", _ms(h.recall_deleted, 0))
            M(f"{pre}InsTwentyDelClean", _ms(g[(g.scenario == "insider k=20") & (g.defence == "none")]
                                             .recall_deleted_clean, 0))
            pv = g[g.defence == "none"].pivot_table(index="seed", columns="scenario", values="f1")
            if "insider k=20" in pv:
                M(f"{pre}LossF", _ms(pv["clean"] - pv["insider k=20"]))
    _tex("t_amlcase", "\n".join([
        r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Case study: insiders at the 20 banks with the most test-period laundering records delete those "
        r"records. F1 (flagged above probability 0.5) and PR-AUC in \%, mean $\pm$ standard deviation over $n$ "
        r"training seeds. Union: a pooled monitor that reads both witnesses' books (pooled only). Restore: the "
        r"records the protocol decodes are put back. A deleted transfer missing from the monitor's books counts as a "
        r"false negative. Every scenario, the hardened monitors, the visible-set metrics, F1 at the "
        r"validation-optimal threshold and the gap to the published F1 are in the supplementary material.}"
        r"\label{tab:amlcase}",
        r"\begin{tabular}{llrrrrrrrr}\toprule",
        r" & & & \multicolumn{4}{c}{F1} & \multicolumn{3}{c}{PR-AUC} \\"
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
        r"Detector & Monitor & $n$ & Clean & Attacked & Union & Restore & Clean & Attacked & Restore \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


def t_amlpna():
    """Supplement: every scenario for the second detector (standard training)."""
    df = _aml_frame()
    if df is None:
        return
    s = df[(df.model == "pna") & (df.variant == "standard")]
    if s.empty:
        return
    order = [x for x in dict.fromkeys(s.scenario) if not x.endswith("b=0.25")]
    rows = []
    for st, label in AML_SETTING.items():
        d = s[s.setting == st]
        if d.empty:
            continue
        pos = d[(d.scenario == "clean") & (d.defence == "none")].laundering_visible.iloc[0]
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for sc in order:
            g = d[d.scenario == sc]
            nd = g[g.defence == "none"].deleted_transfers.mean()
            union = _cell(g, sc, "union of books", "f1") if st == "global" and sc != "clean" else "--"
            rest = ((_cell(g, sc, "protocol restore", "f1"), _cell(g, sc, "protocol restore", "pr_auc"))
                    if sc != "clean" else ("--", "--"))
            rows.append(f"{_aml_label(sc)} & {num(nd)} & {pct(nd / pos, 0) if nd else '--'} & "
                        f"{_cell(g, sc, 'none', 'f1')} & {_cell(g, sc, 'none', 'pr_auc')} & {union} & "
                        f"{rest[0]} & {rest[1]} \\\\")
    _tex("t_amlpna", "\n".join([
        r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Insider deletion and the Multi-PNA+EU monitor (standard training; mean $\pm$ standard deviation "
        f"over {MACROS.get('AmlPnaSeeds', '--')}--{MACROS.get('AmlPnaSeedsMax', '--')} seeds, \\%). Columns as in "
        r"\cref{tab:aml}.}\label{tab:amlpna}",
        r"\begin{tabular}{lrrrrrrr}\toprule",
        r" & \multicolumn{2}{c}{Deleted} & \multicolumn{2}{c}{No defence} & Union & \multicolumn{2}{c}{Restore} \\"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-8}",
        r"Attack & $n$ & share & F1 & PR-AUC & F1 & F1 & PR-AUC \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


def t_amlthr():
    """Supplement: F1 at the reference threshold (0.5) and at the threshold that maximises validation F1."""
    df = _aml_frame()
    if df is None:
        return
    s = df[(df.variant == "standard") & df.f1_vt.notna()]
    rows, diffs = [], []
    for mdl, mlabel in AML_MODEL.items():
        for st, slabel in AML_SETTING.items():
            g = s[(s.model == mdl) & (s.setting == st)]
            if g.empty:
                continue
            tau = g.drop_duplicates("seed").tau
            cells = [f"{tau.mean():.2f}"]
            for sc, f in (("clean", "none"), ("insider k=20", "none"), ("insider k=20", "protocol restore")):
                cells += [_cell(g, sc, f, "f1"), _cell(g, sc, f, "f1_vt")]
            for col in ("f1", "f1_vt"):
                pv = g[g.defence == "none"].pivot_table(index="seed", columns="scenario", values=col)
                if "insider k=20" in pv:
                    diffs.append((col, mdl, st, (pv["clean"] - pv["insider k=20"]).mean()))
            rows.append(f"{mlabel} & {slabel} & {g.seed.nunique()} & " + " & ".join(cells) + r" \\")
    if not rows:
        return
    loss = pd.DataFrame(diffs, columns=["col", "model", "setting", "loss"]).pivot_table(
        index=["model", "setting"], columns="col", values="loss")
    if {"f1", "f1_vt"} <= set(loss.columns):
        M("AmlThrLossDiffMax", f"{100 * (loss.f1_vt - loss.f1).abs().max():.1f}")
    _tex("t_amlthr", "\n".join([
        r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{F1 (\%) at the reference threshold of 0.5 and at the threshold $\tau$ that maximises F1 on the "
        r"validation period (clean books), applied unchanged to the test period; standard training, insiders at "
        r"20 banks. $n$: seeds with both.}\label{tab:amlthr}",
        r"\begin{tabular}{llrrrrrrrr}\toprule",
        r" & & & & \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Attacked} & \multicolumn{2}{c}{Restore} \\"
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"Detector & Monitor & $n$ & $\tau$ & 0.5 & $\tau$ & 0.5 & $\tau$ & 0.5 & $\tau$ \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


def t_anomaly():
    """Why per-bank recall of deleted laundering is 0 with one insider bank and not with five: what the top banks'
    deleted test-period laundering transfers look like (same split and ranking as x_aml.scenarios)."""
    from . import data
    t = data.transfers()
    day = t["minute"].to_numpy().astype(np.int64) * 60 // 86400
    d = np.bincount(day)
    best, score = None, np.inf
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            tot = np.array([d[:i].sum(), d[i:j].sum(), d[j:].sum()], float)
            s = np.max(np.abs(tot / tot.sum() - (0.6, 0.2, 0.2)) / (0.6, 0.2, 0.2))
            if s < score:
                best, score = (i, j), s
    test = day >= best[1]
    L = t["laundering"].to_numpy()
    inter = (t.payer != t.payee).to_numpy()
    rec_bank = np.concatenate([t.payer.to_numpy(), t.payee.to_numpy()[inter]])
    rec_tx = np.concatenate([np.arange(len(t)), np.flatnonzero(inter)])
    sel = test[rec_tx] & L[rec_tx]
    rank = pd.Series(rec_bank[sel]).value_counts().index
    ach = (t["format"] == "ACH").to_numpy()
    M("AnomalyTestAch", pct(ach[test & L].mean()))
    for k, w in ((1, "One"), (5, "Five")):
        tx = np.unique(rec_tx[sel & np.isin(rec_bank, rank[:k])])
        M(f"Anomaly{w}Deleted", num(len(tx)))
        M(f"Anomaly{w}Ach", pct(ach[tx].mean()))
        M(f"Anomaly{w}Payer", pct(np.isin(t.payer.to_numpy()[tx], rank[:k]).mean(), 0))


def t_spill():
    """X11: does deletion also hide the laundering that stays in the books?"""
    files = sorted((RES / "aml").glob("x11_spill_*.csv"))
    if not files:
        return
    d = pd.concat(map(pd.read_csv, files), ignore_index=True)
    d = d[d.n > 0]
    laund = d[d.subset != "legitimate"]
    M("AmlSpillSeeds", d.groupby("setting").seed.nunique().min())
    M("AmlSpillMaxChange", f"{100 * (laund.flagged_attacked - laund.flagged_clean).abs().max():.1f}")
    M("AmlSpillKeptN", num(d[d.subset == "kept laundering"].n.min()))
    rows = []
    for st, label in AML_SETTING.items():
        P = "Pooled" if st == "global" else "Views"
        s = d[d.setting == st]
        if s.empty:
            continue
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for sc in ("insider k=5", "insider k=10", "insider k=20", "aware k=5 b=0.5", "random k=5 b=0.5"):
            g = s[s.scenario == sc]
            k, nb = g[g.subset == "kept laundering"], g[g.subset == "neighbour laundering"]
            rows.append(f"{_aml_label(sc)} & {_ms(k.flagged_clean)} & {_ms(k.flagged_attacked)} & "
                        f"{num(nb.n.mean())} & {_ms(nb.flagged_clean)} & {_ms(nb.flagged_attacked)} \\\\")
        g = s[s.scenario == "insider k=20"]
        for sub, key in (("kept laundering", "Kept"), ("neighbour laundering", "Near"), ("legitimate", "Legit")):
            h = g[g.subset == sub]
            M(f"Aml{P}Spill{key}Clean", _ms(h.flagged_clean, 1 if key != "Legit" else 2))
            M(f"Aml{P}Spill{key}Attacked", _ms(h.flagged_attacked, 1 if key != "Legit" else 2))
        M(f"Aml{P}SpillNearN", num(g[g.subset == "neighbour laundering"].n.mean()))
    _tex("t_spill", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Does deletion also hide the laundering that stays in the books? Recall (\%, mean over "
        f"{MACROS['AmlSpillSeeds']} seeds of the standard model) of the test-period laundering transfers with no "
        r"deleted record, scored by the monitor on the clean and on the attacked books. Neighbours: kept laundering "
        r"transfers that share a payer or payee account with a deleted transfer; $n$ is their number (mean over "
        r"seeds). $\beta$: share of each insider bank's laundering records deleted. Kept laundering numbers "
        f"{MACROS.get('AmlSpillKeptN', '--')} transfers or more per scenario.}}\\label{{tab:spill}}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r" & \multicolumn{2}{c}{Kept laundering} & \multicolumn{3}{c}{Neighbours of deleted transfers} \\"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Attack & Clean & Attacked & $n$ & Clean & Attacked \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


def t_stream():
    files = sorted((RES / "aml").glob("x10_stream_*.csv"))
    if not files:
        return
    d = pd.read_csv(files[0])
    M("StreamEpochs", len(d))
    M("StreamDeleted", num(d.deleted.sum()))
    M("StreamRestored", num(d.restored.sum()))
    M("StreamRestoredWrong", num(d.restored_wrong.sum()))
    M("StreamComplete", pct(d.complete.mean(), 0))
    M("StreamProtocolS", f"{d.protocol_s.median():.1f}")
    M("StreamScoreS", f"{(d.graph_s + d.score_s).median():.1f}")
    M("StreamTotalMaxS", f"{(d.protocol_s + d.graph_s + d.score_s).max():.0f}")
    rows = []
    for k, label in (("none", "No protocol"), ("restore", "Restore"), ("alert", "Restore + alert")):
        lr = d[f"laundering_flagged_{k}"].sum() / d.laundering.sum()
        dr = d[f"deleted_flagged_{k}"].sum() / d.deleted.sum()
        pr = d[f"laundering_flagged_{k}"].sum() / d[f"flagged_{k}"].sum()
        M(f"StreamRecall{k.capitalize()}", pct(lr))
        M(f"StreamDeletedRecall{k.capitalize()}", pct(dr))
        M(f"StreamPrecision{k.capitalize()}", pct(pr))
        rows.append(f"{label} & {pct(lr)} & {pct(dr)} & {pct(pr)} \\\\")
    _tex("t_stream", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Continuous verification, epoch by epoch, with insiders at the five banks with the most laundering "
        f"records ({len(d)} hourly test epochs, per-bank monitors): recall of all laundering transfers and of the "
        r"deleted ones, and precision of the flags. Each epoch is scored on the records up to its end only.}"
        r"\label{tab:stream}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Defence & Laundering recall & Deleted recall & Precision \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for f in (t_coverage, t_detection, f_sizing, f_collusion, t_attacks, t_benign, t_chain, t_scale, t_bench,
              t_aml, t_amlcase, t_amlpna, t_amlthr, t_anomaly, t_spill, t_stream):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items()))
    _tex("macros", "% generated by python -m omission.report\n" + body + "\n")
    print(f"{len(MACROS)} macros")


if __name__ == "__main__":
    main()
