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

from finverity.config import ROOT

from .iblt import threshold

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


def kb(x) -> str:
    return f"{x / 1024:.1f}"


def _tex(name: str, body: str) -> None:
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
    for n, w in ((100, "Hundred"), (500, "FiveHundred"), (1000, "Thousand"), (2000, "TwoThousand")):
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
        rows.append(f"{_label(r.scenario)} & {num(r.attacked_transfers)} & 0\\% & {f(r.totals_tx_detection)} & "
                    f"{f(r.ours_epoch_detection)} & {f(r.ours_item_recall)} & {int(r.ours_false_epochs)} \\\\")
    _tex("t_detection", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        rf"\caption{{Detection on IBM AML HI-Small, consortium of the {num(cfg['members'])} busiest banks, "
        rf"{num(cfg['covered_transfers'])} covered transfers in hourly epochs, sketches of $m={cfg['m']}$ cells. "
        r"Integrity-only anchoring commits each bank's own books and cannot see what was never written. Per-pair "
        r"totals compare the count and amount each side reports per ordered bank pair and epoch. Epoch detection is "
        r"the share of tampered epochs whose aggregate is non-zero; item recall is the share of non-cancelling keys "
        r"that peeling returns with the correct sign; false epochs are untampered epochs flagged.}\label{tab:detection}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r" & Tampered & Integrity & Per-pair & \multicolumn{3}{c}{This protocol} \\\cmidrule(lr){5-7}",
        r"Scenario & transfers & only & totals & Epochs & Items & False \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))
    h = df[df.scenario == "honest"].iloc[0]
    M("Epochs", num(h.epochs))
    M("CoveredTransfers", num(cfg["covered_transfers"]))
    M("Members", num(cfg["members"]))
    M("Cells", cfg["m"])
    M("Kappa", cfg["kappa"])
    M("HonestFalse", int(h.ours_false_epochs))
    n_mem, m = cfg["members"], cfg["m"]
    M("BytesBank", kb(m * 48))
    M("BytesOurs", f"{h.ours_bytes_epoch / 2**20:.1f}")
    M("BytesOursChain", kb(n_mem * 32 + m * 48))
    M("BreakEven", num(m * 48 / 32))
    M("EntriesPerBank", f"{2 * cfg['covered_transfers'] / (n_mem * h.epochs):.0f}")
    M("ChainRatioTags", f"{h.tags_bytes_epoch / (n_mem * 32 + m * 48):.0f}")
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
    co = df[df.scenario.str.startswith("coalition")].iloc[0]
    M("CoalitionLegs", num(co.attacked_transfers))
    M("CoalitionDetect", "0\\%" if co.residual_items == 0 else pct(co.ours_epoch_detection))


# ----------------------------------------------------------------------------------------------------- X3
def f_sizing():
    df = pd.read_csv(RES / "x3_sizing.csv")
    g = df.groupby(["d", "ratio"])["complete"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for d, mk in zip(g.columns, "osD^"):
        ax.plot(g.index, 100 * g[d], marker=mk, ms=4, label=f"$d={d}$")
    ax.axvline(threshold(), color="0.5", ls=":", lw=1)
    ax.text(threshold() + 0.03, 8, "$c_3=1.222$", color="0.35", fontsize=8)
    ax.set_xlabel("Sketch cells per discrepancy, $m/d$")
    ax.set_ylabel("Epochs fully localised (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F2_sizing.pdf")
    fig.savefig(FIG / "F2_sizing.png", dpi=200)
    M("SizingDetect", pct(df.flagged.mean()))
    r15 = df[(df.ratio == 1.5) & (df.d >= 100)].complete.mean()
    M("SizingCompleteOneFiveLarge", pct(r15))
    r3 = df[(df.ratio == 3.0)].complete.mean()
    M("SizingCompleteThree", pct(r3))
    M("SizingTrials", int(df.trial.max() + 1))


# ----------------------------------------------------------------------------------------------------- X4
def f_collusion():
    df = pd.read_csv(RES / "x4_collusion.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(df.c, 100 * df.legs_share_covered, marker="o", ms=3)
    ax.set_xlabel("Colluding member banks $c$ (greedy choice)")
    ax.set_ylabel("Covered laundering legs hidden (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F3_collusion.pdf")
    fig.savefig(FIG / "F3_collusion.png", dpi=200)
    for c, w in ((2, "Two"), (10, "Ten"), (30, "Thirty")):
        r = df[df.c == c].iloc[0]
        M(f"Collude{w}", pct(r.legs_share_covered))
        M(f"ColludeAll{w}", pct(r.legs_share_all))


# ----------------------------------------------------------------------------------------------------- X6
def t_chain():
    d = RES / "chain"
    if not d.exists():
        return
    tx = [pd.read_csv(p) for p in sorted(d.glob("*_tx.csv"))]
    ep = [pd.read_csv(p) for p in sorted(d.glob("*_epochs.csv"))]
    if not tx:
        return
    tx, ep = pd.concat(tx), pd.concat(ep)
    meta = {p.stem.removesuffix("_meta"): json.loads(p.read_text()) for p in d.glob("*_meta.json")}
    name = {"purechain": "PureChain (live)", "besu": "Besu QBFT", "poa2_local": "Clique emulation",
            "ethpos": "Ethereum PoS devnet"}
    rows = []
    for (b, mode, m), g in tx.groupby(["backend", "mode", "m"]):
        base = mode.split("-")[0]   # "commit-aggregate" rows (order -1) belong to the commit run
        e = ep[(ep.backend == b) & (ep["mode"] == base) & (ep.m == m)]
        first = g[g.order <= 0].gas.median()
        later = g[g.order > 0].gas.median() if (g.order > 0).any() else first
        run_meta = next(v for k, v in meta.items() if k.startswith(f"{b}_{base}_m{m}_"))
        usd = later * run_meta["mainnet_gas_gwei"] * 1e-9 * run_meta["eth_usd"]
        ok = ((e.found == e.residual) & (e.spurious == 0)).mean()
        tail = f"{len(e)} & {pct(ok, 0)}" if mode == base else r"\multicolumn{2}{c}{one per epoch}"
        rows.append(f"{name.get(b, b)} & {mode} & {m} & {num(first)} & {num(later)} & "
                    f"{g.latency_s.median():.2f} & {g.latency_s.quantile(0.95):.2f} & \\${usd:.2f} & "
                    f"{tail} \\\\")
    ref = next(iter(meta.values()))
    usd = lambda gas: f"\\${gas * ref['mainnet_gas_gwei'] * 1e-9 * ref['eth_usd']:.2f}"
    for mode, key in (("accumulate", "Acc"), ("publish", "Pub"), ("commit", "Com")):
        g = tx[tx["mode"] == mode]
        M(f"ChainGas{key}First", num(g[g.order <= 0].gas.median()))
        M(f"ChainGas{key}Later", num(g[g.order > 0].gas.median()))
        M(f"ChainFee{key}", usd(g[g.order > 0].gas.median()))
    M("ChainGwei", f"{ref['mainnet_gas_gwei']:g}")
    M("ChainEthUsd", num(ref["eth_usd"]))
    M("ChainDepth", int(ep.confirm_depth.max()))
    M("ChainGasAggregate", num(tx[tx["mode"] == "commit-aggregate"].gas.median()))
    M("ChainFeeAggregate", usd(tx[tx["mode"] == "commit-aggregate"].gas.median()))
    M("ChainRuns", ep.groupby(["backend", "mode"]).ngroups)
    M("ChainEpochs", len(ep))
    M("ChainExact", pct(((ep.found == ep.residual) & (ep.spurious == 0)).mean(), 0))
    for b, key in (("purechain", "Pure"), ("poa2_local", "Emu"), ("besu", "Besu"), ("ethpos", "Pos")):
        if (tx.backend == b).any():
            M(f"ChainLat{key}", f"{tx[tx.backend == b].latency_s.median():.1f}")
        if "confirm_s" in ep and ep[ep.backend == b].confirm_s.notna().any():
            M(f"ChainConfirm{key}", f"{ep[ep.backend == b].confirm_s.median():.1f}")
    hr = sorted((d / "headread").glob("purechain_accumulate_*_epochs.csv"))
    if hr:   # the same runs read at the chain head instead of k confirmations deep
        h = pd.read_csv(hr[0])
        M("ChainHeadReadExact", pct(((h.found == h.residual) & (h.spurious == 0)).mean(), 0))
        M("ChainHeadReadEpochs", len(h))
    _tex("t_chain", "\n".join([
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{The protocol on four ledgers: a consortium of the 16 busiest banks publishes one masked sketch per "
        r"bank and epoch, with injected omissions and fabrications. Gas of the first and of later publications in an "
        r"epoch (median), send-to-receipt latency, the mainnet-equivalent fee of one publication, and the share of "
        r"epochs whose on-chain sum decoded to exactly the injected discrepancies.}\label{tab:chain}",
        r"\begin{tabular}{llrrrrrrrr}\toprule",
        r"Ledger & Mode & $m$ & Gas, first & Gas, later & Lat.\ med (s) & P95 (s) & Fee & Epochs & Exact \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"]))


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
    if v.empty:
        return "--"
    s = f"{100 * v.mean():.{d}f}"
    return s + (f"\\,$\\pm$\\,{100 * v.std():.{d}f}" if len(v) > 1 else "")


def t_aml():
    files = [p for p in sorted((RES / "aml").glob("x8_*.csv")) if not p.stem.endswith("_s99")]
    if not files:
        return
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    M("AmlSeeds", df[df.variant == "standard"].groupby("setting").seed.nunique().min())
    order = list(dict.fromkeys(df.scenario))
    rows = []
    for st, label in AML_SETTING.items():
        d = df[df.setting == st]
        if d.empty:
            continue
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for sc in order:
            g = d[d.scenario == sc]
            cells = [_ms(g[(g.variant == v) & (g.defence == f)].f1) for v, f in AML_COLS]
            dl = [_ms(g[(g.variant == "standard") & (g.defence == f)].recall_deleted.dropna(), 0)
                  for f in ("none", "protocol restore + alert")]
            rows.append(f"{_aml_label(sc)} & " + " & ".join(cells + dl) + " \\\\")
        P = "Pooled" if st == "global" else "Views"
        s = d[d.variant == "standard"]
        for k, w in WORD.items():
            g = s[s.scenario == f"insider k={k}"]
            for f, tag in (("none", ""), ("protocol restore", "Restore"), ("protocol restore + alert", "Alert")):
                h = g[g.defence == f]
                M(f"Aml{P}Ins{w}{tag}F", _ms(h.f1))
                M(f"Aml{P}Ins{w}{tag}Del", _ms(h.recall_deleted, 0))
        M(f"Aml{P}CleanF", _ms(s[s.scenario == "clean"].f1))
        for who in ("aware", "random"):   # five banks, half of their laundering records
            g = s[s.scenario == f"{who} k=5 b=0.5"]
            for f, tag in (("none", ""), ("protocol restore", "Restore"), ("protocol restore + alert", "Alert")):
                M(f"Aml{P}{who.capitalize()}Half{tag}F", _ms(g[g.defence == f].f1))
        for v in ("dropedge", "omission"):
            g = d[(d.variant == v) & (d.defence == "none")]
            M(f"Aml{P}{v.capitalize()}CleanF", _ms(g[g.scenario == "clean"].f1))
            M(f"Aml{P}{v.capitalize()}InsTwentyF", _ms(g[g.scenario == "insider k=20"].f1))
        # what 20 insider banks cost each variant (F1 points, per seed), and PR-AUC for comparison with rewiring
        for v, V in (("standard", ""), ("dropedge", "Dropedge"), ("omission", "Omission")):
            g = d[(d.variant == v) & (d.defence == "none")].pivot_table(index="seed", columns="scenario", values="f1")
            M(f"Aml{P}{V}LossF", f"{100 * (g['clean'] - g['insider k=20']).mean():.1f}")
        pr = s.pivot_table(index="seed", columns=["scenario", "defence"], values="pr_auc")
        clean_pr, hit = pr[("clean", "none")], pr[("insider k=20", "none")]
        M(f"Aml{P}PrClean", f"{clean_pr.mean():.2f}")
        M(f"Aml{P}PrTwenty", f"{hit.mean():.2f}")
        M(f"Aml{P}PrTwentyRestore", f"{pr[('insider k=20', 'protocol restore')].mean():.2f}")
        M(f"Aml{P}PrDrop", pct((1 - hit / clean_pr).mean(), 0))
    M("AmlHardSeeds", df[df.variant != "standard"].groupby(["setting", "variant"]).seed.nunique().min())
    r = df[df.defence == "protocol restore"]
    M("AmlRestoredWrong", num(r.restored_wrong.sum()))
    M("AmlRestoredShare", pct(r.restored_records.sum() / r.deleted_records.sum()))
    _tex("t_aml", "\n".join([
        r"\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Insider omission against the Multi-GIN laundering detector on the HI-Small test period: F1 of the "
        r"laundering class (\%, mean $\pm$ standard deviation over seeds) for the standard model, the two AI-only "
        r"hardened models, and the standard model with the records recovered by the protocol put back (restore) and "
        r"also raised as alerts (+ alert); and recall of the deleted laundering transfers (\%). The standard model "
        f"was trained with {MACROS['AmlSeeds']} seeds, the hardened models with {MACROS['AmlHardSeeds']}."
        r"}\label{tab:aml}",
        r"\begin{tabular}{lrrrrrrr}\toprule",
        r" & \multicolumn{3}{c}{No protocol} & \multicolumn{2}{c}{With protocol} & \multicolumn{2}{c}{Deleted recall} \\"
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Attack & Standard & DropEdge & Omission-aware & Restore & + alert & None & + alert \\\midrule",
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
    for f in (t_coverage, t_detection, f_sizing, f_collusion, t_chain, t_bench, t_aml, t_stream):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items()))
    _tex("macros", "% generated by python -m omission.report\n" + body + "\n")
    print(f"{len(MACROS)} macros")


if __name__ == "__main__":
    main()
