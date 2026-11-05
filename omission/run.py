"""Paper 2 experiments on IBM AML (no blockchain needed; chain costs are in chain.py).

X1 coverage    two-witness coverage of transfers, value and laundering legs vs consortium size
X2 detection   attacks vs the protocol and the baselines over every hourly epoch
X3 sizing      decoding success vs sketch size / residual size
X4 collusion   laundering value a coalition of c member banks can hide
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from finverity.config import RESULTS

from . import data, protocol as P
from .iblt import Sketch, threshold

OUT = RESULTS / "paper2"
N_DEFAULT = 1000
M_DEFAULT = 192
KAPPA = 16


def _low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def x1_coverage():
    t = data.transfers()
    inter = (t["payer"] != t["payee"]).to_numpy()
    L = t["laundering"].to_numpy()
    amt = t["usd"].to_numpy()
    rows = []
    for n in [10, 50, 100, 250, 500, 1000, 2000, 5000, None]:
        mem = data.members(n)
        cov = data.covered(t, mem)
        a = t["payer"].isin(mem).to_numpy()
        b = t["payee"].isin(mem).to_numpy()
        member_any = a | b
        rows.append(dict(members=n or len(data.ranking()), transfers=cov.mean(), laundering=cov[L].mean(),
                         laundering_value=amt[cov & L].sum() / amt[L].sum(),
                         member_transfers=member_any.mean(),
                         covered_of_member=cov.sum() / member_any.sum(),
                         laundering_covered_of_member=(cov & L).sum() / (member_any & L).sum(),
                         intra_bank=(~inter).mean(), laundering_intra=(~inter)[L].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x1_coverage.csv", index=False)
    print(df.round(4).to_string())


def _epochs(books: P.Books):
    order = np.argsort(books.epoch, kind="stable")
    ep = books.epoch[order]
    cuts = np.flatnonzero(np.diff(ep)) + 1
    for part in np.split(order, cuts):
        yield int(books.epoch[part[0]]), books.take(part)


def _scenarios(books: P.Books, sel: pd.DataFrame, rng):
    L = sel["laundering"].to_numpy()
    yield P.Attack("honest", books)
    for p in (1e-4, 1e-3, 1e-2):
        yield P.random_omission(books, p, rng)
    # the member banks with the most covered laundering legs
    lb = pd.Series(books.bank[L[books.tx]]).value_counts()
    for bank in lb.index[:5]:
        yield P.targeted_laundering(books, L, int(bank))
    busiest = int(pd.Series(books.bank).value_counts().index[0])
    yield P.substitution(books, busiest, 5, rng)
    yield P.alteration(books, busiest, 5, rng)
    # the two member banks that share the most laundering legs
    lsel = sel[sel["laundering"]]
    pair = pd.Series(list(zip(np.minimum(lsel.payer, lsel.payee), np.maximum(lsel.payer, lsel.payee)))).value_counts()
    yield P.collusion(books, L, set(int(x) for x in pair.index[0]))


def x2_detection(n_members: int = N_DEFAULT, m: int = M_DEFAULT, kappa: int = KAPPA, mask_every: int = 24):
    sel, keys = data.covered_keys(n_members)
    banks = np.array(sorted(data.members(n_members)), dtype=np.int64)
    books = P.honest_books(sel, keys)
    rng = np.random.default_rng(2026)
    ep_rows, sc_rows = [], []
    for att in _scenarios(books, sel, rng):
        t0 = time.time()
        attacked = att.attacked_tx
        bad_cells = P.baseline_totals(att.books)
        res = []
        every = 1 if att.name == "honest" else mask_every  # the honest run masks every epoch
        for e, b in _epochs(att.books):
            r, _, _ = P.run_epoch(b, banks, m, e, kappa if e % every == 0 else None)
            res.append(r)
            ep_rows.append(dict(scenario=att.name, **r.__dict__, masked=e % every == 0))
        R = pd.DataFrame([r.__dict__ for r in res])
        # attacked transfers caught by each scheme
        a_tx = np.array(sorted(attacked), dtype=np.int64)
        if len(a_tx):
            s = sel.iloc[a_tx]
            cell_hit = np.array([c in bad_cells for c in zip(s["payer"], s["payee"], s["epoch"])])
            tr_keys = {tuple(k.tolist()) for k in keys[a_tx]}
        else:
            cell_hit = np.array([], bool)
            tr_keys = set()
        att_epochs = set(sel["epoch"].to_numpy()[a_tx].tolist()) if len(a_tx) else set()
        s_a = sel.iloc[a_tx]
        att_cells = set(zip(s_a["payer"], s_a["payee"], s_a["epoch"]))
        flagged = set(R.loc[R.flagged, "epoch"])
        sc_rows.append(dict(
            scenario=att.name, attackers=len(att.attackers), attacked_transfers=len(a_tx),
            attacked_laundering=int(sel["laundering"].to_numpy()[a_tx].sum()) if len(a_tx) else 0,
            residual_items=int(R.residual.sum()), epochs=len(R), attacked_epochs=len(att_epochs),
            ours_epoch_detection=len(att_epochs & flagged) / len(att_epochs) if att_epochs else np.nan,
            ours_false_epochs=len(flagged - att_epochs),
            ours_item_recall=R.found.sum() / R.residual.sum() if R.residual.sum() else np.nan,
            ours_spurious=int(R.spurious.sum()),
            ours_complete_share=R.complete.mean(),
            totals_tx_detection=cell_hit.mean() if len(cell_hit) else np.nan,
            totals_false_cells=len(bad_cells - att_cells),
            ours_bytes_epoch=len(banks) * m * P.CELL_BYTES,
            totals_bytes_epoch=P.totals_bytes(att.books) / len(R),
            tags_bytes_epoch=P.TAG_BYTES * len(att.books) / len(R),
            pairs_revealed_epoch=len(pd.DataFrame({"a": att.books.bank, "c": att.books.cpty,
                                                   "e": att.books.epoch}).drop_duplicates()) / 2 / len(R),
            build_s_median=R.t_build_s.median(), decode_s_median=R.t_decode_s.median(),
            wall_s=time.time() - t0))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc_rows[-1].items()}, flush=True)
    pd.DataFrame(ep_rows).to_parquet(OUT / "x2_epochs.parquet", index=False)
    pd.DataFrame(sc_rows).to_csv(OUT / "x2_scenarios.csv", index=False)
    (OUT / "x2_config.json").write_text(json.dumps(dict(members=n_members, m=m, kappa=kappa, mask_every=mask_every,
                                                        covered_transfers=len(sel), epoch_min=data.EPOCH_MIN)))


def x3_sizing(n_members: int = N_DEFAULT, trials: int = 40):
    """Omit d random rows from one busy epoch and decode with m cells."""
    sel, keys = data.covered_keys(n_members)
    banks = np.array(sorted(data.members(n_members)), dtype=np.int64)
    books = P.honest_books(sel, keys)
    busy = int(pd.Series(books.epoch).value_counts().index[0])
    b0 = books.take(books.epoch == busy)
    rng = np.random.default_rng(7)
    rows = []
    for d in (10, 30, 100, 300):
        for ratio in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0):
            m = 3 * int(np.ceil(ratio * d / 3))
            for t in range(trials):
                drop = np.zeros(len(b0), bool)
                drop[rng.choice(len(b0), d, replace=False)] = True
                r, _, _ = P.run_epoch(b0.take(~drop), banks, m, busy, None)
                rows.append(dict(d=d, ratio=ratio, m=m, trial=t, flagged=r.flagged, complete=r.complete,
                                 recall=r.found / r.residual))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x3_sizing.csv", index=False)
    print(df.groupby(["d", "ratio"])[["flagged", "complete", "recall"]].mean().round(3).unstack(0).to_string())
    print("threshold c3 =", threshold())


def x4_collusion(n_members: int = N_DEFAULT, cmax: int = 30):
    """Greedy coalition (an attacker heuristic): add the member bank that hides the most extra laundering legs, i.e.
    covered legs whose payer and payee are both in the coalition. Legs, not value, drive the choice because a single
    synthetic transfer holds 61% of the dataset's laundering value."""
    sel, _ = data.covered_keys(n_members)
    L = sel[sel["laundering"]]
    t = data.transfers()
    total_n, total_v = int(t["laundering"].sum()), t.loc[t["laundering"], "usd"].sum()
    g = L.groupby(["payer", "payee"]).agg(n=("usd", "size"), v=("usd", "sum"))
    adj: dict[int, dict[int, list[float]]] = {}
    for (a, b), r in g.iterrows():
        for x, y in ((a, b), (b, a)):
            e = adj.setdefault(int(x), {}).setdefault(int(y), [0, 0.0])
            e[0] += r.n
            e[1] += r.v
    coal: set[int] = set()
    hid_n, hid_v = 0, 0.0
    rows = [dict(c=0, legs=0, legs_share_covered=0.0, legs_share_all=0.0, value_share_all=0.0)]
    for c in range(1, cmax + 1):
        best, gain = None, (-1, -1.0)
        for x, nb in adj.items():
            if x in coal:
                continue
            gn = sum(v[0] for y, v in nb.items() if y in coal)
            gv = sum(v[1] for y, v in nb.items() if y in coal)
            key = (gn, sum(v[0] for v in nb.values()))
            if key > gain:
                best, gain, bv = x, key, gv
        coal.add(best)
        hid_n += gain[0]
        hid_v += bv
        rows.append(dict(c=c, legs=hid_n, legs_share_covered=hid_n / len(L), legs_share_all=hid_n / total_n,
                         value_share_all=hid_v / total_v, bank=best))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x4_collusion.csv", index=False)
    print(df.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="+", choices=["x1", "x2", "x3", "x4"])
    ap.add_argument("--members", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _low_priority()
    for e in a.exp:
        t = time.time()
        {"x1": x1_coverage, "x2": lambda: x2_detection(a.members), "x3": lambda: x3_sizing(a.members),
         "x4": lambda: x4_collusion(a.members)}[e]()
        print(f"{e} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
