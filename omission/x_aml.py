"""X8-X11: omission as an evasion attack on AML transaction monitoring, and recovery through the protocol.

  train   train a monitor (Multi-GIN+EU or Multi-PNA+EU; standard, DropEdge or omission-aware adversarial training)
  attack  insiders delete laundering records in the test period; score the books as they stand, after the protocol
          has put back the records it decoded, and with the decoded omissions raised as alerts
  spill   whether deletion also hides the laundering that stays in the books (clean vs attacked monitor)
  stream  continuous verification: per-epoch latency of decode, restoration and re-scoring

The protocol is run on the actual books of every affected epoch (consortium of the 1,000 busiest banks, m = 192).
Masks are left out here because they cancel exactly in the aggregate (X2 checks this in every honest epoch).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from finverity.config import RESULTS

from . import aml, data, protocol as P

OUT = RESULTS / "paper2" / "aml"
N_MEMBERS = 1000
M_CELLS = 192


def _name(setting, kind, variant, seed):
    return f"{setting}_{kind}_{variant}_s{seed}"


def _device(gpu: int):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def cmd_train(a):
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    name = _name(a.setting, a.model, a.variant, a.seed)
    logf = open(OUT / "models" / f"{name}.log", "a")

    def log(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{name}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    ckpt = OUT / "models" / f"{name}.ckpt"
    model, hist = aml.train(a.model, a.setting, a.variant, a.seed, a.epochs, _device(a.gpu), log=log, ckpt=ckpt)
    torch.save(model.state_dict(), OUT / "models" / f"{name}.pt")
    hist.to_csv(OUT / "models" / f"{name}_hist.csv", index=False)
    ckpt.unlink(missing_ok=True)
    log("saved")


def load(setting, kind, variant, seed, device):
    e = aml.edges()
    part = aml.split(e.t.to_numpy())
    g = aml.build(aml.all_records().take(part[aml.all_records().tx] == 0), setting, np.flatnonzero(part == 0))
    model = aml.make_model(kind, g)
    model.load_state_dict(torch.load(OUT / "models" / f"{_name(setting, kind, variant, seed)}.pt", map_location="cpu"))
    return model.to(device)


# ----------------------------------------------------------------------------------------------- protocol
class Consortium:
    """The protocol over the covered transfers of the N busiest banks."""

    def __init__(self, n_members=N_MEMBERS, m=M_CELLS):
        self.m = m
        t = data.transfers()
        cov = data.covered(t, data.members(n_members))
        self.sel, self.keys = data.covered_keys(n_members)
        self.tx = np.flatnonzero(cov)                         # covered row -> transfer
        self.pos = -np.ones(len(t), np.int64)
        self.pos[self.tx] = np.arange(len(self.tx))
        self.banks = np.array(sorted(data.members(n_members)), dtype=np.int64)
        self.books = P.honest_books(self.sel, self.keys)
        self.sel = self.sel[["payer", "payee"]]                # the canonical strings are no longer needed
        self.epoch = t["epoch"].to_numpy()
        # key -> covered row through a sorted array of one 64-bit MAC word (no Python objects per key, which
        # data-loader workers would copy); the full key is compared after the lookup
        self._word = np.ascontiguousarray(self.keys[:, 1])
        self._order = np.argsort(self._word, kind="stable")
        self._sorted = self._word[self._order]

    def key_row(self, k) -> int | None:
        j = np.searchsorted(self._sorted, k[1])
        while j < len(self._sorted) and self._sorted[j] == k[1]:
            i = self._order[j]
            if np.array_equal(self.keys[i], k):
                return int(i)
            j += 1
        return None

    def recover(self, deleted: aml.Records):
        """Run every epoch that lost a covered record; return the records the decoded aggregate puts back (the
        record of the witness the sign names as the omitter) and per-epoch outcomes."""
        n = len(self.tx)
        p = self.pos[deleted.tx]
        ok = p >= 0
        payer = self.sel["payer"].to_numpy()
        rows = np.where(deleted.bank[ok] == payer[p[ok]], p[ok], n + p[ok])
        drop = np.zeros(len(self.books), bool)
        drop[rows] = True
        books = self.books.take(~drop)
        restored, ep_rows = [], []
        for e in np.unique(self.epoch[deleted.tx[ok]]):
            b = books.take(books.epoch == e)
            t0 = time.perf_counter()
            r, agg, _ = P.run_epoch(b, self.banks, self.m, int(e), None)
            complete, keys, signs = agg.decode()
            dt = time.perf_counter() - t0
            for k, s in zip(np.asarray(keys).reshape(-1, self.keys.shape[1]), np.asarray(signs).tolist()):
                i = self.key_row(k.astype(self.keys.dtype))
                if i is None:
                    continue
                row = self.sel.iloc[i]
                restored.append((self.tx[i], int(row.payee if s > 0 else row.payer)))
            ep_rows.append(dict(epoch=int(e), residual=r.residual, found=r.found, complete=r.complete, seconds=dt))
        rec = aml.Records(np.array([x for x, _ in restored], np.int64), np.array([b for _, b in restored], np.int64))
        return rec, pd.DataFrame(ep_rows)


# ------------------------------------------------------------------------------------------------ attacks
def scenarios(test_rec: aml.Records, y: np.ndarray, clean: pd.Series, rng):
    """(name, corrupt banks, deleted records). Corrupt banks: most test-period laundering records first."""
    lr = test_rec.take(y[test_rec.tx] == 1)
    rank = pd.Series(lr.bank).value_counts().index.to_numpy()
    yield "clean", [], test_rec.take(np.zeros(len(test_rec), bool))
    for k in (1, 5, 10, 20):
        bad = rank[:k]
        yield f"insider k={k}", bad, lr.take(np.isin(lr.bank, bad))
    bad = rank[:5]
    mine = lr.take(np.isin(lr.bank, bad))
    s = clean.reindex(mine.tx).fillna(0).to_numpy()
    for beta in (0.25, 0.5):
        n = int(round(beta * len(mine)))
        yield f"aware k=5 b={beta:g}", bad, mine.take(np.argsort(-s, kind="stable")[:n])
        yield f"random k=5 b={beta:g}", bad, mine.take(rng.choice(len(mine), n, replace=False))


def metrics(p: pd.Series, test_tx, y, deleted_tx, corrupt_tx, alerts=None) -> dict:
    s = p.reindex(test_tx).fillna(0.0).to_numpy()
    flag = s > 0.5
    if alerts is not None:
        flag |= np.isin(test_tx, alerts)
    yt = y[test_tx] == 1
    tp = (flag & yt).sum()
    dl = np.isin(test_tx, deleted_tx) & yt
    rest = np.isin(test_tx, corrupt_tx) & yt & ~dl
    other = yt & ~np.isin(test_tx, corrupt_tx)
    return dict(f1=2 * tp / max(1, flag.sum() + yt.sum()), precision=tp / max(1, flag.sum()), recall=tp / yt.sum(),
                pr_auc=average_precision_score(yt, s if alerts is None else np.maximum(s, np.isin(test_tx, alerts))),
                recall_deleted=flag[dl].mean() if dl.any() else np.nan,
                recall_corrupt_rest=flag[rest].mean() if rest.any() else np.nan,
                recall_elsewhere=flag[other].mean(), n_flagged=int(flag.sum()))


def cmd_attack(a):
    dev = _device(a.gpu)
    e = aml.edges()
    y = e.y.to_numpy()
    part = aml.split(e.t.to_numpy())
    rec = aml.all_records()
    test_tx = np.flatnonzero(part == 2)
    test_rec = rec.take(part[rec.tx] == 2)
    model = load(a.setting, a.model, a.variant, a.seed, dev)
    cons = Consortium()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    clean = aml.score(model, aml.build(rec, a.setting, test_tx), dev)
    rows = []
    for name, bad, deleted in scenarios(test_rec, y, clean, rng):
        corrupt_tx = np.unique(test_rec.tx[np.isin(test_rec.bank, bad)])
        del_tx = np.unique(deleted.tx)
        base = dict(setting=a.setting, model=a.model, variant=a.variant, seed=a.seed, scenario=name,
                    corrupt_banks=len(bad), deleted_records=len(deleted), deleted_transfers=len(del_tx),
                    deleted_interbank=int((e.payer.to_numpy()[del_tx] != e.payee.to_numpy()[del_tx]).sum()))
        if len(deleted):
            books = rec.minus(deleted)
            p = aml.score(model, aml.build(books, a.setting, test_tx), dev)
        else:
            books, p = rec, clean
        rows.append(dict(base, defence="none", **metrics(p, test_tx, y, del_tx, corrupt_tx)))
        if not len(deleted):
            continue
        restored, ep = cons.recover(deleted)
        truth = set(zip(deleted.tx.tolist(), deleted.bank.tolist()))
        got = set(zip(restored.tx.tolist(), restored.bank.tolist()))
        prot = dict(restored_records=len(got), restored_correct=len(got & truth),
                    restored_wrong=len(got - truth), epochs=len(ep), epochs_complete=float(ep.complete.mean()),
                    decode_s_median=float(ep.seconds.median()))
        p2 = aml.score(model, aml.build(aml.Records.concat([books, restored]), a.setting, test_tx), dev)
        rows.append(dict(base, defence="protocol restore", **prot, **metrics(p2, test_tx, y, del_tx, corrupt_tx)))
        rows.append(dict(base, defence="protocol restore + alert", **prot,
                         **metrics(p2, test_tx, y, del_tx, corrupt_tx, alerts=np.unique(restored.tx))))
        print(f"[{_name(a.setting, a.model, a.variant, a.seed)}] {name}: "
              f"F1 {rows[-3]['f1']:.3f} -> {rows[-2]['f1']:.3f} / {rows[-1]['f1']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"x8_{_name(a.setting, a.model, a.variant, a.seed)}.csv", index=False)


def cmd_spill(a):
    """X11: does deleting laundering records also hide the laundering that stays in the books? Laundering transfers
    with no deleted record are scored by the monitor on the clean and on the attacked books; neighbours share an
    account with a deleted transfer, so their graph context changed most. Legitimate transfers give the false-alarm
    rate. The deletions are those of cmd_attack (same scenarios, same seed)."""
    dev = _device(a.gpu)
    e = aml.edges()
    y = e.y.to_numpy()
    part = aml.split(e.t.to_numpy())
    rec = aml.all_records()
    test_tx = np.flatnonzero(part == 2)
    test_rec = rec.take(part[rec.tx] == 2)
    model = load(a.setting, a.model, a.variant, a.seed, dev)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    clean = aml.score(model, aml.build(rec, a.setting, test_tx), dev)
    pc = clean.reindex(test_tx).fillna(0.0)
    src, dst = e.src.to_numpy(), e.dst.to_numpy()
    laund, legit = test_tx[y[test_tx] == 1], test_tx[y[test_tx] == 0]
    rows = []
    for name, bad, deleted in scenarios(test_rec, y, clean, rng):
        if not len(deleted):
            continue
        del_tx = np.unique(deleted.tx)
        pa = aml.score(model, aml.build(rec.minus(deleted), a.setting, test_tx), dev).reindex(test_tx).fillna(0.0)
        acc = np.union1d(src[del_tx], dst[del_tx])
        kept = laund[~np.isin(laund, del_tx)]
        near = kept[np.isin(src[kept], acc) | np.isin(dst[kept], acc)]
        for subset, ids in (("kept laundering", kept), ("neighbour laundering", near), ("legitimate", legit)):
            c, q = pc[ids].to_numpy(), pa[ids].to_numpy()
            rows.append(dict(setting=a.setting, model=a.model, variant=a.variant, seed=a.seed, scenario=name,
                             subset=subset, n=len(ids), flagged_clean=float((c > 0.5).mean()) if len(ids) else np.nan,
                             flagged_attacked=float((q > 0.5).mean()) if len(ids) else np.nan,
                             lost=int(((c > 0.5) & (q <= 0.5)).sum()), gained=int(((c <= 0.5) & (q > 0.5)).sum()),
                             score_clean=float(c.mean()) if len(ids) else np.nan,
                             score_attacked=float(q.mean()) if len(ids) else np.nan))
        k, nb = rows[-3], rows[-2]
        print(f"[{_name(a.setting, a.model, a.variant, a.seed)}] {name}: kept laundering recall "
              f"{k['flagged_clean']:.3f} -> {k['flagged_attacked']:.3f}; neighbours ({nb['n']}) "
              f"{nb['flagged_clean']:.3f} -> {nb['flagged_attacked']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"x11_spill_{_name(a.setting, a.model, a.variant, a.seed)}.csv", index=False)


def cmd_stream(a):
    """Continuous verification under insiders at the five banks with the most laundering records: every sampled
    test epoch is closed, its masked sketches built and summed, the aggregate decoded, the decoded records put back,
    and the epoch's transfers scored on the graph of everything recorded up to the end of the epoch (no future
    transfers). Times each step."""
    dev = _device(a.gpu)
    e = aml.edges()
    y = e.y.to_numpy()
    t = e.t.to_numpy()
    part = aml.split(t)
    rec = aml.all_records()
    test_rec = rec.take(part[rec.tx] == 2)
    lr = test_rec.take(y[test_rec.tx] == 1)
    bad = pd.Series(lr.bank).value_counts().index.to_numpy()[:5]
    deleted = lr.take(np.isin(lr.bank, bad))
    books = rec.minus(deleted)
    model = load(a.setting, a.model, a.variant, a.seed, dev)
    cons = Consortium()
    ep_of = cons.epoch
    test_epochs = np.unique(ep_of[part == 2])
    out = OUT / f"x10_stream_{_name(a.setting, a.model, a.variant, a.seed)}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []   # resume after an interruption
    done = {r["epoch"] for r in rows}
    for ep in test_epochs[::a.every]:
        if int(ep) in done:
            continue
        mine = deleted.take(ep_of[deleted.tx] == ep)
        in_ep = np.flatnonzero((ep_of == ep) & (part == 2))
        t0 = time.perf_counter()
        n = len(cons.tx)
        p = cons.pos[mine.tx]
        ok = p >= 0
        rows_ = np.where(mine.bank[ok] == cons.sel["payer"].to_numpy()[p[ok]], p[ok], n + p[ok])
        b = cons.books.take(cons.books.epoch == ep)
        drop = np.isin(np.flatnonzero(cons.books.epoch == ep), rows_)
        r, agg, _ = P.run_epoch(b.take(~drop), cons.banks, cons.m, int(ep), 16)
        complete, keys, signs = agg.decode()
        t1 = time.perf_counter()
        restored = []
        for k, s in zip(np.asarray(keys).reshape(-1, cons.keys.shape[1]), np.asarray(signs).tolist()):
            i = cons.key_row(k.astype(cons.keys.dtype))
            if i is not None:
                row = cons.sel.iloc[i]
                restored.append((cons.tx[i], int(row.payee if s > 0 else row.payer)))
        rr = aml.Records(np.array([x for x, _ in restored], np.int64), np.array([q for _, q in restored], np.int64))
        upto = aml.Records.concat([books, rr])
        upto = upto.take(ep_of[upto.tx] <= ep)
        g = aml.build(upto, a.setting, in_ep)
        t2 = time.perf_counter()
        sc = aml.score(model, g, dev)
        t3 = time.perf_counter()
        # the same epoch without the protocol: the monitor sees the attacked books only
        sc0 = aml.score(model, aml.build(books.take(ep_of[books.tx] <= ep), a.setting, in_ep), dev)
        yl = in_ep[y[in_ep] == 1]
        dl = np.unique(mine.tx)
        flag0 = set(sc0[sc0 > 0.5].index.tolist())
        flag1 = set(sc[sc > 0.5].index.tolist())
        flag2 = flag1 | set(rr.tx.tolist())
        truth = set(zip(mine.tx.tolist(), mine.bank.tolist()))
        got = set(zip(rr.tx.tolist(), rr.bank.tolist()))
        rows.append(dict(epoch=int(ep), transfers=len(in_ep), laundering=len(yl), deleted=len(dl),
                         restored=len(got), restored_wrong=len(got - truth), complete=complete,
                         protocol_s=t1 - t0, graph_s=t2 - t1, score_s=t3 - t2,
                         **{f"{k}_{d}": v for d, f in (("none", flag0), ("restore", flag1), ("alert", flag2))
                            for k, v in (("flagged", len(f & set(in_ep.tolist()))),
                                         ("laundering_flagged", sum(int(x in f) for x in yl)),
                                         ("deleted_flagged", sum(int(x in f) for x in dl)))}))
        print(rows[-1], flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "attack", "spill", "stream"])
    ap.add_argument("--every", type=int, default=6, help="stream: score every n-th test epoch")
    ap.add_argument("--setting", default="views", choices=["global", "views"])
    ap.add_argument("--model", default="gin", choices=["gin", "pna"])
    ap.add_argument("--variant", default="standard", choices=["standard", "dropedge", "omission"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    {"train": cmd_train, "attack": cmd_attack, "spill": cmd_spill, "stream": cmd_stream}[a.cmd](a)


if __name__ == "__main__":
    main()
