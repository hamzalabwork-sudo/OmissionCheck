"""Transaction monitoring on IBM AML HI-Small, and the books the monitor reads.

The detector is the reference Multi-GNN of Egressy et al. (AAAI 2024): its model classes are imported unchanged from
the authors' repository (MULTIGNN) and trained with their published hyperparameters, reverse message passing, port
numbering, ego IDs and edge updates. What changes is the input: the graph is assembled from bank records, so a record
an insider deletes is absent from it, and the records our protocol recovers can be put back.

Records: one per (transfer, witness bank); a transfer between two banks has two, an intra-bank transfer one.
  global: one pooled graph; a transfer is in it unless a witness deleted its record (a pooled analytic fed by the
          witness the insider controls, e.g. the originator's report);
  views:  each bank monitors its own books; node (account, bank) is the account as bank sees it, and an interbank
          transfer appears in both witnesses' views. A transfer is flagged if either witness flags it.

Port numbering is computed exactly as in the reference implementation (data_util.ports) but vectorised, so the graph
can be rebuilt after every deletion or restoration.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finverity.config import DATA_PROCESSED

from . import data

MULTIGNN = Path(os.environ.get("MULTIGNN", Path.home() / "omission_ml" / "Multi-GNN"))
EDGES = DATA_PROCESSED / "aml_edges.parquet"
DAY = 86400
# model_settings.json of the reference implementation
SETTINGS = {
    "gin": dict(lr=0.006213266113989207, n_hidden=66, n_gnn_layers=2, w_ce1=1.0000182882773443,
                w_ce2=6.275014431494497, dropout=0.00983468338330501, final_dropout=0.10527690625126304),
    "pna": dict(lr=0.0006116418195373612, n_hidden=20, n_gnn_layers=2, w_ce1=1.0003967674742307,
                w_ce2=7.077633468006714, dropout=0.08340440094051481, final_dropout=0.28812979737686323),
}
NEIGHBOURS = [100, 100]
BATCH = 8192


# --------------------------------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def edges() -> pd.DataFrame:
    """Every transfer in the row order of data.transfers() (row i is transfer i), with the reference features:
    accounts, currencies and payment formats are numbered in order of first appearance as format_kaggle_files.py
    does, and time is in seconds from the first transfer."""
    if EDGES.exists():
        return pd.read_parquet(EDGES)
    df = pd.read_csv(data.RAW, dtype=str)
    n = len(df)

    def interleave(a, b):
        out = np.empty(2 * n, dtype=object)
        out[0::2], out[1::2] = a, b
        return pd.factorize(out)[0].reshape(n, 2)

    acc = interleave((df["From Bank"] + df["Account"]).to_numpy(), (df["To Bank"] + df["Account.1"]).to_numpy())
    cur = interleave(df["Receiving Currency"].to_numpy(), df["Payment Currency"].to_numpy())
    fmt = pd.factorize(df["Payment Format"])[0]
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M").astype("int64") // 10**9
    out = pd.DataFrame({
        "src": acc[:, 0], "dst": acc[:, 1], "t": (ts - ts.min()).astype(np.int64),
        "amount": df["Amount Received"].astype(float), "currency": cur[:, 0], "format": fmt,
        "y": df["Is Laundering"].astype(np.int8), "payer": df["From Bank"].astype(np.int64),
        "payee": df["To Bank"].astype(np.int64)})
    out.to_parquet(EDGES, index=False)
    return out


def split(t: np.ndarray, per=(0.6, 0.2, 0.2)) -> np.ndarray:
    """Reference temporal split (data_loading.get_data): whole days, chosen so the three shares are closest to per.
    Returns 0/1/2 (train/validation/test) per transfer."""
    day = (t // DAY).astype(np.int64)
    d = np.bincount(day)
    best, score = None, np.inf
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            tot = np.array([d[:i].sum(), d[i:j].sum(), d[j:].sum()], float)
            s = np.max(np.abs(tot / tot.sum() - per) / per)
            if s < score:
                best, score = (i, j), s
    i, j = best
    return np.where(day < i, 0, np.where(day < j, 1, 2)).astype(np.int8)


@dataclass
class Records:
    """Bank records of transfers: (transfer, witness bank)."""
    tx: np.ndarray
    bank: np.ndarray

    def __len__(self):
        return len(self.tx)

    def take(self, sel) -> "Records":
        return Records(self.tx[sel], self.bank[sel])

    @staticmethod
    def concat(parts) -> "Records":
        return Records(np.concatenate([p.tx for p in parts]), np.concatenate([p.bank for p in parts]))

    def minus(self, other: "Records") -> "Records":
        return self.take(~np.isin(_rkey(self), _rkey(other)))


def _rkey(r: Records) -> np.ndarray:
    return r.tx.astype(np.int64) * (1 << 24) + r.bank


@lru_cache(maxsize=1)
def all_records() -> Records:
    e = edges()
    tx = np.arange(len(e), dtype=np.int64)
    inter = (e.payer != e.payee).to_numpy()
    return Records(np.concatenate([tx, tx[inter]]),
                   np.concatenate([e.payer.to_numpy(), e.payee.to_numpy()[inter]]))


# ------------------------------------------------------------------------------------------------- graphs
def _ports(a: np.ndarray, b: np.ndarray, t: np.ndarray, n_nodes: int) -> np.ndarray:
    """Rank of b among a's distinct neighbours ordered by first transaction time (reference port numbering)."""
    order = np.lexsort((np.arange(len(a)), t, a))
    pair = a.astype(np.int64) * n_nodes + b
    uniq, first, inv = np.unique(pair[order], return_index=True, return_inverse=True)
    ua = uniq // n_nodes
    o2 = np.lexsort((first, ua))
    ua_s = ua[o2]
    start = np.r_[0, np.flatnonzero(np.diff(ua_s)) + 1]
    rank = np.empty(len(uniq), np.int64)
    rank[o2] = np.arange(len(uniq)) - np.repeat(start, np.diff(np.r_[start, len(ua_s)]))
    out = np.empty(len(a), np.int64)
    out[order] = rank[inv.ravel()]
    return out


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd == 0, 1, sd)


@dataclass
class Graph:
    rec: Records          # one graph edge per record (views) or per transfer (global; bank = -1)
    data: object          # torch_geometric HeteroData
    seeds: np.ndarray     # graph edges to classify


def build(rec: Records, setting: str, seed_tx: np.ndarray) -> Graph:
    """The graph a monitor sees given the surviving records; seeds are the edges of the transfers in seed_tx."""
    from torch_geometric.data import HeteroData
    e = edges()
    if setting == "global":
        # a transfer survives only if every witness still records it
        tx, cnt = np.unique(rec.tx, return_counts=True)
        need = np.where((e.payer.to_numpy() != e.payee.to_numpy())[tx], 2, 1)
        tx = tx[cnt == need]
        g_rec = Records(tx, -np.ones(len(tx), np.int64))
        src, dst = e.src.to_numpy()[tx], e.dst.to_numpy()[tx]
        n_nodes = int(max(e.src.max(), e.dst.max())) + 1
    else:
        g_rec = rec
        tx = rec.tx
        bidx = pd.factorize(rec.bank)[0].astype(np.int64)
        s = e.src.to_numpy()[tx].astype(np.int64) << 20 | bidx
        d = e.dst.to_numpy()[tx].astype(np.int64) << 20 | bidx
        codes, uniq = pd.factorize(np.concatenate([s, d]))
        src, dst = codes[:len(tx)], codes[len(tx):]
        n_nodes = len(uniq)
    t = e.t.to_numpy()[tx]
    attr = np.column_stack([t, e.amount.to_numpy()[tx], e.currency.to_numpy()[tx], e.format.to_numpy()[tx],
                            _ports(dst, src, t, n_nodes), _ports(src, dst, t, n_nodes)]).astype(np.float64)
    attr = torch.tensor(_z(attr), dtype=torch.float32)
    ids = torch.arange(len(tx), dtype=torch.float32).view(-1, 1)
    h = HeteroData()
    h["node"].x = torch.zeros((n_nodes, 1))
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    h["node", "to", "node"].edge_index = ei
    h["node", "rev_to", "node"].edge_index = ei.flipud()
    h["node", "to", "node"].edge_attr = torch.cat([ids, attr], 1)
    h["node", "rev_to", "node"].edge_attr = torch.cat([ids + len(tx), attr], 1)
    h["node", "to", "node"].y = torch.tensor(e.y.to_numpy()[tx], dtype=torch.long)
    seeds = np.flatnonzero(np.isin(tx, seed_tx))
    return Graph(g_rec, h, seeds)


# -------------------------------------------------------------------------------------------------- model
def _multignn():
    if str(MULTIGNN) not in sys.path:
        sys.path.insert(0, str(MULTIGNN))
    import models
    from train_util import AddEgoIds
    return models, AddEgoIds


WORKERS = int(os.environ.get("AML_WORKERS", 2))   # 0 loads in-process: slower, no worker copies


def loader(g: Graph, shuffle: bool, workers: int | None = None):
    from torch_geometric.loader import LinkNeighborLoader
    _, AddEgoIds = _multignn()
    et = ("node", "to", "node")
    idx = torch.tensor(g.seeds, dtype=torch.long)
    return LinkNeighborLoader(g.data, num_neighbors=NEIGHBOURS, edge_label_index=(et, g.data[et].edge_index[:, idx]),
                              edge_label=g.data[et].y[idx], batch_size=BATCH, shuffle=shuffle,
                              transform=AddEgoIds(), num_workers=WORKERS if workers is None else workers,
                              persistent_workers=False)


def make_model(kind: str, g: Graph):
    from torch_geometric.nn import to_hetero
    from torch_geometric.utils import degree
    models, _ = _multignn()
    c = SETTINGS[kind]
    e_dim = g.data["node", "to", "node"].edge_attr.shape[1] - 1
    if kind == "gin":
        m = models.GINe(num_features=2, num_gnn_layers=c["n_gnn_layers"], n_classes=2, n_hidden=c["n_hidden"],
                        residual=False, edge_updates=True, edge_dim=e_dim, dropout=c["dropout"],
                        final_dropout=c["final_dropout"])
    else:
        # degree histogram of a sampled batch, as the reference computes it
        b = next(iter(loader(g, shuffle=True, workers=0)))
        d = degree(torch.cat([b["node", "to", "node"].edge_index[1], b["node", "rev_to", "node"].edge_index[1]]),
                   dtype=torch.long)
        m = models.PNA(num_features=2, num_gnn_layers=c["n_gnn_layers"], n_classes=2, n_hidden=c["n_hidden"],
                       edge_updates=True, edge_dim=e_dim, dropout=c["dropout"], deg=torch.bincount(d),
                       final_dropout=c["final_dropout"])
    return to_hetero(m, g.data.metadata(), aggr="mean")


def _batch(b, g: Graph, ids_np: np.ndarray, node_map: np.ndarray):
    """Seed mask of a sampled batch; seed edges the sampler left out are appended (the reference does this for some
    datasets only). Returns the mask and the graph edge ids of the masked rows."""
    et, rt = ("node", "to", "node"), ("node", "rev_to", "node")
    seed_ids = ids_np[b[et].input_id.numpy()]
    have = b[et].edge_attr[:, 0].long().numpy()
    miss = seed_ids[~np.isin(seed_ids, have)]
    if len(miss):
        n_id = b["node"].n_id.numpy()
        node_map[n_id] = np.arange(len(n_id))
        full = g.data[et]
        add = torch.tensor(node_map[full.edge_index[:, miss].numpy()], dtype=torch.long)
        node_map[n_id] = -1
        b[et].edge_index = torch.cat([b[et].edge_index, add], 1)
        b[et].edge_attr = torch.cat([b[et].edge_attr, full.edge_attr[miss]], 0)
        b[et].y = torch.cat([b[et].y, full.y[miss]])
        have = np.concatenate([have, miss])
    mask = np.isin(have, seed_ids)
    b[et].edge_attr = b[et].edge_attr[:, 1:]
    b[rt].edge_attr = b[rt].edge_attr[:, 1:]
    return torch.tensor(mask), have[mask]


def _run(model, g: Graph, device, train: bool, opt=None, loss_fn=None):
    """One pass over the seeds; returns (graph edge ids, probability of laundering)."""
    et = ("node", "to", "node")
    model.train(train)
    node_map = -np.ones(g.data["node"].x.shape[0], np.int64)
    out_ids, out_p = [], []
    with torch.set_grad_enabled(train):
        for b in loader(g, shuffle=train):
            mask, eid = _batch(b, g, g.seeds, node_map)
            b = b.to(device)
            mask = mask.to(device)
            out = model(b.x_dict, b.edge_index_dict, b.edge_attr_dict)[et][mask]
            if train:
                opt.zero_grad()
                loss = loss_fn(out, b[et].y[mask])
                loss.backward()
                opt.step()
            out_ids.append(eid)
            out_p.append(torch.softmax(out.detach(), -1)[:, 1].float().cpu().numpy())
    ids, p = np.concatenate(out_ids), np.concatenate(out_p)
    s = pd.Series(p).groupby(ids).max()
    return s.index.to_numpy(), s.to_numpy()


def score(model, g: Graph, device) -> pd.Series:
    """Laundering probability per transfer: the maximum over the transfer's seed edges (in views, either witness)."""
    ids, p = _run(model, g, device, train=False)
    return pd.Series(p, index=g.rec.tx[ids]).groupby(level=0).max()


def f1_at_half(y: np.ndarray, p: np.ndarray) -> float:
    pred = p > 0.5
    tp = (pred & (y == 1)).sum()
    return 2 * tp / max(1, pred.sum() + (y == 1).sum())


def train(kind: str, setting: str, variant: str, seed: int, epochs: int, device, log=print,
          drop_p: float = 0.1, omit_share: float = 0.2, ckpt: Path | None = None):
    """Train on the training period, select the epoch with the best validation F1 (reference protocol).
    variant: standard | dropedge (records dropped at random each epoch) | omission (each epoch a random share of
    banks deletes its laundering records: omission-aware adversarial training).
    ckpt: state after every epoch (model, optimiser, selection, random generators); a killed run resumes from it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    e = edges()
    part = split(e.t.to_numpy())
    rec = all_records()
    tr_rec = rec.take(part[rec.tx] == 0)
    va_rec = rec.take(part[rec.tx] <= 1)
    tr_tx, va_tx = np.flatnonzero(part == 0), np.flatnonzero(part == 1)
    y = e.y.to_numpy()
    g_val = build(va_rec, setting, va_tx)
    g_tr = build(tr_rec, setting, tr_tx)
    model = make_model(kind, g_tr).to(device)
    c = SETTINGS[kind]
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([c["w_ce1"], c["w_ce2"]], device=device))
    banks_l = np.unique(tr_rec.bank[y[tr_rec.tx] == 1])
    best, best_state, hist, start = -1.0, None, [], 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best, best_state, hist, start = st["best"], st["best_state"], st["hist"], st["epoch"] + 1
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if st["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(st["cuda_rng"], device)
        log(f"resumed after epoch {st['epoch']} (best val F1 {best:.4f})")
    for ep in range(start, epochs):
        t0 = time.time()
        if variant == "dropedge":
            g_tr = build(tr_rec.take(rng.random(len(tr_rec)) >= drop_p), setting, tr_tx)
        elif variant == "omission":
            bad = rng.choice(banks_l, max(1, int(omit_share * len(banks_l))), replace=False)
            g_tr = build(tr_rec.take(~(np.isin(tr_rec.bank, bad) & (y[tr_rec.tx] == 1))), setting, tr_tx)
        ids, p = _run(model, g_tr, device, train=True, opt=opt, loss_fn=loss_fn)
        tr_f1 = f1_at_half(y[g_tr.rec.tx[ids]], p)
        sv = score(model, g_val, device)
        va_f1 = f1_at_half(y[sv.index.to_numpy()], sv.to_numpy())
        hist.append(dict(epoch=ep, train_f1=tr_f1, val_f1=va_f1, seconds=time.time() - t0))
        log(f"epoch {ep} train F1 {tr_f1:.4f} val F1 {va_f1:.4f} ({time.time() - t0:.0f}s)")
        if va_f1 > best:
            best = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ckpt is not None:
            tmp = ckpt.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), best=best, best_state=best_state,
                            hist=hist, epoch=ep, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None), tmp)
            tmp.replace(ckpt)
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)
