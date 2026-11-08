"""IBM AML HI-Small as an interbank ledger: consortium membership, epochs and transfer tags.

Every transfer between two consortium banks has two independent witnesses, the payer bank and the payee bank. Each
witness derives the same 256-bit key for it: word 0 packs the payer and payee bank identifiers, words 1-3 carry a
192-bit tag, a keyed BLAKE2b MAC of the canonical transfer under the pair's shared key. The shared key stands in for
the X25519 agreement of the deployed protocol (bench.py times the real primitive). Independent sub-keys are derived
from it per purpose (MAC, masks), so the tag key never keys a mask stream.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from finverity.config import DATA_PROCESSED, DATA_RAW

RAW = DATA_RAW / "ibm_aml" / "HI-Small_Trans.csv"
CACHE = DATA_PROCESSED / "ibm_full.parquet"
EPOCH_MIN = 60
MASTER = b"finverity-omission-pairkeys-v1"
CANON = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Payment Currency",
         "Amount Received", "Receiving Currency", "Payment Format"]


def usd_rates(df: pd.DataFrame) -> dict[str, float]:
    """US-dollar value of one unit of each currency, implied by the dataset's own cross-currency transfers (median of
    amount received / amount paid over transfers with exactly one US-dollar side)."""
    x = df[(df["Payment Currency"] != df["Receiving Currency"])]
    a = x[x["Receiving Currency"] == "US Dollar"]
    b = x[x["Payment Currency"] == "US Dollar"]
    r = pd.concat([(a["Amount Received"] / a["Amount Paid"]).groupby(a["Payment Currency"]).median(),
                   (b["Amount Paid"] / b["Amount Received"]).groupby(b["Receiving Currency"]).median()])
    out = r.groupby(level=0).median().to_dict()
    out["US Dollar"] = 1.0
    return out


COLUMNS = ["minute", "epoch", "payer", "payee", "amount", "usd", "laundering", "format"]


@lru_cache(maxsize=1)
def transfers() -> pd.DataFrame:
    """All 5.08M transfers with an epoch index, a laundering flag and the US-dollar value. The canonical strings
    (canon(), with a duplicate rank for rows that are otherwise equal) are cached on disk but loaded only to derive
    tags: they take GBs in memory."""
    if CACHE.exists():
        return pd.read_parquet(CACHE, columns=COLUMNS)
    df = pd.read_csv(RAW, dtype={"Account": str, "Account.1": str})
    rates = usd_rates(df)
    df["usd"] = df["Amount Paid"] * df["Payment Currency"].map(rates)
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["minute"] = ((ts - ts.min()).dt.total_seconds() // 60).astype(np.int32)
    df["epoch"] = (df["minute"] // EPOCH_MIN).astype(np.int32)
    df["payer"] = df["From Bank"].astype(np.int64)
    df["payee"] = df["To Bank"].astype(np.int64)
    df["laundering"] = df["Is Laundering"].astype(bool)
    df["amount"] = df["Amount Paid"].astype(float)
    df["dup"] = df.groupby(CANON, sort=False).cumcount().astype(np.int16)
    canon = df["dup"].astype(str)
    for c in reversed(CANON):
        canon = df[c].astype(str) + "|" + canon
    df["canon"] = canon
    out = df[["minute", "epoch", "payer", "payee", "amount", "usd", "laundering", "Payment Format", "canon"]]
    out = out.rename(columns={"Payment Format": "format"}).reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out[COLUMNS]


def canon(rows: np.ndarray) -> list[str]:
    """Canonical strings of the given transfers (row indices of transfers())."""
    transfers()
    return pd.read_parquet(CACHE, columns=["canon"])["canon"].to_numpy()[rows].tolist()


def ranking() -> pd.Index:
    """Banks ordered by the number of transfers they take part in (either side)."""
    t = transfers()
    return pd.concat([t["payer"], t["payee"]]).value_counts().index


def members(n: int | None) -> set[int]:
    """The consortium: the n busiest banks (None = every bank)."""
    r = ranking()
    return set(r if n is None else r[:n])


def covered(t: pd.DataFrame, mem: set[int]) -> np.ndarray:
    """Transfers with two distinct consortium witnesses."""
    return (t["payer"] != t["payee"]).to_numpy() & t["payer"].isin(mem).to_numpy() & t["payee"].isin(mem).to_numpy()


def pair_key(a: int, b: int) -> bytes:
    lo, hi = (a, b) if a < b else (b, a)
    return hashlib.blake2b(f"{lo}-{hi}".encode(), key=MASTER, digest_size=32).digest()


def subkey(k: bytes, label: str) -> bytes:
    """Purpose-bound sub-key of a pair key (HKDF-Expand with a label in the deployed protocol)."""
    return hashlib.blake2b(label.encode(), key=k, digest_size=32).digest()


def tag_keys(payer: np.ndarray, payee: np.ndarray, canon: list[str]) -> np.ndarray:
    """(n, 4) uint64 keys: [payer<<32 | payee, 192-bit MAC of the canonical transfer under the pair key]."""
    keys = np.empty((len(canon), 4), dtype=np.uint64)
    keys[:, 0] = (payer.astype(np.uint64) << np.uint64(32)) | payee.astype(np.uint64)
    pk: dict[tuple[int, int], bytes] = {}
    buf = bytearray(24 * len(canon))
    for i, (a, b, c) in enumerate(zip(payer.tolist(), payee.tolist(), canon)):
        k = pk.get((a, b))
        if k is None:
            k = pk[(a, b)] = subkey(pair_key(a, b), "mac")
        buf[24 * i:24 * i + 24] = hashlib.blake2b(c.encode(), key=k, digest_size=24).digest()
    keys[:, 1:] = np.frombuffer(bytes(buf), dtype=np.uint64).reshape(-1, 3)
    return keys


def covered_keys(n_members: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Covered transfers of a consortium and their keys (cached per consortium size)."""
    t = transfers()
    mem = members(n_members)
    cov = covered(t, mem)
    sel = t[cov].reset_index(drop=True)
    path = DATA_PROCESSED / f"ibm_keys_v2_N{n_members or 'all'}.npy"
    if path.exists():
        keys = np.load(path)
        if len(keys) == len(sel):
            return sel, keys
    keys = tag_keys(sel["payer"].to_numpy(), sel["payee"].to_numpy(), canon(np.flatnonzero(cov)))
    np.save(path, keys)
    return sel, keys


def fabricate_keys(payer: np.ndarray, payee: np.ndarray, salt: str) -> np.ndarray:
    """Keys of transfers that never happened (an attacker's fabricated entries); unique per salt and row."""
    canon = [f"fabricated|{salt}|{i}" for i in range(len(payer))]
    return tag_keys(payer, payee, canon)
