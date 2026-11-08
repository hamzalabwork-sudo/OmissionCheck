"""X6: the omission check end to end on real ledgers.

A consortium of the B busiest IBM AML banks runs the protocol for several hourly epochs on one chain. Each epoch has
injected discrepancies (one bank omits transfers, another fabricates some). An epoch runs the four phases of the
contract: every bank commits its book root; the seed is taken from the block after the books barrier once that
block is settled; every bank commits to its masked sketch; then sketches are revealed (accumulate: added into
contract storage; publish: in calldata; commit: exchanged off chain, one member posts the sum). The phases of all
epochs of a run are processed together, each bank sending its transactions without waiting for the previous
receipt, so a run needs one settlement wait per phase rather than one per epoch (on the PoS devnet each wait is a
finality wait of 13-19 min). Optionally, the last epoch has a dropout: one bank commits but never reveals, and its
mask neighbours post the masks they shared with it after the reveal window. Gas, effective gas price and
send-to-receipt latency of every transaction are logged.

    python -m omission.chain --backend besu --epochs 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import solcx
from eth_utils import keccak, to_checksum_address
from web3 import Web3

from finverity import accounts
from finverity.backends.evm import Rpc, RpcError
from finverity.config import (BESU, BUILD, CONTRACTS, ETHPOS, EVM_VERSION, POA2_LOCAL, PURECHAIN, RESULTS,
                              SOLC_VERSION, ETH_USD, MAINNET_GAS_GWEI)

from . import data, protocol as P
from .iblt import Sketch

SOURCE = CONTRACTS / "OmissionOracle.sol"
ARTIFACT = BUILD / "OmissionOracle.json"
DEPLOYMENTS = BUILD / "omission_deployments.json"
OUT = RESULTS / "paper2" / "chain"
CHAINS = {"purechain": PURECHAIN, "besu": BESU, "poa2_local": POA2_LOCAL, "ethpos": ETHPOS}
# confirmations before state is read: Clique-style chains fork, and a read at the
# head can come from a node that has not seen every publication yet. The PoS devnet reads at the finalized block.
DEPTH = {"purechain": 8, "poa2_local": 8}   # Clique blocks can be replaced by reorganisations: read this deep
FINALIZED = {"ethpos"}
WINDOW = 30                     # blocks per phase window
ACC_GAS = 9_300_000             # accumulate: the first writer of an epoch pays 2m zero-to-nonzero stores
ACC_LATER_GAS = 2_800_000
_M64 = 2**64 - 1


def compile_oracle() -> dict:
    src = SOURCE.read_text()
    h = hashlib.sha256(src.encode()).hexdigest()
    if ARTIFACT.exists():
        art = json.loads(ARTIFACT.read_text())
        if art["source_sha256"] == h:
            return art
    out = solcx.compile_standard({
        "language": "Solidity", "sources": {"OmissionOracle.sol": {"content": src}},
        "settings": {"optimizer": {"enabled": True, "runs": 200}, "evmVersion": EVM_VERSION,
                     "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}}}, solc_version=SOLC_VERSION)
    c = out["contracts"]["OmissionOracle.sol"]["OmissionOracle"]
    art = {"source_sha256": h, "abi": c["abi"], "bytecode": "0x" + c["evm"]["bytecode"]["object"]}
    ARTIFACT.write_text(json.dumps(art))
    return art


# ------------------------------------------------------------------------------------------ word encoding
def to_words(s: Sketch) -> list[int]:
    out = []
    for i in range(s.m):
        k = [int(x) for x in s.keysum[i]]
        out.append((k[0] << 192) | (k[1] << 128) | (k[2] << 64) | k[3])
        out.append((int(s.count[i]) << 128) | (int(s.chk[i, 0]) << 64) | int(s.chk[i, 1]))
    return out


def from_words(w: list[int], seed: bytes) -> Sketch:
    m = len(w) // 2
    s = Sketch.empty(m, seed)
    for i in range(m):
        a, b = w[2 * i], w[2 * i + 1]
        s.keysum[i] = [(a >> 192) & _M64, (a >> 128) & _M64, (a >> 64) & _M64, a & _M64]
        s.count[i] = (b >> 128) & _M64
        s.chk[i] = [(b >> 64) & _M64, b & _M64]
    return s


def _hash(words: list[int]) -> bytes:
    return keccak(b"".join(w.to_bytes(32, "big") for w in words))


# ------------------------------------------------------------------------------------------------- chain
class Ledger:
    def __init__(self, backend: str):
        self.backend = backend
        self.cfg = CHAINS[backend]
        self.rpc = Rpc(self.cfg.rpc_url, timeout=60)
        self.w3 = Web3()
        self.art = compile_oracle()
        self.admin = accounts.admin()
        self.nonce: dict[str, int] = {}

    def gas_price(self) -> int:
        if self.cfg.gas_price >= 0:
            return self.cfg.gas_price
        return 2 * int(self.rpc.call("eth_gasPrice"), 16)

    def submit(self, acct, to: str | None, data: str, gas: int | None = None) -> tuple[str, float, str]:
        """Sign and send without waiting; nonces are tracked locally so one account can have several in flight."""
        if acct.address not in self.nonce:
            self.nonce[acct.address] = int(self.rpc.call("eth_getTransactionCount", [acct.address, "pending"]), 16)
        tx = {"from": acct.address, "data": data, **({"to": to} if to else {})}
        if gas is None:
            gas = int(int(self.rpc.call("eth_estimateGas", [tx]), 16) * 1.2) + 20_000
        tx = {"gasPrice": self.gas_price(), "chainId": self.cfg.chain_id, "nonce": self.nonce[acct.address],
              "gas": gas, "data": data, **({"to": to} if to else {})}
        raw = acct.sign_transaction(tx).rawTransaction
        t0 = time.time()
        txh = self.rpc.call("eth_sendRawTransaction", ["0x" + raw.hex().removeprefix("0x")])
        self.nonce[acct.address] += 1
        return txh, t0, data

    def wait(self, pending: list[tuple[str, float, str]]) -> list[dict]:
        """Receipts of submitted transactions, in order, with send-to-receipt latency (polled every 0.1 s)."""
        out: dict[str, dict] = {}
        t_start = time.time()
        while len(out) < len(pending):
            for txh, t0, data in pending:
                if txh in out:
                    continue
                rc = self.rpc.call("eth_getTransactionReceipt", [txh])
                if rc:
                    if int(rc["status"], 16) != 1:
                        raise RpcError(f"{txh} reverted")
                    rc["latency_s"] = time.time() - t0
                    raw = bytes.fromhex(data[2:])
                    rc["calldata_bytes"] = len(raw)
                    rc["calldata_zero"] = raw.count(0)   # EIP-7623 prices zero and non-zero bytes apart
                    out[txh] = rc
            if time.time() - t_start > 1200:
                raise TimeoutError([p[0] for p in pending if p[0] not in out][:3])
            time.sleep(0.1)
        return [out[p[0]] for p in pending]

    def send(self, acct, to, data, gas=None) -> dict:
        return self.wait([self.submit(acct, to, data, gas)])[0]

    def deploy(self, m: int, banks: list, window: int) -> str:
        deps = json.loads(DEPLOYMENTS.read_text()) if DEPLOYMENTS.exists() else {}
        key = f"{self.cfg.name}:{self.cfg.chain_id}:{self.art['source_sha256'][:16]}:m{m}:B{len(banks)}:w{window}"
        info = deps.get(key)
        if info and self.rpc.call("eth_getCode", [info["address"], "latest"]) not in ("0x", "0x0"):
            self.c = self.w3.eth.contract(address=info["address"], abi=self.art["abi"])
            return info["address"]
        f = self.w3.eth.contract(abi=self.art["abi"], bytecode=self.art["bytecode"])
        rc = self.send(self.admin, None, f.constructor(m, window).data_in_transaction)
        addr = to_checksum_address(rc["contractAddress"])
        self.c = self.w3.eth.contract(address=addr, abi=self.art["abi"])
        dh = [keccak(text=f"x25519-pub-{a.address}") for a in banks]  # placeholder agreement keys
        self.send(self.admin, addr, self.c.encodeABI(fn_name="addMembers", args=[[a.address for a in banks], dh]))
        deps[key] = {"address": addr, "deploy_gas": int(rc["gasUsed"], 16)}
        DEPLOYMENTS.write_text(json.dumps(deps, indent=1))
        return addr

    def head(self) -> int:
        return int(self.rpc.call("eth_blockNumber"), 16)

    def settled(self) -> int:
        """The newest block treated as final: the finalized block on PoS, else head minus the confirmation depth."""
        if self.backend in FINALIZED:
            return int(self.rpc.call("eth_getBlockByNumber", ["finalized", False])["number"], 16)
        return self.head() - DEPTH.get(self.backend, 0)

    def settle(self, block: int) -> tuple[str, float]:
        """Wait until `block` is settled; return (block to read state at, seconds waited)."""
        t0 = time.time()
        while self.settled() < block:
            time.sleep(1.0)
        return hex(block), time.time() - t0

    def settle_txs(self, txs: list[str]) -> tuple[str, float]:
        t0 = time.time()
        while True:
            rcs = [self.rpc.call("eth_getTransactionReceipt", [h]) for h in txs]
            if all(rcs):   # a receipt can vanish while its block is reorganised out
                tip = max(int(r["blockNumber"], 16) for r in rcs)
                if self.settled() >= tip:
                    return hex(tip), time.time() - t0
            time.sleep(1.0)

    def until_block(self, n: int) -> None:
        while self.head() < n:
            time.sleep(0.5)

    def call(self, fn: str, *args, block: str = "latest"):
        data = self.c.encodeABI(fn_name=fn, args=list(args))
        for attempt in range(20):   # a node behind the hub may not have `block` yet
            try:
                out = self.rpc.call("eth_call", [{"to": self.c.address, "data": data}, block])
                break
            except RpcError:
                if attempt == 19:
                    raise
                time.sleep(1)
        abi = next(x for x in self.art["abi"] if x.get("name") == fn)
        return self.w3.codec.decode([o["type"] for o in abi["outputs"]], bytes.fromhex(out[2:]))


def _calldata_words(ledger: Ledger, txh: str, types: list[str]) -> list[int]:
    tx = ledger.rpc.call("eth_getTransactionByHash", [txh])
    args = ledger.w3.codec.decode(types, bytes.fromhex(tx["input"][2:])[4:])
    return list(next(a for a in args if isinstance(a, (list, tuple))))


def run(backend: str, epochs: int, n_banks: int, m: int, mode: str, omit_k: int, fab_k: int, seed: int,
        dropout: bool = False, tag: str = "") -> None:
    bank_ids = list(data.ranking()[:n_banks])
    sel, keys = data.covered_keys(n_banks)
    books = P.honest_books(sel, keys)
    banks_sorted = np.array(sorted(bank_ids), dtype=np.int64)
    accts = {int(b): (accounts.derive("bank", i) if backend == "purechain" else accounts.derive("submitter", i))
             for i, b in enumerate(banks_sorted)}
    ledger = Ledger(backend)
    window = WINDOW
    addr = ledger.deploy(m, list(accts.values()), window)
    rng = np.random.default_rng(seed)
    busy = pd.Series(books.epoch).value_counts()
    chosen = sorted(rng.choice(busy.index[:max(epochs * 3, 30)], epochs, replace=False).tolist())
    base = int(time.time())  # contract epochs are unique per run (sketches cannot be re-published)
    kappa = 2 * n_banks      # full mesh of masks at this size
    rows, ep = [], []

    def log(phase, j, bank, rc):
        rows.append(dict(backend=backend, mode=mode, m=m, banks=n_banks, epoch=ep[j]["epoch"],
                         contract_epoch=ep[j]["cid"], bank=bank, phase=phase, gas=int(rc["gasUsed"], 16),
                         gas_price=int(rc.get("effectiveGasPrice", "0x0"), 16), latency_s=rc["latency_s"],
                         block=int(rc["blockNumber"], 16), calldata_bytes=rc["calldata_bytes"],
                         calldata_zero=rc["calldata_zero"]))

    def phase(name, items):
        """items: (epoch index, bank, account, calldata, gas); returns receipts and the phase wall time."""
        t0 = time.time()
        pend = [ledger.submit(acct, addr, d, g) for _, _, acct, d, g in items]
        rcs = ledger.wait(pend)
        for (j, bank, *_), rc in zip(items, rcs):
            log(name, j, bank, rc)
        return rcs, time.time() - t0

    # books, discrepancies and book roots
    for j, e in enumerate(chosen):
        b = books.take(books.epoch == e)
        # the first bank omits omit_k of its rows, the second fabricates fab_k rows
        victims = [int(x) for x in pd.Series(b.bank).value_counts().index[:2]]
        drop = np.zeros(len(b), bool)
        mine = np.flatnonzero(b.bank == victims[0])
        drop[rng.choice(mine, min(omit_k, len(mine)), replace=False)] = True
        fab = b.take(rng.choice(np.flatnonzero(b.bank == victims[1]), fab_k, replace=False))
        payer = np.where(fab.sign > 0, fab.bank, fab.cpty)
        payee = np.where(fab.sign > 0, fab.cpty, fab.bank)
        fab.keys = data.fabricate_keys(payer, payee, f"chain-{seed}-{e}")
        fab.tx = -np.ones(len(fab), np.int64)
        tampered = P.Books.concat([b.take(~drop), fab])
        absent = int(banks_sorted[-1]) if dropout and j == len(chosen) - 1 else None
        roots = {int(k): P.merkle_root(P.book_leaves(tampered.take(tampered.bank == k), f"salt-{k}".encode()))
                 for k in banks_sorted}
        ep.append(dict(epoch=int(e), cid=base + j, books=tampered, rows=len(b), roots=roots, absent=absent))

    # phase 1: book roots
    t_run = time.time()
    rcs, t_books = phase("books", [(j, int(k), accts[int(k)],
                                    ledger.c.encodeABI(fn_name="commitBooks", args=[x["cid"], x["roots"][int(k)]]),
                                    None)
                                   for j, x in enumerate(ep) for k in banks_sorted])
    # phase 2: the barrier is read at a settled block, and the seed block (the first block after it) must be
    # settled too before anyone uses it
    t0 = time.time()
    at, _ = ledger.settle_txs([rc["transactionHash"] for rc in rcs])
    done = [int(ledger.call("epochs", x["cid"], block=at)[1]) for x in ep]
    _, _ = ledger.settle(max(done) + 1)
    t_seed_wait = time.time() - t0
    for x, d in zip(ep, done):
        x["books_done"] = d
        bh = bytes.fromhex(ledger.rpc.call("eth_getBlockByNumber", [hex(d + 1), False])["hash"][2:])
        x["seed"] = keccak(bh + x["cid"].to_bytes(32, "big"))
    t_seed = time.time() - t0
    # phase 3: masked sketches under the epoch seed, then their commitments
    for x in ep:
        flat = P.bank_sketches(x["books"], banks_sorted, m, x["seed"])
        P.apply_masks(flat, banks_sorted, m, x["cid"], kappa, x["seed"])
        x["flat"] = flat
        x["words"] = {int(k): to_words(Sketch(flat.count[i * m:(i + 1) * m].copy(),
                                              flat.keysum[i * m:(i + 1) * m].copy(),
                                              flat.chk[i * m:(i + 1) * m].copy(), x["seed"]))
                      for i, k in enumerate(banks_sorted)}
    _, t_sketch = phase("sketch", [(j, int(k), accts[int(k)],
                                    ledger.c.encodeABI(fn_name="commitSketch", args=[x["cid"], _hash(x["words"][int(k)])]),
                                    None)
                                   for j, x in enumerate(ep) for k in banks_sorted])
    # phase 4: reveal (accumulate / publish) or post the off-chain sum (commit)
    reveal_tx: dict[tuple[int, int], str] = {}
    if mode in ("accumulate", "publish"):
        fn = "accumulate" if mode == "accumulate" else "reveal"
        items = [(j, int(k), accts[int(k)], ledger.c.encodeABI(fn_name=fn, args=[x["cid"], x["words"][int(k)]]), None)
                 for j, x in enumerate(ep) for k in banks_sorted if int(k) != x["absent"]]
        if mode == "accumulate":
            # the first sketch of an epoch pays 2m zero-to-nonzero stores; the rest are estimated after it landed
            first = [i for i, it in enumerate(items) if i == 0 or items[i - 1][0] != it[0]]
            r1, t1 = phase("reveal", [(*items[i][:4], ACC_GAS) for i in first])
            # later sketches add into warm storage (about 2.3M gas); a fixed limit avoids one slow gas estimate per
            # transaction, which on Besu can outlast the reveal window
            r2, t2 = phase("reveal", [(*it[:4], ACC_LATER_GAS) for i, it in enumerate(items) if i not in first])
            order = first + [i for i in range(len(items)) if i not in first]
            rcs, t_reveal = [None] * len(items), t1 + t2
            for i, rc in zip(order, r1 + r2):
                rcs[i] = rc
        else:
            rcs, t_reveal = phase("reveal", items)
        for (j, k, *_), rc in zip(items, rcs):
            reveal_tx[(j, k)] = rc["transactionHash"]
    else:
        t0 = time.time()
        for j, x in enumerate(ep):   # each sketch is checked against its commitment before it enters the sum
            agg = Sketch.empty(m, x["seed"])
            for k in banks_sorted:
                if int(k) == x["absent"]:
                    continue
                assert ledger.call("sketchHash", x["cid"], accts[int(k)].address)[0] == _hash(x["words"][int(k)])
                agg = agg + from_words(x["words"][int(k)], x["seed"])
            x["agg_offchain"] = agg
        t_reveal = time.time() - t0
    # dropout: after the reveal window, the absent bank's mask neighbours post the masks they shared with it
    t_dropout = 0.0
    mask_tx: dict[int, list[str]] = {}
    for j, x in enumerate(ep):
        if x["absent"] is None:
            continue
        t0 = time.time()
        sd = int(ledger.call("epochs", x["cid"])[2])
        ledger.until_block(sd + window + 1)
        pos = int(np.searchsorted(banks_sorted, x["absent"]))
        items = []
        for lo, hi, mk in P.masks(banks_sorted, m, x["cid"], kappa, x["seed"]):
            if pos not in (lo, hi):
                continue
            other = hi if lo == pos else lo
            fix = -mk if hi == pos else mk    # cancels the mask the present side added
            fix.seed = x["seed"]
            w = to_words(fix)
            if mode == "commit":
                x["agg_offchain"] = x["agg_offchain"] + from_words(w, x["seed"])
                continue
            k = int(banks_sorted[other])
            items.append((j, k, accts[k], ledger.c.encodeABI(
                fn_name="revealMask", args=[x["cid"], accts[x["absent"]].address, w, mode == "accumulate"]),
                ACC_GAS if mode == "accumulate" else None))
        if items:
            rcs, _ = phase("mask", items)
            mask_tx[j] = [rc["transactionHash"] for rc in rcs]
        t_dropout = time.time() - t0
    if mode == "commit":
        items = [(j, int(banks_sorted[0]), accts[int(banks_sorted[0])],
                  ledger.c.encodeABI(fn_name="postAggregate", args=[x["cid"], to_words(x["agg_offchain"])]), None)
                 for j, x in enumerate(ep)]
        _, t_post = phase("aggregate", items)
        t_reveal += t_post
    # settle the last transaction, read the sums back, decode
    at, t_final = ledger.settle(max(r["block"] for r in rows))
    t_total = time.time() - t_run
    ep_rows = []
    for j, x in enumerate(ep):
        t0 = time.time()
        if mode == "accumulate":
            agg = from_words(list(ledger.call("aggregate", x["cid"], block=at)[0]), x["seed"])
        elif mode == "publish":
            agg = Sketch.empty(m, x["seed"])
            for (jj, k), h in reveal_tx.items():
                if jj == j:
                    agg = agg + from_words(_calldata_words(ledger, h, ["uint256", "uint256[]"]), x["seed"])
            for h in mask_tx.get(j, []):
                agg = agg + from_words(_calldata_words(ledger, h, ["uint256", "address", "uint256[]", "bool"]),
                                       x["seed"])
        else:
            agg = x["agg_offchain"]
            assert ledger.call("aggregateHash", x["cid"], block=at)[0] == _hash(to_words(agg))
        st = ledger.call("epochs", x["cid"], block=at)
        complete, k, s = agg.decode()
        t_read = time.time() - t0
        b = x["books"] if x["absent"] is None else x["books"].take(x["books"].bank != x["absent"])
        tr = P.truth(b)
        dec = {tuple(y.tolist()): int(v) for y, v in zip(k, s)}
        found = sum(1 for kk, v in dec.items() if tr.get(kk) == v)
        ep_rows.append(dict(backend=backend, mode=mode, m=m, banks=n_banks, epoch=x["epoch"], rows=x["rows"],
                            residual=len(tr), found=found, spurious=len(dec) - found, complete=complete,
                            dropout=x["absent"] is not None, seed_matches=bytes(st[6]) == x["seed"],
                            books_done=x["books_done"], sketches_done=int(st[2]), reveals=int(st[5]),
                            read_decode_s=t_read))
        print(ep_rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{backend}_{mode}_m{m}_B{n_banks}{tag}"
    pd.DataFrame(rows).to_csv(OUT / f"{stem}_tx.csv", index=False)
    pd.DataFrame(ep_rows).to_csv(OUT / f"{stem}_epochs.csv", index=False)
    meta = dict(backend=backend, chain_id=ledger.cfg.chain_id, contract=addr, mode=mode, m=m, banks=n_banks,
                epochs=epochs, omit_k=omit_k, fab_k=fab_k, seed=seed, dropout=dropout, window=window,
                confirm_depth=DEPTH.get(backend, 0), finalized=backend in FINALIZED,
                phase_s=dict(books=t_books, seed_settle=t_seed_wait, seed=t_seed, sketch=t_sketch, reveal=t_reveal,
                             dropout=t_dropout, final_settle=t_final, total=t_total),
                gas_price_wei=int(np.median([r["gas_price"] for r in rows])),
                mainnet_gas_gwei=MAINNET_GAS_GWEI, eth_usd=ETH_USD, client=ledger.rpc.call("web3_clientVersion"))
    (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["phase_s"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(CHAINS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--banks", type=int, default=16)
    ap.add_argument("--m", type=int, default=192)
    ap.add_argument("--mode", default="accumulate", choices=["accumulate", "publish", "commit"])
    ap.add_argument("--omit", type=int, default=5)
    ap.add_argument("--fabricate", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dropout", action="store_true", help="last epoch: one bank commits but never reveals")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.backend, a.epochs, a.banks, a.m, a.mode, a.omit, a.fabricate, a.seed, a.dropout, a.tag)


if __name__ == "__main__":
    main()
