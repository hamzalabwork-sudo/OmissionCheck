"""X6: the omission check end to end on real ledgers.

A consortium of the B busiest IBM AML banks runs the protocol for several hourly epochs on one chain. Each epoch has
injected discrepancies (one bank omits transfers, another fabricates some). Every bank publishes its masked sketch
in its own transaction; afterwards the sum is read back (accumulate mode: from contract storage; publish mode: from
the transactions' calldata), decoded, and compared with the ground truth. Gas and send-to-receipt latency of every
transaction are logged.

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
from .iblt import WORDS, Sketch

SOURCE = CONTRACTS / "OmissionOracle.sol"
ARTIFACT = BUILD / "OmissionOracle.json"
DEPLOYMENTS = BUILD / "omission_deployments.json"
OUT = RESULTS / "paper2" / "chain"
CHAINS = {"purechain": PURECHAIN, "besu": BESU, "poa2_local": POA2_LOCAL, "ethpos": ETHPOS}
# confirmations before the sum is read: Clique-style chains fork (live PureChain reached depth 6), and a read at the
# head can come from a node that has not seen every publication yet
DEPTH = {"purechain": 7, "poa2_local": 7}
_SH = [np.uint64(192), np.uint64(128), np.uint64(64)]


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
        out.append((int(s.count[i]) << 64) | int(s.chk[i]))
    return out


def from_words(w: list[int]) -> Sketch:
    m = len(w) // 2
    s = Sketch.empty(m)
    for i in range(m):
        a, b = w[2 * i], w[2 * i + 1]
        s.keysum[i] = [(a >> 192) & 2**64 - 1, (a >> 128) & 2**64 - 1, (a >> 64) & 2**64 - 1, a & 2**64 - 1]
        s.count[i] = (b >> 64) & 2**64 - 1
        s.chk[i] = b & 2**64 - 1
    return s


# ------------------------------------------------------------------------------------------------- chain
class Ledger:
    def __init__(self, backend: str):
        self.cfg = CHAINS[backend]
        self.rpc = Rpc(self.cfg.rpc_url, timeout=60)
        self.w3 = Web3()
        self.art = compile_oracle()
        self.admin = accounts.admin()

    def gas_price(self) -> int:
        if self.cfg.gas_price >= 0:
            return self.cfg.gas_price
        return 2 * int(self.rpc.call("eth_gasPrice"), 16)

    def send(self, acct, to: str | None, data: str, gas: int | None = None) -> dict:
        nonce = int(self.rpc.call("eth_getTransactionCount", [acct.address, "pending"]), 16)
        tx = {"from": acct.address, "data": data, **({"to": to} if to else {})}
        if gas is None:
            gas = int(int(self.rpc.call("eth_estimateGas", [tx]), 16) * 1.2) + 20_000
        tx = {"gasPrice": self.gas_price(), "chainId": self.cfg.chain_id, "nonce": nonce, "gas": gas, "data": data,
              **({"to": to} if to else {})}
        raw = acct.sign_transaction(tx).rawTransaction
        t0 = time.time()
        txh = self.rpc.call("eth_sendRawTransaction", ["0x" + raw.hex().removeprefix("0x")])
        while True:
            rc = self.rpc.call("eth_getTransactionReceipt", [txh])
            if rc:
                break
            if time.time() - t0 > 600:
                raise TimeoutError(txh)
            time.sleep(0.1)
        if int(rc["status"], 16) != 1:
            raise RpcError(f"{txh} reverted")
        rc["latency_s"] = time.time() - t0
        rc["calldata_bytes"] = (len(data) - 2) // 2
        return rc

    def deploy(self, m: int, banks: list) -> str:
        deps = json.loads(DEPLOYMENTS.read_text()) if DEPLOYMENTS.exists() else {}
        key = f"{self.cfg.name}:{self.cfg.chain_id}:{self.art['source_sha256'][:16]}:m{m}:B{len(banks)}"
        info = deps.get(key)
        if info and self.rpc.call("eth_getCode", [info["address"], "latest"]) not in ("0x", "0x0"):
            self.c = self.w3.eth.contract(address=info["address"], abi=self.art["abi"])
            return info["address"]
        f = self.w3.eth.contract(abi=self.art["abi"], bytecode=self.art["bytecode"])
        rc = self.send(self.admin, None, f.constructor(m).data_in_transaction)
        addr = to_checksum_address(rc["contractAddress"])
        self.c = self.w3.eth.contract(address=addr, abi=self.art["abi"])
        dh = [keccak(text=f"x25519-pub-{a.address}") for a in banks]  # placeholder agreement keys
        self.send(self.admin, addr, self.c.encodeABI(fn_name="addMembers", args=[[a.address for a in banks], dh]))
        deps[key] = {"address": addr, "deploy_gas": int(rc["gasUsed"], 16)}
        DEPLOYMENTS.write_text(json.dumps(deps, indent=1))
        return addr

    def settle(self, txs: list[str], depth: int) -> str:
        """Wait until every transaction is `depth` blocks deep; return the block to read state at."""
        if depth == 0:
            return "latest"
        while True:
            rcs = [self.rpc.call("eth_getTransactionReceipt", [h]) for h in txs]
            if all(rcs):   # a receipt can vanish while its block is reorganised out
                tip = max(int(r["blockNumber"], 16) for r in rcs) + depth
                if int(self.rpc.call("eth_blockNumber"), 16) >= tip:
                    return hex(tip)
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


def _sum_calldata(ledger: Ledger, txs: list[str], m: int) -> Sketch:
    agg = Sketch.empty(m)
    for h in txs:
        tx = ledger.rpc.call("eth_getTransactionByHash", [h])
        raw = bytes.fromhex(tx["input"][2:])
        _, sk = ledger.w3.codec.decode(["uint256", "uint256[]"], raw[4:])
        agg = agg + from_words(list(sk))
    return agg


def run(backend: str, epochs: int, n_banks: int, m: int, mode: str, omit_k: int, fab_k: int, seed: int,
        tag: str = "") -> None:
    bank_ids = list(data.ranking()[:n_banks])
    sel, keys = data.covered_keys(n_banks)
    books = P.honest_books(sel, keys)
    banks_sorted = np.array(sorted(bank_ids), dtype=np.int64)
    accts = {int(b): (accounts.derive("bank", i) if backend == "purechain" else accounts.derive("submitter", i))
             for i, b in enumerate(banks_sorted)}
    ledger = Ledger(backend)
    addr = ledger.deploy(m, list(accts.values()))
    rng = np.random.default_rng(seed)
    busy = pd.Series(books.epoch).value_counts()
    chosen = sorted(rng.choice(busy.index[:max(epochs * 3, 30)], epochs, replace=False).tolist())
    base = int(time.time())  # contract epochs are unique per run (sketches cannot be re-published)
    rows, ep_rows = [], []
    for j, e in enumerate(chosen):
        b = books.take(books.epoch == e)
        # discrepancies: the first bank omits omit_k of its rows, the second fabricates fab_k rows
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
        flat = P.bank_sketches(tampered, banks_sorted, m)
        P.apply_masks(flat, banks_sorted, m, e, kappa=2 * n_banks)   # full mesh of masks
        cid = base + j
        txs, sketches = [], []
        t_ep = time.time()
        for i, bank in enumerate(banks_sorted):
            s = Sketch(flat.count[i * m:(i + 1) * m].copy(), flat.keysum[i * m:(i + 1) * m].copy(),
                       flat.chk[i * m:(i + 1) * m].copy())
            words = to_words(s)
            sketches.append(words)
            args = [cid, keccak(b"".join(w.to_bytes(32, "big") for w in words))] if mode == "commit" else [cid, words]
            rc = ledger.send(accts[int(bank)], addr, ledger.c.encodeABI(fn_name=mode, args=args))
            txs.append(rc["transactionHash"])
            rows.append(dict(backend=backend, mode=mode, m=m, banks=n_banks, epoch=e, contract_epoch=cid, bank=int(bank),
                             order=i, gas=int(rc["gasUsed"], 16), latency_s=rc["latency_s"],
                             block=int(rc["blockNumber"], 16), calldata_bytes=rc["calldata_bytes"]))
        t_pub = time.time() - t_ep
        t0 = time.time()
        at = ledger.settle(txs, DEPTH.get(backend, 0))
        t_conf = time.time() - t0
        t0 = time.time()
        if mode == "accumulate":
            agg = from_words(list(ledger.call("aggregate", cid, block=at)[0]))
        elif mode == "publish":
            agg = _sum_calldata(ledger, txs, m)
        else:  # commit: sum the off-chain sketches after checking them against the committed hashes, post the sum
            agg = Sketch.empty(m)
            for bank, words in zip(banks_sorted, sketches):
                h = ledger.call("sketchHash", cid, accts[int(bank)].address, block=at)[0]
                assert h == keccak(b"".join(w.to_bytes(32, "big") for w in words))
                agg = agg + from_words(words)
            rc = ledger.send(accts[int(banks_sorted[0])], addr,
                             ledger.c.encodeABI(fn_name="postAggregate", args=[cid, to_words(agg)]))
            rows.append(dict(backend=backend, mode="commit-aggregate", m=m, banks=n_banks, epoch=e, contract_epoch=cid,
                             bank=int(banks_sorted[0]), order=-1, gas=int(rc["gasUsed"], 16),
                             latency_s=rc["latency_s"], block=int(rc["blockNumber"], 16),
                             calldata_bytes=rc["calldata_bytes"]))
        complete, k, s = agg.decode()
        t_read = time.time() - t0
        tr = P.truth(tampered)
        dec = {tuple(x.tolist()): int(y) for x, y in zip(k, s)}
        found = sum(1 for kk, v in dec.items() if tr.get(kk) == v)
        ep_rows.append(dict(backend=backend, mode=mode, m=m, banks=n_banks, epoch=e, rows=len(b), residual=len(tr),
                            found=found, spurious=len(dec) - found, complete=complete, publish_s=t_pub,
                            confirm_depth=DEPTH.get(backend, 0), confirm_s=t_conf, read_decode_s=t_read))
        print(ep_rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{backend}_{mode}_m{m}_B{n_banks}{tag}"
    pd.DataFrame(rows).to_csv(OUT / f"{stem}_tx.csv", index=False)
    pd.DataFrame(ep_rows).to_csv(OUT / f"{stem}_epochs.csv", index=False)
    meta = dict(backend=backend, chain_id=ledger.cfg.chain_id, contract=addr, mode=mode, m=m, banks=n_banks,
                omit_k=omit_k, fab_k=fab_k, seed=seed, mainnet_gas_gwei=MAINNET_GAS_GWEI, eth_usd=ETH_USD,
                client=ledger.rpc.call("web3_clientVersion"))
    (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))


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
    a = ap.parse_args()
    run(a.backend, a.epochs, a.banks, a.m, a.mode, a.omit, a.fabricate, a.seed)


if __name__ == "__main__":
    main()
