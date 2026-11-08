"""Local PureChain network: the released PureChain client with its own on-demand sealing policy.

Runs the published purechain-geth v1.0.1 binary (github.com/gieworld/purechain_testnet, SHA-256 checked against the
release's SHA256SUMS.txt) as N Clique signers with the public network's parameters (1 s period, 30M gas limit, zero
base fee, Shanghai/Cancun from genesis). Validators are started without --mine: each one runs the client's own Smart
Auto-Mining v2 policy (rehearsal/scripts/autoMine-v2.js, unmodified), which starts the miner while transactions are
pending or a block within the last 15 s carried transactions, and stops it otherwise. This gives controlled
experiments with the real demand-driven sealing and tail that the live network shows.

    python -m infra.purechain_local up [n] | down
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import requests
from eth_account import Account

from finverity.config import PURECHAIN_LOCAL, ROOT

IMAGE = "ethereum/client-go:v1.13.15"          # alpine userland only; the PureChain binary is mounted in
BIN_DIR = ROOT / "build" / "purechain"
BINARY = "purechain-geth-v1.0.1-linux-amd64"
SHA256 = "a251a17ca6d716d7518c258480c4a7a5380b1003cc4291b68be86d387265db25"
SAM = "autoMine-v2.js"
DIR = ROOT / "infra" / "purechain_network"
NET = "finverity-pcl"
SUBNET = "172.31.0"
N_SIGNERS = 4
CPUS = os.environ.get("PCL_CPUS_PER_VALIDATOR", "1.0")
NAMES = [f"pcl-signer{i}" for i in range(1, 11)]
# EIP-4788 beacon-roots contract, deployed at genesis on Cancun Clique chains (network/genesis.json of the client)
BEACON_ROOTS = {"0x000F3df6D732807Ef1319fB7B8bB8522d0Beac02": {"balance": "0", "code": (
    "0x3373fffffffffffffffffffffffffffffffffffffffe14604d57602036146024575f5ffd5b5f35801560495762001fff810690815414"
    "603c575f5ffd5b62001fff01545f5260205ff35b5f5ffd5b62001fff42064281555f359062001fff015500")}}


def _docker(*args, check=True):
    r = subprocess.run(["docker", *args], text=True, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])}: {r.stderr[-600:]}")
    return r


def _check_binary() -> None:
    b = BIN_DIR / BINARY
    if not b.exists() or hashlib.sha256(b.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(f"{b} missing or not the released v1.0.1 binary (sha256 mismatch)")


def down() -> None:
    _docker("rm", "-f", *NAMES, check=False)


def names(n: int) -> list[str]:
    return NAMES[:n]


GENESIS_LEAD_S = 45      # genesis timestamp this far in the future: nothing can be sealed during the warm start


def _genesis(signers: list[str], ts: int) -> dict:
    from finverity import accounts
    extra = "0x" + "00" * 32 + "".join(a[2:].lower() for a in signers) + "00" * 65
    alloc = {a.address[2:].lower(): {"balance": hex(10**24)} for a in [accounts.admin(), *accounts.roles(16).submitters]}
    return {
        "config": {"chainId": PURECHAIN_LOCAL.chain_id, "homesteadBlock": 0, "eip150Block": 0, "eip155Block": 0,
                   "eip158Block": 0, "byzantiumBlock": 0, "constantinopleBlock": 0, "petersburgBlock": 0,
                   "istanbulBlock": 0, "berlinBlock": 0, "londonBlock": 0, "shanghaiTime": 0, "cancunTime": 0,
                   "zeroBaseFee": True, "clique": {"period": 1, "epoch": 30000}},
        "timestamp": hex(ts), "difficulty": "0x1", "gasLimit": hex(30_000_000), "extradata": extra, "alloc": {**alloc, **BEACON_ROOTS},
    }


def up(n: int = N_SIGNERS, attempts: int = 3) -> dict:
    _check_binary()
    down()
    if DIR.exists():
        shutil.rmtree(DIR)
    DIR.mkdir(parents=True)
    keys = [Account.create() for _ in range(n)]
    node_keys = [os.urandom(32).hex() for _ in range(n)]
    ts = int(time.time()) + GENESIS_LEAD_S
    (DIR / "genesis.json").write_text(json.dumps(_genesis([k.address for k in keys], ts), indent=1))
    (DIR / "password.txt").write_text("finverity")
    if _docker("network", "inspect", NET, check=False).returncode != 0:
        _docker("network", "create", "--subnet", f"{SUBNET}.0/24", NET)
    from eth_keys import keys as ekeys
    enodes, mounts = [], ["-v", f"{BIN_DIR}:/pc:ro"]
    for i, (k, nk) in enumerate(zip(keys, node_keys), start=1):
        d = DIR / f"signer{i}"
        (d / "keystore").mkdir(parents=True)
        (d / "keystore" / f"key-{k.address}.json").write_text(json.dumps(Account.encrypt(k.key, "finverity")))
        pub = ekeys.PrivateKey(bytes.fromhex(nk)).public_key.to_hex()[2:]
        enodes.append(f"enode://{pub}@{SUBNET}.{10 + i}:30303")
        _docker("run", "--rm", *mounts, "-v", f"{d}:/data", "-v", f"{DIR / 'genesis.json'}:/genesis.json:ro",
                "--entrypoint", f"/pc/{BINARY}", IMAGE, "--datadir=/data", "init", "/genesis.json")
    for i, (k, nk) in enumerate(zip(keys, node_keys), start=1):
        d = DIR / f"signer{i}"
        geth = " ".join([
            f"/pc/{BINARY}", "--datadir=/data", f"--networkid={PURECHAIN_LOCAL.chain_id}", f"--nodekeyhex={nk}",
            "--port=30303", f"--bootnodes={','.join(e for j, e in enumerate(enodes) if j != i - 1)}",
            f"--netrestrict={SUBNET}.0/24", "--syncmode=full", f"--miner.etherbase={k.address}",
            f"--unlock={k.address}", "--password=/password.txt", "--allow-insecure-unlock",
            "--rpc.enabledeprecatedpersonal", "--miner.gasprice=0", "--txpool.pricelimit=0",
            "--miner.gaslimit=30000000", "--txpool.accountslots=1024", "--txpool.globalslots=20000",
            "--txpool.accountqueue=1024", "--txpool.globalqueue=20000",
            "--http", "--http.addr=0.0.0.0", "--http.vhosts=*", "--http.api=eth,net,web3,txpool,clique"])
        # start the client, wait for its IPC, then run the client's own sealing policy (no --mine). Geth discards
        # propagated blocks until a node counts as synced, which on a fresh chain only starting the miner sets
        # (a validator of a running network has history and is synced): every validator starts its miner once at
        # launch, and the policy stops it on its first poll when there is no work. The genesis timestamp lies in the
        # future, so no validator can seal (and fork) a first block before every validator is synced; _live checks it.
        sam = (f"until /pc/{BINARY} attach --exec eth.blockNumber /data/geth.ipc >/dev/null 2>&1; do sleep 1; done; "
               f"sleep 4; /pc/{BINARY} attach --exec 'miner.start()' /data/geth.ipc >/dev/null 2>&1; "
               f"/pc/{BINARY} attach --exec 'loadScript(\"/pc/{SAM}\")' /data/geth.ipc > /data/sam.log 2>&1")
        args = ["run", "-d", "--name", NAMES[i - 1], "--network", NET, "--ip", f"{SUBNET}.{10 + i}",
                f"--cpus={CPUS}", *mounts, "-v", f"{d}:/data", "-v", f"{DIR / 'password.txt'}:/password.txt:ro"]
        if i == 1:
            args += ["-p", "127.0.0.1:8566:8545"]
        args += ["--entrypoint", "sh", IMAGE, "-c", f"{geth} & ({sam}) & wait"]
        _docker(*args)
    try:
        _wait_ready(n)
        time.sleep(max(0.0, ts + 2 - time.time()))          # the first block can be sealed only after genesis time
        if not _live(n):
            raise TimeoutError("probe transaction not sealed or validators disagree on the head (Clique stall)")
    except TimeoutError as e:
        if attempts <= 1:
            raise
        print(f"{e}; recreating the network", file=sys.stderr)
        return up(n, attempts - 1)
    state = {"validators": n, "signers": [k.address for k in keys], "client": _rpc("web3_clientVersion"),
             "sealing": "Smart Auto-Mining v2 (on demand + 15 s idle window)", "cpus_per_validator": CPUS}
    (DIR / "current.json").write_text(json.dumps(state, indent=1))
    return state


def _rpc(method, params=None):
    return requests.post(PURECHAIN_LOCAL.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method,
                                                         "params": params or []}, timeout=5).json()["result"]


def _wait_ready(n: int, timeout: float = 180) -> None:
    end, last = time.time() + timeout, None
    while time.time() < end:
        try:
            peers = int(_rpc("net_peerCount"), 16)
            last = peers
            if peers >= n - 1 and all("Polling" in _docker("exec", c, "cat", "/data/sam.log", check=False).stdout
                                      for c in names(n)):
                return                                      # every validator runs its sealing policy
        except Exception as e:
            last = repr(e)
        time.sleep(2)
    raise TimeoutError(f"local PureChain not ready: {last}")


def _live(n: int, timeout: float = 60) -> bool:
    """Send a zero-value transaction and check that it is sealed and that every validator holds the same head."""
    from finverity import accounts
    a = accounts.admin()
    tx = {"to": a.address, "value": 0, "gas": 21_000, "gasPrice": 0, "chainId": PURECHAIN_LOCAL.chain_id,
          "nonce": int(_rpc("eth_getTransactionCount", [a.address, "pending"]), 16)}
    txh = _rpc("eth_sendRawTransaction", ["0x" + a.sign_transaction(tx).rawTransaction.hex().removeprefix("0x")])
    end = time.time() + timeout
    while time.time() < end:
        try:
            rc = _rpc("eth_getTransactionReceipt", [txh])
        except Exception:
            rc = None
        if rc:
            b = int(rc["blockNumber"], 16)
            time.sleep(5)                                    # the policy keeps sealing: the chain must move on
            js = f"eth.getBlock({b}) ? eth.getBlock({b}).hash : 'none'"
            hashes = {_docker("exec", c, f"/pc/{BINARY}", "attach", "--exec", js, "/data/geth.ipc",
                              check=False).stdout.strip() for c in names(n)}
            return len(hashes) == 1 and "none" not in str(hashes) and int(_rpc("eth_blockNumber"), 16) >= b + 2
        time.sleep(1)
    return False


def ensure(n: int = N_SIGNERS) -> None:
    running = _docker("ps", "--format", "{{.Names}}", check=False).stdout.split()
    st = json.loads((DIR / "current.json").read_text()) if (DIR / "current.json").exists() else {}
    if st.get("validators") == n and all(x in running for x in names(n)) and not any(x in running for x in NAMES[n:]):
        try:
            if int(_rpc("net_peerCount"), 16) >= n - 1 and _live(n):
                return
        except Exception:
            pass
    up(n)


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_SIGNERS
    print(json.dumps(up(n) if sys.argv[1] == "up" else down(), indent=1))
