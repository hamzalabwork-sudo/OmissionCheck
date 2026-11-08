"""On-chain cost of verifying the binding proof: a Groth16 verifier over BN254 deployed on a local ledger.

The verifier is the standard one (EIP-196/197 precompiles): it folds the public inputs into the verification key with
one ecMul and one ecAdd each and checks e(-A, B) e(alpha, beta) e(vk_x, gamma) e(C, delta) = 1 with one 4-pair
pairing call. Precompile prices do not depend on whether the pairing holds, so the gas is measured with verification
key and proof points on the curve that do not form a valid proof (the call returns false, it does not revert).

    python -m omission.zk.gas --backend poa2_local
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import solcx
from py_ecc.bn128 import G1, G2, multiply

from finverity.config import EVM_VERSION, ROOT, SOLC_VERSION
from omission.chain import Ledger

from .field import P

Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
OUT = ROOT / "paper2" / "zk"


def g1(k: int) -> tuple[int, int]:
    p = multiply(G1, k)
    return int(p[0].n), int(p[1].n)


def g2(k: int) -> tuple[int, int, int, int]:
    p = multiply(G2, k)
    x, y = p[0].coeffs, p[1].coeffs        # c0 + c1 i; the precompile takes (c1, c0)
    return int(x[1]), int(x[0]), int(y[1]), int(y[0])


def _snark_g2(p: list[list[str]]) -> tuple[int, int, int, int]:
    """snarkjs G2 point [[x.c0, x.c1], [y.c0, y.c1], ...] in precompile order (x.c1, x.c0, y.c1, y.c0)."""
    return int(p[0][1]), int(p[0][0]), int(p[1][1]), int(p[1][0])


def snark_key(vkey: dict) -> dict:
    return {"al": tuple(int(v) for v in vkey["vk_alpha_1"][:2]), "be": _snark_g2(vkey["vk_beta_2"]),
            "ga": _snark_g2(vkey["vk_gamma_2"]), "de": _snark_g2(vkey["vk_delta_2"]),
            "ic": [tuple(int(v) for v in p[:2]) for p in vkey["IC"]]}


def snark_proof(proof: dict) -> tuple[list[int], list[int], list[int]]:
    return ([int(v) for v in proof["pi_a"][:2]], list(_snark_g2(proof["pi_b"])), [int(v) for v in proof["pi_c"][:2]])


def source(n_pub: int, key: dict | None = None) -> str:
    """Verifier source; key is a snarkjs verification key (snark_key), or None for a benchmark key."""
    if key is None:
        key = {"al": g1(2), "be": g2(3), "ga": g2(5), "de": g2(7), "ic": [g1(11 + i) for i in range(n_pub + 1)]}
    al, be, ga, de, ic = key["al"], key["be"], key["ga"], key["de"], key["ic"]
    assert len(ic) == n_pub + 1
    fold = "\n".join(f"        require(x[{i}] < R, \"input\");\n"
                     f"        (vx, vy) = _addmul(vx, vy, {ic[i + 1][0]}, {ic[i + 1][1]}, x[{i}]);"
                     for i in range(n_pub))
    return f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// Groth16 verifier (BN254) for the sketch-binding circuit with {n_pub} public inputs; benchmark key.
contract SketchBindingVerifier {{
    uint256 constant R = {P};
    uint256 constant Q = {Q};
    bool public last;
    uint256 public lastGas;
    event Proof(bytes32 indexed tag, uint256[8] proof);

    function verify(uint256[2] calldata a, uint256[4] calldata b, uint256[2] calldata c,
                    uint256[{n_pub}] calldata x) public view returns (bool) {{
        uint256 vx = {ic[0][0]};
        uint256 vy = {ic[0][1]};
{fold}
        uint256[24] memory p;
        p[0] = a[0]; p[1] = (Q - a[1]) % Q;
        p[2] = b[0]; p[3] = b[1]; p[4] = b[2]; p[5] = b[3];
        p[6] = {al[0]}; p[7] = {al[1]};
        p[8] = {be[0]}; p[9] = {be[1]}; p[10] = {be[2]}; p[11] = {be[3]};
        p[12] = vx; p[13] = vy;
        p[14] = {ga[0]}; p[15] = {ga[1]}; p[16] = {ga[2]}; p[17] = {ga[3]};
        p[18] = c[0]; p[19] = c[1];
        p[20] = {de[0]}; p[21] = {de[1]}; p[22] = {de[2]}; p[23] = {de[3]};
        uint256[1] memory out;
        bool ok;
        assembly {{ ok := staticcall(gas(), 8, p, 768, out, 32) }}
        require(ok, "pairing");
        return out[0] == 1;
    }}

    function _addmul(uint256 x1, uint256 y1, uint256 x2, uint256 y2, uint256 s) internal view
            returns (uint256, uint256) {{
        uint256[3] memory i = [x2, y2, s];
        uint256[2] memory r;
        bool ok;
        assembly {{ ok := staticcall(gas(), 7, i, 96, r, 64) }}
        require(ok, "ecmul");
        uint256[4] memory j = [x1, y1, r[0], r[1]];
        assembly {{ ok := staticcall(gas(), 6, j, 128, r, 64) }}
        require(ok, "ecadd");
        return (r[0], r[1]);
    }}

    /// verify and record the result and the execution gas of verify()
    function check(uint256[2] calldata a, uint256[4] calldata b, uint256[2] calldata c,
                   uint256[{n_pub}] calldata x) external {{
        uint256 g = gasleft();
        last = verify(a, b, c, x);
        lastGas = g - gasleft();
    }}

    /// post the proof without verifying it on chain (readers verify it off chain)
    function post(bytes32 tag, uint256[8] calldata proof) external {{
        emit Proof(tag, proof);
    }}
}}
"""


def compile_verifier(n_pub: int, key: dict | None = None) -> dict:
    out = solcx.compile_standard({
        "language": "Solidity", "sources": {"V.sol": {"content": source(n_pub, key)}},
        "settings": {"optimizer": {"enabled": True, "runs": 200}, "evmVersion": EVM_VERSION,
                     "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}}}, solc_version=SOLC_VERSION)
    c = out["contracts"]["V.sol"]["SketchBindingVerifier"]
    return {"abi": c["abi"], "bytecode": "0x" + c["evm"]["bytecode"]["object"]}


def measure(backend: str, n_pubs: list[int]) -> list[dict]:
    led = Ledger(backend)
    rng = random.Random(5)
    a, b, c = g1(13), g2(17), g1(19)
    rows = []
    for n_pub in n_pubs:
        art = compile_verifier(n_pub)
        rc = led.wait([led.submit(led.admin, None, art["bytecode"], 6_000_000)])[0]
        addr = led.w3.to_checksum_address(rc["contractAddress"])
        v = led.w3.eth.contract(address=addr, abi=art["abi"])
        x = [rng.randrange(P) for _ in range(n_pub)]
        args = ([a[0], a[1]], list(b), [c[0], c[1]], x)
        rc2 = led.wait([led.submit(led.admin, addr, v.encodeABI(fn_name="check", args=args), 2_000_000)])[0]
        verify_gas = int(led.rpc.call("eth_call", [{"to": addr, "data": v.encodeABI(fn_name="lastGas")}, "latest"]), 16)
        result = int(led.rpc.call("eth_call", [{"to": addr, "data": v.encodeABI(fn_name="last")}, "latest"]), 16)
        proof = [a[0], a[1], *b, c[0], c[1]]
        rc3 = led.wait([led.submit(led.admin, addr, v.encodeABI(fn_name="post", args=[b"\x01" * 32, proof]),
                                   200_000)])[0]
        rows.append({"backend": backend, "n_public": n_pub, "deploy_gas": int(rc["gasUsed"], 16),
                     "check_tx_gas": int(rc2["gasUsed"], 16), "verify_exec_gas": verify_gas,
                     "check_calldata_bytes": rc2["calldata_bytes"], "post_tx_gas": int(rc3["gasUsed"], 16),
                     "post_calldata_bytes": rc3["calldata_bytes"], "pairing_returned": result})
        print(rows[-1], flush=True)
    return rows


def measure_real(backend: str, build: Path) -> dict:
    """Deploy the verifier for a real snarkjs key and verify the real proof and a proof with a tampered input."""
    led = Ledger(backend)
    key = snark_key(json.loads((build / "vkey.json").read_text()))
    a, b, c = snark_proof(json.loads((build / "proof.json").read_text()))
    x = [int(v) for v in json.loads((build / "public.json").read_text())]
    art = compile_verifier(len(x), key)
    rc = led.wait([led.submit(led.admin, None, art["bytecode"], 6_000_000)])[0]
    addr = led.w3.to_checksum_address(rc["contractAddress"])
    v = led.w3.eth.contract(address=addr, abi=art["abi"])
    row = {"backend": backend, "build": build.name, "n_public": len(x), "deploy_gas": int(rc["gasUsed"], 16)}
    for label, inputs in (("valid", x), ("tampered", [(x[0] + 1) % P, *x[1:]])):
        rc2 = led.wait([led.submit(led.admin, addr, v.encodeABI(fn_name="check", args=(a, b, c, inputs)),
                                   2_000_000)])[0]
        row[f"{label}_check_tx_gas"] = int(rc2["gasUsed"], 16)
        row[f"{label}_verify_exec_gas"] = int(led.rpc.call("eth_call", [{"to": addr, "data": v.encodeABI(
            fn_name="lastGas")}, "latest"]), 16)
        row[f"{label}_accepted"] = int(led.rpc.call("eth_call", [{"to": addr, "data": v.encodeABI(fn_name="last")},
                                                                  "latest"]), 16)
    if row["valid_accepted"] != 1 or row["tampered_accepted"] != 0:
        raise AssertionError(row)
    print(row, flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="poa2_local")
    ap.add_argument("--n-pub", type=int, nargs="+", default=[5, 21])
    ap.add_argument("--real", type=Path, nargs="*", help="snarkjs build directories holding vkey, proof and public")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.real:
        rows = [measure_real(a.backend, d) for d in a.real]
        (OUT / "verifier_gas_real.json").write_text(json.dumps(rows, indent=1))
        return
    rows = measure(a.backend, a.n_pub)
    (OUT / "verifier_gas.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
