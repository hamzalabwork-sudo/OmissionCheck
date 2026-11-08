"""Merkle commitments for batched evidence recording (mirrors FinVerityRegistry.verifyInclusion).

leaf = keccak256(0x00 || recordId || evidenceHash || metaHash); internal nodes are keccak256(0x01 || a || b) over the
sorted pair, so a proof is just the list of sibling hashes and no node can be presented as a leaf. An odd node at
any level is promoted unchanged (never duplicated, so two different leaf lists cannot share a root). The registry
stores the number of leaves and marks every committed record ID, so a root cannot hide a second version of a record.
"""
from __future__ import annotations

from eth_utils import keccak

LEAF, NODE = b"\x00", b"\x01"


def leaf(record_id: bytes, evidence_hash: bytes, meta_hash: bytes) -> bytes:
    return keccak(LEAF + record_id + evidence_hash + meta_hash)


def _pair(a: bytes, b: bytes) -> bytes:
    return keccak(NODE + a + b) if a < b else keccak(NODE + b + a)


def build(leaves: list[bytes]) -> tuple[bytes, list[list[bytes]]]:
    """Return (root, proofs) where proofs[i] authenticates leaves[i]."""
    if not leaves:
        raise ValueError("empty batch")
    proofs: list[list[bytes]] = [[] for _ in leaves]
    level = list(leaves)
    pos = list(range(len(leaves)))          # index of each original leaf at the current level
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_pair(level[i], level[i + 1]))
            else:
                nxt.append(level[i])
        for j, p in enumerate(pos):
            sib = p ^ 1
            if sib < len(level):
                proofs[j].append(level[sib])
            pos[j] = p // 2
        level = nxt
    return level[0], proofs


def verify(leaf_hash: bytes, proof: list[bytes], root: bytes) -> bool:
    h = leaf_hash
    for p in proof:
        h = _pair(h, p)
    return h == root
