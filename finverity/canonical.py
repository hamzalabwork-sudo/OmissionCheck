"""Canonicalization E* = Canon(E) and commitment H = SHA256(E*).

Implements the RFC 8785 (JSON Canonicalization Scheme) profile used by FinVerity:
  * object members sorted by key (keys are restricted to ASCII, so code-point order
    equals the UTF-16 order required by RFC 8785),
  * no insignificant whitespace, UTF-8 output,
  * strings normalized to Unicode NFC before serialization,
  * floating-point numbers are rejected: monetary amounts must be decimal strings,
    which removes the float-formatting ambiguity that RFC 8785 otherwise has to specify.

Evidence arriving as JSON text is parsed with `parse`, which accepts only I-JSON (RFC 7493): duplicate member
names, NaN/Infinity and lone surrogates are rejected, so the hashed object is exactly the object an auditor reads.
Value formats (two-decimal amount strings, UTC timestamps ending in Z) are fixed by the evidence schema, so two
spellings of the same value are rejected at entry rather than hashed differently.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


class CanonicalizationError(ValueError):
    pass


def _normalize(value: Any, path: str = "$") -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or not k.isascii():
                raise CanonicalizationError(f"{path}: keys must be ASCII strings, got {k!r}")
            out[k] = _normalize(v, f"{path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise CanonicalizationError(f"{path}: integer outside IEEE-754 safe range")
        return value
    if isinstance(value, float):
        raise CanonicalizationError(f"{path}: floats are not allowed; use decimal strings")
    raise CanonicalizationError(f"{path}: unsupported type {type(value).__name__}")


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise CanonicalizationError(f"duplicate member name(s): {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _reject_constant(name: str):
    raise CanonicalizationError(f"non-I-JSON number {name}")


def parse(text: str | bytes) -> Any:
    """Parse JSON text as I-JSON (RFC 7493) or raise CanonicalizationError."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    obj = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    _check_strings(obj)
    return obj


def _check_strings(v: Any) -> None:
    if isinstance(v, str):
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            raise CanonicalizationError("lone surrogate in string") from None
    elif isinstance(v, dict):
        for k, x in v.items():
            _check_strings(k)
            _check_strings(x)
    elif isinstance(v, list):
        for x in v:
            _check_strings(x)


def canonicalize(obj: Any) -> bytes:
    return json.dumps(
        _normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def commit(obj: Any) -> bytes:
    """H_i = SHA256(Canon(E_i))"""
    return sha256(canonicalize(obj))
