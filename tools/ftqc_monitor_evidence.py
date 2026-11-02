"""Bounded, record-only identity for RSS failures after owned-process cleanup."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from types import CodeType
from typing import TypedDict

_NAMES = ("communicate", "_communicate", "_check_timeout", "wait", "_wait")
_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_name",
    "co_qualname",
    "co_firstlineno",
    "co_linetable",
    "co_exceptiontable",
)
_SCHEMA = "cpython312_code_structure_v1_filename_separate"


class _FunctionIdentity(TypedDict):
    function: str
    available: bool
    filename_sha256: str | None
    structural_sha256: str | None


class _FrameIdentity(TypedDict):
    ordinal: int
    function: str
    line: int
    filename_sha256: str
    filename_matches_runtime_module: bool
    matches_runtime_code_object: bool


class FailureIdentity(TypedDict):
    capture_status: str
    checkpoint: str
    structure_schema: str | None
    structure_status: str
    runtime_module_filename_sha256: str | None
    functions: list[_FunctionIdentity]
    frames: list[_FrameIdentity]
    traceback_truncated: bool


def _packed(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _filename_digest(value: str) -> str:
    if len(value) > 4096:
        raise ValueError("filename identity exceeds4096characters")
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _supports_structure() -> bool:
    return sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)


def _structure_digest(code: CodeType) -> str:
    """Reuse the reviewed field schema; filenames remain separate identities."""
    if not _supports_structure():
        raise ValueError("unsupported runtime structure schema")
    nodes, size = 0, 0

    def encode(value: object, depth: int) -> object:
        nonlocal nodes, size
        nodes += 1
        if depth > 32 or nodes > 4096 or size > 262144:
            raise ValueError("code structure bound exceeded")
        if value is None:
            return ["none"]
        if value is Ellipsis:
            return ["ellipsis"]
        if type(value) is bool:
            return ["bool", value]
        if isinstance(value, int) and type(value) is int:
            if value.bit_length() > 4096:
                raise ValueError("integer constant exceeds4096bits")
            return ["int", str(value)]
        if isinstance(value, float) and type(value) is float:
            if not math.isfinite(value):
                raise ValueError("nonfinite float constant")
            return ["float", value.hex()]
        if isinstance(value, complex) and type(value) is complex:
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("nonfinite complex constant")
            return ["complex", value.real.hex(), value.imag.hex()]
        if isinstance(value, str) and type(value) is str:
            size += len(value) * 6
            if size > 262144:
                raise ValueError("code string bound exceeded")
            return ["str", value]
        if isinstance(value, bytes) and type(value) is bytes:
            size += len(value) * 2
            if size > 262144:
                raise ValueError("code byte bound exceeded")
            return ["bytes", value.hex()]
        if isinstance(value, (tuple, frozenset)) and type(value) in (tuple, frozenset):
            if len(value) > 4096:
                raise ValueError("code container bound exceeded")
            children = [encode(item, depth + 1) for item in value]
            if isinstance(value, frozenset):
                children.sort(key=_packed)
            return ["tuple" if isinstance(value, tuple) else "frozenset", children]
        if isinstance(value, CodeType) and type(value) is CodeType:
            return [
                "code",
                [[name, encode(getattr(value, name), depth + 1)] for name in _FIELDS],
            ]
        raise ValueError("unsupported code constant")

    data = _packed([_SCHEMA, encode(code, 0)])
    if len(data) > 262144:
        raise ValueError("code serialization exceeds256KiB")
    return hashlib.sha256(data).hexdigest()


def collect_failure_identity(error: BaseException) -> FailureIdentity:
    """Record scalar labels only; never classify a checkpoint or mask a failure."""
    node = None
    try:
        supported = _supports_structure()
        module_file = getattr(subprocess, "__file__", None)
        module_digest = (
            _filename_digest(module_file) if isinstance(module_file, str) else None
        )
        functions: list[_FunctionIdentity] = []
        codes: dict[str, CodeType] = {}
        for name in _NAMES:
            code = getattr(getattr(subprocess.Popen, name, None), "__code__", None)
            if isinstance(code, CodeType):
                codes[name] = code
                filename = _filename_digest(code.co_filename)
                structure = _structure_digest(code) if supported else None
            else:
                filename, structure = None, None
            functions.append(
                {
                    "function": name,
                    "available": isinstance(code, CodeType),
                    "filename_sha256": filename,
                    "structural_sha256": structure,
                }
            )
        frames: list[_FrameIdentity] = []
        node = error.__traceback__
        candidates = 0
        while node is not None and len(frames) < 16 and candidates < 8:
            code = node.tb_frame.f_code
            name = code.co_name if code.co_name in _NAMES else "other"
            filename = _filename_digest(code.co_filename)
            line = node.tb_lineno
            if type(line) is not int or not 0 <= line <= 1000000:
                raise ValueError("traceback line exceeds scalar bound")
            frames.append(
                {
                    "ordinal": len(frames),
                    "function": name,
                    "line": line,
                    "filename_sha256": filename,
                    "filename_matches_runtime_module": filename == module_digest,
                    "matches_runtime_code_object": name in codes
                    and code is codes[name],
                }
            )
            candidates += int(name != "other")
            node = node.tb_next
        result: FailureIdentity = {
            "capture_status": "captured",
            "checkpoint": "unknown",
            "structure_schema": _SCHEMA if supported else None,
            "structure_status": (
                "unsupported_runtime_schema"
                if not supported
                else (
                    "available"
                    if len(codes) == len(_NAMES)
                    else "incomplete_runtime_identity"
                )
            ),
            "runtime_module_filename_sha256": module_digest,
            "functions": functions,
            "frames": frames,
            "traceback_truncated": node is not None,
        }
        if len(_packed(result)) > 16384:
            raise ValueError("failure identity exceeds16KiB")
        return result
    except BaseException:
        # This diagnostic boundary must preserve the primary process/cleanup failure.
        return unavailable_failure_identity()
    finally:
        node = None


def unavailable_failure_identity() -> FailureIdentity:
    return {
        "capture_status": "capture_failed",
        "checkpoint": "unknown",
        "structure_schema": None,
        "structure_status": "unavailable",
        "runtime_module_filename_sha256": None,
        "functions": [],
        "frames": [],
        "traceback_truncated": False,
    }
