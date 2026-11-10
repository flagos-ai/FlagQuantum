"""Read real restricted FCIDUMP integrals into the native chemistry boundary."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TextIO

import torch

from ...algorithms.chemistry_domain.problem import MolecularProblem
from ...core.algebra.pauli import count


def _header(stream: TextIO) -> dict[str, tuple[int, ...]]:
    fragments: list[str] = []
    size = 0
    for line in stream:
        content = line.partition("!")[0].strip().upper()
        size += len(line)
        if size > 65536:
            raise ValueError("FCIDUMP header exceeds 65536 characters")
        if not content:
            continue
        fragments.append(content)
        end = re.search(r"&END\b|/", content)
        if end:
            if content[end.end() :].strip(" ,"):
                raise ValueError("FCIDUMP header terminator must end its line")
            fragments[-1] = content[: end.start()]
            break
    else:
        raise ValueError("FCIDUMP header requires &END or / terminator")
    text = " ".join(fragments).strip()
    start = re.match(r"&FCI\b", text)
    if start is None:
        raise ValueError("FCIDUMP header must start with &FCI")
    text = text[start.end() :]
    keys = list(re.finditer(r"([A-Z][A-Z0-9_]*)\s*=", text))
    if not keys or text[: keys[0].start()].strip(" ,"):
        raise ValueError("FCIDUMP header requires named integer fields")
    result: dict[str, tuple[int, ...]] = {}
    for index, key in enumerate(keys):
        name = key.group(1)
        if name not in {"NORB", "NELEC", "MS2", "ORBSYM", "ISYM", "IUHF"}:
            raise ValueError(f"unsupported FCIDUMP header field: {name}")
        if name in result:
            raise ValueError(f"duplicate FCIDUMP header field: {name}")
        stop = keys[index + 1].start() if index + 1 < len(keys) else len(text)
        values = text[key.end() : stop].strip(" ,")
        if not re.fullmatch(r"[+-]?\d+(?:[\s,]+[+-]?\d+)*", values):
            raise ValueError(f"FCIDUMP {name} requires integer values")
        result[name] = tuple(int(value) for value in re.split(r"[\s,]+", values))
        if name != "ORBSYM" and len(result[name]) != 1:
            raise ValueError(f"FCIDUMP {name} requires one integer")
    if not {"NORB", "NELEC", "MS2"} <= result.keys():
        raise ValueError("FCIDUMP requires NORB, NELEC and MS2")
    return result


def _record(line: str, line_number: int) -> tuple[float, tuple[int, ...]] | None:
    content = line.partition("!")[0].strip()
    if not content:
        return None
    fields = content.split()
    if len(fields) != 5:
        raise ValueError(
            f"FCIDUMP record {line_number} requires value and four indices"
        )
    try:
        value = float(fields[0].replace("D", "E").replace("d", "e"))
        indices = tuple(int(field) for field in fields[1:])
    except ValueError as error:
        raise ValueError(f"invalid FCIDUMP record {line_number}") from error
    if not math.isfinite(value):
        raise ValueError(f"FCIDUMP record {line_number} must be finite")
    return value, indices


def _read(stream: TextIO, max_orbitals: int) -> MolecularProblem:
    header = _header(stream)
    n = header["NORB"][0]
    electrons = header["NELEC"][0]
    if not 1 <= n <= max_orbitals:
        raise ValueError("FCIDUMP NORB exceeds the positive max_orbitals budget")
    if header["MS2"] != (0,) or header.get("IUHF", (0,)) != (0,):
        raise ValueError(
            "FCIDUMP supports restricted closed-shell MS2=0 integrals only"
        )
    if not 0 <= electrons <= 2 * n or electrons % 2:
        raise ValueError("FCIDUMP NELEC must be even and fit the orbital space")
    if "ORBSYM" in header and (
        len(header["ORBSYM"]) != n or any(label < 0 for label in header["ORBSYM"])
    ):
        raise ValueError("FCIDUMP ORBSYM must contain NORB nonnegative labels")
    if header.get("ISYM", (1,))[0] < 0:
        raise ValueError("FCIDUMP ISYM must be nonnegative")
    one = torch.zeros((n, n), dtype=torch.float64, device="cpu")
    two = torch.zeros((n, n, n, n), dtype=torch.float64, device="cpu")
    seen: dict[tuple[int, ...], float] = {}
    offset = 0.0
    for line_number, line in enumerate(stream, start=1):
        record = _record(line, line_number)
        if record is None:
            continue
        value, indices = record
        i, j, k, last = indices
        if any(q < 0 or q > n for q in indices):
            raise ValueError(f"FCIDUMP record {line_number} index outside NORB")
        if i == j == k == last == 0:
            key: tuple[int, ...] = ()
        elif i and j and not k and not last:
            key = tuple(sorted((i, j)))
        elif all(indices):
            left, right = sorted((tuple(sorted((i, j))), tuple(sorted((k, last)))))
            key = (*left, *right)
        else:
            raise ValueError(
                f"unsupported FCIDUMP index pattern at record {line_number}; "
                "orbital-energy records and unrestricted blocks are unsupported"
            )
        if key in seen and not math.isclose(
            seen[key], value, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"conflicting FCIDUMP symmetry duplicate at record {line_number}"
            )
        if key in seen:
            continue
        seen[key] = value
        if not key:
            offset = value
        elif len(key) == 2:
            one[i - 1, j - 1] = one[j - 1, i - 1] = value
        else:
            for p, q in ((i - 1, j - 1), (j - 1, i - 1)):
                for r, s in ((k - 1, last - 1), (last - 1, k - 1)):
                    two[p, q, r, s] = two[r, s, p, q] = value
    return MolecularProblem(one, two, electrons, core_energy=offset)


def read_fcidump(path: str | Path, *, max_orbitals: int = 32) -> MolecularProblem:
    """Read a real, restricted, closed-shell spatial-orbital FCIDUMP file.

    Integral records use one-based indices and real chemist notation ``(ij|kl)``.
    Missing records are zero; eightfold ERI symmetry is restored. Fortran D
    exponents, blank lines and ``!`` comments are accepted. Conflicting symmetry
    duplicates, unknown header fields and unsupported record types raise
    ValueError. File and decoding errors propagate to the caller.

    The undivided constant enters ``core_energy`` once; it is not assumed to be
    nuclear repulsion. Symmetry labels are validated but not used for tapering
    or reference selection. The reference occupies the first NELEC/2 orbitals,
    without an HF energy claim. Callers can select a different native reference.

    Args:
        path: UTF-8 FCIDUMP file.
        max_orbitals: Explicit dense-integral allocation limit. The ERI tensor
            alone costs ``8 * NORB**4`` bytes; native snapshots add overhead.

    Examples:
        problem = read_fcidump("FCIDUMP")
        assert problem.n_modes == 2 * problem.n_orbitals
    """
    limit = count(max_orbitals, "max_orbitals")
    if limit == 0:
        raise ValueError("max_orbitals must be positive")
    with Path(path).open(encoding="utf-8") as stream:
        return _read(stream, limit)
