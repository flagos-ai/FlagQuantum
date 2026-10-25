"""Private conversion from maintained Quafu calibration to QSteed topology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, cast


def qsteed_chip_info(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal legacy-shaped topology currently consumed by QSteed.

    The bridge copies authoritative SDK fields only.  In particular, it does not
    invent a one-qubit fidelity when the maintained calibration omits one.
    """

    calibrated_at = str(calibration.get("calibrated_at", "")).strip()
    if not calibrated_at:
        raise RuntimeError("Quafu calibration is missing calibrated_at")

    raw_qubits = calibration.get("qubits")
    if not _is_sequence(raw_qubits):
        raise RuntimeError("Quafu calibration qubits are not a sequence")
    qubit_rows = cast(Sequence[Any], raw_qubits)
    qubits: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(qubit_rows):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Quafu calibration qubits[{position}] is not a mapping")
        index = _qubit_index(raw.get("index"), field=f"qubits[{position}].index")
        label = f"Q{index}"
        if label in qubits:
            raise RuntimeError(f"Quafu calibration repeats physical qubit {index}")
        item = dict(raw)
        item["index"] = index
        fidelity = item.get("fidelity")
        if fidelity is not None:
            item["fidelity"] = _fidelity(fidelity, field=f"qubits[{position}].fidelity")
        qubits[label] = item
    if not qubits:
        raise RuntimeError("Quafu calibration contains no physical qubits")

    raw_couplers = calibration.get("couplers")
    if not _is_sequence(raw_couplers):
        raise RuntimeError("Quafu calibration couplers are not a sequence")
    coupler_rows = cast(Sequence[Any], raw_couplers)
    couplers: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    available = {item["index"] for item in qubits.values()}
    for position, raw in enumerate(coupler_rows):
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"Quafu calibration couplers[{position}] is not a mapping"
            )
        pair = raw.get("qubits")
        if not _is_sequence(pair):
            raise RuntimeError(
                f"Quafu calibration couplers[{position}].qubits must contain two indices"
            )
        pair_items = cast(Sequence[Any], pair)
        if len(pair_items) != 2:
            raise RuntimeError(
                f"Quafu calibration couplers[{position}].qubits must contain two indices"
            )
        left = _qubit_index(pair_items[0], field=f"couplers[{position}].qubits[0]")
        right = _qubit_index(pair_items[1], field=f"couplers[{position}].qubits[1]")
        if left == right:
            raise RuntimeError(
                "Quafu calibration coupler cannot join a qubit to itself"
            )
        normalized = (min(left, right), max(left, right))
        if normalized in seen_pairs:
            raise RuntimeError(f"Quafu calibration repeats coupler {normalized}")
        if left not in available or right not in available:
            raise RuntimeError(
                f"Quafu calibration coupler {normalized} references an unknown qubit"
            )
        fidelity = _fidelity(
            raw.get("cz_fidelity"),
            field=f"couplers[{position}].cz_fidelity",
        )
        item = dict(raw)
        item["qubits_index"] = [left, right]
        item["fidelity"] = fidelity
        couplers[f"C{position}"] = item
        seen_pairs.add(normalized)

    result: dict[str, Any] = {
        "calibration_time": calibrated_at,
        "qubits_info": qubits,
        "couplers_info": couplers,
    }
    basis_gates = calibration.get("basis_gates")
    if basis_gates is not None:
        if not _is_sequence(basis_gates) or any(
            not isinstance(gate, str) or not gate.strip() for gate in basis_gates
        ):
            raise RuntimeError("Quafu calibration basis_gates are invalid")
        result["basis_gates"] = [gate.strip().lower() for gate in basis_gates]
    return result


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _qubit_index(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Quafu calibration {field} is not a nonnegative integer")
    return value


def _fidelity(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"Quafu calibration {field} is not a number")
    resolved = float(value)
    if not 0 < resolved <= 1:
        raise RuntimeError(f"Quafu calibration {field} must be in (0, 1]")
    return resolved


__all__ = ("qsteed_chip_info",)
