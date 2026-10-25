"""Verify the pinned Ising detector layout without sampling or training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from examples.qec import compare_ising as comparison
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("layout identities and coordinates must be integers")
    return int(value)


def _coordinate(value: Any, distance: int, *, data: bool) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("layout requires two-dimensional coordinates")
    x, y = (_integer(v) for v in value)
    if any(not 0 <= v <= 2 * distance or v % 2 != int(data) for v in (x, y)):
        raise ValueError("invalid data/check coordinate parity or bounds")
    return x, y


def _checks(code: Any, distance: int) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    dict[int, tuple[int, int, frozenset[int]]],
]:
    """Validate vendor supports against geometric neighbors, without reordering."""
    data = tuple(_integer(q) for q in code.data_qubits)
    xs = tuple(_integer(q) for q in code.xcheck_qubits)
    zs = tuple(_integer(q) for q in code.zcheck_qubits)
    half = (distance**2 - 1) // 2
    if (
        len(data) != distance**2
        or len(xs) != half
        or len(zs) != half
        or len(set(data + xs + zs)) != len(data + xs + zs)
    ):
        raise ValueError("incomplete or overlapping data/check identities")
    metadata = code.code_dict
    coordinates = {
        q: _coordinate(metadata["data"][q]["coord"], distance, data=True) for q in data
    }
    expected_data = {
        (x, y) for x in range(1, 2 * distance, 2) for y in range(1, 2 * distance, 2)
    }
    if set(coordinates.values()) != expected_data:
        raise ValueError("data coordinates must cover the complete square")
    checks: dict[int, tuple[int, int, frozenset[int]]] = {}
    occupied: set[int] = set()
    for kind, ids, matrix, key in (
        (1, xs, code.hx, "syndrome_X"),
        (2, zs, code.hz, "syndrome_Z"),
    ):
        rows = matrix.tolist()
        if len(rows) != half:
            raise ValueError("stabilizer row count mismatch")
        for q, row in zip(ids, rows, strict=True):
            if len(row) != len(data) or any(
                not isinstance(v, Real) or v not in (0, 1) for v in row
            ):
                raise ValueError("requires binary stabilizer rows in data order")
            x, y = _coordinate(metadata[key][q]["coord"], distance, data=False)
            site = (y // 2) * (distance + 1) + x // 2
            if site in occupied:
                raise ValueError("colliding check coordinates")
            occupied.add(site)
            support = frozenset(qd for qd, bit in zip(data, row, strict=True) if bit)
            neighbors = frozenset(
                qd
                for qd, (dx, dy) in coordinates.items()
                if abs(dx - x) == abs(dy - y) == 1
            )
            if len(support) not in (2, 4) or support != neighbors:
                raise ValueError("stabilizer support disagrees with check coordinates")
            checks[q] = (kind, site, support)
    return data, xs, zs, checks


def resolve_layout(
    circuit: Any, code: Any, *, distance: int, rounds: int
) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int, ...]]:
    """Resolve every original detector from its absolute measurement records.

    This parser's small synthetic fixtures do not authenticate vendor source.
    Use build_model for the pinned d9/R9 study boundary.
    """
    if (
        type(distance) is not int
        or distance < 3
        or distance % 2 != 1
        or type(rounds) is not int
        or rounds < 2
    ):
        raise ValueError("requires odd distance >=3 and rounds >=2")
    data, xs, zs, checks = _checks(code, distance)
    records: list[tuple[int, str]] = []
    parities: list[tuple[int, ...]] = []
    allowed = {
        "RX",
        "R",
        "Z_ERROR",
        "X_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "CX",
        "TICK",
        "MRX",
        "MR",
        "MX",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
    }
    for instruction in circuit.flattened():
        name = instruction.name
        if name not in allowed:
            raise ValueError("unsupported Ising instruction schedule: " + str(name))
        targets = instruction.targets_copy()
        if name in ("MRX", "MR", "MX"):
            for target in targets:
                if not target.is_qubit_target or target.is_inverted_result_target:
                    raise ValueError("unsupported measurement target")
                records.append((_integer(target.value), name))
        elif name == "DETECTOR":
            parity: list[int] = []
            for target in targets:
                if not target.is_measurement_record_target:
                    raise ValueError("detector must reference measurement records")
                offset = _integer(target.value)
                absolute = len(records) + offset
                if offset >= 0 or absolute < 0 or absolute >= len(records):
                    raise ValueError("detector references an unavailable measurement")
                parity.append(absolute)
            if not parity or len(parity) != len(set(parity)):
                raise ValueError("empty or duplicate detector measurement references")
            parities.append(tuple(sorted(parity)))
    expected_records = [
        (q, gate)
        for _ in range(rounds)
        for ids, gate in ((xs, "MRX"), (zs, "MR"))
        for q in ids
    ] + [(q, "MX") for q in data]
    if records != expected_records or circuit.num_measurements != len(records):
        raise ValueError("measurement identity/order/reset schedule mismatch")
    width = distance**2 - 1
    locations = {q: i for i, q in enumerate(xs + zs)}
    data_records = {q: rounds * width + i for i, q in enumerate(data)}
    expected: list[tuple[int, ...]] = []
    sites: list[int] = []
    kinds: list[int] = []
    plane = (distance + 1) ** 2
    for time in range(rounds + 1):
        ids = xs if time in (0, rounds) else xs + zs
        for q in ids:
            kind, site, support = checks[q]
            if time == 0:
                signature: tuple[int, ...] = (locations[q],)
            elif time == rounds:
                signature = tuple(data_records[d] for d in support) + (
                    (rounds - 1) * width + locations[q],
                )
            else:
                signature = (
                    (time - 1) * width + locations[q],
                    time * width + locations[q],
                )
            expected.append(tuple(sorted(signature)))
            sites.append(time * plane + site)
            kinds.append(kind)
    if parities != expected or circuit.num_detectors != len(expected):
        raise ValueError("detector identity/order/support/time or boundary mismatch")
    return (rounds + 1, distance + 1, distance + 1), tuple(sites), tuple(kinds)


def build_model(source: Path) -> tuple[SpacetimePredecoder, dict[str, object]]:
    """Construct the pinned cell and verify archived input hashes; draw no data."""
    config = comparison.Config(True, distance=9, rounds=9)
    reference = comparison.reference(source, None, config, load_model=False)
    root = (source / "code").resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("qec", "model", "evaluation", "export"):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(root):
                raise ValueError("vendor module origin is outside the pinned source")
    vendor = importlib.import_module("qec.surface_code.memory_circuit")
    code = vendor.SurfaceCode(9, first_bulk_syndrome_type="X", rotated_type="V")
    shape, sites, kinds = resolve_layout(reference.circuit, code, distance=9, rounds=9)
    dem = comparison.training_dem(reference.circuit)
    circuit_sha = hashlib.sha256(str(reference.circuit).encode()).hexdigest()
    dem_sha = hashlib.sha256(dem.to_stim_text().encode()).hexdigest()
    anchor = json.loads((archive.ARCHIVE / "results.json").read_text())
    if (
        circuit_sha != anchor["circuit_sha256"]
        or dem_sha != anchor["training_dem_sha256"]
    ):
        raise ValueError(
            "circuit or DEM differs from archived cell; check dependencies"
        )
    binding = {
        "grid_shape": shape,
        "sites": sites,
        "kinds": kinds,
        "circuit_sha256": circuit_sha,
        "dem_sha256": dem_sha,
        "source_sha256": comparison.ISING_SOURCE_SHA256,
    }
    layout_sha = hashlib.sha256(
        json.dumps(binding, sort_keys=True).encode()
    ).hexdigest()
    model = SpacetimePredecoder.from_dem(
        dem, grid_shape=shape, sites=sites, kinds=kinds
    )
    return model, {
        "purpose": "verify_layout_only",
        "detectors": len(sites),
        "grid_shape": shape,
        "x_detectors": kinds.count(1),
        "z_detectors": kinds.count(2),
        "bulk_order": "X_then_Z_from_measurement_records",
        "circuit_sha256": circuit_sha,
        "dem_sha256": dem_sha,
        "layout_sha256": layout_sha,
        "new_samples": 0,
        "new_optimizer_steps": 0,
    }


def verify_encoding(model: SpacetimePredecoder) -> None:
    """Check zero/one/every one-hot in bounded batches without model inference."""
    width = model.h.shape[0]
    for offset in range(-2, width, 32):
        indices = range(offset, min(offset + 32, width))
        bits = torch.zeros(len(indices), width)
        for row, index in enumerate(indices):
            if index == -1:
                bits[row] = 1
            elif index >= 0:
                bits[row, index] = 1
        encoded = model.encode_input(bits)
        if not torch.equal(encoded[:, 0].flatten(1)[:, model.sites], bits):
            raise ValueError("detector encoding roundtrip failed")
        if bool((encoded[:, 0] * (1 - model.masks[0])).any()):
            raise ValueError("detector encoding leaked into invalid sites")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ising-source", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, report = build_model(args.ising_source)
    verify_encoding(model)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
