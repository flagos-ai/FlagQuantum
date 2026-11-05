"""Independent dense Grover Circuits and analytic amplitudes for unchanged originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
DTYPE = torch.complex128
LOGGER = logging.getLogger(__name__)


def marked_states(case: str) -> tuple[int, int, tuple[int, ...]]:
    if case == "whitebox_fuzzing":
        return (
            12,
            10,
            tuple(
                x + 64 * y
                for y in range(64)
                for x in range(64)
                if x + y < 9 and (x % 4) * (y % 2) == 1
            ),
        )
    if case != "grover_graph_coloring":
        raise ValueError("unknown original decision case")
    marked = []
    for word in range(1024):
        a, b, c, d, e = ((word >> (2 * i)) & 3 for i in range(5))
        if (
            a != 0
            and a != 1
            and a != b
            and b != 0
            and b != c
            and b != e
            and b != 2
            and c != 0
            and c != d
            and c != e
            and d != 1
            and d != 2
            and e != 3
        ):
            marked.append(word)
    return 10, 1, tuple(marked)


def grover_power_matrix(
    width: int, marked: tuple[int, ...], rounds: int
) -> torch.Tensor:
    """Build the full operator power, including both orthogonal complements.

    Only the two uniform good/bad axes rotate. The good orthogonal complement
    has eigenvalue +1; the bad one has (-1)**rounds. This preserves the complete
    operator and avoids repeating a 256 MiB matrix conversion ten times.
    """
    size, good = 1 << width, len(marked)
    bad = size - good
    if (
        not 0 < good < size
        or len(set(marked)) != good
        or any(not 0 <= i < size for i in marked)
    ):
        raise ValueError("reference requires distinct nondegenerate marked words")
    theta = math.asin(math.sqrt(good / size))
    cosine, sine = math.cos(2 * rounds * theta), math.sin(2 * rounds * theta)
    bad_sign = (-1) ** rounds
    matrix = torch.full((size, size), (cosine - bad_sign) / bad, dtype=DTYPE)
    rows = torch.tensor(marked, dtype=torch.long)
    matrix[rows, :] = sine / math.sqrt(good * bad)
    matrix[:, rows] = -sine / math.sqrt(good * bad)
    matrix[rows[:, None], rows[None, :]] = (cosine - 1) / good
    diagonal = torch.full((size,), bad_sign, dtype=DTYPE)
    diagonal[rows] = 1
    matrix.diagonal().add_(diagonal)
    return matrix


def reference_circuit(case: str) -> fq.Circuit:
    """Use the full dense Grover power, independent of BDD and cube kernels.

    The largest validation-only buffer is 256 MiB, separate from production's
    64 MiB state budget. All original rounds are retained in the operator power.
    """
    width, rounds, marked = marked_states(case)
    step = grover_power_matrix(width, marked, rounds)
    circuit = fq.Circuit(width, dtype=DTYPE)
    for bit in range(width):
        circuit.h(bit)
    circuit.any(*range(width), matrix=step)
    return circuit


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    rounds: int
    data_bits: int
    marked_states: tuple[int, ...]
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    analytic_max_abs_error: float
    workspace_nonzero_probability: float
    state_norm: float
    seconds: float
    reference_matrix_bytes: int
    validation: str = "passed"


def validate(case: str) -> Validation:
    started = time.perf_counter()
    LOGGER.info("Synthesizing %s", case)
    manifest = json.loads((FIXTURES / "decision_sources.json").read_text())
    entry = next(row for row in manifest["fixtures"] if row["file"] == case + ".qmod")
    path = FIXTURES / entry["file"]
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if (
        blob != entry["git_blob_sha1"]
        or hashlib.sha256(raw).hexdigest() != entry["sha256"]
    ):
        raise ValueError("original source digest mismatch")
    if (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        != manifest["license_sha256"]
    ):
        raise ValueError("original license digest mismatch")
    program = synth.synthesize(path, memory_limit_bytes=64 * 1024**2)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resource report")
    width, rounds, marked = marked_states(case)
    scratch = resources.n_qubits - width
    LOGGER.info(
        "Executing production: %s qubits, %s operations",
        resources.n_qubits,
        resources.n_operations,
    )
    actual = fq.run(program).statevector().reshape(-1)
    LOGGER.info("Executing independent dense Circuit reference")
    expected = (
        fq.run(
            reference_circuit(case),
            options=fq.ExecutionOptions(
                mode="statevector",
                backend="torch",
                device="cpu",
                require_gradients=False,
                allow_backend_fallback=False,
            ),
        )
        .statevector()
        .reshape(-1)
    )
    # Every physical scratch word is compared; only the zero word may be populated.
    data = actual.reshape(1 << width, 1 << scratch)
    dirty = float(data[:, 1:].abs().square().sum())
    error = float((data[:, 0] - expected).abs().max())
    theta = math.asin(math.sqrt(len(marked) / (1 << width)))
    good = math.sin((2 * rounds + 1) * theta) / math.sqrt(len(marked))
    bad = math.cos((2 * rounds + 1) * theta) / math.sqrt((1 << width) - len(marked))
    marked_set = set(marked)
    closed = torch.tensor(
        [good if word in marked_set else bad for word in range(1 << width)], dtype=DTYPE
    )
    analytic_error = float((data[:, 0] - closed).abs().max())
    norm = float(actual.abs().square().sum())
    if max(error, analytic_error) > 1e-12 or dirty > 1e-24 or abs(norm - 1) > 1e-9:
        raise AssertionError(
            f"full-state errors {error}/{analytic_error}, dirty={dirty}, norm={norm}"
        )
    return Validation(
        case,
        blob,
        rounds,
        width,
        marked,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        scratch,
        error,
        analytic_error,
        dirty,
        norm,
        time.perf_counter() - started,
        (1 << width) ** 2 * 16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("grover_graph_coloring", "whitebox_fuzzing"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    report = asdict(validate(args.case))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
