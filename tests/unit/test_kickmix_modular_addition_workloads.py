"""Compare the locked modular workloads after resolving classical dataflow."""

import json
from pathlib import Path

import pytest

from benchmarks.benchmark_mixed_modular_addition import locked_modular_workloads
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from tests.unit.fixtures.kickmix_modular_feedback import feedback_digest


@pytest.mark.parametrize(
    ("kind", "native_records"), [("add", 13119), ("sub", 15641), ("approx", 11529)]
)
def test_modular_workload_effects(kind: str, native_records: int) -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("kickmix_modular_addition_workloads.json")
        .read_text()
    )
    assert fixture["source_blob"] == "ba7e69f03ad225a13ca996bb44f23c36a6f4f28c"
    expected = fixture["cases"][kind]
    program = locked_modular_workloads()[kind]()
    assert program.n_qubits == 768
    assert program.input_bits == (() if kind == "approx" else tuple(range(1, 255)))
    text = encode_kmx(program)
    records = text.splitlines()[1:]
    assert len(records) == native_records
    assert decode_kmx(text) == program
    assert feedback_digest(records, program.input_bits) == (
        expected["feedback_effects"],
        expected["feedback_sha256"],
    )


def test_feedback_captures_scope_snapshot_and_live_initialization() -> None:
    prefix = ["HMR q0 b0", "PUSH_CONDITION if b0", "HMR q1 b0"]
    snapshot = [*prefix, "CX q0 q1", "POP_CONDITION"]
    changed = [*prefix, "CX q0 q1 if b0", "POP_CONDITION"]
    assert feedback_digest(snapshot, ()) != feedback_digest(changed, ())
    erase = ["HMR q0 b0", "BIT_STORE0 b0", "Z q0 if b0"]
    assert feedback_digest(erase, ()) != feedback_digest(erase[::2], ())
    assert feedback_digest(["BIT_STORE0 b0", *snapshot], ()) == feedback_digest(
        snapshot, ()
    )
