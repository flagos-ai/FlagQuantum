# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted assertions from frozen Qualtran v0.7.0 Hamming tests.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight_test.py

"""Port every frozen Hamming test assertion to the existing native owners."""

from __future__ import annotations

import ast
import csv
import hashlib
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_ir,
    hamming_weight_requirements,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = (
    ("test_hamming_weight_t_complexity", 3),
    ("test_hamming_weight_t_complexity", 4),
    ("test_hamming_weight_t_complexity", 5),
    ("test_hamming_weight_compute", 3),
    ("test_hamming_weight_compute", 4),
)


@pytest.fixture(scope="module", autouse=True)
def unchanged_complete_source_files() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for filename in ("hamming_weight.py", "hamming_weight_test.py"):
        name = "qualtran/bloqs/arithmetic/" + filename
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )
        if filename.endswith("_test.py"):
            definitions = {
                node.name
                for node in ast.parse(raw).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            assert definitions == {name for name, _ in CASES}


@pytest.mark.parametrize("name,bitsize", CASES)
def test_all_unmodified_original_assertions(name: str, bitsize: int) -> None:
    from qualtran.bloqs.arithmetic import hamming_weight_test as original

    getattr(original, name)(bitsize)


@pytest.mark.parametrize("bitsize", [3, 4, 5])
def test_original_t_complexity_assertion_with_native_budget(bitsize: int) -> None:
    from qualtran.bloqs.arithmetic import HammingWeightCompute
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity, t_complexity

    and_t = And().t_complexity()
    junk_bitsize = bitsize - bitsize.bit_count()
    num_clifford = junk_bitsize * (5 + and_t.clifford) + bitsize.bit_count()
    num_t = junk_bitsize * and_t.t
    call = ArithmeticCall("hamming_weight", (bitsize,))
    model = query_qec_gate_cost(
        call, declared_calls=arithmetic_call_graph(call), legacy_shims=True
    )
    budget = legacy_gate_budget(model)
    assert budget == {"t": num_t, "clifford": num_clifford, "rotations": 0}
    assert t_complexity(HammingWeightCompute(bitsize)) == TComplexity(
        t=num_t, clifford=num_clifford
    )
    emitted = hamming_weight_ir(
        tuple(range(bitsize)),
        tuple(range(bitsize, bitsize + junk_bitsize)),
        tuple(
            range(bitsize + junk_bitsize, bitsize + junk_bitsize + bitsize.bit_length())
        ),
    )
    assert sum(gate.name in {"t", "tdg"} for gate in emitted) == num_t
    assert sum(gate.name in {"cx", "h", "s"} for gate in emitted) == num_clifford


@pytest.mark.parametrize("bitsize", [3, 4])
def test_original_compute_and_inverse_assertions_with_native_append(
    bitsize: int,
) -> None:
    junk_bitsize, out_bitsize = hamming_weight_requirements(bitsize)
    width = bitsize + junk_bitsize + out_bitsize
    source = tuple(range(bitsize))
    junk = tuple(range(bitsize, bitsize + junk_bitsize))
    out = tuple(range(bitsize + junk_bitsize, width))
    compute = hamming_weight_ir(source, junk, out)
    erase = hamming_weight_ir(
        source, junk, out, adjoint=True, classical_bits=tuple(range(junk_bitsize))
    )
    for inp in range(2**bitsize):
        initial = torch.zeros(1 << width, dtype=torch.complex128)
        initial[inp << (junk_bitsize + out_bitsize)] = 1
        circuit = Circuit(width, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(width, compute)
        )
        forward = circuit.state().flatten()
        position = int(forward.abs().argmax().item())
        actual_bits = f"{position & ((1 << out_bitsize) - 1):0{out_bitsize}b}"
        assert actual_bits == f"{inp.bit_count():0{out_bitsize}b}"
        assert forward[position].abs().item() == pytest.approx(1, abs=1e-12)
        for history in product((0, 1), repeat=junk_bitsize):
            restored = forced_branch(forward, erase, history)
            torch.testing.assert_close(
                restored,
                initial / 2 ** (junk_bitsize / 2),
                rtol=0,
                atol=1e-12,
            )
