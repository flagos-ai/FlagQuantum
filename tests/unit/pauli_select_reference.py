"""Shared locked Pauli SELECT fixture and source-generator comparison support."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._select_pauli_lcu import select_pauli_lcu_from_cirq
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies


def assert_all_branch_operators(
    ir: CircuitIR, logical: int, expected: torch.Tensor
) -> None:
    """Check every logical basis column in every unnormalized measured branch."""
    count = sum(g.name == "measure" for g in ir.instructions)
    stride = 1 << (ir.n_wires - logical)
    for history in product((0, 1), repeat=count):
        for column in range(1 << logical):
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            target = torch.zeros_like(initial)
            target[::stride] = expected[:, column] / sqrt(1 << count)
            torch.testing.assert_close(
                forced_branch(initial, ir.instructions, history),
                target,
                atol=3e-11,
                rtol=0,
            )


@pytest.fixture(scope="module")
def source() -> Any:
    pytest.importorskip("qualtran")
    import kickmix
    from qualtran.bloqs.multiplexers import select_pauli_lcu

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(select_pauli_lcu.__file__).parent
    for suffix, digest in (
        (".ipynb", "6513d5faf7b227cd26534f30e05a7d3b55a29f5a"),
        (".py", "41e2f8ddb341fbbdd2d004ef533041bd24a5898c"),
        ("_test.py", "81ccdd543c60783cecee6a87ae238b050ed3b28b"),
    ):
        data = (directory / f"select_pauli_lcu{suffix}").read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return select_pauli_lcu


def compile_pauli_select(
    original: Any, words: tuple[str, ...]
) -> tuple[CircuitIR, int]:
    width, target, polarity = (
        original.selection_bitsize,
        original.target_bitsize,
        original.control_val,
    )
    offset = int(polarity is not None)
    stop = offset + width + target
    work = unary_iteration_workspace(width, polarity)
    ports = {
        "selection": tuple(range(offset, offset + width)),
        "target": tuple(range(offset + width, stop)),
    }
    controls = (0,) if offset else ()
    if controls:
        ports["control"] = controls
    scratch = tuple(range(stop, stop + work))
    slots = tuple(range(work))
    native_select = select_pauli_lcu_from_cirq(original)
    assert native_select.select_unitaries == words
    op = native_select.make_bounded_on(
        selection=ports["selection"],
        target=ports["target"],
        control=controls,
        workspace=scratch,
        classical_bits=slots,
    )
    circuit = DynamicCircuit(stop + work)
    circuit.append(op)
    ir = circuit.to_ir()
    reference = reference_arithmetic_gates(
        original,
        ports,
        scratch,
        dict(zip(scratch, slots, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(ir.instructions, reference)
    assert_source_model(op.resource_model(), original)
    return ir, work
