"""Bind complete callable target-position sources to owned SELECT consumers."""

from collections import Counter
from dataclasses import replace
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_target_position_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "shape,controls", (((2,), 0), ((2,), 1), ((2,), 2), ((2, 2), 1))
)
@pytest.mark.parametrize("family", ("HS", "ST"))
def test_callable_source_directions_native_replacement_and_models(
    source: Any, shape: tuple[int, ...], controls: int, family: str
) -> None:
    import cirq
    import numpy as np
    from qualtran import BQUInt, QAny, Register

    gates = (cirq.H, cirq.S) if family == "HS" else (cirq.S, cirq.T)

    def nth_gate(*indices: int) -> Any:
        return gates[int(np.ravel_multi_index(indices, shape)) % len(gates)]

    original = source.ApplyGateToLthQubit(
        tuple(Register(f"axis{i}", BQUInt(1, size)) for i, size in enumerate(shape)),
        nth_gate,
        control_regs=() if not controls else (Register("enable", QAny(controls)),),
    )
    emitter = import_register_emitter(original)
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    selection, targets = emitter.layout[:-1], emitter.layout[-1:]
    native = SelectBlockEncoding(
        selection, targets, emitter, BlackBoxPrepare(selection, (), _identity_prepare)
    )
    count = prod(shape)
    address_width = len(shape)
    logical = controls + address_width + count
    work, bits = emitter.requirements()
    identity = torch.eye(2, dtype=torch.complex128)
    operator = torch.zeros((1 << logical, 1 << logical), dtype=torch.complex128)
    block_size = 1 << count
    for enabled, row in product(range(1 << controls), range(count)):
        selected = torch.as_tensor(cirq.unitary(gates[row % 2]), dtype=torch.complex128)
        factors = [
            selected if enabled == (1 << controls) - 1 and i == row else identity
            for i in range(count)
        ]
        block = factors[0]
        for factor in factors[1:]:
            block = torch.kron(block, factor)
        offset = (enabled * count + row) * block_size
        operator[offset : offset + block_size, offset : offset + block_size] = block
    generator = torch.Generator().manual_seed(1084)
    coherent = torch.randn(1 << logical, generator=generator, dtype=torch.complex128)
    coherent /= coherent.norm()
    columns = [coherent]
    if count == 2 and controls <= 1:
        columns.extend(torch.eye(1 << logical, dtype=torch.complex128).unbind(1))
    costs = {
        "H": (2, GateCost(clifford=4, rotation=2)),
        "S": (3, GateCost(and_bloq=1, clifford=2, measurement=1)),
        "T": (3, GateCost(t=1, and_bloq=1, clifford=1, measurement=1)),
    }
    children = tuple(
        ComponentCost(str(gates[i % 2]), *costs[str(gates[i % 2])])
        for i in range(count)
    )
    for adjoint in (False, True):
        ir = (
            Circuit(logical + work)
            .append(
                native.make_on(
                    selection=tuple(range(controls + address_width)),
                    target=tuple(range(controls + address_width, logical)),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(bits)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        assert CircuitIR.from_json(ir.to_json()).instructions == ir.instructions
        model = indexed_component_cost(
            (1,) * len(shape),
            shape,
            children,
            target_width=count,
            child_target_width=1,
            control_count=controls,
            adjoint=adjoint,
        )
        assert_source_model(model, original.adjoint() if adjoint else original)
        inventory = Counter(gate.name for gate in ir.instructions)
        events = inventory["measure"]
        # S/T's actual inverse expands the source controlled-phase leaf through And.
        assert events == count + controls - 2 + int(adjoint) * (
            count if family == "ST" else count // 2
        )
        assert inventory["t"] + inventory["tdg"] == 4 * events + (
            count // 2 if adjoint and family == "ST" else 0
        )
        expected_operator = operator.mH if adjoint else operator
        for column in columns:
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[:: 1 << work] = column
            expected = torch.zeros_like(initial)
            expected[:: 1 << work] = expected_operator @ column
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial, ir.instructions, history),
                    expected / 2 ** (events / 2),
                    atol=4e-12,
                    rtol=0,
                )


def test_symbolic_width_is_bound_before_capturing_the_callable(source: Any) -> None:
    import attrs
    import cirq
    from qualtran import BQUInt, Register
    from sympy import Symbol

    original = source.ApplyGateToLthQubit(
        Register("address", BQUInt(Symbol("n", positive=True, integer=True), 2)),
        lambda index: cirq.S if index else cirq.H,
    )
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(original)
    resolved = attrs.evolve(
        original, selection_regs=(Register("address", BQUInt(1, 2)),)
    )
    emitter = import_register_emitter(resolved)
    assert emitter.layout == (("control", 1), ("address", 1), ("target", 2))
    assert emitter.snapshots[0].n_wires >= 4

    symbolic_extent = attrs.evolve(
        original,
        selection_regs=(
            Register("address", BQUInt(2, Symbol("L", positive=True, integer=True))),
        ),
    )
    with pytest.raises(ValueError, match="Symbolic iteration size"):
        import_register_emitter(symbolic_extent)
