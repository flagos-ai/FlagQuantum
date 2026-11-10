"""Original field erasure builds and executes through owned native components."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.finite_field.product_erasure_program import (
    product_erasure_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,modulus", [(1, 3), (2, 7), (3, 11), (3, 13), (5, 37), (8, 283), (8, 285)]
)
def test_unmodified_source_builder_replaces_native_construction(
    width: int, modulus: int
) -> None:
    from galois import Poly
    from qualtran import QGF
    from qualtran.bloqs.basic_gates import CZ, Discard, MeasureX
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulMBUC, Parity
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    polynomial = Poly.Int(modulus)
    dtype = FiniteFieldEncoding(2, width, polynomial).register_type()
    native = product_erasure_program(dtype)
    main = native.functions[0]
    assert main.registers is not None
    builder = ComponentProgramBuilder(Signature(main.registers), native.functions[1:])

    class SourceBuilder:
        """Translate external source bloqs into existing native component calls."""

        def split(self, reg: Any) -> Any:
            return builder.call_dict("split", {"reg": reg})["reg"]

        def join(self, reg: Any, *, dtype: Any) -> Any:
            return builder.call_dict("join", {"reg": reg})["reg"]

        def add(self, bloq: Any, **registers: Any) -> Any:
            if isinstance(bloq, MeasureX):
                return builder.call_dict("measure_x", registers)["c"]
            if isinstance(bloq, Discard):
                assert builder.call_dict("discard", registers) == {}
                return None
            if isinstance(bloq, Parity):
                return builder.call_tuple(f"parity_{bloq.n}", registers)
            assert bloq.subbloq == CZ()
            return builder.call_tuple("controlled_cz", registers)

    source = GF2MulMBUC(QGF(2, width, polynomial))
    outputs = source.build_composite_bloq(SourceBuilder(), **builder.inputs)
    assert builder.finalize(outputs) == native
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    bound = bind_register_program(native, inputs)
    assert bound.outputs == inputs[:2]
    assert bound.classical_outputs == ()
    assert bound.events == ()
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": width,
        "measure": width,
        "classical_xor": width,
        "cz": width**2,
    }
    source_cost = get_cost_value(source, QECGatesCost())
    assert source_cost.clifford == width**2
    assert source_cost.measurement == source_cost.total_t_count() == 0


def state_with_known_product(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    field = FiniteFieldEncoding(2, width).gf_type
    generator = torch.Generator().manual_seed(108430 + width)
    logical = torch.randn(
        1 << (2 * width + 1), dtype=torch.complex128, generator=generator
    )
    logical /= logical.norm()
    state = torch.zeros(1 << (3 * width + 2), dtype=torch.complex128)
    for a, b, spectator in product(range(1 << width), range(1 << width), range(2)):
        logical_index = ((a << width | b) << 1) | spectator
        target = int(field(a) * field(b))
        physical_index = (((a << width | b) << width | target) << 2) | (spectator << 1)
        state[physical_index] = logical[logical_index]
    return state, logical


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("rebuild", [False, True])
def test_every_complex_branch_retains_phase_and_unmeasured_work_zero(
    width: int,
    rebuild: bool,
) -> None:
    dtype = FiniteFieldEncoding(2, width).register_type()
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    program = product_erasure_program(dtype)
    if rebuild:
        copied = copy_component_program(program)
        assert bind_register_program(copied, inputs) == bind_register_program(
            program, inputs
        )
        program = copied
    bound = bind_register_program(program, inputs, clean=(3 * width + 1,))
    gates = (
        Circuit(3 * width + 2)
        .append(FieldOperation(3 * width + 2, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    measured = [g.wires[0] for g in gates if g.name == "measure"]
    state, logical = state_with_known_product(width)
    total = 0.0
    for history in product((0, 1), repeat=width):
        branch = forced_branch(state, gates, history).reshape((2,) * (3 * width + 2))
        indices: list[int | slice] = [slice(None)] * (3 * width + 2)
        for wire, outcome in zip(measured, history, strict=True):
            indices[wire] = outcome
        indices[-1] = 0
        actual = branch[tuple(indices)].flatten()
        expected = logical / (2 ** (width / 2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-15)
        assert torch.count_nonzero(branch[..., 1]) == 0
        # Every destroyed physical wire retains the branch's Z projection.
        assert torch.count_nonzero(branch) == torch.count_nonzero(actual)
        total += float(branch.abs().square().sum())
    assert total == pytest.approx(1)


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_actual_append_runtime_executes_owned_erasure(strategy: str) -> None:
    width = 2
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    bound = bind_register_program(
        product_erasure_program(FiniteFieldEncoding(2, width).register_type()),
        inputs,
        clean=(7,),
    )
    state, _ = state_with_known_product(width)
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(8, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=32, seed=1084, strategy=strategy)
    for bits, actual in zip(
        result.classical_bits, result.final_states.reshape(32, 256), strict=True
    ):
        history = tuple(int(bits[i]) for i in range(width))
        expected = forced_branch(state, bound.circuit.instructions, history)
        expected /= expected.norm()
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-15)
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("uint", 2),
        RegisterType(
            "finite_field",
            2,
            parameters=(("characteristic", 2), ("degree", 2), ("order", 4)),
        ),
    ],
)
def test_erasure_requires_resolved_nominal_polynomial_identity(
    dtype: RegisterType,
) -> None:
    with pytest.raises(ValueError, match="resolved binary polynomial"):
        product_erasure_program(dtype)


def test_owned_erasure_keeps_workspace_retirement_refusal() -> None:
    dtype = FiniteFieldEncoding(2, 2).register_type()
    with pytest.raises(ValueError, match="retirement ownership"):
        bind_register_program(
            product_erasure_program(dtype),
            ((0, 1), (2, 3), (4, 5)),
            input_ownership=((4, False), (5, False)),
        )
