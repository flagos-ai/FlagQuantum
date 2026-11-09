"""Append an Add/T tensor block and verify complex amplitudes and work cleanup.

Run python -m examples.algorithms.tensor_product_block_encoding.
"""

from cmath import exp
from collections.abc import Callable, Mapping
from math import pi

import torch

from examples.algorithms.unitary_block_encoding import child as add_child
from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.circuit import _CircuitOperation
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def t_child(
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return (Instruction("tdg" if adjoint else "t", ports["q"]),)


def main() -> None:
    blocks = (
        UnitaryBlockEncoding((("a", 2), ("b", 2)), add_child),
        UnitaryBlockEncoding((("q", 1),), t_child),
    )
    tensor = TensorProductBlockEncoding(blocks)

    def factory(block: UnitaryBlockEncoding) -> Callable[
        [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        _CircuitOperation,
    ]:
        def bind(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            inverse: bool,
        ) -> _CircuitOperation:
            return block.make_on(
                system=ports["system"],
                ancilla=ports["ancilla"],
                resource=ports["resource"],
                workspace=work,
                classical_bits=slots,
                adjoint=inverse,
            )

        return bind

    binders = tuple(factory(block) for block in blocks)
    amplitudes = torch.complex(torch.arange(1, 33).double(), torch.arange(32).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    for value in range(32):
        left, right, target = value >> 3, (value >> 1) & 3, value & 1
        output = (left << 3) | (((left + right) % 4) << 1) | target
        expected[output << 1] = amplitudes[value] * (exp(1j * pi / 4) if target else 1)
    op = tensor.make_on(
        binders,
        system=(0, 1, 2, 3, 4),
        workspaces=((5,), ()),
        classical_bits=((0,), ()),
    )
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    forward_ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=16, seed=61, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=4e-13,
        rtol=4e-13,
    )
    inverse = tensor.make_on(
        binders,
        system=(0, 1, 2, 3, 4),
        workspaces=((5,), ()),
        classical_bits=((0,), ()),
        adjoint=True,
    )
    circuit.append(inverse)
    restored = run_dynamic(circuit, shots=16, seed=67, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=4e-13,
        rtol=4e-13,
    )
    costs = (
        blocks[0].resource_model(source_add_cost(2, 2)),
        blocks[1].resource_model(ComponentCost("T", 1, GateCost(t=1))),
    )
    report = estimate_ftqc_resources(
        tensor.resource_model(costs), emitted_ir=forward_ir
    )
    assert report.emitted is not None
    print("Model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/T-dagger instructions", report.emitted.t_count)
    print("Model logical peak / reserved wires", report.logical.peak_logical_qubits, 6)
    print("Cross-block complex amplitudes, exact inverse and clean work verified")


if __name__ == "__main__":
    main()
