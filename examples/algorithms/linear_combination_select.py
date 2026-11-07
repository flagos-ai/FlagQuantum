"""Append signed SELECT on a coherent address, then its exact source inverse.

Run python -m examples.algorithms.linear_combination_select.
"""

from collections.abc import Callable, Mapping

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_select import (
    LinearCombinationSelect,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    def block(label: str) -> UnitaryBlockEncoding:
        def emit(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> tuple[Instruction, ...]:
            return (Instruction(label, ports["q"]),)

        return UnitaryBlockEncoding((("q", 1),), emit)

    def factory(
        label: str,
    ) -> Callable[
        [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        FieldOperation,
    ]:
        def bind(
            predicate: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> FieldOperation:
            return FieldOperation(
                4, (Instruction("c" + label, (predicate, *ports["system"])),)
            )

        return bind

    select = LinearCombinationSelect(
        tuple(block(label) for label in ("x", "z", "x")), (1.0, -1.0, 1.0)
    )
    factories = tuple(factory(label) for label in ("x", "z", "x"))
    initial = torch.zeros(16, dtype=torch.complex128)
    amplitudes = torch.tensor(
        [1 + 2j, 3 - 1j, -2 + 4j, 1j, 2, -3j, 1 - 1j, -2], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    initial[::2] = amplitudes
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    forward = select.make_on(
        factories,
        selection=(0, 1),
        system=(2,),
        traversal_workspace=(3,),
        traversal_classical_bits=(0,),
        workspaces=((),) * 3,
        classical_bits=((),) * 3,
    )
    circuit.append(forward)
    ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=8, seed=43, strategy="batched")
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    minus_z = torch.diag(torch.tensor([-1, 1], dtype=torch.complex128))
    desired = torch.zeros_like(initial)
    # The pinned bounded tree aliases padded address 3 to term 2.
    for index, matrix in enumerate((x, minus_z, x, x)):
        desired.reshape(4, 2, 2)[index, :, 0] = matrix @ amplitudes.reshape(4, 2)[index]
    torch.testing.assert_close(
        result.final_states, desired.expand_as(result.final_states), atol=3e-12, rtol=0
    )
    circuit.append(
        select.make_on(
            factories,
            selection=(0, 1),
            system=(2,),
            traversal_workspace=(3,),
            traversal_classical_bits=(0,),
            workspaces=((),) * 3,
            classical_bits=((),) * 3,
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=8, seed=47, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=3e-12,
        rtol=0,
    )
    costs = (ComponentCost("CX/CZ", 2, GateCost(clifford=1)),) * 3
    report = estimate_ftqc_resources(select.resource_model(costs), emitted_ir=ir)
    assert report.emitted is not None
    print("Model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/Tdag", report.emitted.t_count)
    print("Model measurements", report.logical.gates.measurement)
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print(
        "Full complex signed SELECT, padded address and private work verified; inverse restored"
    )


if __name__ == "__main__":
    main()
