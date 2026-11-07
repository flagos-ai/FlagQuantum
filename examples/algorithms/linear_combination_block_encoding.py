"""Append the block encoding of (X - 3 Z)/4 and verify its exact inverse.

Run python -m examples.algorithms.linear_combination_block_encoding.
"""

from collections.abc import Callable, Mapping
from math import sqrt

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
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
                max(predicate, *ports["system"]) + 1,
                (Instruction("c" + label, (predicate, *ports["system"])),),
            )

        return bind

    encoding = LinearCombinationBlockEncoding((block("x"), block("z")), (1.0, -3.0), 1)
    binders = (factory("x"), factory("z"))
    psi = torch.tensor([1, 1j], dtype=torch.complex128) / sqrt(2)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0], initial[128] = psi[0], psi[1]
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    operation = encoding.make_on(
        binders,
        system=(0,),
        ancilla=(1,),
        resource=(2, 3, 4, 5),
        prepare_workspace=(6, 7),
        prepare_classical_bits=(0,),
        traversal_workspace=(),
        traversal_classical_bits=(),
        workspaces=((), ()),
        classical_bits=((), ()),
    )
    circuit.append(operation)
    ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=8, seed=53, strategy="batched")
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    expected = ((x - 3 * z) @ psi) / 4
    torch.testing.assert_close(
        result.final_states[:, [0, 128]], expected.expand(8, 2), atol=4e-12, rtol=0
    )
    torch.testing.assert_close(
        result.final_states.reshape(8, -1, 4)[:, :, 1:],
        torch.zeros_like(result.final_states.reshape(8, -1, 4)[:, :, 1:]),
        atol=4e-12,
        rtol=0,
    )
    # Retain the failure amplitudes; only private work is required to clear.
    circuit.append(
        encoding.make_on(
            binders,
            system=(0,),
            ancilla=(1,),
            resource=(2, 3, 4, 5),
            prepare_workspace=(6, 7),
            prepare_classical_bits=(0,),
            traversal_workspace=(),
            traversal_classical_bits=(),
            workspaces=((), ()),
            classical_bits=((), ()),
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=8, seed=59, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=4e-12,
        rtol=0,
    )
    prep = encoding.coefficient_preparation.resource_model()
    select = encoding.select.resource_model(
        (ComponentCost("CX/CZ", 2, GateCost(clifford=1)),) * 2
    )
    report = estimate_ftqc_resources(
        encoding.resource_model(select, prep, prep), emitted_ir=ir
    )
    assert report.emitted is not None
    print("Normalization alpha", encoding.alpha)
    print(
        "Model T equivalents (abstract Fredkin)", report.logical.gates.total_t_count()
    )
    print("Emitted T/Tdag (exact Fredkin)", report.emitted.t_count)
    print("Model measurements", report.logical.gates.measurement)
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print("Projected (X-3Z)/4, private work and full complex inverse verified")


if __name__ == "__main__":
    main()
