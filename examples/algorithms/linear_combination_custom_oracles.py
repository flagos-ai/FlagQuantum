"""Supply PREPARE and SELECT to encode (X+Z)/2, retaining circuit.append(op).

Run python -m examples.algorithms.linear_combination_custom_oracles.
"""

from collections.abc import Mapping
from math import sqrt

import torch

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_ir,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    def prepare(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        wire = ports["address"][0]
        gates = (
            Instruction("h", (wire,)),
            Instruction("s", (wire,)),
            Instruction("cx", (wire, *ports["tag"])),
        )
        return (
            (
                Instruction("cx", (wire, *ports["tag"])),
                Instruction("sdg", (wire,)),
                Instruction("h", (wire,)),
            )
            if inverse
            else gates
        )

    prepared = BlackBoxPrepare((("address", 2),), (("tag", 1),), prepare, 1.0)

    def select(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        def child(index: int, predicate: int) -> tuple[Instruction, ...]:
            return (
                Instruction(
                    "cx" if index < 2 else "cz", (predicate, ports["system"][0])
                ),
            )

        return indexed_component_ir(
            (ports["branch"],),
            (4,),
            child,
            workspace=work,
            classical_bits=bits,
            adjoint=inverse,
        )

    selected = SelectBlockEncoding((("branch", 2),), (("system", 1),), select, prepared)

    def block(label: str) -> UnitaryBlockEncoding:
        def emit(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> tuple[Instruction, ...]:
            return (Instruction(label, ports["q"]),)

        return UnitaryBlockEncoding((("q", 1),), emit)

    encoding = LinearCombinationBlockEncoding(
        (block("x"), block("z")),
        (1.0, 1.0),
        8,
        prepare_override=prepared,
        select_override=selected,
    )
    initial = torch.zeros(32, dtype=torch.complex128)
    psi = torch.tensor([1, 1j], dtype=torch.complex128) / sqrt(2)
    initial[0], initial[16] = psi[0], psi[1]
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(
        encoding.make_on(
            system=(0,),
            ancilla=(1, 2),
            resource=(3,),
            prepare_workspace=(),
            prepare_classical_bits=(),
            traversal_workspace=(4,),
            traversal_classical_bits=(0,),
        )
    )
    ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=8, seed=61, strategy="batched")
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    desired = ((x + z) @ psi) / 2
    torch.testing.assert_close(
        result.final_states[:, [0, 16]], desired.expand(8, 2), atol=3e-12, rtol=0
    )
    torch.testing.assert_close(
        result.final_states[:, 1::2],
        torch.zeros_like(result.final_states[:, 1::2]),
        atol=3e-12,
        rtol=0,
    )
    circuit.append(
        encoding.make_on(
            system=(0,),
            ancilla=(1, 2),
            resource=(3,),
            prepare_workspace=(),
            prepare_classical_bits=(),
            traversal_workspace=(4,),
            traversal_classical_bits=(0,),
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=8, seed=67, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=3e-12,
        rtol=0,
    )
    prepare_cost = ComponentCost("SuppliedPrepare", 3, GateCost(clifford=3))
    select_cost = indexed_component_cost(
        (2,),
        (4,),
        (ComponentCost("CX/CZ", 2, GateCost(clifford=1)),) * 4,
        target_width=1,
    )
    report = estimate_ftqc_resources(
        encoding.resource_model(select_cost, prepare_cost, prepare_cost), emitted_ir=ir
    )
    assert report.emitted is not None
    print("Supplied selection width", encoding.prepare.selection_width)
    print("Model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/Tdag", report.emitted.t_count)
    print("Model measurements", report.logical.gates.measurement)
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print("Custom (X+Z)/2 projection, clean work and full complex inverse verified")


if __name__ == "__main__":
    main()
