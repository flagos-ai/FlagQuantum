"""Append a product with nonzero failure branches and verify its projected block.

Run python -m examples.algorithms.product_block_encoding.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import pi, sqrt
from typing import Literal

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class SignalBlock:
    """Example block with encoded T/sqrt(2) or H/sqrt(2), not a SELECT oracle."""

    kind: Literal["t", "h"]
    ancilla_bitsize: int = 3
    system_bitsize: int = 1
    resource_bitsize: int = 0
    alpha: float = 1.0
    epsilon: float = 0.0

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(self) -> ComponentCost:
        return ComponentCost(
            "SignalBlock",
            1 + self.ancilla_bitsize,
            GateCost(t=1, clifford=2) if self.kind == "t" else GateCost(clifford=3),
        )

    def make_on(
        self,
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        resource: tuple[int, ...] = (),
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> FieldOperation:
        if len(system) != 1 or len(ancilla) != self.ancilla_bitsize:
            raise ValueError("Signal block ports must match their widths")
        rotate = Instruction("h", (ancilla[0],))
        couple = Instruction("cx", (ancilla[0], system[0]))
        gate = Instruction("tdg" if adjoint and self.kind == "t" else self.kind, system)
        gates = (gate, couple, rotate) if adjoint else (rotate, couple, gate)
        return FieldOperation(max((*system, *ancilla)) + 1, gates)


def main() -> None:
    blocks = (SignalBlock("t"), SignalBlock("h"))
    root = ProductBlockEncoding(blocks)

    def factory(
        block: SignalBlock,
    ) -> Callable[
        [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        FieldOperation,
    ]:
        def bind(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            bits: tuple[int, ...],
            inverse: bool,
        ) -> FieldOperation:
            return block.make_on(
                system=ports["system"], ancilla=ports["ancilla"], adjoint=inverse
            )

        return bind

    binders = tuple(factory(block) for block in blocks)
    initial = torch.zeros(128, dtype=torch.complex128)
    psi = torch.tensor([1, 1j], dtype=torch.complex128) / sqrt(2)
    initial[0], initial[64] = psi[0], psi[1]
    forward = root.make_on(
        binders,
        system=(0,),
        ancilla=(1, 2, 3, 4),
        workspaces=((), ()),
        classical_bits=((), ()),
        flag_workspace=(5, 6),
        flag_classical_bits=(0, 1),
    )
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(forward)
    forward_ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=16, seed=71, strategy="batched")
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / sqrt(2)
    t = torch.diag(
        torch.tensor(
            [1, complex(torch.exp(torch.tensor(1j * pi / 4, dtype=torch.complex128)))],
            dtype=torch.complex128,
        )
    )
    expected = (t @ h @ psi) / 2
    torch.testing.assert_close(
        result.final_states[:, [0, 64]], expected.expand(16, 2), atol=5e-13, rtol=5e-13
    )
    probability = (result.final_states[:, [0, 64]].abs() ** 2).sum(-1)
    torch.testing.assert_close(
        probability, torch.full_like(probability, 0.25), atol=5e-13, rtol=0
    )
    inverse = root.make_on(
        binders,
        system=(0,),
        ancilla=(1, 2, 3, 4),
        workspaces=((), ()),
        classical_bits=((), ()),
        flag_workspace=(5, 6),
        flag_classical_bits=(0, 1),
        adjoint=True,
    )
    circuit.append(inverse)
    restored = run_dynamic(circuit, shots=16, seed=73, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=5e-13,
        rtol=5e-13,
    )
    costs = tuple(block.resource_model() for block in blocks)
    model = estimate_ftqc_resources(root.resource_model(costs), emitted_ir=forward_ir)
    decomposed = estimate_ftqc_resources(
        root.resource_model(costs, flag_model="decomposition")
    )
    assert model.emitted is not None
    print(
        "Model T equivalents / emitted T instructions",
        model.logical.gates.total_t_count(),
        model.emitted.t_count,
    )
    print(
        "Abstract / decomposition model Clifford counts",
        model.logical.gates.clifford,
        decomposed.logical.gates.clifford,
    )
    print("Model logical peak / reserved wires", model.logical.peak_logical_qubits, 7)
    print(
        "Emitted measurement instructions",
        sum(g.name == "measure" for g in forward_ir.instructions),
    )
    print(
        "Emitted primitive instruction counts",
        dict(Counter(g.name for g in forward_ir.instructions)),
    )
    print("Zero-signal success probability", float(probability[0]))
    print("Failure flags are logical outputs; private work and exact inverse verified")


if __name__ == "__main__":
    main()
