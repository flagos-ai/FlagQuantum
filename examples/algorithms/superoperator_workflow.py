# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition adapted from Qualtran v0.7.0 simulation/supertensor.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/supertensor.ipynb

"""Coherent evolution, measurement dilation and explicit environment tracing.

Run ``python -m examples.algorithms.superoperator_workflow``. This example
composes existing tensor owners; no foreign graph or second executor is used.
"""

from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.primitives.measurement import (
    discard_qubit_ir,
    measure_z_clear_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.conjugate_pairs import tensor_conjugate_pairs
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices
from flagquantum.simulation.unitary import get_unitary


@dataclass(frozen=True)
class OpenSystemExample:
    """Explicit numerical leaves and logical ports; integer labels are not wires.

    Ports are row-major/MSB-first. Discarded labels have no logical output.
    Physical allocation/reset is separate from this numerical representation.
    """

    nodes: tuple[TensorNetworkNode, ...]
    signature: Signature
    incoming: dict[str, tuple[tuple[int, ...], ...]]
    outgoing: dict[str, tuple[tuple[int, ...], ...]]
    discarded: tuple[int, ...] = ()

    def pure(self) -> torch.Tensor:
        """Contract an ordinary state/isometry; refuse open-system discard."""
        if self.discarded:
            raise ValueError("Pure tensor simulation cannot discard an environment")
        return contract_register_tensors(
            self.nodes, self.signature, self.incoming, self.outgoing
        )

    def doubled_nodes(self) -> tuple[TensorNetworkNode, ...]:
        """Double every leaf and trace only declared environment edges."""
        return trace_conjugate_indices(
            tensor_conjugate_pairs(self.nodes), self.discarded
        )

    def channel(self) -> torch.Tensor:
        """Order output-forward/backward, then input-forward/backward axes."""
        return contract_register_superoperator(
            self.doubled_nodes(), self.signature, self.incoming, self.outgoing
        )


def superoperator_example(
    kind: Literal["plus", "hadamard", "coin", "dilation", "measurement"],
) -> OpenSystemExample:
    """Compose the notebook's five concrete graphs from native numerical leaves.

    Zero preparation owns a logical result. The dilation's fresh c output is
    retained, not scratch; measurement discards q and retains only c. Cast is an
    identity on numerical data, not dephasing or a hardware instruction.
    """
    like = torch.empty((), dtype=torch.complex128)
    h = get_unitary(CircuitIR(1, (Instruction("h", (0,)),)), dtype=like.dtype)
    zero = torch.tensor([1, 0], dtype=like.dtype)
    q = quantum_bit_type()
    c = classical_bit_type()
    if kind == "hadamard":
        return OpenSystemExample(
            (TensorNetworkNode(h, (1, 0), "H"),),
            Signature([Register("q", q)]),
            {"q": ((0,),)},
            {"q": ((1,),)},
        )
    if kind in {"plus", "coin"}:
        nodes: tuple[TensorNetworkNode, ...] = (
            TensorNetworkNode(zero, (0,), "Zero"),
            TensorNetworkNode(h, (1, 0), "H"),
        )
        if kind == "plus":
            return OpenSystemExample(
                nodes,
                Signature([Register("q", q, side=RegisterSide.RIGHT)]),
                {},
                {"q": ((1,),)},
            )
        return OpenSystemExample(
            (*nodes, z_measurement_copy_node(1, 2, 3, like)),
            Signature([Register("c", c, side=RegisterSide.RIGHT)]),
            {},
            {"c": ((2,),)},
            (3,),
        )
    cz = get_unitary(
        CircuitIR(2, (Instruction("cz", (0, 1)),)), dtype=like.dtype
    ).reshape(2, 2, 2, 2)
    nodes = (
        TensorNetworkNode(zero, (1,), "Zero"),
        TensorNetworkNode(h, (2, 1), "H"),
        TensorNetworkNode(cz, (3, 4, 2, 0), "CZ"),
        TensorNetworkNode(h, (5, 3), "H"),
        TensorNetworkNode(torch.eye(2, dtype=like.dtype), (6, 5), "Cast QBit to CBit"),
    )
    if kind not in {"dilation", "measurement"}:
        raise ValueError(f"Unknown superoperator example {kind!r}")
    discarded = kind == "measurement"
    signature = Signature(
        [
            Register(
                "q", q, side=RegisterSide.LEFT if discarded else RegisterSide.THRU
            ),
            Register("c", c, side=RegisterSide.RIGHT),
        ]
    )
    outgoing: dict[str, tuple[tuple[int, ...], ...]] = (
        {"c": ((6,),)} if discarded else {"q": ((4,),), "c": ((6,),)}
    )
    return OpenSystemExample(
        nodes, signature, {"q": ((0,),)}, outgoing, (4,) if discarded else ()
    )


@dataclass(frozen=True)
class ZMeasurementDilation:
    """Record Z through H-CZ-H, optionally materialize c and clear consumed slots.

    Caller owns q=0, fresh zero record=1, spectator=2. The coherent form retains
    both outputs. The measured form records c in classical slot0 and clears q
    and record; it has no coherent q output and never resets the spectator.
    """

    measured: bool = False

    def instructions(self) -> tuple[Instruction, ...]:
        gates: tuple[Instruction, ...] = (
            Instruction("h", (1,)),
            Instruction("cz", (1, 0)),
            Instruction("h", (1,)),
        )
        if self.measured:
            gates += (*measure_z_clear_ir(1, 0), *discard_qubit_ir(0))
        return gates

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        """Append the same explicit primitive stream to an ordinary circuit."""
        return circuit.compose(CircuitIR(circuit.n_qubits, self.instructions()))


def main() -> None:
    coherent = superoperator_example("plus").channel()
    coin = superoperator_example("coin").channel()
    h = superoperator_example("hadamard").channel().reshape(4, 4)
    torch.testing.assert_close(
        (h @ coherent.reshape(4)).reshape(2, 2),
        torch.diag(torch.tensor([1, 0], dtype=torch.complex128)),
    )
    torch.testing.assert_close((h @ coin.reshape(4)).reshape(2, 2), coin)
    print("Coherent density:", coherent.tolist())
    print("Coin density:", coin.tolist())
    print(
        "Measurement channel shape:",
        tuple(superoperator_example("measurement").channel().shape),
    )


if __name__ == "__main__":
    main()
