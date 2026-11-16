# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 FindECCPrivateKey.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/find_ecc_private_key.py

"""Two consecutive point phase estimations over the same coordinate state."""

from dataclasses import dataclass
from functools import cached_property

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ...dynamic import DynamicCircuit
from .elliptic_curve_phase_estimation import EllipticCurvePhaseEstimation
from .elliptic_curve_point import ECPoint


def elliptic_curve_double_phase_source_graph() -> SymbolicCallGraph[str]:
    """Declare the source's two generic phase-estimation calls."""
    root = "FindECCPrivateKey"
    return symbolic_call_graph(root, {root: {"ECPhaseEstimateR": 2}})


@dataclass(frozen=True)
class _BoundEllipticCurveDoublePhase:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Elliptic-curve double phase requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveDoublePhase:
    """Compose the locked raw initialization, two phase operations and dirty exits.

    Both phase stages share the coordinate state and use independent zeroed
    phase qubits and output slots. Scratch is reused on the clean child
    domain. The source consumes the coordinate registers with dirty release;
    the native caller owns those wires and may inspect them after measurement.
    When the Montgomery factor differs from one, raw initialization may
    encode a different starting point. This preserves the stored source word.
    """

    width: int
    base_point: ECPoint
    public_key: ECPoint
    window: int
    multiplication_window: int = 1

    def __post_init__(self) -> None:
        if self.base_point.mod != self.public_key.mod:
            raise ValueError("Inconsistent moduli in the two points.")
        if self.base_point.curve_a != self.public_key.curve_a:
            raise ValueError("Inconsistent curve parameters in the two points.")

    @property
    def source_signature(self) -> Signature:
        return Signature(())

    @cached_property
    def _stages(
        self,
    ) -> tuple[EllipticCurvePhaseEstimation, EllipticCurvePhaseEstimation]:
        return (
            EllipticCurvePhaseEstimation(
                self.width, self.base_point, self.window, self.multiplication_window
            ),
            EllipticCurvePhaseEstimation(
                self.width, self.public_key, self.window, self.multiplication_window
            ),
        )

    def workspace_requirements(self) -> tuple[int, int]:
        """Count reusable arithmetic scratch, separate from four data/phase words."""
        first, second = self._stages
        first_q, first_c = first.workspace_requirements()
        second_q, second_c = second.workspace_requirements()
        return max(first_q, second_q), max(first_c, second_c)

    def make_on(
        self,
        *,
        x: tuple[int, ...],
        y: tuple[int, ...],
        first_phase: tuple[int, ...],
        second_phase: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        first_results: tuple[int, ...],
        second_results: tuple[int, ...],
    ) -> _BoundEllipticCurveDoublePhase:
        n = self.width
        q, c = self.workspace_requirements()
        if any(
            len(word) != n
            for word in (x, y, first_phase, second_phase, first_results, second_results)
        ):
            raise ValueError(
                "Double phase requires equal data, phase and result word widths"
            )
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Double phase requires exact arithmetic workspace sizes")
        wires = (*x, *y, *first_phase, *second_phase, *workspace)
        for values in (wires, (*classical_bits, *first_results, *second_results)):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Double phase requires distinct nonnegative wires and slots"
                )
        raw_x, raw_y = int(self.base_point.x), int(self.base_point.y)
        if any(not 0 <= value < 1 << n for value in (raw_x, raw_y)):
            raise ValueError("Raw source coordinates must fit the declared width")
        x, y, first_phase, second_phase = (
            tuple(range(i * n, (i + 1) * n)) for i in range(4)
        )
        scratch = tuple(range(4 * n, len(wires)))
        circuit = DynamicCircuit(len(wires))
        for word, value in ((x, raw_x), (y, raw_y)):
            for i, wire in enumerate(word):
                if value >> (n - i - 1) & 1:
                    circuit.x(wire)
        for stage, phase, results in zip(
            self._stages,
            (first_phase, second_phase),
            (first_results, second_results),
            strict=True,
        ):
            stage_q, stage_c = stage.workspace_requirements()
            circuit.append(
                stage.make_on(
                    x=x,
                    y=y,
                    phase=phase,
                    workspace=scratch[:stage_q],
                    classical_bits=classical_bits[:stage_c],
                    result_bits=results,
                )
            )
        return _BoundEllipticCurveDoublePhase(wires, circuit.to_ir().instructions)
