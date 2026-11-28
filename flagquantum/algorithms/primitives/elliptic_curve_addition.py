# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECAdd.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add.py

"""Bind all six stages of reversible affine elliptic-curve addition."""

from collections.abc import Hashable
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import montgomery_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from ...simulation.elliptic_curve_values import elliptic_curve_addition_values
from .elliptic_curve_addition_cleanup import EllipticCurveAdditionCleanup
from .elliptic_curve_addition_coordinates import EllipticCurveAdditionCoordinates
from .elliptic_curve_addition_finish import EllipticCurveAdditionFinish
from .elliptic_curve_addition_flags import EllipticCurveAdditionFlags
from .elliptic_curve_addition_model import (
    _STAGE_NAMES,
    elliptic_curve_addition_parameterized_graph,
)
from .elliptic_curve_addition_products import EllipticCurveAdditionProducts
from .elliptic_curve_addition_slope import EllipticCurveAdditionSlope
from .elliptic_curve_window_product import CurveWindowProduct


def elliptic_curve_addition_signature(width: SymbolicInt) -> Signature:
    """Declare the five source Montgomery through registers."""
    word = montgomery_register_type(width)
    return Signature(Register(name, word) for name in ("a", "b", "x", "y", "lam_r"))


def elliptic_curve_addition_source_graph() -> SymbolicCallGraph[str]:
    """Declare the six direct calls shared by symbolic and concrete instances."""
    return symbolic_call_graph("ECAdd", {"ECAdd": dict.fromkeys(_STAGE_NAMES, 1)})


@dataclass(frozen=True)
class _BoundEllipticCurveAddition:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Elliptic-curve addition requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveAddition:
    """Add the first encoded point into the second with a supplied tangent.

    Coordinates and the precomputed tangent use big-endian Montgomery words,
    with (0, 0) reserved for infinity. Reserve clean work for the five flags,
    the slope, and the shared stage scratch. Clean output is conditional on
    the locked circuit's arithmetic and release promises. In particular,
    source decompositions can produce noncanonical residues and retain a flag
    on some curve input pairs. The caller-owned wires expose these residuals.
    """

    width: int
    modulus: int
    window: int = 1

    def __post_init__(self) -> None:
        CurveWindowProduct(self.width, self.modulus, self.window)

    @property
    def source_signature(self) -> Signature:
        return elliptic_curve_addition_signature(self.width)

    def source_call_graph(self) -> SymbolicCallGraph[Hashable]:
        """Return parameterized source estimates, separately from emitted gates."""
        return elliptic_curve_addition_parameterized_graph(
            self.width, self.modulus, self.window
        )

    def _stages(self) -> tuple[
        EllipticCurveAdditionFlags,
        EllipticCurveAdditionSlope,
        EllipticCurveAdditionCoordinates,
        EllipticCurveAdditionProducts,
        EllipticCurveAdditionCleanup,
        EllipticCurveAdditionFinish,
    ]:
        n, p, w = self.width, self.modulus, self.window
        return (
            EllipticCurveAdditionFlags(n, p),
            EllipticCurveAdditionSlope(n, p, w),
            EllipticCurveAdditionCoordinates(n, p, w),
            EllipticCurveAdditionProducts(n, p, w),
            EllipticCurveAdditionCleanup(n, p, w),
            EllipticCurveAdditionFinish(n, p),
        )

    def classical_values(
        self, a: int, b: int, x: int, y: int, lam_r: int
    ) -> tuple[int, int, int, int, int]:
        """Evaluate the source affine callback on five stored Montgomery words.

        The tangent determines the curve coefficient. The decomposition's
        arithmetic and clean-release promises have separate execution evidence.
        """
        return elliptic_curve_addition_values(
            self.width, self.modulus, a, b, x, y, lam_r
        )

    def workspace_requirements(self) -> tuple[int, int]:
        requirements = tuple(stage.workspace_requirements() for stage in self._stages())
        return self.width + 5 + max(q for q, _ in requirements), max(
            c for _, c in requirements
        )

    def make_on(
        self,
        *,
        a: tuple[int, ...],
        b: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        lam_r: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveAddition:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (a, b, x, y, lam_r)):
            raise ValueError("Each curve addition operand must have the declared width")
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Elliptic-curve addition requires exact workspace sizes")
        wires = (*a, *b, *x, *y, *lam_r, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Elliptic-curve addition requires distinct nonnegative wires and slots"
                )

        a, b, x, y, lam_r = (tuple(range(i * n, (i + 1) * n)) for i in range(5))
        flags = tuple(range(5 * n, 5 * n + 5))
        lam = tuple(range(5 * n + 5, 6 * n + 5))
        scratch = tuple(range(6 * n + 5, len(wires)))
        coordinates = {"a": a, "b": b, "x": x, "y": y}
        init, choose, update, products, cleanup, finish = self._stages()
        circuit = DynamicCircuit(len(wires))

        def bind_work(requirements: tuple[int, int]) -> dict[str, tuple[int, ...]]:
            qubits, slots = requirements
            return {
                "workspace": scratch[:qubits],
                "classical_bits": classical_bits[:slots],
            }

        circuit.append(
            init.make_on(
                **coordinates, flags=flags, **bind_work(init.workspace_requirements())
            )
        )
        circuit.append(
            choose.make_on(
                **coordinates,
                f1=flags[0],
                ctrl=flags[-1],
                lam=lam,
                lam_r=lam_r,
                **bind_work(choose.workspace_requirements()),
            )
        )
        circuit.append(
            update.make_on(
                **coordinates,
                ctrl=flags[-1],
                lam=lam,
                **bind_work(update.workspace_requirements()),
            )
        )
        circuit.append(
            products.make_on(
                x=x,
                y=y,
                lam=lam,
                **bind_work(products.workspace_requirements()),
            )
        )
        circuit.append(
            cleanup.make_on(
                **coordinates,
                ctrl=flags[-1],
                lam=lam,
                lam_r=lam_r,
                **bind_work(cleanup.workspace_requirements()),
            )
        )
        circuit.append(
            finish.make_on(
                **coordinates,
                flags=flags,
                **bind_work(finish.workspace_requirements()),
            )
        )
        return _BoundEllipticCurveAddition(wires, circuit.to_ir().instructions)
