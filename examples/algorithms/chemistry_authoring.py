# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's chemistry authoring tutorial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/writing_algorithms.ipynb

"""Dense preparation sketches and register wiring for the authoring tutorial."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flagquantum.algorithms.primitives.alias_discretization import (
    reversible_sampling_rows,
)
from flagquantum.algorithms.primitives.chemistry_qroam_model import (
    source_chemistry_qroam_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.comparison.less_equal import (
    source_equal_width_less_equal_cost,
)
from flagquantum.compiler.registers.lookup.qroam_resource_model import (
    source_select_swap_qrom_cost,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    optimal_qroam_clean_log_block_size,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)

_ORBITALS = ("p", "q", "r", "s")
_ALTERNATES = tuple(f"alt_pqrs[{i}]" for i in range(4))


@dataclass(frozen=True)
class DensePreparationSketch:
    """Source tutorial declarations with the original floor division of spin orbitals."""

    spin_orbitals: int
    precision: int

    def __post_init__(self) -> None:
        if self.spin_orbitals < 4 or self.precision < 1:
            raise ValueError("Use at least four spin orbitals and positive precision")

    @property
    def spatial(self) -> int:
        return self.spin_orbitals // 2

    @property
    def rows(self) -> int:
        return self.spatial**2 + self.spatial**4

    @property
    def orbital_width(self) -> int:
        return (self.spatial - 1).bit_length()

    @property
    def address(self) -> int:
        return (self.rows - 1).bit_length()

    def signature(self, *, detailed: bool = False) -> Signature:
        word = bounded_unsigned_register_type(self.orbital_width, self.spatial)
        orbital_ports = tuple(Register(name, word) for name in _ORBITALS)
        spin_ports = tuple(
            Register(name, bounded_unsigned_register_type(1))
            for name in ("alpha", "beta")
        )
        if not detailed:
            return Signature(
                (
                    *orbital_ports,
                    *spin_ports,
                    Register("sign", bounded_unsigned_register_type(1)),
                )
            )
        return Signature(
            (
                *orbital_ports,
                Register("l", bounded_unsigned_register_type(self.address, self.rows)),
                *spin_ports,
                Register("sigma", bounded_unsigned_register_type(self.precision)),
                Register(
                    "alt_pqrs", opaque_register_type(self.orbital_width), shape=(4,)
                ),
                Register("keep", opaque_register_type(self.precision)),
                Register("less_than", quantum_bit_type()),
            )
        )

    def contiguous_signature(self) -> Signature:
        """The original opaque index builder supplies a cost and through ports."""
        return Signature.build(
            **dict.fromkeys(_ORBITALS, self.orbital_width), l=self.address
        )

    def cost(self, block_size: int | None = None) -> ComponentCost:
        """Compose the estimate before introducing concrete lookup data or wiring."""
        lookup = source_chemistry_qroam_cost(
            self.rows,
            4 * self.orbital_width + 2 + self.precision,
            block_size=block_size,
        )
        return self._model(lookup, detailed=False)

    def detailed_cost(self, rows: tuple[tuple[int, ...], ...]) -> ComponentCost:
        """Count the original detailed example, including its forward and erase comparison."""
        widths = (self.orbital_width,) * 4 + (self.precision,)
        log = optimal_qroam_clean_log_block_size(self.rows, 2 * sum(widths))
        table = QroamCleanTable((self.rows,), (log,), rows)
        lookup = source_select_swap_qrom_cost(
            table, 0, (self.address,), widths, dirty=False
        )
        return self._model(lookup, detailed=True)

    def _model(self, lookup: ComponentCost, *, detailed: bool) -> ComponentCost:
        width = self.orbital_width
        logical = int(self.signature(detailed=detailed).n_qubits())
        uniform = CostCall(source_uniform_preparation_cost(self.spatial), 4)
        compare = CostCall(source_equal_width_less_equal_cost(self.precision))
        swap = CostCall(ComponentCost("CSwap", 1 + 2 * width, GateCost(cswap=width)), 4)
        if detailed:
            index = ComponentCost(
                "ContiguousIndexFourD",
                int(self.contiguous_signature().n_qubits()),
                GateCost(toffoli=3 * (width**2 - width) + 4 * (width - 1)),
            )
            hadamard = ComponentCost("Hadamard", 1, GateCost(clifford=1))
            lookup_ports = self.address + 4 * width + self.precision
            compare_ports = 2 * self.precision + 1
            compare = CostCall(
                compare.component, bystander_qubits=logical - compare_ports
            )
            calls: tuple[CostCall, ...] = (
                CostCall(uniform.component, 4, logical - width),
                CostCall(index, bystander_qubits=logical - index.qubits),
                CostCall(lookup, bystander_qubits=logical - lookup_ports),
                CostCall(hadamard, self.precision, logical - 1),
                compare,
                CostCall(swap.component, 4, logical - swap.component.qubits),
                compare,
            )
        else:
            calls = (uniform, CostCall(lookup), compare, swap)
        return ComponentCost(
            (
                "PrepareSecondQuantizationDetailed"
                if detailed
                else "PrepareSecondQuantization"
            ),
            logical,
            calls=calls,
        )

    def register_score(self) -> RegisterScore:
        """Display the modeled wiring, retaining each alternate orbital lane."""
        widths = {
            reg.name: int(reg.bitsize)
            for reg in self.signature(detailed=True)
            if reg.name != "alt_pqrs"
        }
        widths.update(dict.fromkeys(_ALTERNATES, self.orbital_width))

        def step(label: str, *ports: str) -> RegisterStep:
            return RegisterStep(label, ports, ports)

        compare = step("LessThanEqual", "keep", "sigma", "less_than")
        return layout_register_score(
            widths,
            (
                *_ORBITALS,
                "l",
                "alpha",
                "beta",
                "sigma",
                *_ALTERNATES,
                "keep",
                "less_than",
            ),
            (
                *(step("Uniform", port) for port in _ORBITALS),
                step("ContiguousIndexFourD", *_ORBITALS, "l"),
                step("SelectSwapQROM", "l", *_ALTERNATES, "keep"),
                step("Hadamards", "sigma"),
                compare,
                *(
                    step("CSwap", "less_than", alternate, orbital)
                    for alternate, orbital in zip(_ALTERNATES, _ORBITALS, strict=True)
                ),
                compare,
            ),
        )


def authoring_integrals(
    spatial: int, *, seed: int = 7
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Use the tutorial's distributions and tensor symmetrization with an explicit seed."""
    generator = np.random.RandomState(seed)
    one = generator.random((spatial, spatial))
    one = (one + one.T) / 2
    two = generator.normal(size=(spatial,) * 4)
    two += two.transpose(0, 1, 3, 2)
    two += two.transpose(1, 0, 2, 3)
    two += two.transpose(2, 3, 0, 1)
    return one, two


def dense_alias_rows(
    one: NDArray[np.float64], two: NDArray[np.float64], precision: int
) -> tuple[tuple[int, ...], ...]:
    """Map alternate row numbers back to four orbital axes, keeping one-body padding."""
    spatial = one.shape[0]
    weights = np.abs(np.concatenate((one.ravel(), two.ravel()))).tolist()
    alt, keep, _ = reversible_sampling_rows(
        weights, epsilon=2**-precision / len(weights)
    )
    pairs = spatial**2
    indices = [
        (
            (*np.unravel_index(row, (spatial,) * 2), 0, 0)
            if row < pairs
            else np.unravel_index(row - pairs, (spatial,) * 4)
        )
        for row in alt
    ]
    return (
        *(
            tuple(int(value) for value in column)
            for column in zip(*indices, strict=True)
        ),
        tuple(keep),
    )


def tutorial_ccz_equivalents(model: ComponentCost) -> int:
    """Apply the original approximate-swap convention and omit arbitrary rotations."""
    gates = estimate_component_resources(model).gates
    groups, remainder = divmod(gates.t, 4)
    if remainder:
        raise ValueError("Tutorial T costs must be grouped in four-T swap units")
    return groups + gates.toffoli + gates.and_bloq + gates.cswap
