"""Coherent factor aliases with blocked lookup and a matching adjoint."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.comparison.less_equal import (
    less_equal_ir,
    less_equal_requirements,
)
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.lookup.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
)
from ...compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
    source_qroam_clean_uncompute_ir,
)
from ...compiler.registers.preparation.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .alias_sampling import alias_numerators


@dataclass(frozen=True)
class FactorAliasPreparation:
    """Sample a factor with an optional nonzero flag carried through the swap.

    All words are big-endian. The lookup and comparison outputs stay live
    through SELECT and are consumed by the matching adjoint. The success
    flag is deterministic because the shared uniform preparation is exact.
    """

    table: QroamCleanTable
    probability_bits: int
    normalization: float

    _track_nonzero: ClassVar[bool] = False

    @classmethod
    def from_weights(
        cls,
        weights: Sequence[float],
        *,
        probability_bits: int = 8,
        log_block_size: int = 0,
    ) -> Self:
        """Build alias numerators from the nonnegative factor weights."""
        alternates, keep = alias_numerators(weights, mu=probability_bits)
        rows = len(weights)
        datasets: tuple[tuple[int, ...], ...] = (
            (alternates, keep) if rows > 1 else (keep,)
        )
        if cls._track_nonzero:
            nonzero = (0,) + (1,) * (rows - 1)
            datasets += (nonzero, tuple(nonzero[i] for i in alternates))
        return cls(
            QroamCleanTable((rows,), (log_block_size,), datasets),
            probability_bits,
            sum(weights),
        )

    @property
    def _targets(self) -> RegisterLayout:
        width = (self.table.shape[0] - 1).bit_length()
        return (
            *((("alt_l", width),) if width else ()),
            ("keep", self.probability_bits),
            *((("l_ne_zero", 1), ("alt_l_ne_zero", 1)) if self._track_nonzero else ()),
        )

    @property
    def layout(self) -> RegisterLayout:
        """Name the selection, success and retained QROAM/alias words."""
        width = (self.table.shape[0] - 1).bit_length()
        return (
            ("l", width),
            ("succ_l", 1),
            ("sigma", self.probability_bits),
            ("choose_alt", 1),
            *clean_bank_output_layout(
                self._targets, self.table.block_shape[0], junk_prefix="junk_"
            ),
        )

    def requirements(self) -> tuple[int, int]:
        """Return scratch qubits and feedback slots, excluding live outputs."""
        width, uniform_work, uniform_slots = uniform_preparation_requirements(
            self.table.shape[0]
        )
        lookup = self.table.workspace_size(0, (width,))
        compare_work, compare_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        return max(uniform_work, lookup, compare_work), max(
            uniform_slots, lookup, compare_slots
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundOuterPrepare":
        """Bind factor preparation or its inverse to a dynamic circuit."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if type(adjoint) is not bool or len(workspace) != self.requirements()[0]:
            raise ValueError(
                "Factor PREPARE requires matching clean work and a direction"
            )
        return _BoundOuterPrepare(
            self, dict(ports), workspace, adjoint, max((*logical, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        rows = self.table.shape[0]
        width, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
        uniform = uniform_preparation_ir(
            rows,
            ports["l"],
            work[:uniform_work],
            slots[:uniform_slots],
            adjoint=adjoint,
        )
        lookup_layout = clean_bank_output_layout(
            self._targets, self.table.block_shape[0], junk_prefix="junk_"
        )
        banks = bind_clean_banks(
            {name: ports[name] for name, _ in lookup_layout},
            self._targets,
            self.table.block_shape[0],
            junk_prefix="junk_",
        )
        lookup_work = self.table.workspace_size(0, (width,))
        load = source_qroam_clean_uncompute_ir if adjoint else source_qroam_clean_ir
        lookup = load(
            self.table,
            (),
            (ports["l"],),
            banks,
            work[:lookup_work],
            slots[:lookup_work],
        )
        sigma = tuple(Instruction("h", (wire,)) for wire in ports["sigma"])
        (flag,) = ports["choose_alt"]
        cmp_work, cmp_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        compare = less_equal_ir(
            ports["keep"], ports["sigma"], flag, work[:cmp_work], slots[:cmp_slots]
        )
        left = (*ports["l"], *ports.get("l_ne_zero", ()))
        right = (*ports.get("alt_l", ()), *ports.get("alt_l_ne_zero", ()))
        swap = controlled_swap_ir(flag, left, right) if left else ()
        success = (Instruction("x", ports["succ_l"]),)
        stages = (uniform, sigma, lookup, compare, swap, success)
        return tuple(
            gate
            for stage in (reversed(stages) if adjoint else stages)
            for gate in stage
        )


@dataclass(frozen=True)
class _BoundOuterPrepare:
    prepare: FactorAliasPreparation
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "Factor PREPARE needs a DynamicCircuit with sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        slots = tuple(range(start, start + self.prepare.requirements()[1]))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.prepare._emit(self.ports, self.workspace, slots, self.adjoint),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
