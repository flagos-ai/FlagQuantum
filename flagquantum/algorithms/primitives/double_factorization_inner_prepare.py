"""Signed inner alias preparation for Appendix C step 3 of arXiv:2011.03494."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from ...circuit import Circuit
from ...compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from ...compiler.registers.bits.bitwise import _validate_wires, xor_ir
from ...compiler.registers.comparison.less_equal import (
    less_equal_ir,
    less_equal_requirements,
)
from ...compiler.registers.control.control_predicate import ControlRegister
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
from ...compiler.registers.preparation.rank_uniform_preparation import (
    rank_rotation_word,
    rank_uniform_ir,
    rank_uniform_requirements,
)
from ...compiler.registers.primitives.controlled_swap import (
    controlled_swap_with_predicate_ir,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .alias_sampling import alias_numerators
from .double_factorization_indexed_data import DoubleFactorizationIndexedData


@dataclass(frozen=True)
class PrepareDoubleFactorizationInner:
    """Bind the rank, angle and offset words supplied by indexed factor data.

    Factor zero is the one-body row. Alternates stay within their factor's
    local index space. The address and loaded banks remain live for the
    adjoint, while the selected sign accompanies p through SELECT. Failed
    rank preparation keeps its index and is excluded by the success flag.
    """

    ranks: tuple[int, ...]
    probability_bits: int
    rotation_bits: int
    table: QroamCleanTable

    @classmethod
    def from_eigenvalues(
        cls,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        *,
        probability_bits: int = 8,
        rotation_bits: int = 8,
        log_block_size: int = 0,
    ) -> "PrepareDoubleFactorizationInner":
        """Construct a concatenated signed alias table using factor-local rows."""
        spatial = len(one_body)
        rows = (one_body, *factors)
        if any(not 1 <= len(row) <= spatial for row in rows):
            raise ValueError("DF eigenvalue rows must have a positive spatial rank")
        if type(rotation_bits) is not int or rotation_bits < 1:
            raise ValueError("DF inner PREPARE requires positive rotation precision")
        alt: list[int] = []
        keep: list[int] = []
        sign: list[int] = []
        alt_sign: list[int] = []
        for row in rows:
            if any(not isfinite(value) for value in row):
                raise ValueError("DF eigenvalues must be finite and real")
            weights = tuple(map(abs, row))
            a, k = alias_numerators(
                weights if any(weights) else (1.0,) * len(row), mu=probability_bits
            )
            alt.extend(a)
            keep.extend(k)
            sign.extend(int(value < 0) for value in row)
            alt_sign.extend(int(row[i] < 0) for i in a)
        ranks = tuple(map(len, rows))
        columns = (alt, keep, sign, alt_sign) if spatial > 1 else (keep, sign, alt_sign)
        table = QroamCleanTable(
            (sum(ranks),), (log_block_size,), tuple(map(tuple, columns))
        )
        return cls(ranks, probability_bits, rotation_bits, table)

    def indexed_data(
        self, *, log_block_size: int = 0
    ) -> DoubleFactorizationIndexedData:
        """Build matching read-only metadata with rounded rank rotation angles."""
        words = tuple(
            rank_rotation_word(rank, self.rotation_bits) for rank in self.ranks
        )
        return DoubleFactorizationIndexedData(
            self.ranks, words, self.rotation_bits, log_block_size
        )

    @property
    def _p_width(self) -> int:
        return (self.ranks[0] - 1).bit_length()

    @property
    def _address_width(self) -> int:
        return (self.table.shape[0] - 1).bit_length()

    @property
    def _targets(self) -> RegisterLayout:
        return (
            *((("alt_p", self._p_width),) if self._p_width else ()),
            ("keep", self.probability_bits),
            ("theta", 1),
            ("alt_theta", 1),
        )

    @property
    def _outputs(self) -> RegisterLayout:
        return clean_bank_output_layout(
            self._targets, self.table.block_shape[0], junk_prefix="junk_"
        )

    @property
    def layout(self) -> RegisterLayout:
        """Name the input metadata, signal and retained alias registers."""
        return (
            *((("xi", self._p_width),) if self._p_width else ()),
            *((("offset", self._address_width),) if self._address_width else ()),
            ("rot_data", self.rotation_bits),
            ("p", self._p_width),
            ("rotation", 1),
            ("succ_p", 1),
            ("address", self._address_width),
            ("sigma", self.probability_bits),
            ("choose_alt", 1),
            *self._outputs,
        )

    def requirements(self) -> tuple[int, int]:
        """Return clean scratch and reusable measurement slots."""
        rank_work, rank_slots = rank_uniform_requirements(self._p_width)
        lookup = self.table.workspace_size(0, (self._address_width,))
        compare_work, compare_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        carries = self._address_width - 1 if self._p_width else 0
        return max(rank_work, lookup, compare_work, carries), max(
            rank_slots, lookup, compare_slots, carries
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundInnerPrepare":
        """Bind the inner preparation or its matching adjoint."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if type(adjoint) is not bool or len(workspace) != self.requirements()[0]:
            raise ValueError(
                "DF inner PREPARE requires matching clean work and a direction"
            )
        return _BoundInnerPrepare(
            self, dict(ports), workspace, adjoint, max((*logical, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        p, address = ports["p"], ports["address"]
        offset = ports.get("offset", ())
        success = ports["succ_p"][0]
        rank_work, rank_slots = rank_uniform_requirements(len(p))
        rank = rank_uniform_ir(
            ports.get("xi", ()),
            ports["rot_data"],
            p,
            ports["rotation"][0],
            success,
            work[:rank_work],
            slots[:rank_slots],
            adjoint=adjoint,
        )
        copy_offset = xor_ir(offset, address) if address else ()
        arithmetic = add_adjoint_ir if adjoint else add_ir
        carries = len(address) - 1
        addition = arithmetic(p, address, work[:carries], slots[:carries]) if p else ()
        index = (*addition, *copy_offset) if adjoint else (*copy_offset, *addition)

        banks = bind_clean_banks(
            {name: ports[name] for name, _ in self._outputs},
            self._targets,
            self.table.block_shape[0],
            junk_prefix="junk_",
        )
        lookup_work = self.table.workspace_size(0, (len(address),))
        load = source_qroam_clean_uncompute_ir if adjoint else source_qroam_clean_ir
        lookup = load(
            self.table, (), (address,), banks, work[:lookup_work], slots[:lookup_work]
        )
        sigma = tuple(Instruction("h", (wire,)) for wire in ports["sigma"])
        (flag,) = ports["choose_alt"]
        compare_work, compare_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        compare = less_equal_ir(
            ports["keep"],
            ports["sigma"],
            flag,
            work[:compare_work],
            slots[:compare_slots],
        )
        swap = controlled_swap_with_predicate_ir(
            ((success, flag),),
            (ControlRegister(1, (1, 1)),),
            (*p, *ports["theta"]),
            (*ports.get("alt_p", ()), *ports["alt_theta"]),
            work[:1],
            slots[:1],
        )
        stages = (rank, index, sigma, lookup, compare, swap)
        return tuple(
            gate
            for stage in (reversed(stages) if adjoint else stages)
            for gate in stage
        )


@dataclass(frozen=True)
class _BoundInnerPrepare:
    prepare: PrepareDoubleFactorizationInner
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "DF inner PREPARE needs a DynamicCircuit with sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        slots = tuple(range(start, start + self.prepare.requirements()[1]))
        gates = self.prepare._emit(self.ports, self.workspace, slots, self.adjoint)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
