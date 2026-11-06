"""Factor-indexed signed triangular alias PREPARE for Eq. B11."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from ...circuit import Circuit
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.contiguous_index import (
    contiguous_index_columns_ir,
    contiguous_index_columns_workspace_size,
)
from ...compiler.registers.controlled_swap import controlled_swap_ir
from ...compiler.registers.less_equal import less_equal_ir, less_equal_requirements
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
)
from ...compiler.registers.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
    source_qroam_clean_uncompute_ir,
)
from ...compiler.registers.triangular_uniform_preparation import (
    triangular_uniform_ir,
    triangular_uniform_requirements,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .alias_sampling import alias_numerators


@dataclass(frozen=True)
class PrepareSingleFactorizationInner:
    """Prepare an alias signal at a read-only factor index l.

    Input rows use ascending triangular address q*(q+1)//2+p, p<=q.
    All quantum words are big-endian. The selected sign is a live output,
    and the caller applies its phase between this operation and the adjoint.
    Address, comparison and lookup junk remain live until the adjoint.
    """

    orbitals: int
    probability_bits: int
    table: QroamCleanTable

    @classmethod
    def from_coefficients(
        cls,
        orbitals: int,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        *,
        probability_bits: int = 8,
        log_blocks: tuple[int, int] = (0, 0),
    ) -> "PrepareSingleFactorizationInner":
        """Build factor-major QROAM columns with coherent sign exchange."""
        triangular_uniform_requirements(orbitals)
        coordinates = tuple((p, q) for q in range(orbitals) for p in range(q + 1))
        rows = (one_body, *factors)
        columns: tuple[list[int], ...] = tuple([] for _ in range(5))
        alt_p, alt_q, keep_data, sign, alt_sign = columns
        for row in rows:
            if len(row) != len(coordinates) or any(
                not isfinite(value) for value in row
            ):
                raise ValueError(
                    "Each SF factor requires one finite coefficient per triangular address"
                )
            weights = tuple(map(abs, row))
            # A zero factor has no outer support. Give its inner register a
            # defined identity-alias table so coherent callers can still bind it.
            alternate, keep = alias_numerators(
                weights if any(weights) else (1.0,) * len(row), mu=probability_bits
            )
            alt_p.extend(coordinates[i][0] for i in alternate)
            alt_q.extend(coordinates[i][1] for i in alternate)
            keep_data.extend(keep)
            sign.extend(int(value < 0) for value in row)
            alt_sign.extend(int(row[i] < 0) for i in alternate)
        datasets = columns if orbitals > 1 else columns[2:]
        table = QroamCleanTable(
            (len(rows), len(coordinates)), log_blocks, tuple(map(tuple, datasets))
        )
        return cls(orbitals, probability_bits, table)

    @property
    def _targets(self) -> RegisterLayout:
        width = (self.orbitals - 1).bit_length()
        return (
            *((("alt_p", width), ("alt_q", width)) if width else ()),
            ("keep", self.probability_bits),
            ("theta", 1),
            ("alt_theta", 1),
        )

    @property
    def _banks(self) -> RegisterLayout:
        first, second = self.table.block_shape
        return clean_bank_output_layout(
            self._targets, first * second, junk_prefix="junk_"
        )

    @property
    def layout(self) -> RegisterLayout:
        """Name the signal, the rotation wire and retained alias registers."""
        factors, pairs = self.table.shape
        width = (self.orbitals - 1).bit_length()
        return (
            ("l", (factors - 1).bit_length()),
            ("p", width),
            ("q", width),
            ("succ_pq", 1),
            ("rotation", 1),
            ("s", (pairs - 1).bit_length()),
            ("sigma", self.probability_bits),
            ("choose_alt", 1),
            *self._banks,
        )

    def requirements(self) -> tuple[int, int]:
        """Return borrowed qubits and feedback slots, excluding named outputs."""
        factors, pairs = self.table.shape
        width = (self.orbitals - 1).bit_length()
        triangle_work, triangle_slots = triangular_uniform_requirements(self.orbitals)
        index_work = contiguous_index_columns_workspace_size(width) if width else 0
        lookup = self.table.workspace_size(
            0, ((factors - 1).bit_length(), (pairs - 1).bit_length())
        )
        compare_work, compare_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        return max(triangle_work, index_work, lookup, compare_work), max(
            triangle_slots, int(bool(width)), lookup, compare_slots
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundInnerPrepare":
        """Bind either direction for circuit.append on a DynamicCircuit."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if type(adjoint) is not bool or len(workspace) != self.requirements()[0]:
            raise ValueError(
                "SF inner PREPARE requires matching clean work and a direction"
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
        p, q, address = ports["p"], ports["q"], ports["s"]
        triangle_work, triangle_slots = triangular_uniform_requirements(self.orbitals)
        triangle = triangular_uniform_ir(
            self.orbitals,
            p,
            q,
            ports["rotation"][0],
            work[:triangle_work],
            slots[:triangle_slots],
            adjoint=adjoint,
        )
        index = (
            contiguous_index_columns_ir(
                p,
                q,
                address,
                work[: contiguous_index_columns_workspace_size(len(p))],
                measurement_bit=slots[0],
            )
            if p
            else ()
        )
        selections = (ports["l"], address)
        lookup_work = self.table.workspace_size(0, tuple(map(len, selections)))
        first, second = self.table.block_shape
        banks = bind_clean_banks(
            {name: ports[name] for name, _ in self._banks},
            self._targets,
            first * second,
            junk_prefix="junk_",
        )
        load = source_qroam_clean_uncompute_ir if adjoint else source_qroam_clean_ir
        lookup = load(
            self.table, (), selections, banks, work[:lookup_work], slots[:lookup_work]
        )
        sigma = tuple(Instruction("h", (wire,)) for wire in ports["sigma"])
        (flag,) = ports["choose_alt"]
        cmp_work, cmp_slots = less_equal_requirements(
            self.probability_bits, self.probability_bits
        )
        compare = less_equal_ir(
            ports["keep"], ports["sigma"], flag, work[:cmp_work], slots[:cmp_slots]
        )
        swap = controlled_swap_ir(
            flag,
            (*p, *q, *ports["theta"]),
            (*ports.get("alt_p", ()), *ports.get("alt_q", ()), *ports["alt_theta"]),
        )
        success = (Instruction("x", ports["succ_pq"]),)
        stages = (triangle, index, sigma, lookup, compare, swap, success)
        return tuple(
            gate
            for stage in (reversed(stages) if adjoint else stages)
            for gate in stage
        )


@dataclass(frozen=True)
class _BoundInnerPrepare:
    prepare: PrepareSingleFactorizationInner
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "SF inner PREPARE needs a DynamicCircuit with sufficient width"
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
