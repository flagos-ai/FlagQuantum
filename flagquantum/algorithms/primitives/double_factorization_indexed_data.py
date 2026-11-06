"""Executable indexed factor metadata from arXiv:2011.03494, Eq. C29."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import accumulate

from ...circuit import Circuit
from ...compiler.registers.bitwise import _validate_wires
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
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit


@dataclass(frozen=True)
class DoubleFactorizationIndexedData:
    """Read factor metadata while preserving the big-endian factor index.

    Factor zero has full spatial rank and covers the one-body term. Each rank
    word stores rank minus one so full rank fits the source register width.
    Offsets index the concatenated rows, including the one-body row at zero.
    Rotation words are supplied already quantized by the caller.
    """

    ranks: tuple[int, ...]
    rotation_words: tuple[int, ...]
    rotation_bits: int
    log_block_size: int = 0
    _table: QroamCleanTable = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.ranks or any(
            type(rank) is not int or not 1 <= rank <= self.ranks[0]
            for rank in self.ranks
        ):
            raise ValueError(
                "DF ranks must be positive and bounded by the one-body rank"
            )
        if (
            type(self.rotation_bits) is not int
            or self.rotation_bits < 1
            or len(self.ranks) != len(self.rotation_words)
            or any(
                type(word) is not int or not 0 <= word < 1 << self.rotation_bits
                for word in self.rotation_words
            )
        ):
            raise ValueError("DF rotation words must fit the declared precision")
        object.__setattr__(
            self,
            "_table",
            QroamCleanTable(
                (len(self.ranks),),
                (self.log_block_size,),
                tuple(column for _, width, column in self._columns if width),
            ),
        )

    @property
    def _columns(self) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
        return (
            ("l_ne_zero", 1, (0,) + (1,) * (len(self.ranks) - 1)),
            (
                "xi",
                (self.ranks[0] - 1).bit_length(),
                tuple(rank - 1 for rank in self.ranks),
            ),
            ("rot_data", self.rotation_bits, self.rotation_words),
            (
                "offset",
                (sum(self.ranks) - 1).bit_length(),
                tuple(accumulate(self.ranks, initial=0))[:-1],
            ),
        )

    @property
    def _targets(self) -> RegisterLayout:
        return tuple((name, width) for name, width, _ in self._columns if width)

    @property
    def _outputs(self) -> RegisterLayout:
        return clean_bank_output_layout(
            self._targets, self._table.block_shape[0], junk_prefix="junk_"
        )

    @property
    def layout(self) -> RegisterLayout:
        """Return the address and retained selected and block-bank outputs."""
        return (("l", (len(self.ranks) - 1).bit_length()), *self._outputs)

    def workspace_size(self) -> int:
        """Return the shared QROAM scratch width, also the feedback slot count."""
        return self._table.workspace_size(0, (self.layout[0][1],))

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        adjoint: bool = False,
    ) -> "_BoundIndexedData":
        """Bind a clean load or the inverse on its matching signed banks."""
        wires = unpartition_register(ports, self.layout)
        _validate_wires((*wires, *workspace))
        if type(adjoint) is not bool or len(workspace) != self.workspace_size():
            raise ValueError(
                "DF indexed data requires matching clean work and a direction"
            )
        banks = bind_clean_banks(
            {name: ports[name] for name, _ in self._outputs},
            self._targets,
            self._table.block_shape[0],
            junk_prefix="junk_",
        )
        return _BoundIndexedData(
            self._table,
            ports["l"],
            banks,
            workspace,
            adjoint,
            max((*wires, *workspace)) + 1,
        )


@dataclass(frozen=True)
class _BoundIndexedData:
    table: QroamCleanTable
    address: tuple[int, ...]
    banks: tuple[tuple[tuple[int, ...], ...], ...]
    workspace: tuple[int, ...]
    adjoint: bool
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "DF indexed data needs a DynamicCircuit with sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        gates = self._emit(start)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )

    def _emit(self, start: int) -> tuple[Instruction, ...]:
        emit = (
            source_qroam_clean_uncompute_ir if self.adjoint else source_qroam_clean_ir
        )
        return emit(
            self.table,
            (),
            (self.address,),
            self.banks,
            self.workspace,
            tuple(range(start, start + len(self.workspace))),
        )
