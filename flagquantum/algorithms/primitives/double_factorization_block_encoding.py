"""Executable double-factorization block encoding from Appendix C, Fig. 16."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.multi_control import multi_and_ir
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.prepare_reflection import prepare_reflection_ir
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .double_factorization_angles import orbital_rotation_words
from .double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
)
from .double_factorization_indexed_data import DoubleFactorizationIndexedData
from .double_factorization_inner_prepare import PrepareDoubleFactorizationInner
from .double_factorization_outer_prepare import PrepareDoubleFactorizationOuter
from .multiplexers.select_double_factorization import SelectDoubleFactorization

_METADATA = {"xi", "offset", "rot_data"}
_SELECT_SHARED = {
    "offset",
    "p",
    "succ_p",
    "theta",
    "succ_l",
    "l_ne_zero",
    "system",
    "spin",
}


@dataclass(frozen=True)
class DoubleFactorizationBlockEncoding:
    """Compose the factor signal, signed number selection and middle reflection.

    All non-system ports start in zero. Project them onto zero to read the
    encoded block. Angle rows follow the concatenated eigenvalue order, and
    the system is spin-major.
    """

    outer: PrepareDoubleFactorizationOuter
    inner: PrepareDoubleFactorizationInner
    metadata: DoubleFactorizationIndexedData
    select: SelectDoubleFactorization

    def __post_init__(self) -> None:
        if (
            self.outer.table.shape != (len(self.inner.ranks),)
            or self.metadata.ranks != self.inner.ranks
            or self.metadata.rotation_bits != self.inner.rotation_bits
            or self.select.orbitals != self.inner.ranks[0]
            or self.select.table.shape != (sum(self.inner.ranks),)
        ):
            raise ValueError(
                "DF block encoding requires compatible factor and orbital layouts"
            )

    @classmethod
    def from_eigenvalues(
        cls,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        angle_rows: Sequence[Sequence[int]],
        *,
        outer_bits: int = 8,
        inner_bits: int = 8,
        rank_rotation_bits: int = 8,
        basis_rotation_bits: int = 8,
        outer_log_block: int = 0,
        metadata_log_block: int = 0,
        inner_log_block: int = 0,
        basis_log_block: int = 0,
    ) -> "DoubleFactorizationBlockEncoding":
        """Use effective one-body eigenvalues and fixed-point basis rotation rows."""
        outer = PrepareDoubleFactorizationOuter.from_eigenvalues(
            one_body,
            factors,
            probability_bits=outer_bits,
            log_block_size=outer_log_block,
        )
        # The number operator n=(I-Z)/2 gives a negative Z in the one-body
        # term. The factor squares retain their original eigenvalue signs.
        inner = PrepareDoubleFactorizationInner.from_eigenvalues(
            tuple(-value for value in one_body),
            factors,
            probability_bits=inner_bits,
            rotation_bits=rank_rotation_bits,
            log_block_size=inner_log_block,
        )
        select = SelectDoubleFactorization.from_angle_words(
            len(one_body),
            angle_rows,
            angle_bits=basis_rotation_bits,
            log_block_size=basis_log_block,
        )
        return cls(
            outer, inner, inner.indexed_data(log_block_size=metadata_log_block), select
        )

    @classmethod
    def from_eigensystems(
        cls,
        one_body: Sequence[tuple[float, Sequence[float]]],
        factors: Sequence[Sequence[tuple[float, Sequence[float]]]],
        *,
        outer_bits: int = 8,
        inner_bits: int = 8,
        rank_rotation_bits: int = 8,
        basis_rotation_bits: int = 8,
        outer_log_block: int = 0,
        metadata_log_block: int = 0,
        inner_log_block: int = 0,
        basis_log_block: int = 0,
    ) -> "DoubleFactorizationBlockEncoding":
        """Use ordered eigenvalue and real unit eigenvector pairs.

        Give one vector per retained eigenvalue. Every vector is in the
        same spatial orbital basis, with one component per one-body pair.
        """
        orbitals = len(one_body)
        vectors = tuple(vector for row in (one_body, *factors) for _, vector in row)
        if any(len(vector) != orbitals for vector in vectors):
            raise ValueError("DF orbital vectors must match the spatial orbital count")
        return cls.from_eigenvalues(
            tuple(value for value, _ in one_body),
            tuple(tuple(value for value, _ in factor) for factor in factors),
            tuple(
                orbital_rotation_words(vector, basis_rotation_bits)
                for vector in vectors
            ),
            outer_bits=outer_bits,
            inner_bits=inner_bits,
            rank_rotation_bits=rank_rotation_bits,
            basis_rotation_bits=basis_rotation_bits,
            outer_log_block=outer_log_block,
            metadata_log_block=metadata_log_block,
            inner_log_block=inner_log_block,
            basis_log_block=basis_log_block,
        )

    @property
    def layout(self) -> RegisterLayout:
        """Name the prepared signal, system and temporary lookup banks."""
        return (
            *self.outer.layout,
            *(
                (f"metadata_{name}", width)
                for name, width in self.metadata.layout
                if name != "l"
            ),
            *(
                (f"inner_{name}", width)
                for name, width in self.inner.layout
                if name not in _METADATA
            ),
            *(
                (f"select_{name}", width)
                for name, width in self.select.layout()
                if name not in _SELECT_SHARED
            ),
            ("spin", 1),
            ("system", 2 * self.select.orbitals),
        )

    def requirements(self) -> tuple[int, int]:
        """Return reusable scratch wires and feedback slots for the full block."""
        outer_work, outer_slots = self.outer.requirements()
        inner_work, inner_slots = self.inner.requirements()
        indexed = self.metadata.workspace_size()
        selected = self.select.workspace_size(additional_control=True)
        reflection = (
            (self.select.orbitals - 1).bit_length() + self.inner.probability_bits + 2
        )
        return (
            max(outer_work, inner_work, indexed, selected, reflection),
            max(outer_slots, inner_slots, indexed, selected, reflection),
        )

    def source_resource_model(self) -> ComponentCost:
        """Return the locked analytical baseline before emitting a circuit.

        This model uses the source's QROAM block selection and reflection.
        It requires the source's shared outer and inner alias precision and
        inherits the source model's QROAM shape limits.
        """
        if self.outer.probability_bits != self.inner.probability_bits:
            raise ValueError("The source DF model requires shared alias precision")
        return source_double_factorization_block_encoding_cost(
            auxiliaries=len(self.inner.ranks) - 1,
            spin_orbitals=2 * self.select.orbitals,
            eigenvalues=sum(self.inner.ranks[1:]),
            alias_bits=self.inner.probability_bits,
            basis_rotation_bits=self.select.angle_bits,
            inner_rotation_bits=self.inner.rotation_bits,
        )

    def make_on(
        self,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
    ) -> "_BoundDoubleFactorization":
        """Bind the block encoding for circuit.append on a DynamicCircuit."""
        wires = unpartition_register(ports, self.layout)
        _validate_wires((*wires, *workspace))
        if len(workspace) != self.requirements()[0]:
            raise ValueError("DF block encoding requires matching clean workspace")
        return _BoundDoubleFactorization(
            self, dict(ports), workspace, max((*wires, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        start: int,
    ) -> tuple[Instruction, ...]:
        slots = tuple(range(start, start + self.requirements()[1]))
        outer_ports = {name: ports[name] for name, _ in self.outer.layout}
        indexed_ports = {
            name: ports[name if name == "l" else f"metadata_{name}"]
            for name, _ in self.metadata.layout
        }
        inner_ports = {
            name: ports[f"{'metadata' if name in _METADATA else 'inner'}_{name}"]
            for name, _ in self.inner.layout
        }
        outer_work, outer_slots = self.outer.requirements()
        inner_work, inner_slots = self.inner.requirements()
        indexed_work = self.metadata.workspace_size()
        load = self.metadata.make_on(indexed_ports, workspace=work[:indexed_work])
        unload = self.metadata.make_on(
            indexed_ports, workspace=work[:indexed_work], adjoint=True
        )
        prepare = self.inner._emit(
            inner_ports, work[:inner_work], slots[:inner_slots], False
        )
        unprepare = self.inner._emit(
            inner_ports, work[:inner_work], slots[:inner_slots], True
        )
        hadamard = Instruction("h", ports["spin"])
        select_shared = {
            "offset": indexed_ports.get("offset", ()),
            "p": inner_ports["p"],
            "succ_p": inner_ports["succ_p"],
            "theta": inner_ports["theta"],
            "succ_l": ports["succ_l"],
            "l_ne_zero": indexed_ports["l_ne_zero"],
            "system": ports["system"],
            "spin": ports["spin"],
        }

        def body(extra: bool) -> tuple[Instruction, ...]:
            selected_ports = {
                name: (
                    ()
                    if not width
                    else (
                        select_shared[name]
                        if name in _SELECT_SHARED
                        else ports[f"select_{name}"]
                    )
                )
                for name, width in self.select.layout(additional_control=extra)
            }
            size = self.select.workspace_size(additional_control=extra)
            selected = self.select._emit(selected_ports, work[:size], slots[:size])
            return (*prepare, hadamard, *selected, hadamard, *unprepare)

        reflected = (
            *inner_ports["p"],
            *inner_ports["sigma"],
            *inner_ports["rotation"],
            *ports["spin"],
        )
        controls = (*ports["succ_l"], *indexed_ports["l_ne_zero"])
        predicate = work[:1]
        count = len(reflected)
        reflection = (
            *multi_and_ir(controls, (1, 1), predicate),
            *prepare_reflection_ir(
                reflected,
                work[1:count],
                slots[1:count],
                control=predicate[0],
                global_phase=-1,
            ),
            *multi_and_ir(
                controls, (1, 1), predicate, classical_bits=slots[:1], uncompute=True
            ),
        )
        return (
            *self.outer._emit(
                outer_ports, work[:outer_work], slots[:outer_slots], False
            ),
            *load._emit(start),
            *body(False),
            *reflection,
            *body(True),
            *unload._emit(start),
            *self.outer._emit(
                outer_ports, work[:outer_work], slots[:outer_slots], True
            ),
        )


@dataclass(frozen=True)
class _BoundDoubleFactorization:
    encoding: DoubleFactorizationBlockEncoding
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "DF block encoding needs a DynamicCircuit with sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        gates = self.encoding._emit(self.ports, self.workspace, start)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
