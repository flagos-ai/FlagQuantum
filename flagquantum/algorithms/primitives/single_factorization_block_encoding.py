"""Executable single-factorization block encoding from Fig. 15 of arXiv:2011.03494."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.control.multi_control import multi_and_ir
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import prepare_reflection_ir
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .multiplexers.select_single_factorization import SelectSingleFactorization
from .single_factorization_inner_prepare import PrepareSingleFactorizationInner
from .single_factorization_outer_prepare import PrepareSingleFactorizationOuter

_REFLECTED_PORTS = ("p", "q", "sigma", "rotation", "spin", "swap_pq")


@dataclass(frozen=True)
class SingleFactorizationOneBody:
    """Apply the signed one-body SELECT conjugated by inner preparation.

    A standalone one-body table has only row zero. With a multirow table,
    the input factor address selects the row. The factor address and outer
    success flag are retained. The inner signal starts at zero, as do the
    spin, swap and caller-owned work wires. The signal can leave the zero
    subspace. The projected zero-signal block is the signed Pauli average.
    """

    inner: PrepareSingleFactorizationInner

    @property
    def layout(self) -> RegisterLayout:
        return (
            ("succ_l", 1),
            *self.inner.layout,
            ("spin", 1),
            ("swap_pq", 1),
            ("system", 2 * self.inner.orbitals),
        )

    def requirements(self) -> tuple[int, int]:
        """Return borrowed qubits and feedback slots for the standalone body."""
        inner_work, inner_slots = self.inner.requirements()
        select_work = SelectSingleFactorization.workspace_size(2 * self.inner.orbitals)
        select_slots = select_work - int(self.inner.orbitals > 1)
        return max(inner_work, select_work), max(inner_slots, select_slots)

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundSingleFactorization":
        """Bind the body to a DynamicCircuit."""
        return _BoundSingleFactorization.bind(self, ports, workspace)

    def _emit(
        self, ports: Mapping[str, tuple[int, ...]], work: tuple[int, ...], start: int
    ) -> tuple[Instruction, ...]:
        return self._prepare_body(ports, work, start)(None)

    def _prepare_body(
        self, ports: Mapping[str, tuple[int, ...]], work: tuple[int, ...], start: int
    ) -> Callable[[int | None], tuple[Instruction, ...]]:
        inner_ports = {name: ports[name] for name, _ in self.inner.layout}
        inner_work, inner_slots = self.inner.requirements()
        slots = tuple(range(start, start + inner_slots))
        prepare = self.inner._emit(inner_ports, work[:inner_work], slots, False)
        unprepare = self.inner._emit(inner_ports, work[:inner_work], slots, True)
        p, q = inner_ports["p"], inner_ports["q"]
        spin, exchange = ports["spin"][0], ports["swap_pq"][0]
        system = ports["system"]
        hadamards = (Instruction("h", (spin,)), Instruction("h", (exchange,)))
        swap = controlled_swap_ir(exchange, p, q) if p else ()

        def emit(factor_control: int | None) -> tuple[Instruction, ...]:
            size = SelectSingleFactorization.workspace_size(
                len(system), additional_control=factor_control is not None
            )
            select = SelectSingleFactorization(
                p,
                q,
                spin,
                inner_ports["succ_pq"][0],
                ports["succ_l"][0],
                system,
                work[:size],
                factor_control,
                coefficient_sign=inner_ports["theta"][0],
            )
            return (
                *prepare,
                *hadamards,
                *swap,
                *select._emit(start),
                *swap,
                *reversed(hadamards),
                *unprepare,
            )

        return emit


@dataclass(frozen=True)
class SingleFactorizationBlockEncoding:
    """Compose the prepared signal, signed SELECT and controlled reflection.

    Coefficients are triangular words with the inner preparation's ordering.
    The zero signal block represents the alias-quantized Hamiltonian with
    the constant shift from the two applications around the reflection.
    """

    outer: PrepareSingleFactorizationOuter
    inner: PrepareSingleFactorizationInner

    def __post_init__(self) -> None:
        if self.outer.table.shape[0] != self.inner.table.shape[0]:
            raise ValueError("SF preparations must have the same factor count")

    @classmethod
    def from_coefficients(
        cls,
        orbitals: int,
        one_body: Sequence[float],
        factors: Sequence[Sequence[float]],
        *,
        outer_bits: int = 8,
        inner_bits: int = 8,
        outer_log_block: int = 0,
        inner_log_blocks: tuple[int, int] = (0, 0),
    ) -> "SingleFactorizationBlockEncoding":
        """Build matched outer and inner alias tables from the coefficients."""
        inner = PrepareSingleFactorizationInner.from_coefficients(
            orbitals,
            one_body,
            factors,
            probability_bits=inner_bits,
            log_blocks=inner_log_blocks,
        )
        outer = PrepareSingleFactorizationOuter.from_coefficients(
            one_body,
            factors,
            probability_bits=outer_bits,
            log_block_size=outer_log_block,
        )
        return cls(outer, inner)

    @property
    def layout(self) -> RegisterLayout:
        """Name zero-initialized signal wires and the orbital-major system."""
        return (
            *self.outer.layout,
            *(
                (f"inner_{name}", width)
                for name, width in self.inner.layout
                if name != "l"
            ),
            ("spin", 1),
            ("swap_pq", 1),
            ("system", 2 * self.inner.orbitals),
        )

    def requirements(self) -> tuple[int, int]:
        """Return the reusable clean qubits and classical feedback slots."""
        outer_work, outer_slots = self.outer.requirements()
        inner_work, inner_slots = self.inner.requirements()
        select_work = SelectSingleFactorization.workspace_size(
            2 * self.inner.orbitals, additional_control=True
        )
        reflection_width = (
            2 * (self.inner.orbitals - 1).bit_length() + self.inner.probability_bits + 3
        )
        return (
            max(outer_work, inner_work, select_work, reflection_width),
            max(outer_slots, inner_slots, select_work, reflection_width),
        )

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundSingleFactorization":
        """Bind the block encoding to a DynamicCircuit."""
        return _BoundSingleFactorization.bind(self, ports, workspace)

    def _emit(
        self, ports: Mapping[str, tuple[int, ...]], work: tuple[int, ...], start: int
    ) -> tuple[Instruction, ...]:
        slots = tuple(range(start, start + self.requirements()[1]))
        outer_ports = {name: ports[name] for name, _ in self.outer.layout}
        inner_ports = {
            name: ports["l" if name == "l" else f"inner_{name}"]
            for name, _ in self.inner.layout
        }
        outer_work, outer_slots = self.outer.requirements()
        succ_l, nonzero = ports["succ_l"][0], ports["l_ne_zero"][0]
        body_ports = inner_ports | {
            name: ports[name] for name in ("succ_l", "spin", "swap_pq", "system")
        }
        body = SingleFactorizationOneBody(self.inner)._prepare_body(
            body_ports, work, start
        )
        reflected = tuple(
            wire for name in _REFLECTED_PORTS for wire in body_ports[name]
        )
        count = len(reflected)
        controls = (succ_l, nonzero)
        polarity = (1, 1)
        predicate = work[:1]
        reflection = (
            *multi_and_ir(controls, polarity, predicate),
            *prepare_reflection_ir(
                reflected,
                work[1:count],
                slots[1:count],
                control=predicate[0],
                global_phase=-1,
            ),
            *multi_and_ir(
                controls, polarity, predicate, classical_bits=slots[:1], uncompute=True
            ),
        )
        return (
            *self.outer._emit(
                outer_ports, work[:outer_work], slots[:outer_slots], False
            ),
            *body(None),
            *reflection,
            *body(nonzero),
            *self.outer._emit(
                outer_ports, work[:outer_work], slots[:outer_slots], True
            ),
        )


@dataclass(frozen=True)
class _BoundSingleFactorization:
    encoding: SingleFactorizationBlockEncoding | SingleFactorizationOneBody
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    @classmethod
    def bind(
        cls,
        encoding: SingleFactorizationBlockEncoding | SingleFactorizationOneBody,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
    ) -> "_BoundSingleFactorization":
        logical = unpartition_register(ports, encoding.layout)
        _validate_wires((*logical, *workspace))
        if len(workspace) != encoding.requirements()[0]:
            raise ValueError("SF block encoding requires matching clean workspace")
        return cls(encoding, dict(ports), workspace, max((*logical, *workspace)) + 1)

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "SF block encoding needs a DynamicCircuit with sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.encoding._emit(self.ports, self.workspace, start),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
