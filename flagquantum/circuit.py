"""FlagQuantum native circuit API.

This module is intentionally implemented on top of FlagQuantum's own IR and
PyTorch tensor operations. It does not depend on tensor-network, graph, or
scientific-computing helper packages, which keeps the core runtime portable to
AI accelerators that already support mainstream deep-learning frameworks.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from .core.ir import CircuitIR, Instruction
from .core.operator_schema import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
    inverse_operator,
)
from .core.parameters import (
    bind_parameter_value,
    parameter_names_in_value,
)
from .core.qubit_mapping import normalize_qubits, qubit_map_from, remap_qubits
from .core.runtime_config import RuntimeConfig, get_runtime_config
from .errors import CapabilityError, ValidationError

if TYPE_CHECKING:
    from .noise import NoiseModel
    from .observables import OutputRequest
    from .runtime.builder_compilation import BuilderBindings
    from .runtime.execution_plan import CircuitAnalysis, ExecutionPlan
    from .runtime.options import ExecutionOptions
    from .runtime.planner.models import RuntimeSelectionPlan
    from .runtime.result import ExecutionResult


def _count_argument(name: str, value: Any, *, minimum: int = 1) -> int:
    """Read an integer count without quietly rewriting the caller's value.

    ``int(value)`` accepted ``2.7`` as ``2`` and ``"3"`` as ``3``, so a mistyped
    count changed the program instead of failing. Only values that already denote an
    integer are accepted, through the interpreter's own ``__index__`` protocol, which
    admits NumPy integers and zero-dimensional torch integer tensors while refusing
    floats and strings. ``bool`` is refused because ``True`` is a flag rather than a
    count, which is the rule ``ExecutionOptions`` already applies to ``batch_size``.
    """

    if isinstance(value, bool):
        raise TypeError(f"Circuit {name} must be an integer, got {value!r}")
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(f"Circuit {name} must be an integer, got {value!r}") from None
    if count < minimum:
        raise ValidationError(f"Circuit {name} must be >= {minimum}, got {count}")
    return count


def _inverted_instruction(instruction: Instruction) -> Instruction:
    """Return the single gate that undoes one recorded instruction.

    The matrix recorded on an instruction is what the simulator executes, so it decides
    the inverse whenever it is present: an operation with an explicit matrix is inverted
    by conjugate transpose, which is the only rule that inverts a user-supplied unitary.
    Opcode rules apply to the remaining instructions, and an instruction that neither
    route can invert is refused instead of being copied forward unchanged.

    ``CapabilityError`` is the class the errors-module boundary reserves for a capability
    that is absent rather than a value that is wrong, which is the case here: the program
    is well formed, and undoing it is the part that is not available.
    """

    reason: str | None = None
    if instruction.metadata.get("is_channel"):
        reason = "it is a noise channel, which has no unitary inverse"
    elif instruction.metadata.get("is_dynamic"):
        reason = "it is a dynamic operation, which has no fixed inverse"
    elif instruction.metadata.get("conditions"):
        reason = "it is classically conditioned, which has no fixed inverse"

    if reason is None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        if matrix is not None:
            inverted_matrix = getattr(matrix, "mH", None)
            if inverted_matrix is None:
                reason = "its matrix does not expose a conjugate transpose"
            else:
                return Instruction(
                    name=instruction.name,
                    wires=instruction.wires,
                    params=dict(instruction.params),
                    matrix=inverted_matrix,
                    metadata=dict(instruction.metadata),
                )

    if reason is None:
        schema = get_operator_schema(instruction.name)
        if schema is None:
            reason = "it is an unknown opcode with no matrix"
        else:
            inverted = inverse_operator(schema, instruction.params)
            if inverted is None:
                reason = (
                    f"its adjoint declaration {schema.adjoint!r} names no inverse gate"
                )
            else:
                opcode, params = inverted
                return Instruction(
                    name=opcode,
                    wires=instruction.wires,
                    params=params,
                    metadata=dict(instruction.metadata),
                )

    raise CapabilityError(
        f"Cannot invert instruction {instruction.name!r} on qubits "
        f"{instruction.wires}: {reason}."
    )


def _composition_source(other: Any) -> tuple[int, Any, tuple[Instruction, ...]]:
    """Read the width, batch size, and instructions of a program being composed.

    The instructions are read into a tuple, so composing a circuit onto itself
    appends a snapshot rather than iterating the list that is growing under it. A
    hand-built :class:`CircuitIR` that declares no batch size is read as ``bsz=1``,
    which is what a program with no batch dimension means; a mismatch against the
    target circuit is refused by the caller rather than resolved.
    """

    if isinstance(other, Circuit):
        return other.n_qubits, other.bsz, tuple(other._instructions)
    if isinstance(other, CircuitIR):
        return other.n_wires, other.metadata.get("batch_size", 1), other.instructions
    raise TypeError(
        "Circuit.compose accepts a Circuit or a CircuitIR, got "
        f"{type(other).__name__}"
    )


class _CircuitConstructionOptions(TypedDict):
    """Configuration retained when binding or copying a circuit."""

    n_wires: int
    bsz: int
    device: torch.device | str | None
    dtype: torch.dtype
    inputs: torch.Tensor | None
    config: RuntimeConfig


class _StatevectorExecutionStatistics(TypedDict, total=False):
    """Counters populated as local statevector execution progresses."""

    triton_single_qubit_loop_enabled: bool
    triton_single_qubit_loop_regions: int
    triton_single_qubit_loop_gates: int
    triton_ry_rz_pair_candidates: int
    triton_ry_rz_pair_executed: int
    triton_single_qubit_matrix_regions: int
    diagonal_elementwise_gates: int
    diagonal_fused_regions: int
    permutation_gates: int
    triton_cx_sequence_regions: int
    fixed_single_qubit_specialized_gates: int
    fused_gate_regions: int
    fused_gate_count: int
    dependency_reordered_single_qubit_regions: int
    statevector_apply_count: int
    batched_rx_ry_rz_regions: int
    batched_rotation_sequence_regions: int
    native_cpu_one_qubit_layer_regions: int
    native_cpu_parameterized_one_qubit_layer_regions: int
    native_cpu_clifford_matching_regions: int
    statevector_batch_chunk_size: int
    statevector_batch_chunk_count: int
    statevector_batch_chunk_budget_bytes: int
    statevector_batch_assembly: str


class Circuit:
    """FlagQuantum native differentiable circuit.

    Examples:
        Build a two-qubit Bell circuit using the fluent gate API:

        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2).h(0).cx(0, 1)
        >>> circuit.n_qubits
        2
    """

    _parameter_bindings: BuilderBindings
    _n_wires: int
    _bsz: int

    def __init__(
        self,
        n_qubits: int | None = None,
        *,
        n_wires: int | None = None,
        nqubits: int | None = None,
        bsz: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        inputs: torch.Tensor | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        from .core._qubit_aliases import warn_qubit_alias

        for alias, value in (("n_wires", n_wires), ("nqubits", nqubits)):
            if value is not None:
                warn_qubit_alias(alias, "n_qubits")
        counts = {
            name: _count_argument(name, value)
            for name, value in (
                ("n_qubits", n_qubits),
                ("n_wires", n_wires),
                ("nqubits", nqubits),
            )
            if value is not None
        }
        if not counts:
            raise ValidationError(
                "Circuit requires n_qubits (or legacy n_wires/nqubits)."
            )
        if len(set(counts.values())) != 1:
            rendered = ", ".join(f"{name}={value}" for name, value in counts.items())
            raise ValidationError(
                f"Circuit received conflicting qubit counts: {rendered}."
            )
        # The backing fields are written directly here: the setters below refresh
        # derived state that does not exist yet, and both values were already read
        # through ``_count_argument`` when the aliases were resolved.
        self._n_wires = next(iter(counts.values()))
        self._nqubits = self._n_wires
        self._bsz = _count_argument("bsz", bsz)
        self.runtime_config = config or get_runtime_config()
        if dtype is not None:
            self.runtime_config = self.runtime_config.with_overrides(
                complex_dtype=str(dtype).removeprefix("torch.")
            )
        self.device = device if device is not None else self.runtime_config.device
        self.dtype = dtype or getattr(torch, self.runtime_config.complex_dtype)
        self._instructions: list[Instruction] = []
        self._state_cache: torch.Tensor | None = None
        self._last_statevector_runtime: _StatevectorExecutionStatistics = {}
        self._initial_state_workspace: torch.Tensor | None = None
        self._initial_state_batch_window_workspace: torch.Tensor | None = None
        self._ir_cache: CircuitIR | None = None
        self._backend_programs: dict[tuple[Any, ...], Any] = {}
        self._statevector_constant_parameters: dict[
            tuple[int, str, torch.dtype, int], torch.Tensor
        ] = {}
        self._statevector_cx_masks: dict[
            tuple[tuple[int, ...], tuple[int, ...], str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._statevector_fused_matrices: dict[
            tuple[int, str, torch.dtype, int], torch.Tensor
        ] = {}
        self._statevector_z_signs: dict[
            tuple[tuple[int, ...], str, torch.dtype], torch.Tensor
        ] = {}
        self._inputs = inputs
        self.circuit_param: _CircuitConstructionOptions = {
            "n_wires": self.n_wires,
            "bsz": self.bsz,
            "device": self.device,
            "dtype": self.dtype,
            "inputs": inputs,
            "config": self.runtime_config,
        }

    def _invalidate_execution_cache(self, *, instructions_changed: bool) -> None:
        """Invalidate cached execution state after a circuit mutation."""

        self._state_cache = None
        if not instructions_changed:
            return
        self._ir_cache = None
        self._backend_programs.clear()
        self._statevector_constant_parameters.clear()
        self._statevector_cx_masks.clear()
        self._statevector_fused_matrices.clear()

    def _set_declared_count(self, name: str, value: Any) -> None:
        """Apply one declared count under the rule the constructor already applies.

        ``bsz`` and ``n_wires`` are read by the program builder and by every
        executor, so the rule that reads them from ``Circuit(...)`` has to be the
        rule that reads them from an assignment. Without this, ``circuit.bsz = 2.7``
        was stored as ``2.7``, reported back as ``2.7``, and executed as a
        two-entry batch; ``circuit.bsz = 0`` was accepted and failed later inside a
        run with an error naming neither the circuit nor ``bsz``.
        """

        count = _count_argument(name, value)
        if name == "n_wires":
            self._require_width_covers_recorded_wires(count)
        if getattr(self, f"_{name}") == count:
            return
        setattr(self, f"_{name}", count)
        if name == "n_wires":
            self._nqubits = count
        self._after_declared_count_change()

    def _require_width_covers_recorded_wires(self, n_wires: int) -> None:
        """Refuse a width that would strand an instruction already recorded."""

        for index, instruction in enumerate(self._instructions):
            stranded = tuple(wire for wire in instruction.wires if wire >= n_wires)
            if stranded:
                raise ValidationError(
                    f"Circuit n_wires must cover every recorded wire: instruction "
                    f"{index} references wire {stranded[0]}, which needs n_wires >= "
                    f"{stranded[0] + 1}."
                )

    def _after_declared_count_change(self) -> None:
        """Discard everything derived from ``n_wires`` and ``bsz``.

        A count change reaches further than a gate edit. The cached IR carries both
        counts, the cached backend programs are built from them, and the statevector
        workspaces are shaped ``(bsz, 2**n_wires)`` — including
        :attr:`_initial_state_workspace` and the Z-sign table, which a gate edit may
        keep but a width or batch change cannot.
        """

        self._invalidate_execution_cache(instructions_changed=True)
        self._initial_state_workspace = None
        self._initial_state_batch_window_workspace = None
        self._statevector_z_signs.clear()
        self.circuit_param = {
            **self.circuit_param,
            "n_wires": self._n_wires,
            "bsz": self._bsz,
        }

    @property
    def n_wires(self) -> int:
        """Number of wires the recorded instructions may reference."""

        return self._n_wires

    @n_wires.setter
    def n_wires(self, value: Any) -> None:
        self._set_declared_count("n_wires", value)

    @property
    def bsz(self) -> int:
        """Declared batch size: one amplitude set per entry."""

        return self._bsz

    @bsz.setter
    def bsz(self, value: Any) -> None:
        self._set_declared_count("bsz", value)

    @property
    def n_qubits(self) -> int:
        """Number of public circuit qubits."""

        return self.n_wires

    @property
    def num_qubits(self) -> int:
        """Qiskit-compatible alias for :attr:`n_qubits`."""

        return self.n_wires

    def gate(
        self,
        name: str,
        wires: Iterable[int] | int,
        *,
        params: Mapping[str, Any] | None = None,
        matrix: Any | None = None,
        **kwargs: Any,
    ) -> "Circuit":
        merged_params = dict(params or {})
        merged_params.update(kwargs)
        normalized_wires = normalize_qubits(wires, owner=f"Gate {name!r}")
        outside = tuple(wire for wire in normalized_wires if wire >= self.n_wires)
        if outside:
            raise ValidationError(
                f"Gate {name!r} references wire(s) {outside} outside circuit "
                f"range [0, {self.n_wires - 1}]."
            )
        opcode = canonical_opcode(name)
        schema = get_operator_schema(opcode)
        metadata: dict[str, Any] = {}
        if schema is not None and schema.channel:
            if matrix is not None:
                raise ValidationError(
                    f"Channel {opcode!r} derives its Kraus operators from its "
                    "declared parameters; pass parameters instead of a matrix."
                )
            from .noise.channels import channel_from_parameters

            channel = channel_from_parameters(opcode, merged_params)
            matrix = channel.kraus
            # Keep the declared parameters on the instruction at their stored
            # value, so an inline channel and a channel the noise model lowered
            # are the same instruction and the plan identity covers both alike.
            merged_params = dict(channel.parameters)
            metadata["is_channel"] = True
        instruction = Instruction(
            name=opcode,
            wires=normalized_wires,
            params=merged_params,
            matrix=matrix,
            metadata=metadata,
        )
        self._instructions.append(instruction)
        self._invalidate_execution_cache(instructions_changed=True)
        return self

    def any(self, *wires: int, unitary: Any, name: str = "any") -> "Circuit":
        return self.gate(name, wires, matrix=unitary)

    unitary = any

    def adjoint(self) -> "Circuit":
        """Return the circuit that undoes this one.

        The instructions are emitted in reverse order, and each one is replaced by the
        single gate that inverts it. The rule belongs to the opcode, is declared once in
        :data:`flagquantum.core.OPERATOR_SCHEMAS`, and is read from there: a gate that is
        its own inverse is kept as it is, an angle is negated, and ``s``/``t``/``sx``
        become ``sdg``/``tdg``/``sxdg``. The qubits of every instruction are unchanged,
        because the inverse of a gate acts on the same qubits in the same order.

        A custom operation recorded through :meth:`any` is inverted through its own
        matrix rather than through an opcode rule, since that matrix is what executes.

        This is a construction-time operation: it emits instructions the IR already
        describes, so ``IR_VERSION`` does not change. The result is a new circuit, so
        the circuit being inverted stays usable, and a reusable block can be inverted
        before it is appended a second time.

        Returns:
            A new circuit with the same qubit count, batch size, device, and dtype.

        Raises:
            CapabilityError: If an instruction is not an invertible fixed gate: a noise
                channel, a mid-circuit measurement or reset, a classically conditioned
                gate, or a custom operation whose matrix is not a unitary.

        Examples:
            >>> import flagquantum as fq
            >>> circuit = fq.Circuit(2).h(0).s(0).cx(0, 1)
            >>> [item.name for item in circuit.adjoint().to_ir().instructions]
            ['cx', 'sdg', 'h']
        """

        inverse = type(self)(
            n_qubits=self.n_wires,
            bsz=self.bsz,
            device=self.device,
            dtype=self.dtype,
            inputs=self._inputs,
            config=self.runtime_config,
        )
        inverse._instructions.extend(
            _inverted_instruction(instruction)
            for instruction in reversed(self._instructions)
        )
        if self._instructions:
            inverse._invalidate_execution_cache(instructions_changed=True)
        return inverse

    def compose(
        self,
        other: "Circuit | CircuitIR",
        *,
        qubits: Iterable[int] | int | None = None,
        qubit_map: Mapping[int, int] | None = None,
    ) -> "Circuit":
        """Continue this circuit with another program, placed on chosen qubits.

        Every instruction of ``other`` is rewritten onto the target qubits and
        appended, so the result is the circuit that building those instructions by
        hand in the same order would have produced. Composition is a construction-time
        operation: it emits instructions the IR already describes, so ``IR_VERSION``
        does not change.

        Args:
            other: The program to append, as a :class:`Circuit` or a
                :class:`CircuitIR`.
            qubits: Target qubit for each qubit of ``other``, in ``other``'s own
                order. A single qubit is accepted for a one-qubit program.
            qubit_map: Target qubit of each qubit of ``other``, keyed by the qubit
                label ``other`` uses. At most one of ``qubits`` and ``qubit_map`` may
                be given.

        Returns:
            This circuit, so that calls chain.

        Raises:
            TypeError: If both ``qubits`` and ``qubit_map`` are given, if ``other`` is
                neither a :class:`Circuit` nor a :class:`CircuitIR`, or if a label is
                not an integer.
            ValidationError: If the two programs disagree on batch size, if the target
                map does not name every qubit of ``other``, if it names one target
                qubit twice, or if a target qubit is outside this circuit.

        Examples:
            >>> import flagquantum as fq
            >>> bell = fq.Circuit(2).h(0).cx(0, 1)
            >>> fq.Circuit(4).x(0).compose(bell, qubits=(1, 2)).to_ir().instructions[-1].wires
            (1, 2)
        """

        local_qubits, other_bsz, instructions = _composition_source(other)
        if other_bsz != self.bsz:
            raise ValidationError(
                f"Circuit.compose cannot mix batch sizes: this circuit has "
                f"bsz={self.bsz}, the composed program has bsz={other_bsz}."
            )
        target = qubit_map_from(
            local_qubits,
            qubits=qubits,
            qubit_map=qubit_map,
            owner="Circuit.compose",
        )
        negative = tuple(qubit for qubit in target if qubit < 0)
        if negative:
            raise ValidationError(
                f"Circuit.compose target qubit(s) {negative} must be non-negative."
            )
        outside = tuple(qubit for qubit in target if qubit >= self.n_wires)
        if outside:
            raise ValidationError(
                f"Circuit.compose target qubit(s) {outside} outside circuit range "
                f"[0, {self.n_wires - 1}]."
            )
        for instruction in instructions:
            self._instructions.append(
                Instruction(
                    name=instruction.name,
                    wires=remap_qubits(
                        instruction.wires, target, owner="Circuit.compose"
                    ),
                    params=dict(instruction.params),
                    matrix=instruction.matrix,
                    metadata=dict(instruction.metadata),
                )
            )
        if instructions:
            self._invalidate_execution_cache(instructions_changed=True)
        return self

    def to_ir(self) -> CircuitIR:
        if self._ir_cache is None:
            # A decimal JSON encoding of ``2**n_wires`` is irrelevant to
            # MPS/TN execution and eventually hits Python's large-integer
            # string guard. Keep dense shapes for statevector-sized circuits
            # and describe larger logical amplitude spaces symbolically.
            dense_shape_available = self.n_wires <= 4096
            self._ir_cache = CircuitIR(
                self.n_wires,
                tuple(self._instructions),
                dtype=str(self.dtype).removeprefix("torch."),
                shape=(
                    (self.bsz, 2**self.n_wires)
                    if dense_shape_available
                    else (self.bsz,)
                ),
                metadata={
                    "batch_size": self.bsz,
                    "logical_state_shape": (
                        None
                        if dense_shape_available
                        else {
                            "batch_size": self.bsz,
                            "amplitude_dimension": "power_of_two",
                            "amplitude_exponent": self.n_wires,
                        }
                    ),
                    "runtime_config": self.runtime_config.to_manifest(),
                },
            )
        return self._ir_cache

    def to_qir(self) -> list[dict[str, Any]]:
        return [
            {
                "name": inst.name,
                "index": inst.wires,
                "parameters": dict(inst.params),
                "gate": inst.matrix,
                "mpo": False,
                "split": None,
            }
            for inst in self._instructions
        ]

    def draw(self, format: str = "text", **kwargs: Any) -> Any:
        """Draw this circuit through FlagQuantum's unified drawer."""

        from .drawer import draw

        return draw(self, format=format, **kwargs)

    @classmethod
    def from_ir(cls, ir: CircuitIR, **kwargs: Any) -> "Circuit":
        kwargs.pop("n_qubits", None)
        kwargs.pop("n_wires", None)
        kwargs.pop("nqubits", None)
        if "dtype" not in kwargs and "config" not in kwargs:
            kwargs["dtype"] = getattr(torch, ir.dtype)
        circuit = cls(ir.n_wires, **kwargs)
        circuit._instructions.extend(ir.instructions)
        return circuit

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of symbolic parameters used by this circuit."""

        names: set[str] = set()
        for instruction in self._instructions:
            for value in instruction.params.values():
                names.update(parameter_names_in_value(value))
        return tuple(sorted(names))

    def is_parameterized(self) -> bool:
        """Return whether the circuit contains unbound symbolic parameters."""

        return bool(self.parameter_names)

    def bind_parameters(self, values: Mapping[str | Any, Any]) -> "Circuit":
        """Return a copy with named symbolic parameters replaced by values."""

        bound = type(self)(**self.circuit_param)
        bound._instructions.extend(
            Instruction(
                name=instruction.name,
                wires=instruction.wires,
                params=bind_parameter_value(instruction.params, values),
                matrix=instruction.matrix,
                metadata=instruction.metadata,
            )
            for instruction in self._instructions
        )
        return bound

    @classmethod
    def from_qir(cls, qir: Sequence[Mapping[str, Any]], **kwargs: Any) -> "Circuit":
        explicit_counts = {
            name: kwargs.pop(name)
            for name in ("n_qubits", "n_wires", "nqubits")
            if name in kwargs
        }
        if explicit_counts:
            circuit = cls(**explicit_counts, **kwargs)
        else:
            inferred = 1 + max(max(item["index"]) for item in qir if item.get("index"))
            circuit = cls(inferred, **kwargs)
        for item in qir:
            circuit.gate(
                str(item["name"]),
                item["index"],
                params=item.get("parameters", {}),
                matrix=item.get("gate"),
            )
        return circuit

    def initial_state(self) -> torch.Tensor:
        from .simulation.statevector.batching import _initial_state

        # Statevector/TN kernels are functional: they never mutate this leaf.
        # It is therefore safe to share it across autograd graphs and avoid a
        # zero-fill allocation on every training step.
        return _initial_state(self)

    def state(self, *, refresh: bool = False) -> torch.Tensor:
        from .simulation.statevector.local import state

        return state(self, refresh=refresh)

    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.state()) ** 2

    probability = probabilities

    def density_matrix(self) -> torch.Tensor:
        from .simulation.density_matrix import density_matrix

        return density_matrix(self)

    def noisy_density_matrix(
        self, noise_model: NoiseModel | None = None
    ) -> torch.Tensor:
        from .runtime.noise_registry import noisy_density_matrix

        return noisy_density_matrix(self, noise_model)

    def run(
        self,
        *,
        options: ExecutionOptions | None = None,
        outputs: OutputRequest | Sequence[OutputRequest] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> ExecutionResult:
        """Execute the circuit and return an :class:`ExecutionResult`.

        This is exactly the object-oriented spelling of
        ``flagquantum.run(circuit, ...)``. Backend-native controls belong to
        :mod:`flagquantum.runtime`.
        """

        from ._api import run

        return run(
            self,
            options=options,
            outputs=outputs,
            noise_model=noise_model,
            compiler=None,
            target=None,
            target_qubits=None,
            shots=None,
            name=None,
        )

    def expectation_z(self, wires: Iterable[int] | int | None = None) -> torch.Tensor:
        from .simulation.statevector.local import _expectation_z

        if wires is None:
            wires = range(self.n_wires)
        return _expectation_z(self, normalize_qubits(wires, owner="expectation_z"))

    def expectation_ps(
        self,
        *,
        z: Sequence[int] | None = None,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
    ) -> torch.Tensor:
        from .simulation.statevector.local import _expectation_pauli_string

        x_set = set(x or ())
        y_set = set(y or ())
        z_set = set(z or ())
        if (x_set & y_set) or (x_set & z_set) or (y_set & z_set):
            raise ValidationError("A wire can appear in only one of x, y, or z.")

        return _expectation_pauli_string(
            self,
            x=tuple(x or ()),
            y=tuple(y or ()),
            z=tuple(z or ()),
        )

    def sample(
        self,
        shots: int = 1,
        *,
        generator: torch.Generator | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        """Sample computational-basis bitstrings from the circuit state."""

        from .simulation.statevector.local import _sample_statevector

        if format not in {"bits", "index"}:
            raise ValidationError("sample format must be 'bits' or 'index'.")
        return _sample_statevector(
            self,
            shots=shots,
            generator=generator,
            return_bits=format == "bits",
        )

    def counts(
        self,
        shots: int,
        *,
        generator: torch.Generator | None = None,
        format: str = "bin",
    ) -> list[dict[str | int, int]]:
        """Return batched sample counts."""

        # Validate the format before sampling. The loop below only sees a
        # sampled row, so a request that sampled no rows — or one whose budget
        # is spent before the first row is formatted — spent the whole shot
        # budget and then refused the request it was always going to refuse.
        # The MPS and tensor-network states check this first for the same
        # reason; see tests/unit/test_counts_histogram_format.py.
        if format not in ("bin", "int"):
            raise ValidationError("counts format must be 'bin' or 'int'.")

        samples = self.sample(shots, generator=generator, format="index")
        outputs: list[dict[str | int, int]] = []
        for row in samples:
            unique, counts = torch.unique(row, return_counts=True)
            batch_counts: dict[str | int, int] = {}
            for key, count in zip(unique.tolist(), counts.tolist(), strict=True):
                if format == "int":
                    out_key: str | int = int(key)
                else:
                    out_key = f"{int(key):0{self.n_wires}b}"
                batch_counts[out_key] = int(count)
            outputs.append(batch_counts)
        return outputs

    def run_distributed(self, **options: Any) -> Any:
        """Run through the distributed statevector engine."""

        from .runtime.execution import run_distributed

        return run_distributed(self.to_ir(), **options)

    def to_device(self, **options: Any) -> Any:
        return self.run_distributed(**options)

    def compile(self, **options: Any) -> "Circuit":
        from .compiler import compile as compile_program

        compiled_ir = compile_program(self, **options)
        compiled = type(self).from_ir(
            compiled_ir,
            **dict(self.circuit_param),
        )
        compiled._ir_cache = compiled_ir
        return compiled

    def layers(self) -> list[list[Instruction]]:
        from .compiler import schedule_layers

        return schedule_layers(self.to_ir())

    def analysis(self) -> CircuitAnalysis:
        from .runtime.planner import analyze

        return analyze(self.to_ir())

    def plan(
        self,
        *,
        options: ExecutionOptions | None = None,
        outputs: OutputRequest | Sequence[OutputRequest] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> ExecutionPlan:
        """Plan this circuit using stable backend-neutral execution options."""

        from ._api import plan

        return plan(
            self,
            options=options,
            outputs=outputs,
            noise_model=noise_model,
        )

    def runtime_plan(self, **options: Any) -> RuntimeSelectionPlan:
        """Explain the best local, JAX, or distributed runtime for this circuit."""

        from .runtime.planner import plan_runtime_selection

        options.setdefault("bsz", self.bsz)
        return plan_runtime_selection(self.to_ir(), **options)

    def copy(self) -> "Circuit":
        return type(self).from_ir(self.to_ir(), **dict(self.circuit_param))

    def __len__(self) -> int:
        return len(self._instructions)

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, instructions={len(self)})"


def expectation(*ops: tuple[Any, Sequence[int]], ket: torch.Tensor) -> torch.Tensor:
    """Compute a small dense expectation value with native PyTorch tensors."""

    from .simulation.statevector.local import _expectation_from_operators

    return _expectation_from_operators(*ops, ket=ket)


_CONTROLLED_TWO_QUBIT_GATES = {
    "cx",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "cphase",
}


def _public_qubit_keywords(opcode: str, arity: int) -> tuple[tuple[str, ...], ...]:
    """Return public keyword aliases without changing internal wire vocabulary."""

    if arity == 1:
        return (("qubit", "target"),)
    if arity == 2 and opcode in _CONTROLLED_TWO_QUBIT_GATES:
        return (("control",), ("target",))
    if arity == 2:
        return (("qubit1", "left"), ("qubit2", "right"))
    if opcode == "ccx":
        return (("control1",), ("control2",), ("target",))
    if opcode == "cswap":
        return (("control",), ("target1",), ("target2",))
    return tuple((f"qubit{index + 1}",) for index in range(arity))


def _install_gate_method(name: str) -> None:
    schema = get_operator_schema(name)

    def method(self: Circuit, *args: Any, **kwargs: Any) -> Circuit:
        arity = schema.arity if schema is not None else len(args)
        positional = list(args)
        wires: list[Any | None] = [None] * arity
        if schema is not None:
            for index, aliases in enumerate(
                _public_qubit_keywords(schema.opcode, schema.arity)
            ):
                supplied = [alias for alias in aliases if alias in kwargs]
                if len(supplied) > 1:
                    rendered = ", ".join(supplied)
                    raise TypeError(
                        f"{name} got multiple names for qubit {index}: {rendered}"
                    )
                if supplied:
                    wires[index] = kwargs.pop(supplied[0])
        for index, wire in enumerate(wires):
            if wire is None and positional:
                wires[index] = positional.pop(0)
        resolved_wires = []
        for wire in wires:
            if wire is None:
                expected = ", ".join(
                    aliases[0] for aliases in _public_qubit_keywords(name, arity)
                )
                raise TypeError(f"{name} requires qubit arguments: {expected}")
            resolved_wires.append(wire)

        values = positional
        parameter_names = schema.parameters if schema is not None else ()
        if len(values) > len(parameter_names):
            raise TypeError(
                f"{name} accepts {arity} qubit(s) and "
                f"{len(parameter_names)} parameter(s)"
            )
        # Positional values may cover only a prefix of the schema: the remaining
        # parameters are allowed to arrive as keywords, as in `rx(0, theta=0.5)`.
        # A short `values` is therefore expected here, and the missing names are
        # caught by `IRValidationError` below rather than by `strict=True`.
        for parameter_name, value in zip(parameter_names, values, strict=False):
            if parameter_name in kwargs:
                raise TypeError(f"{name} got multiple values for {parameter_name!r}")
            kwargs[parameter_name] = value
        return self.gate(name, resolved_wires, **kwargs)

    method.__name__ = name
    setattr(Circuit, name, method)
    setattr(Circuit, name.upper(), method)


for _gate_name in sorted(set(OPERATOR_SCHEMAS) | set(OPERATOR_ALIASES)):
    _install_gate_method(_gate_name)


__all__ = ["Circuit", "expectation"]
