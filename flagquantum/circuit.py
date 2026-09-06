"""FlagQuantum native circuit API.

This module is intentionally implemented on top of FlagQuantum's own IR and
PyTorch tensor operations. It does not depend on tensor-network, graph, or
scientific-computing helper packages, which keeps the core runtime portable to
AI accelerators that already support mainstream deep-learning frameworks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch

from .core.ir import CircuitIR, Instruction, MeasurementNode
from .core.operator_schema import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    canonical_opcode,
    get_operator_schema,
)
from .core.parameters import (
    bind_parameter_value,
    parameter_names_in_value,
)
from .core.runtime_config import RuntimeConfig, get_runtime_config
from .errors import ValidationError
from .ops.matrices import GATE_MAT_DICT

if TYPE_CHECKING:
    from .noise import NoiseModel
    from .runtime.execution_plan import ExecutionPlan
    from .runtime.options import ExecutionOptions
    from .runtime.result import ExecutionResult


def _normalize_wires(wires: Iterable[int] | int) -> tuple[int, ...]:
    if isinstance(wires, int):
        return (wires,)
    return tuple(int(wire) for wire in wires)


class Circuit:
    """FlagQuantum native differentiable circuit."""

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
        counts = {
            name: int(value)
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
        self.n_wires = next(iter(counts.values()))
        if self.n_wires <= 0:
            raise ValidationError(
                f"Circuit n_qubits must be positive, got {self.n_wires}."
            )
        self._nqubits = self.n_wires
        self.bsz = int(bsz)
        self.runtime_config = config or get_runtime_config()
        if dtype is not None:
            self.runtime_config = self.runtime_config.with_overrides(
                complex_dtype=str(dtype).removeprefix("torch.")
            )
        self.device = device if device is not None else self.runtime_config.device
        self.dtype = dtype or getattr(torch, self.runtime_config.complex_dtype)
        self._instructions: list[Instruction] = []
        self._state_cache: torch.Tensor | None = None
        self._initial_state_workspace: torch.Tensor | None = None
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
        self.circuit_param = {
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
        normalized_wires = _normalize_wires(wires)
        outside = tuple(wire for wire in normalized_wires if wire >= self.n_wires)
        if outside:
            raise ValidationError(
                f"Gate {name!r} references wire(s) {outside} outside circuit "
                f"range [0, {self.n_wires - 1}]."
            )
        instruction = Instruction(
            name=canonical_opcode(name),
            wires=normalized_wires,
            params=merged_params,
            matrix=matrix,
        )
        self._instructions.append(instruction)
        self._invalidate_execution_cache(instructions_changed=True)
        return self

    def any(self, *wires: int, unitary: Any, name: str = "any") -> "Circuit":
        return self.gate(name, wires, matrix=unitary)

    unitary = any

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

        bound = type(self)(**dict(self.circuit_param))
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
        from .simulation.statevector import _initial_state

        # Statevector/TN kernels are functional: they never mutate this leaf.
        # It is therefore safe to share it across autograd graphs and avoid a
        # zero-fill allocation on every training step.
        return _initial_state(self)

    def state(self, *, refresh: bool = False) -> torch.Tensor:
        from .simulation.statevector import state

        return state(self, refresh=refresh)

    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.state()) ** 2

    probability = probabilities

    def density_matrix(self) -> torch.Tensor:
        from .simulation.density_matrix import density_matrix

        return density_matrix(self)

    def noisy_density_matrix(self, noise_model=None) -> torch.Tensor:
        from .runtime.noise_registry import noisy_density_matrix

        return noisy_density_matrix(self, noise_model)

    def run(
        self,
        *,
        options: ExecutionOptions | None = None,
        measurements: Sequence[MeasurementNode] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> ExecutionResult:
        """Execute the circuit and return an :class:`ExecutionResult`.

        This is exactly the object-oriented spelling of
        ``flagquantum.run(circuit, ...)``. Backend-native controls belong to
        :mod:`flagquantum.backends`.
        """

        from .runtime.execution import run as run_circuit

        return run_circuit(
            self,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    def expectation_z(self, wires: Iterable[int] | int | None = None) -> torch.Tensor:
        from .simulation.statevector import _expectation_z

        if wires is None:
            wires = range(self.n_wires)
        return _expectation_z(self, _normalize_wires(wires))

    def expectation_ps(
        self,
        *,
        z: Sequence[int] | None = None,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
    ) -> torch.Tensor:
        from .simulation.statevector import _expectation_pauli_string

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

        from .simulation.statevector import _sample_statevector

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

        samples = self.sample(shots, generator=generator, format="index")
        outputs: list[dict[str | int, int]] = []
        for row in samples:
            unique, counts = torch.unique(row, return_counts=True)
            batch_counts: dict[str | int, int] = {}
            for key, count in zip(unique.tolist(), counts.tolist()):
                if format == "int":
                    out_key: str | int = int(key)
                elif format == "bin":
                    out_key = f"{int(key):0{self.n_wires}b}"
                else:
                    raise ValidationError("counts format must be 'bin' or 'int'.")
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

    def analysis(self):
        from .runtime.planner import analyze

        return analyze(self.to_ir())

    def plan(
        self,
        *,
        options: ExecutionOptions | None = None,
        measurements: Sequence[MeasurementNode] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> ExecutionPlan:
        """Plan this circuit using stable backend-neutral execution options."""

        from .runtime.planner import plan

        return plan(
            self,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    def runtime_plan(self, **options: Any):
        """Explain the best local, JAX, or distributed runtime for this circuit."""

        from .runtime.planner import plan_runtime_selection

        options.setdefault("bsz", self.bsz)
        return plan_runtime_selection(self.to_ir(), **options)

    def copy(self) -> "Circuit":
        return type(self).from_ir(self.to_ir(), **dict(self.circuit_param))

    def __len__(self) -> int:
        return len(self._instructions)

    def __repr__(self) -> str:
        return f"Circuit(n_wires={self.n_wires}, instructions={len(self)})"


def expectation(*ops: tuple[Any, Sequence[int]], ket: torch.Tensor) -> torch.Tensor:
    """Compute a small dense expectation value with native PyTorch tensors."""

    from .simulation.statevector import _expectation_from_operators

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
        missing = [index for index, wire in enumerate(wires) if wire is None]
        if missing:
            expected = ", ".join(
                aliases[0] for aliases in _public_qubit_keywords(name, arity)
            )
            raise TypeError(f"{name} requires qubit arguments: {expected}")

        values = positional
        parameter_names = schema.parameters if schema is not None else ()
        if len(values) > len(parameter_names):
            raise TypeError(
                f"{name} accepts {arity} wire(s) and {len(parameter_names)} parameter(s)"
            )
        for parameter_name, value in zip(parameter_names, values):
            if parameter_name in kwargs:
                raise TypeError(f"{name} got multiple values for {parameter_name!r}")
            kwargs[parameter_name] = value
        return self.gate(name, wires, **kwargs)

    method.__name__ = name
    setattr(Circuit, name, method)
    setattr(Circuit, name.upper(), method)


for _gate_name in sorted(
    set(GATE_MAT_DICT) | set(OPERATOR_SCHEMAS) | set(OPERATOR_ALIASES)
):
    _install_gate_method(_gate_name)


__all__ = ["Circuit", "expectation"]
