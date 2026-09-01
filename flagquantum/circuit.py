"""FlagQuantum native circuit API.

This module is intentionally implemented on top of FlagQuantum's own IR and
PyTorch tensor operations. It does not depend on tensor-network, graph, or
scientific-computing helper packages, which keeps the core runtime portable to
AI accelerators that already support mainstream deep-learning frameworks.
"""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch

from .circuit_statevector import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_cx_permutation,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_rx_rz_loop,
    _apply_single_qubit_fixed,
    _batched_rotation_sequence_matrices,
    _batched_rx_ry_rz_matrices,
    _bits_from_indices,
    _canonical_name,
    _compile_statevector_program,
    _fused_gate_matrix,
    _gate_matrix,
    _gate_parameter_tensor,
    _normalize_wires,
    _StatevectorCXSequenceStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorRXRZLoopStep,
    _triton_parameterized_single_qubit_matrix_enabled,
    _triton_ry_rz_pair_enabled,
    _triton_single_qubit_loop_enabled,
    _triton_single_qubit_matrix_enabled,
)
from .core.ir import CircuitIR, Instruction, MeasurementNode
from .core.operator_schema import (
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    get_operator_schema,
)
from .core.parameters import (
    bind_parameter_value,
    parameter_names_in_value,
)
from .core.runtime_config import RuntimeConfig, get_runtime_config, runtime_config
from .ops.complex_ops import complex_conj, complex_mul
from .ops.matrices import GATE_MAT_DICT

if TYPE_CHECKING:
    from .compilation.planner import ExecutionPlan
    from .noise import NoiseModel
    from .runtime.options import ExecutionOptions
    from .runtime.result import ExecutionResult


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
            raise ValueError("Circuit requires n_qubits (or legacy n_wires/nqubits).")
        if len(set(counts.values())) != 1:
            rendered = ", ".join(f"{name}={value}" for name, value in counts.items())
            raise ValueError(f"Circuit received conflicting qubit counts: {rendered}.")
        self.n_wires = next(iter(counts.values()))
        if self.n_wires <= 0:
            raise ValueError(f"Circuit n_qubits must be positive, got {self.n_wires}.")
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
            raise ValueError(
                f"Gate {name!r} references wire(s) {outside} outside circuit "
                f"range [0, {self.n_wires - 1}]."
            )
        instruction = Instruction(
            name=_canonical_name(name),
            wires=normalized_wires,
            params=merged_params,
            matrix=matrix,
        )
        self._instructions.append(instruction)
        self._state_cache = None
        self._ir_cache = None
        self._backend_programs.clear()
        self._statevector_constant_parameters.clear()
        self._statevector_cx_masks.clear()
        self._statevector_fused_matrices.clear()
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
        if self._inputs is not None:
            state = self._inputs.to(device=self.device, dtype=self.dtype)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            return state
        if self._initial_state_workspace is None:
            state = torch.zeros(
                self.bsz, 2**self.n_wires, dtype=self.dtype, device=self.device
            )
            state[:, 0] = 1
            self._initial_state_workspace = state
        # Statevector/TN kernels are functional: they never mutate this leaf.
        # It is therefore safe to share it across autograd graphs and avoid a
        # zero-fill allocation on every training step.
        return self._initial_state_workspace

    def _statevector_gate_parameters(
        self,
        instruction: Instruction,
        state: torch.Tensor,
        parameter_bindings: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor | None:
        """Resolve gate parameters, caching immutable constants on device."""

        cacheable = parameter_bindings is None and all(
            isinstance(value, Number) for value in instruction.params.values()
        )
        if not cacheable:
            return _gate_parameter_tensor(
                instruction,
                state,
                parameter_bindings=parameter_bindings,
            )
        key = (
            id(instruction),
            str(state.device),
            state.dtype,
            int(state.shape[0]),
        )
        cached = self._statevector_constant_parameters.get(key)
        if cached is None:
            cached = _gate_parameter_tensor(
                instruction,
                state,
                parameter_bindings=None,
            )
            if cached is not None:
                self._statevector_constant_parameters[key] = cached
        return cached

    def _statevector_cx_sequence_masks(
        self, step: _StatevectorCXSequenceStep, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (step.controls, step.targets, str(device))
        cached = self._statevector_cx_masks.get(key)
        if cached is None:
            controls = torch.tensor(
                [1 << (self.n_wires - 1 - wire) for wire in step.controls],
                dtype=torch.int64,
                device=device,
            )
            targets = torch.tensor(
                [1 << (self.n_wires - 1 - wire) for wire in step.targets],
                dtype=torch.int64,
                device=device,
            )
            cached = (controls, targets, controls.flip(0), targets.flip(0))
            self._statevector_cx_masks[key] = cached
        return cached

    def _statevector_fused_constant_matrix(
        self,
        step: _StatevectorFusedGateStep,
        state: torch.Tensor,
        parameter_bindings: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor | None:
        """Return a cached matrix for an immutable one-qubit fused region."""

        cacheable = (
            len(step.wires) == 1
            and parameter_bindings is None
            and all(
                instruction.matrix is None
                and all(
                    isinstance(value, Number) for value in instruction.params.values()
                )
                for instruction in step.instructions
            )
        )
        if not cacheable:
            return None
        key = (id(step), str(state.device), state.dtype, int(state.shape[0]))
        cached = self._statevector_fused_matrices.get(key)
        if cached is None:
            cached = _fused_gate_matrix(
                step,
                bsz=state.shape[0],
                device=state.device,
                dtype=state.dtype,
                parameter_bindings=None,
            ).detach()
            self._statevector_fused_matrices[key] = cached
        return cached

    def state(self, *, refresh: bool = False) -> torch.Tensor:
        if self._state_cache is not None and not refresh:
            return self._state_cache
        with runtime_config(self.runtime_config):
            state = self.initial_state()
            binding_source = getattr(self, "_parameter_bindings", None)
            parameter_bindings = (
                None if binding_source is None else binding_source.values()
            )
            enable_triton_loop = (
                _triton_single_qubit_loop_enabled()
                and state.is_cuda
                and state.dtype == torch.complex64
            )
            program_key = ("statevector", "triton_rx_rz_loop", enable_triton_loop)
            program = self._backend_programs.get(program_key)
            if program is None:
                program = _compile_statevector_program(
                    self._instructions,
                    self.n_wires,
                    enable_triton_loop=enable_triton_loop,
                )
                self._backend_programs[program_key] = program
            fused_steps = tuple(
                step for step in program if isinstance(step, _StatevectorRXRZLoopStep)
            )
            triton_ry_rz_pairs = sum(
                isinstance(step, _StatevectorFusedGateStep)
                and tuple(item.name for item in step.instructions) == ("ry", "rz")
                for step in program
            )
            self._last_statevector_runtime = {
                "triton_single_qubit_loop_enabled": enable_triton_loop,
                "triton_single_qubit_loop_regions": len(fused_steps),
                "triton_single_qubit_loop_gates": sum(
                    2 * len(step.pairs) for step in fused_steps
                ),
                "triton_ry_rz_pair_candidates": triton_ry_rz_pairs,
                "triton_ry_rz_pair_executed": 0,
                "triton_single_qubit_matrix_regions": 0,
                "diagonal_elementwise_gates": sum(
                    isinstance(step, _StatevectorGateStep)
                    and _canonical_name(step.instruction.name)
                    in _DIAGONAL_STATEVECTOR_GATES
                    for step in program
                ),
                "permutation_gates": sum(
                    isinstance(step, _StatevectorGateStep)
                    and _canonical_name(step.instruction.name) in {"x", "cx", "swap"}
                    for step in program
                )
                + sum(
                    len(step.controls)
                    for step in program
                    if isinstance(step, _StatevectorCXSequenceStep)
                ),
                "triton_cx_sequence_regions": sum(
                    isinstance(step, _StatevectorCXSequenceStep) for step in program
                ),
                "fixed_single_qubit_specialized_gates": sum(
                    isinstance(step, _StatevectorGateStep)
                    and _canonical_name(step.instruction.name) == "y"
                    for step in program
                ),
                "fused_gate_regions": sum(
                    isinstance(step, _StatevectorFusedGateStep) for step in program
                ),
                "fused_gate_count": sum(
                    len(step.instructions)
                    for step in program
                    if isinstance(step, _StatevectorFusedGateStep)
                ),
                "dependency_reordered_single_qubit_regions": sum(
                    isinstance(step, _StatevectorFusedGateStep)
                    and step.dependency_reordered
                    for step in program
                ),
                "statevector_apply_count": len(program),
            }
            rx_ry_rz_steps = tuple(
                step
                for step in program
                if isinstance(step, _StatevectorFusedGateStep)
                and tuple(item.name for item in step.instructions) == ("rx", "ry", "rz")
                and len(step.wires) == 1
            )
            batched_rx_ry_rz_matrices: dict[int, torch.Tensor] = {}
            if rx_ry_rz_steps:
                region_angles = []
                for step in rx_ry_rz_steps:
                    parameters = tuple(
                        self._statevector_gate_parameters(
                            instruction, state, parameter_bindings
                        )
                        for instruction in step.instructions
                    )
                    if any(parameter is None for parameter in parameters):
                        region_angles = []
                        break
                    region_angles.append(
                        torch.stack(
                            tuple(parameter[:, 0] for parameter in parameters), dim=-1
                        )
                    )
                if region_angles:
                    matrices = _batched_rx_ry_rz_matrices(
                        torch.stack(region_angles, dim=0)
                    ).to(dtype=state.dtype)
                    batched_rx_ry_rz_matrices = {
                        id(step): matrices[index]
                        for index, step in enumerate(rx_ry_rz_steps)
                    }
            self._last_statevector_runtime["batched_rx_ry_rz_regions"] = len(
                batched_rx_ry_rz_matrices
            )
            batched_rotation_matrices: dict[int, torch.Tensor] = {}
            rotation_patterns = {
                tuple(item.name for item in step.instructions)
                for step in program
                if isinstance(step, _StatevectorFusedGateStep)
                and len(step.instructions) >= 2
                and all(item.name in {"rx", "ry", "rz"} for item in step.instructions)
            }
            for names in rotation_patterns:
                rotation_steps = tuple(
                    step
                    for step in program
                    if isinstance(step, _StatevectorFusedGateStep)
                    and tuple(item.name for item in step.instructions) == names
                )
                region_angles = []
                for step in rotation_steps:
                    parameters = tuple(
                        self._statevector_gate_parameters(
                            instruction, state, parameter_bindings
                        )
                        for instruction in step.instructions
                    )
                    if any(parameter is None for parameter in parameters):
                        region_angles = []
                        break
                    region_angles.append(
                        torch.stack(
                            tuple(parameter[:, 0] for parameter in parameters),
                            dim=-1,
                        )
                    )
                if region_angles:
                    matrices = _batched_rotation_sequence_matrices(
                        torch.stack(region_angles, dim=0),
                        names=names,
                        dtype=state.dtype,
                    )
                    batched_rotation_matrices.update(
                        {
                            id(step): matrices[index]
                            for index, step in enumerate(rotation_steps)
                        }
                    )
            self._last_statevector_runtime["batched_rotation_sequence_regions"] = len(
                batched_rotation_matrices
            )
            for step in program:
                if isinstance(step, _StatevectorCXSequenceStep):
                    if state.is_cuda and state.dtype == torch.complex64:
                        from .simulation.triton_kernels import cx_sequence

                        (
                            control_masks,
                            target_masks,
                            reverse_control_masks,
                            reverse_target_masks,
                        ) = self._statevector_cx_sequence_masks(step, state.device)
                        state = cx_sequence(
                            state,
                            control_masks=control_masks,
                            target_masks=target_masks,
                            reverse_control_masks=reverse_control_masks,
                            reverse_target_masks=reverse_target_masks,
                            n_wires=self.n_wires,
                        )
                    else:
                        for control, target in zip(
                            step.controls, step.targets, strict=True
                        ):
                            state = _apply_cx_permutation(
                                state, (control, target), self.n_wires
                            )
                    continue
                if isinstance(step, _StatevectorRXRZLoopStep):
                    state = _apply_rx_rz_loop(
                        state,
                        step,
                        self.n_wires,
                        parameter_bindings,
                    )
                    continue
                if isinstance(step, _StatevectorFusedGateStep):
                    if (
                        state.is_cuda
                        and state.dtype == torch.complex64
                        and _triton_ry_rz_pair_enabled()
                        and tuple(item.name for item in step.instructions)
                        == ("ry", "rz")
                    ):
                        ry_angles = self._statevector_gate_parameters(
                            step.instructions[0],
                            state,
                            parameter_bindings,
                        )
                        rz_angles = self._statevector_gate_parameters(
                            step.instructions[1],
                            state,
                            parameter_bindings,
                        )
                        if (
                            ry_angles is not None
                            and rz_angles is not None
                            and not state.requires_grad
                            and not ry_angles.requires_grad
                            and not rz_angles.requires_grad
                        ):
                            from .simulation.triton_kernels import ry_rz_pair

                            state = ry_rz_pair(
                                state,
                                ry_angles,
                                rz_angles,
                                wire=step.wires[0],
                                n_wires=self.n_wires,
                            )
                            self._last_statevector_runtime[
                                "triton_ry_rz_pair_executed"
                            ] += 1
                            continue
                    constant_matrix = self._statevector_fused_constant_matrix(
                        step, state, parameter_bindings
                    )
                    if (
                        constant_matrix is not None
                        and state.is_cuda
                        and state.dtype == torch.complex64
                        and _triton_single_qubit_matrix_enabled()
                    ):
                        from .simulation.triton_kernels import single_qubit_matrix

                        state = single_qubit_matrix(
                            state,
                            constant_matrix,
                            wire=step.wires[0],
                            n_wires=self.n_wires,
                        )
                        self._last_statevector_runtime[
                            "triton_single_qubit_matrix_regions"
                        ] += 1
                        continue
                    matrix = batched_rotation_matrices.get(id(step))
                    if matrix is None:
                        matrix = batched_rx_ry_rz_matrices.get(id(step))
                    if matrix is None:
                        matrix = _fused_gate_matrix(
                            step,
                            bsz=state.shape[0],
                            device=state.device,
                            dtype=state.dtype,
                            parameter_bindings=parameter_bindings,
                        )
                    if (
                        state.is_cuda
                        and state.dtype == torch.complex64
                        and len(step.wires) == 1
                        and _triton_single_qubit_matrix_enabled()
                        and (
                            not matrix.requires_grad
                            or _triton_parameterized_single_qubit_matrix_enabled()
                        )
                    ):
                        from .simulation.triton_kernels import single_qubit_matrix

                        state = single_qubit_matrix(
                            state,
                            matrix,
                            wire=step.wires[0],
                            n_wires=self.n_wires,
                        )
                        self._last_statevector_runtime[
                            "triton_single_qubit_matrix_regions"
                        ] += 1
                        continue
                    state = _apply_matrix(
                        state,
                        matrix,
                        step.wires,
                        self.n_wires,
                        layout=step.layout,
                    )
                    continue
                instruction = step.instruction
                name = _canonical_name(instruction.name)
                if name in {"x", "cx", "swap"}:
                    state = _apply_fixed_permutation(
                        state, name, instruction.wires, self.n_wires
                    )
                elif name == "y":
                    state = _apply_single_qubit_fixed(
                        state, name, instruction.wires[0], self.n_wires
                    )
                else:
                    matrix = _gate_matrix(
                        instruction,
                        bsz=state.shape[0],
                        device=state.device,
                        dtype=state.dtype,
                        parameter_bindings=parameter_bindings,
                    )
                    apply_gate = (
                        _apply_diagonal_matrix
                        if name in _DIAGONAL_STATEVECTOR_GATES
                        else _apply_matrix
                    )
                    state = apply_gate(
                        state,
                        matrix,
                        instruction.wires,
                        self.n_wires,
                        layout=step.layout,
                    )
        self._state_cache = state
        return state

    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.state()) ** 2

    probability = probabilities

    def density_matrix(self) -> torch.Tensor:
        from .runtime.backends.density_matrix import density_matrix

        return density_matrix(self)

    def noisy_density_matrix(self, noise_model=None) -> torch.Tensor:
        from .runtime.backends.density_matrix import noisy_density_matrix

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
        if wires is None:
            wires = range(self.n_wires)
        wires = _normalize_wires(wires)
        probs = self.probabilities()
        key = (wires, str(probs.device), probs.dtype)
        signs = self._statevector_z_signs.get(key)
        if signs is None:
            basis = torch.arange(
                probs.shape[-1], dtype=torch.int64, device=probs.device
            )
            signs = torch.stack(
                tuple(
                    1 - 2 * ((basis >> (self.n_wires - 1 - wire)) & 1) for wire in wires
                ),
                dim=-1,
            ).to(dtype=probs.dtype)
            self._statevector_z_signs[key] = signs
        return probs @ signs

    def expectation_ps(
        self,
        *,
        z: Sequence[int] | None = None,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
    ) -> torch.Tensor:
        x_set = set(x or ())
        y_set = set(y or ())
        z_set = set(z or ())
        if (x_set & y_set) or (x_set & z_set) or (y_set & z_set):
            raise ValueError("A wire can appear in only one of x, y, or z.")

        state = self.state()
        transformed = state
        for wire in x or ():
            transformed = _apply_matrix(
                transformed,
                GATE_MAT_DICT["x"].to(device=state.device, dtype=state.dtype),
                (wire,),
                self.n_wires,
            )
        for wire in y or ():
            transformed = _apply_matrix(
                transformed,
                GATE_MAT_DICT["y"].to(device=state.device, dtype=state.dtype),
                (wire,),
                self.n_wires,
            )
        for wire in z or ():
            transformed = _apply_matrix(
                transformed,
                GATE_MAT_DICT["z"].to(device=state.device, dtype=state.dtype),
                (wire,),
                self.n_wires,
            )
        value = complex_mul(complex_conj(state), transformed).sum(dim=-1)
        return torch.real(value)

    def sample(
        self,
        shots: int = 1,
        *,
        generator: torch.Generator | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        """Sample computational-basis bitstrings from the circuit state."""

        probs = self.probabilities()
        samples = torch.multinomial(
            probs, num_samples=shots, replacement=True, generator=generator
        )
        if format == "index":
            return samples
        if format != "bits":
            raise ValueError("sample format must be 'bits' or 'index'.")
        return _bits_from_indices(samples, self.n_wires)

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
                    raise ValueError("counts format must be 'bin' or 'int'.")
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
        from .compilation.compiler import compile_for_backend

        compiled_ir = compile_for_backend(self, **options)
        compiled = type(self).from_ir(
            compiled_ir,
            **dict(self.circuit_param),
        )
        compiled._ir_cache = compiled_ir
        return compiled

    def layers(self) -> list[list[Instruction]]:
        from .compilation.compiler import schedule_layers

        return schedule_layers(self.to_ir())

    def analysis(self):
        from .compilation.planner import analyze

        return analyze(self.to_ir())

    def plan(
        self,
        *,
        options: ExecutionOptions | None = None,
        measurements: Sequence[MeasurementNode] | None = None,
        noise_model: NoiseModel | None = None,
    ) -> ExecutionPlan:
        """Plan this circuit using stable backend-neutral execution options."""

        from .compilation.planner import plan

        return plan(
            self,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )

    def runtime_plan(self, **options: Any):
        """Explain the best local, JAX, or distributed runtime for this circuit."""

        from .compilation.planner import plan_runtime_selection

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

    state = ket.reshape(1, -1) if ket.ndim == 1 else ket
    n_wires = int(torch.log2(torch.tensor(state.shape[-1], dtype=torch.float32)).item())
    circuit = Circuit(n_wires, bsz=state.shape[0], device=state.device, inputs=state)
    for matrix, wires in ops:
        circuit.any(*wires, unitary=matrix)
    return complex_mul(complex_conj(state), circuit.state()).sum(dim=-1)


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
