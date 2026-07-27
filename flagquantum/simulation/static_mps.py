"""Functional fixed-shape MPS program inspired by TensorCircuit-NG."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

import torch

from ..core.ir import CircuitIR, ensure_circuit_ir

TensorTuple = tuple[torch.Tensor, ...]
RealImagTuple = tuple[torch.Tensor, ...]


def _bucket_recompile_context():
    try:
        from torch._dynamo import config

        limit = 1024
        return config.patch(
            recompile_limit=max(int(config.recompile_limit), limit),
            accumulated_recompile_limit=max(
                int(config.accumulated_recompile_limit), limit
            ),
        )
    except (ImportError, AttributeError):
        return nullcontext()


def _ri_conj(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value[..., 0], -value[..., 1]), dim=-1)


def _ri_einsum(equation: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    real = torch.einsum(equation, left[..., 0], right[..., 0]) - torch.einsum(
        equation, left[..., 1], right[..., 1]
    )
    imag = torch.einsum(equation, left[..., 0], right[..., 1]) + torch.einsum(
        equation, left[..., 1], right[..., 0]
    )
    return torch.stack((real, imag), dim=-1)


@dataclass(frozen=True)
class StaticMPSProgram:
    n_wires: int
    max_bond: int
    bond_dims: tuple[int, ...]
    crossing_counts: tuple[int, ...] = ()

    @classmethod
    def compile(cls, n_wires: int, max_bond: int) -> "StaticMPSProgram":
        if n_wires < 1 or max_bond < 1:
            raise ValueError("n_wires and max_bond must be positive")
        bonds = tuple(
            min(2 ** min(index, n_wires - index), max_bond)
            for index in range(n_wires + 1)
        )
        return cls(int(n_wires), int(max_bond), bonds, (0,) * (n_wires + 1))

    @classmethod
    def from_ir(
        cls, circuit_or_ir: object, *, max_bond: int | None = None
    ) -> "StaticMPSProgram":
        """Infer exact cut capacities from two-qubit gates in Circuit IR."""
        ir: CircuitIR = ensure_circuit_ir(circuit_or_ir)
        capacities = [1] * (ir.n_wires + 1)
        crossings = [0] * (ir.n_wires + 1)
        schmidt_ranks = {"cx": 2, "cy": 2, "cz": 2, "swap": 4}
        for instruction in ir.instructions:
            wires = tuple(sorted(instruction.wires))
            if len(wires) == 1:
                continue
            if len(wires) != 2:
                raise ValueError(
                    "static MPS rank inference currently supports one- and two-qubit gates"
                )
            rank = int(instruction.metadata.get("operator_schmidt_rank", 0))
            if rank <= 0:
                rank = schmidt_ranks.get(instruction.name, 4)
            for cut in range(wires[0] + 1, wires[1] + 1):
                crossings[cut] += 1
                capacities[cut] = min(
                    capacities[cut] * rank,
                    2 ** min(cut, ir.n_wires - cut),
                )
        required = max(capacities)
        if max_bond is not None and required > int(max_bond):
            raise ValueError(
                f"Circuit IR requires exact bond {required}, exceeding max_bond={max_bond}"
            )
        return cls(
            ir.n_wires,
            required if max_bond is None else int(max_bond),
            tuple(capacities),
            tuple(crossings),
        )

    def zero(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> TensorTuple:
        tensors = []
        for wire in range(self.n_wires):
            tensor = torch.zeros(
                batch_size,
                self.bond_dims[wire],
                2,
                self.bond_dims[wire + 1],
                device=device,
                dtype=dtype,
            )
            tensor[:, 0, 0, 0] = 1
            tensors.append(tensor)
        return tuple(tensors)

    def zero_real_imag(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> RealImagTuple:
        tensors = []
        for wire in range(self.n_wires):
            tensor = torch.zeros(
                batch_size,
                self.bond_dims[wire],
                2,
                self.bond_dims[wire + 1],
                2,
                device=device,
                dtype=dtype,
            )
            tensor[:, 0, 0, 0, 0] = 1
            tensors.append(tensor)
        return tuple(tensors)

    @staticmethod
    def apply_ry_real_imag(
        tensors: RealImagTuple, angle: torch.Tensor, wire: int
    ) -> RealImagTuple:
        current = tensors[wire]
        cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
        zero = cosine * current[:, :, 0] - sine * current[:, :, 1]
        one = sine * current[:, :, 0] + cosine * current[:, :, 1]
        updated = torch.stack((zero, one), dim=2)
        return tensors[:wire] + (updated,) + tensors[wire + 1 :]

    @staticmethod
    def apply_rz_real_imag(
        tensors: RealImagTuple, angle: torch.Tensor, wire: int
    ) -> RealImagTuple:
        current = tensors[wire]
        cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
        signs = torch.tensor(
            [-1.0, 1.0], device=current.device, dtype=current.dtype
        ).reshape(1, 1, 2, 1)
        real = cosine * current[..., 0] - signs * sine * current[..., 1]
        imag = signs * sine * current[..., 0] + cosine * current[..., 1]
        updated = torch.stack((real, imag), dim=-1)
        return tensors[:wire] + (updated,) + tensors[wire + 1 :]

    def apply_cx_mpo_real_imag(
        self,
        tensors: RealImagTuple,
        left_wire: int,
        active_dims: tuple[int, int, int],
    ) -> RealImagTuple:
        """Apply CX by its exact rank-two MPO without an SVD split."""
        left, right = tensors[left_wire], tensors[left_wire + 1]
        left_active, center_active, right_active = active_dims
        center_capacity = self.bond_dims[left_wire + 1]
        new_center = center_active * 2
        if new_center > center_capacity:
            left_core = left[:, :left_active, :, :center_active]
            right_core = right[:, :center_active, :, :right_active]
            theta = _ri_einsum("blsm,bmtr->blstr", left_core, right_core).reshape(
                left.shape[0], left_active, 4, right_active, 2
            )
            pair = theta[:, :, (0, 1, 3, 2)].reshape(
                left.shape[0], left_active * 2, 2 * right_active, 2
            )
            rows, columns = left_active * 2, 2 * right_active
            if center_capacity < min(rows, columns):
                raise RuntimeError(
                    "Circuit IR bond profile requires a rank-revealing exact split"
                )
            if rows <= columns:
                real = torch.eye(
                    rows, center_capacity, device=pair.device, dtype=pair.dtype
                ).expand(left.shape[0], -1, -1)
                left_out = torch.stack((real, torch.zeros_like(real)), dim=-1)
                right_out = torch.nn.functional.pad(
                    pair, (0, 0, 0, 0, 0, center_capacity - rows)
                )
            else:
                left_out = torch.nn.functional.pad(
                    pair, (0, 0, 0, center_capacity - columns)
                )
                real = torch.eye(
                    center_capacity, columns, device=pair.device, dtype=pair.dtype
                ).expand(left.shape[0], -1, -1)
                right_out = torch.stack((real, torch.zeros_like(real)), dim=-1)
            left_out = left_out.reshape(
                left.shape[0], left_active, 2, center_capacity, 2
            )
            right_out = right_out.reshape(
                right.shape[0], center_capacity, 2, right_active, 2
            )
            new_center = center_capacity
            left_out = torch.nn.functional.pad(
                left_out,
                (0, 0, 0, 0, 0, 0, 0, self.bond_dims[left_wire] - left_active),
            )
            right_out = torch.nn.functional.pad(
                right_out,
                (
                    0,
                    0,
                    0,
                    self.bond_dims[left_wire + 2] - right_active,
                    0,
                    0,
                    0,
                    0,
                ),
            )
            return (
                tensors[:left_wire] + (left_out, right_out) + tensors[left_wire + 2 :]
            )
        left = left[:, :left_active, :, :center_active]
        right = right[:, :center_active, :, :right_active]
        left_zero = torch.stack((left[:, :, 0], torch.zeros_like(left[:, :, 1])), dim=2)
        left_one = torch.stack((torch.zeros_like(left[:, :, 0]), left[:, :, 1]), dim=2)
        left_out = torch.stack((left_zero, left_one), dim=4).reshape(
            left.shape[0], left_active, 2, new_center, 2
        )
        right_out = torch.stack((right, torch.flip(right, dims=(2,))), dim=2).reshape(
            right.shape[0], new_center, 2, right_active, 2
        )
        left_out = torch.nn.functional.pad(
            left_out,
            (
                0,
                0,
                0,
                center_capacity - new_center,
                0,
                0,
                0,
                self.bond_dims[left_wire] - left_active,
            ),
        )
        right_out = torch.nn.functional.pad(
            right_out,
            (
                0,
                0,
                0,
                self.bond_dims[left_wire + 2] - right_active,
                0,
                0,
                0,
                center_capacity - new_center,
            ),
        )
        return tensors[:left_wire] + (left_out, right_out) + tensors[left_wire + 2 :]

    @staticmethod
    def _step_real_imag(
        environment: torch.Tensor,
        tensor: torch.Tensor,
        insert_z: bool = False,
    ) -> torch.Tensor:
        ket = tensor
        if insert_z:
            signs = torch.tensor(
                [1.0, -1.0], device=tensor.device, dtype=tensor.dtype
            ).reshape(1, 1, 2, 1, 1)
            ket = tensor * signs
        partial = _ri_einsum("bij,bipr->bjpr", environment, _ri_conj(tensor))
        return _ri_einsum("bjpr,bjps->brs", partial, ket)

    def expectation_z_zz_chain_real_imag(
        self,
        tensors: RealImagTuple,
        *,
        z_weight: float = 0.1,
        zz_weight: float = -1.0,
    ) -> torch.Tensor:
        batch = int(tensors[0].shape[0])
        real = torch.ones(batch, 1, 1, device=tensors[0].device, dtype=tensors[0].dtype)
        boundary = torch.stack((real, torch.zeros_like(real)), dim=-1)
        left = [boundary]
        for tensor in tensors:
            left.append(self._step_real_imag(left[-1], tensor))
        right: list[torch.Tensor] = [boundary] * (self.n_wires + 1)
        for wire in range(self.n_wires - 1, -1, -1):
            partial = _ri_einsum(
                "bipr,brs->bips", _ri_conj(tensors[wire]), right[wire + 1]
            )
            right[wire] = _ri_einsum("bips,bjps->bij", partial, tensors[wire])
        energy = torch.zeros(batch, device=tensors[0].device, dtype=tensors[0].dtype)
        for wire, tensor in enumerate(tensors):
            inserted = self._step_real_imag(left[wire], tensor, True)
            value = _ri_einsum("bij,bij->b", inserted, right[wire + 1])
            energy = energy + float(z_weight) * value[:, 0]
            if wire + 1 < self.n_wires:
                inserted = self._step_real_imag(inserted, tensors[wire + 1], True)
                value = _ri_einsum("bij,bij->b", inserted, right[wire + 2])
                energy = energy + float(zz_weight) * value[:, 0]
        return energy

    @staticmethod
    def apply_one(tensors: TensorTuple, matrix: torch.Tensor, wire: int) -> TensorTuple:
        current = tensors[int(wire)]
        equation = "pq,blqr->blpr" if matrix.ndim == 2 else "bpq,blqr->blpr"
        updated = torch.einsum(equation, matrix, current)
        return tensors[:wire] + (updated,) + tensors[wire + 1 :]

    def apply_two(
        self, tensors: TensorTuple, matrix: torch.Tensor, left_wire: int
    ) -> TensorTuple:
        left_wire = int(left_wire)
        left, right = tensors[left_wire], tensors[left_wire + 1]
        batch, left_dim, _, _ = left.shape
        right_dim = int(right.shape[-1])
        theta = torch.einsum("blsm,bmtr->blstr", left, right).reshape(
            batch, left_dim, 4, right_dim
        )
        equation = "ij,bljr->blir" if matrix.ndim == 2 else "bij,bljr->blir"
        pair = torch.einsum(equation, matrix, theta).reshape(
            batch, left_dim * 2, 2 * right_dim
        )
        rank = self.bond_dims[left_wire + 1]
        rows, columns = left_dim * 2, 2 * right_dim
        full_rank = min(rows, columns)
        if rank >= full_rank:
            # An exact fixed-shape factorization avoids the undefined gradient of
            # singular vectors at the repeated zero singular values common in
            # product/low-entanglement states.
            if rows <= columns:
                left_matrix = torch.eye(
                    rows, rank, device=pair.device, dtype=pair.dtype
                ).expand(batch, -1, -1)
                right_matrix = torch.nn.functional.pad(pair, (0, 0, 0, rank - rows))
            else:
                left_matrix = torch.nn.functional.pad(pair, (0, rank - columns))
                right_matrix = torch.eye(
                    rank, columns, device=pair.device, dtype=pair.dtype
                ).expand(batch, -1, -1)
        else:
            u, singular_values, vh = torch.linalg.svd(pair, full_matrices=False)
            left_matrix = u[:, :, :rank]
            if pair.requires_grad:
                # PyTorch's complex SVD pullback is undefined for repeated
                # singular values. Keep the selected subspace fixed during the
                # pullback and differentiate its projection. This is stable but
                # is an approximate truncated-MPS gradient.
                left_matrix = left_matrix.detach()
                right_matrix = torch.conj(left_matrix).transpose(-2, -1) @ pair
            else:
                right_matrix = singular_values[:, :rank, None] * vh[:, :rank, :]
        left_out = left_matrix.reshape(batch, left_dim, 2, rank)
        right_out = right_matrix.reshape(batch, rank, 2, right_dim)
        return tensors[:left_wire] + (left_out, right_out) + tensors[left_wire + 2 :]

    @staticmethod
    def _step(
        environment: torch.Tensor,
        tensor: torch.Tensor,
        operator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ket = (
            tensor
            if operator is None
            else torch.einsum("pq,blqr->blpr", operator, tensor)
        )
        return torch.einsum("bij,bipr,bjps->brs", environment, torch.conj(tensor), ket)

    def expectation_z_zz_chain(
        self,
        tensors: TensorTuple,
        *,
        z_weight: float = 0.1,
        zz_weight: float = -1.0,
    ) -> torch.Tensor:
        batch = int(tensors[0].shape[0])
        boundary = torch.ones(
            batch, 1, 1, device=tensors[0].device, dtype=tensors[0].dtype
        )
        left = [boundary]
        for tensor in tensors:
            left.append(self._step(left[-1], tensor))
        right: list[torch.Tensor] = [boundary] * (self.n_wires + 1)
        for wire in range(self.n_wires - 1, -1, -1):
            tensor = tensors[wire]
            right[wire] = torch.einsum(
                "bipr,bjps,brs->bij", torch.conj(tensor), tensor, right[wire + 1]
            )
        z = torch.tensor(
            [[1, 0], [0, -1]], device=tensors[0].device, dtype=tensors[0].dtype
        )
        energy = torch.zeros(
            batch, device=tensors[0].device, dtype=tensors[0].real.dtype
        )
        for wire, tensor in enumerate(tensors):
            inserted = self._step(left[wire], tensor, z)
            energy = energy + float(z_weight) * torch.real(
                torch.einsum("bij,bij->b", inserted, right[wire + 1])
            )
            if wire + 1 < self.n_wires:
                inserted = self._step(inserted, tensors[wire + 1], z)
                energy = energy + float(zz_weight) * torch.real(
                    torch.einsum("bij,bij->b", inserted, right[wire + 2])
                )
        return energy

    def to_statevector(self, tensors: TensorTuple) -> torch.Tensor:
        merged = tensors[0]
        for tensor in tensors[1:]:
            merged = torch.einsum("blpm,bmqr->blpqr", merged, tensor).reshape(
                merged.shape[0], merged.shape[1], -1, tensor.shape[-1]
            )
        return merged[:, 0, :, 0]


def ry_matrix(angle: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
    return torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine)))).to(
        torch.complex64
    )


def rz_matrix(angle: torch.Tensor) -> torch.Tensor:
    phase = angle / 2
    return torch.diag(torch.stack((torch.exp(-1j * phase), torch.exp(1j * phase))))


CX = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=torch.complex64,
)


def run_static_brickwork(
    program: StaticMPSProgram, parameters: torch.Tensor
) -> TensorTuple:
    tensors = program.zero(device=parameters.device)
    layers = int(parameters.shape[0])
    for layer in range(layers):
        for wire in range(program.n_wires):
            tensors = program.apply_one(
                tensors, ry_matrix(parameters[layer, wire, 0]), wire
            )
            tensors = program.apply_one(
                tensors, rz_matrix(parameters[layer, wire, 1]), wire
            )
        for wire in range(layer % 2, program.n_wires - 1, 2):
            tensors = program.apply_two(tensors, CX.to(parameters.device), wire)
    return tensors


def run_static_brickwork_real_imag(
    program: StaticMPSProgram, parameters: torch.Tensor
) -> RealImagTuple:
    tensors = program.zero_real_imag(device=parameters.device, dtype=parameters.dtype)
    active = [1] * (program.n_wires + 1)
    layers = int(parameters.shape[0])
    for layer in range(layers):
        for wire in range(program.n_wires):
            tensors = program.apply_ry_real_imag(
                tensors, parameters[layer, wire, 0], wire
            )
            tensors = program.apply_rz_real_imag(
                tensors, parameters[layer, wire, 1], wire
            )
        for wire in range(layer % 2, program.n_wires - 1, 2):
            tensors = program.apply_cx_mpo_real_imag(
                tensors,
                wire,
                (active[wire], active[wire + 1], active[wire + 2]),
            )
            active[wire + 1] = min(active[wire + 1] * 2, program.bond_dims[wire + 1])
    return tensors


@lru_cache(maxsize=128)
def _compiled_site_bucket(
    left_capacity: int, right_capacity: int, backend: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def site(tensor: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        tensors = (tensor,)
        tensors = StaticMPSProgram.apply_ry_real_imag(tensors, angles[0], 0)
        return StaticMPSProgram.apply_rz_real_imag(tensors, angles[1], 0)[0]

    del left_capacity, right_capacity
    return torch.compile(site, backend=backend, dynamic=False, fullgraph=True)


@lru_cache(maxsize=256)
def _compiled_cx_bucket(
    capacities: tuple[int, int, int],
    active_dims: tuple[int, int, int],
    backend: str,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    program = StaticMPSProgram(2, max(capacities), capacities)

    def cx(
        left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated = program.apply_cx_mpo_real_imag((left, right), 0, active_dims)
        return updated[0], updated[1]

    return torch.compile(cx, backend=backend, dynamic=False, fullgraph=True)


@lru_cache(maxsize=256)
def _compiled_environment_bucket(
    environment_shape: tuple[int, ...],
    tensor_shape: tuple[int, ...],
    insert_z: bool,
    backend: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    del environment_shape, tensor_shape

    def step(environment: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return StaticMPSProgram._step_real_imag(environment, tensor, insert_z)

    return torch.compile(step, backend=backend, dynamic=False, fullgraph=True)


@lru_cache(maxsize=128)
def _compiled_right_environment_bucket(
    tensor_shape: tuple[int, ...],
    environment_shape: tuple[int, ...],
    backend: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    del tensor_shape, environment_shape

    def step(tensor: torch.Tensor, environment: torch.Tensor) -> torch.Tensor:
        partial = _ri_einsum("bipr,brs->bips", _ri_conj(tensor), environment)
        return _ri_einsum("bips,bjps->bij", partial, tensor)

    return torch.compile(step, backend=backend, dynamic=False, fullgraph=True)


@lru_cache(maxsize=128)
def _compiled_inner_bucket(
    shape: tuple[int, ...], backend: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    del shape

    def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return _ri_einsum("bij,bij->b", left, right)[:, 0]

    return torch.compile(inner, backend=backend, dynamic=False, fullgraph=True)


def run_bucketed_static_brickwork_real_imag(
    program: StaticMPSProgram,
    parameters: torch.Tensor,
    *,
    backend: str = "inductor",
) -> RealImagTuple:
    """Run a static circuit using small compiled kernels shared by shape bucket."""
    tensors = list(
        program.zero_real_imag(device=parameters.device, dtype=parameters.dtype)
    )
    active = [1] * (program.n_wires + 1)
    for layer in range(int(parameters.shape[0])):
        for wire in range(program.n_wires):
            bucket = _compiled_site_bucket(
                program.bond_dims[wire], program.bond_dims[wire + 1], backend
            )
            tensors[wire] = bucket(tensors[wire], parameters[layer, wire])
        for wire in range(layer % 2, program.n_wires - 1, 2):
            capacities = program.bond_dims[wire : wire + 3]
            active_dims = (active[wire], active[wire + 1], active[wire + 2])
            bucket = _compiled_cx_bucket(capacities, active_dims, backend)
            tensors[wire], tensors[wire + 1] = bucket(tensors[wire], tensors[wire + 1])
            active[wire + 1] = min(active[wire + 1] * 2, program.bond_dims[wire + 1])
    return tuple(tensors)


def bucketed_expectation_z_zz_chain_real_imag(
    program: StaticMPSProgram,
    tensors: RealImagTuple,
    *,
    backend: str = "inductor",
    z_weight: float = 0.1,
    zz_weight: float = -1.0,
) -> torch.Tensor:
    """Evaluate the shared environment sweep through compiled shape buckets."""
    batch = int(tensors[0].shape[0])
    real = torch.ones(batch, 1, 1, device=tensors[0].device, dtype=tensors[0].dtype)
    boundary = torch.stack((real, torch.zeros_like(real)), dim=-1)
    left = [boundary]
    for tensor in tensors:
        bucket = _compiled_environment_bucket(
            tuple(left[-1].shape), tuple(tensor.shape), False, backend
        )
        left.append(bucket(left[-1], tensor))
    right: list[torch.Tensor] = [boundary] * (program.n_wires + 1)
    for wire in range(program.n_wires - 1, -1, -1):
        bucket = _compiled_right_environment_bucket(
            tuple(tensors[wire].shape), tuple(right[wire + 1].shape), backend
        )
        right[wire] = bucket(tensors[wire], right[wire + 1])
    energy = torch.zeros(batch, device=tensors[0].device, dtype=tensors[0].dtype)
    for wire, tensor in enumerate(tensors):
        step = _compiled_environment_bucket(
            tuple(left[wire].shape), tuple(tensor.shape), True, backend
        )
        inserted = step(left[wire], tensor)
        inner = _compiled_inner_bucket(tuple(inserted.shape), backend)
        energy = energy + float(z_weight) * inner(inserted, right[wire + 1])
        if wire + 1 < program.n_wires:
            step = _compiled_environment_bucket(
                tuple(inserted.shape), tuple(tensors[wire + 1].shape), True, backend
            )
            inserted = step(inserted, tensors[wire + 1])
            inner = _compiled_inner_bucket(tuple(inserted.shape), backend)
            energy = energy + float(zz_weight) * inner(inserted, right[wire + 2])
    return energy


def build_bucketed_static_vqe_loss(
    program: StaticMPSProgram,
    layers: int,
    *,
    backend: str = "inductor",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a Python scheduler over reusable compiled shape buckets."""
    layers = int(layers)

    def loss(parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape != (layers, program.n_wires, 2):
            raise ValueError("parameters must have shape [layers, n_wires, 2]")
        with _bucket_recompile_context():
            tensors = run_bucketed_static_brickwork_real_imag(
                program, parameters, backend=backend
            )
            return bucketed_expectation_z_zz_chain_real_imag(
                program, tensors, backend=backend
            ).sum()

    return loss


def build_static_vqe_loss(
    program: StaticMPSProgram, layers: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a shape-specialized pure loss function for graph capture."""

    layers = int(layers)

    def loss(parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape != (layers, program.n_wires, 2):
            raise ValueError("parameters must have shape [layers, n_wires, 2]")
        tensors = run_static_brickwork(program, parameters)
        return program.expectation_z_zz_chain(tensors).sum()

    return loss


def build_static_vqe_loss_real_imag(
    program: StaticMPSProgram, layers: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a float-only, shape-specialized loss for Inductor capture."""

    layers = int(layers)

    def loss(parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape != (layers, program.n_wires, 2):
            raise ValueError("parameters must have shape [layers, n_wires, 2]")
        tensors = run_static_brickwork_real_imag(program, parameters)
        return program.expectation_z_zz_chain_real_imag(tensors).sum()

    return loss


@lru_cache(maxsize=32)
def compile_static_vqe_loss(
    program: StaticMPSProgram,
    layers: int,
    *,
    backend: str = "inductor",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile and cache a whole fixed-shape VQE forward/backward graph."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    return torch.compile(
        build_static_vqe_loss_real_imag(program, layers),
        backend=backend,
        dynamic=False,
        fullgraph=True,
    )


__all__ = [
    "StaticMPSProgram",
    "bucketed_expectation_z_zz_chain_real_imag",
    "build_bucketed_static_vqe_loss",
    "build_static_vqe_loss",
    "build_static_vqe_loss_real_imag",
    "compile_static_vqe_loss",
    "run_static_brickwork",
    "run_static_brickwork_real_imag",
    "run_bucketed_static_brickwork_real_imag",
]
