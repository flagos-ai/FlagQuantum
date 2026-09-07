"""Native matrix-product-state execution for FlagQuantum."""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import torch

from ..core.ir import Instruction
from ..ops.gate_matrix import gate_matrix
from ..ops.matrices import GATE_MAT_DICT, get_global_precision
from .mps.models import (
    MPSConfig,
    MPSTruncationRecord,
)
from .mps_factorization import (
    _discarded_weight,
    _select_rank,
    _split_pair_matrix,
    _split_pair_matrix_bucket,
    _z_sum_dense_weights,
)
from .mps_planning_mixin import MPSPlanningMixin
from .real_imag_kernels import complex_einsum_pair
from .statevector_ops import _apply_matrix, _bits_from_indices


class MPSState(MPSPlanningMixin):
    """Batched MPS state with tensors shaped [batch, left, physical, right]."""

    def __init__(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        config: MPSConfig | None = None,
    ) -> None:
        if not tensors:
            raise ValueError("MPSState requires at least one tensor.")
        self.tensors = list(tensors)
        self.config = config or MPSConfig()
        self.local_swap_count = 0
        self.truncation_errors: list[float] = []
        self.truncation_records: list[MPSTruncationRecord] = []
        self.orthogonality_center: int | None = None
        self.triton_two_site_regions = 0
        self.eager_two_site_regions = 0
        self.svd_gradient_method = "not_used"
        self.fixed_rank_qr_regions = 0
        self.spatial_two_site_bucket_count = 0
        self.spatial_two_site_bucketed_gate_count = 0

    @property
    def n_wires(self) -> int:
        return len(self.tensors)

    @property
    def bsz(self) -> int:
        return int(self.tensors[0].shape[0])

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensors[0].dtype

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(int(tensor.shape[-1]) for tensor in self.tensors[:-1])

    @property
    def max_bond(self) -> int:
        return max(self.bond_dims, default=1)

    @property
    def parameter_count(self) -> int:
        return int(sum(tensor.numel() for tensor in self.tensors))

    def state_norm(self) -> torch.Tensor:
        return torch.real(self._expectation_product_ops({}))

    @classmethod
    def zero(
        cls,
        n_wires: int,
        *,
        bsz: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        config: MPSConfig | None = None,
    ) -> "MPSState":
        out_dtype = dtype or get_global_precision()
        tensors = []
        for _ in range(int(n_wires)):
            tensor = torch.zeros(bsz, 1, 2, 1, dtype=out_dtype, device=device)
            tensor[:, 0, 0, 0] = 1
            tensors.append(tensor)
        out = cls(tensors, config=config)
        out.orthogonality_center = 0
        return out

    @classmethod
    def from_statevector(
        cls,
        state: torch.Tensor,
        n_wires: int,
        *,
        config: MPSConfig | None = None,
    ) -> "MPSState":
        if state.ndim == 1:
            state = state.reshape(1, -1)
        cfg = config or MPSConfig()
        bsz = state.shape[0]
        rest = state.reshape(bsz, 1, 2 ** int(n_wires))
        tensors: list[torch.Tensor] = []
        truncation_errors: list[float] = []
        truncation_records: list[MPSTruncationRecord] = []
        left_dim = 1
        for wire in range(int(n_wires) - 1):
            rest = rest.reshape(bsz, left_dim * 2, -1)
            per_batch = []
            next_parts = []
            ranks = []
            svds = [torch.linalg.svd(rest[b], full_matrices=False) for b in range(bsz)]
            for u, s, vh in svds:
                rank = _select_rank(s, cfg.max_bond, cfg.cutoff)
                ranks.append(rank)
            rank = min(ranks) if cfg.cutoff > 0 else ranks[0]
            step_error = 0.0
            for u, s, vh in svds:
                step_error = max(step_error, _discarded_weight(s, rank))
                u = u[:, :rank]
                s = s[:rank]
                vh = vh[:rank, :]
                per_batch.append(u.reshape(left_dim, 2, rank))
                next_parts.append(s[:, None] * vh)
            if step_error > 0:
                truncation_errors.append(step_error)
            truncation_records.append(
                MPSTruncationRecord(
                    bond=wire,
                    kept_rank=int(rank),
                    original_rank=max(int(s.shape[0]) for _, s, _ in svds),
                    discarded_weight=float(step_error),
                    max_bond=cfg.max_bond,
                    cutoff=cfg.cutoff,
                    source="from_statevector",
                )
            )
            tensors.append(torch.stack(per_batch, dim=0))
            rest = torch.stack(next_parts, dim=0)
            left_dim = rank
        tensors.append(rest.reshape(bsz, left_dim, 2, 1))
        out = cls(tensors, config=cfg)
        out.truncation_errors.extend(truncation_errors)
        out.truncation_records.extend(truncation_records)
        out.orthogonality_center = int(n_wires) - 1
        return out

    def to_statevector(self) -> torch.Tensor:
        state = self.tensors[0][:, 0, :, :]
        for tensor in self.tensors[1:]:
            state = torch.einsum("b...l,blsr->b...sr", state, tensor)
        return state.reshape(self.bsz, -1)

    state = to_statevector

    def amplitudes(
        self,
        bitstrings: Sequence[int | str | Sequence[int]],
    ) -> torch.Tensor:
        """Contract selected amplitudes directly from the MPS."""

        normalized = []
        for bitstring in bitstrings:
            if isinstance(bitstring, int):
                if bitstring < 0 or bitstring >= 2**self.n_wires:
                    raise ValueError("integer bitstring is outside the state space")
                bits = tuple(int(bit) for bit in f"{bitstring:0{self.n_wires}b}")
            elif isinstance(bitstring, str):
                if len(bitstring) != self.n_wires or set(bitstring) - {"0", "1"}:
                    raise ValueError(
                        "bitstring must contain exactly n_wires binary digits"
                    )
                bits = tuple(int(bit) for bit in bitstring)
            else:
                bits = tuple(int(bit) for bit in bitstring)
                if len(bits) != self.n_wires or any(bit not in {0, 1} for bit in bits):
                    raise ValueError(
                        "bitstring must contain exactly n_wires binary values"
                    )
            normalized.append(bits)
        if not normalized:
            raise ValueError("bitstrings must contain at least one target")

        values = []
        for bits in normalized:
            environment = torch.ones(
                self.bsz,
                1,
                dtype=self.dtype,
                device=self.device,
            )
            for bit, tensor in zip(bits, self.tensors):
                environment = torch.einsum(
                    "bl,blr->br",
                    environment,
                    tensor[:, :, bit, :],
                )
            values.append(environment[:, 0])
        return torch.stack(values, dim=-1)

    def amplitude(self, bitstring: int | str | Sequence[int]) -> torch.Tensor:
        """Contract one selected amplitude directly from the MPS."""

        return self.amplitudes((bitstring,))[:, 0]

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.to_statevector()) ** 2

    def copy(self) -> "MPSState":
        out = type(self)(
            [tensor.clone() for tensor in self.tensors],
            config=self.config,
        )
        out.local_swap_count = self.local_swap_count
        out.truncation_errors = list(self.truncation_errors)
        out.truncation_records = list(self.truncation_records)
        out.orthogonality_center = self.orthogonality_center
        return out

    def canonicalize(self) -> "MPSState":
        rebuilt = type(self).from_statevector(
            self.to_statevector(),
            self.n_wires,
            config=self.config,
        )
        self.tensors = rebuilt.tensors
        self.truncation_errors.extend(rebuilt.truncation_errors)
        self.truncation_records.extend(rebuilt.truncation_records)
        self.orthogonality_center = rebuilt.orthogonality_center
        return self

    def orthogonalize_left(self) -> "MPSState":
        """Move the MPS into left-canonical form with local QR sweeps."""

        if self.bsz != 1:
            rebuilt = type(self).from_statevector(
                self.to_statevector(),
                self.n_wires,
                config=self.config,
            )
            self.tensors = rebuilt.tensors
            self.truncation_errors.extend(rebuilt.truncation_errors)
            self.truncation_records.extend(rebuilt.truncation_records)
            self.orthogonality_center = rebuilt.orthogonality_center
            return self
        for wire in range(self.n_wires - 1):
            tensor = self.tensors[wire]
            _, left_dim, physical_dim, right_dim = tensor.shape
            matrix = tensor[0].reshape(left_dim * physical_dim, right_dim)
            q, r = torch.linalg.qr(matrix, mode="reduced")
            new_right_dim = q.shape[-1]
            self.tensors[wire] = q.reshape(1, left_dim, physical_dim, new_right_dim)
            next_tensor = self.tensors[wire + 1]
            self.tensors[wire + 1] = torch.einsum("ab,cbsr->casr", r, next_tensor)
        self.orthogonality_center = self.n_wires - 1
        return self

    def orthogonalize_right(self) -> "MPSState":
        """Move the MPS into right-canonical form with local QR sweeps."""

        if self.bsz != 1:
            rebuilt = (
                type(self)
                .from_statevector(
                    self.to_statevector(),
                    self.n_wires,
                    config=self.config,
                )
                .move_orthogonality_center(0)
            )
            self.tensors = rebuilt.tensors
            self.truncation_errors.extend(rebuilt.truncation_errors)
            self.truncation_records.extend(rebuilt.truncation_records)
            self.orthogonality_center = rebuilt.orthogonality_center
            return self
        self._sweep_center_left(0)
        return self

    def move_orthogonality_center(self, site: int) -> "MPSState":
        """Move the mixed-canonical orthogonality center to ``site``."""

        site = int(site)
        if site < 0 or site >= self.n_wires:
            raise ValueError(
                f"Orthogonality center must be in [0, {self.n_wires - 1}], got {site}."
            )
        if self.bsz != 1:
            rebuilt = type(self).from_statevector(
                self.to_statevector(),
                self.n_wires,
                config=self.config,
            )
            self.tensors = rebuilt.tensors
            self.truncation_errors.extend(rebuilt.truncation_errors)
            self.truncation_records.extend(rebuilt.truncation_records)
            self.orthogonality_center = rebuilt.orthogonality_center
        if self.orthogonality_center is None or self.orthogonality_center < site:
            self.orthogonalize_left()
        self._sweep_center_left(site)
        return self

    def _sweep_center_left(self, target_site: int) -> None:
        for wire in range(self.n_wires - 1, int(target_site), -1):
            tensor = self.tensors[wire]
            _, left_dim, physical_dim, right_dim = tensor.shape
            matrix = tensor[0].reshape(left_dim, physical_dim * right_dim)
            q, r = torch.linalg.qr(matrix.transpose(-1, -2), mode="reduced")
            right_orthogonal = q.transpose(-1, -2)
            transfer = r.transpose(-1, -2)
            new_left_dim = right_orthogonal.shape[0]
            self.tensors[wire] = right_orthogonal.reshape(
                1, new_left_dim, physical_dim, right_dim
            )
            previous = self.tensors[wire - 1]
            self.tensors[wire - 1] = torch.einsum("blpa,ac->blpc", previous, transfer)
        self.orthogonality_center = int(target_site)

    def left_canonical_residual(self) -> float:
        residual = 0.0
        for tensor in self.tensors[:-1]:
            for batch_tensor in tensor:
                matrix = batch_tensor.reshape(-1, batch_tensor.shape[-1])
                gram = torch.conj(matrix).transpose(-1, -2) @ matrix
                eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
                residual = max(
                    residual,
                    float(torch.max(torch.abs(gram - eye)).detach().cpu()),
                )
        return residual

    def right_canonical_residual(self) -> float:
        residual = 0.0
        for tensor in self.tensors[1:]:
            for batch_tensor in tensor:
                matrix = batch_tensor.reshape(batch_tensor.shape[0], -1)
                gram = matrix @ torch.conj(matrix).transpose(-1, -2)
                eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
                residual = max(
                    residual,
                    float(torch.max(torch.abs(gram - eye)).detach().cpu()),
                )
        return residual

    def mixed_canonical_residual(self, center: int | None = None) -> float:
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return max(self.left_canonical_residual(), self.right_canonical_residual())
        center = int(center)
        residual = 0.0
        for tensor in self.tensors[:center]:
            for batch_tensor in tensor:
                matrix = batch_tensor.reshape(-1, batch_tensor.shape[-1])
                gram = torch.conj(matrix).transpose(-1, -2) @ matrix
                eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
                residual = max(
                    residual, float(torch.max(torch.abs(gram - eye)).detach().cpu())
                )
        for tensor in self.tensors[center + 1 :]:
            for batch_tensor in tensor:
                matrix = batch_tensor.reshape(batch_tensor.shape[0], -1)
                gram = matrix @ torch.conj(matrix).transpose(-1, -2)
                eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
                residual = max(
                    residual, float(torch.max(torch.abs(gram - eye)).detach().cpu())
                )
        return residual

    def expectation_z(self, wires: Iterable[int] | int | None = None) -> torch.Tensor:
        if wires is None:
            target_wires = tuple(range(self.n_wires))
        elif isinstance(wires, int):
            target_wires = (wires,)
        else:
            target_wires = tuple(int(wire) for wire in wires)
        if len(target_wires) > 1:
            return self._expectation_single_site_z(target_wires)
        z_op = GATE_MAT_DICT["z"].to(device=self.device, dtype=self.dtype)
        values = []
        for wire in target_wires:
            values.append(self._expectation_product_ops({wire: z_op}))
        return torch.stack(values, dim=-1)

    def expectation_z_sum(
        self, wires: Iterable[int] | int | None = None
    ) -> torch.Tensor:
        if wires is None:
            targets = tuple(range(self.n_wires))
        elif isinstance(wires, int):
            targets = (int(wires),)
        else:
            targets = tuple(int(wire) for wire in wires)
        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
        if not targets:
            return torch.zeros(self.bsz, dtype=real_dtype, device=self.device)
        if any(wire < 0 or wire >= self.n_wires for wire in targets):
            raise ValueError("expectation_z_sum wire index out of range.")
        if self.n_wires <= int(self.config.dense_observable_wires):
            return self._expectation_z_sum_dense(targets)
        target_counts: dict[int, int] = {}
        for wire in targets:
            target_counts[wire] = target_counts.get(wire, 0) + 1

        env = torch.ones(self.bsz, 1, 1, dtype=self.dtype, device=self.device)
        acc = torch.zeros_like(env)
        for wire, tensor in enumerate(self.tensors):
            next_acc = torch.einsum(
                "bij,bipr,bjps->brs",
                acc,
                torch.conj(tensor),
                tensor,
            )
            count = target_counts.get(wire, 0)
            if count:
                z_env = torch.einsum(
                    "bij,bir,bjs->brs",
                    env,
                    torch.conj(tensor[:, :, 0, :]),
                    tensor[:, :, 0, :],
                ) - torch.einsum(
                    "bij,bir,bjs->brs",
                    env,
                    torch.conj(tensor[:, :, 1, :]),
                    tensor[:, :, 1, :],
                )
                next_acc = next_acc + count * z_env
            env = torch.einsum(
                "bij,bipr,bjps->brs",
                env,
                torch.conj(tensor),
                tensor,
            )
            acc = next_acc
        return torch.real(acc[:, 0, 0])

    def expectation_z_zz_chain(
        self,
        *,
        z_coefficients: dict[int, Any],
        zz_coefficients: dict[int, Any],
    ) -> torch.Tensor:
        """Evaluate weighted ``Z_i`` and adjacent ``Z_i Z_{i+1}`` in one sweep."""

        env = torch.ones(self.bsz, 1, 1, dtype=self.dtype, device=self.device)
        pending_z = torch.zeros_like(env)
        total = torch.zeros_like(env)

        def transfer(source: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
            return torch.einsum(
                "bij,bipr,bjps->brs", source, torch.conj(tensor), tensor
            )

        def transfer_z(source: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
            return torch.einsum(
                "bij,bir,bjs->brs",
                source,
                torch.conj(tensor[:, :, 0, :]),
                tensor[:, :, 0, :],
            ) - torch.einsum(
                "bij,bir,bjs->brs",
                source,
                torch.conj(tensor[:, :, 1, :]),
                tensor[:, :, 1, :],
            )

        for wire, tensor in enumerate(self.tensors):
            next_total = transfer(total, tensor)
            if wire in z_coefficients:
                coefficient = torch.as_tensor(
                    z_coefficients[wire], dtype=self.dtype, device=self.device
                )
                next_total = next_total + coefficient * transfer_z(env, tensor)
            if wire - 1 in zz_coefficients:
                coefficient = torch.as_tensor(
                    zz_coefficients[wire - 1], dtype=self.dtype, device=self.device
                )
                next_total = next_total + coefficient * transfer_z(pending_z, tensor)
            pending_z = transfer_z(env, tensor)
            env = transfer(env, tensor)
            total = next_total
        return torch.real(total[:, 0, 0])

    def _expectation_z_sum_dense(self, wires: Sequence[int]) -> torch.Tensor:
        state = self.to_statevector()
        probs = torch.abs(state) ** 2
        weights = _z_sum_dense_weights(
            self.n_wires,
            tuple(int(wire) for wire in wires),
            device=self.device,
            dtype=probs.real.dtype,
        )
        return probs @ weights

    def _expectation_single_site_z(self, wires: Sequence[int]) -> torch.Tensor:
        targets = tuple(int(wire) for wire in wires)
        target_set = set(targets)
        if any(wire < 0 or wire >= self.n_wires for wire in targets):
            raise ValueError("expectation_z wire index out of range.")

        left_envs: list[torch.Tensor] = [
            torch.ones(self.bsz, 1, 1, dtype=self.dtype, device=self.device)
        ]
        for tensor in self.tensors:
            left_envs.append(
                torch.einsum(
                    "bij,bipr,bjps->brs",
                    left_envs[-1],
                    torch.conj(tensor),
                    tensor,
                )
            )

        right_envs: list[torch.Tensor | None] = [None] * (self.n_wires + 1)
        right_envs[self.n_wires] = torch.ones(
            self.bsz, 1, 1, dtype=self.dtype, device=self.device
        )
        for wire in range(self.n_wires - 1, -1, -1):
            tensor = self.tensors[wire]
            right_envs[wire] = torch.einsum(
                "bipr,bmps,brs->bim",
                torch.conj(tensor),
                tensor,
                right_envs[wire + 1],
            )

        values: dict[int, torch.Tensor] = {}
        for wire in sorted(target_set):
            tensor = self.tensors[wire]
            right_env = right_envs[wire + 1]
            assert right_env is not None
            values[wire] = torch.real(
                torch.einsum(
                    "bij,bir,bjs,brs->b",
                    left_envs[wire],
                    torch.conj(tensor[:, :, 0, :]),
                    tensor[:, :, 0, :],
                    right_env,
                )
                - torch.einsum(
                    "bij,bir,bjs,brs->b",
                    left_envs[wire],
                    torch.conj(tensor[:, :, 1, :]),
                    tensor[:, :, 1, :],
                    right_env,
                )
            )
        return torch.stack([values[wire] for wire in targets], dim=-1)

    def expectation_z_and_nearest_neighbor_zz(
        self, wires: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate many local Z and adjacent ZZ terms in one environment sweep.

        This avoids a complete MPS contraction for every ZZ term. It is
        particularly important for batched spacetime-observation workloads.
        """

        targets = tuple(int(wire) for wire in wires)
        if any(wire < 0 or wire >= self.n_wires for wire in targets):
            raise ValueError("observable wire index out of range")
        left_envs = [torch.ones(self.bsz, 1, 1, dtype=self.dtype, device=self.device)]
        for tensor in self.tensors:
            left_envs.append(
                torch.einsum(
                    "bij,bipr,bjps->brs",
                    left_envs[-1],
                    torch.conj(tensor),
                    tensor,
                )
            )
        right_envs: list[torch.Tensor | None] = [None] * (self.n_wires + 1)
        right_envs[-1] = torch.ones(
            self.bsz, 1, 1, dtype=self.dtype, device=self.device
        )
        for wire in range(self.n_wires - 1, -1, -1):
            tensor = self.tensors[wire]
            right_env = right_envs[wire + 1]
            assert right_env is not None
            right_envs[wire] = torch.einsum(
                "bipr,bmps,brs->bim",
                torch.conj(tensor),
                tensor,
                right_env,
            )
        z_op = GATE_MAT_DICT["z"].to(device=self.device, dtype=self.dtype)
        z_values = []
        zz_values = []
        for wire in targets:
            tensor = self.tensors[wire]
            right_env = right_envs[wire + 1]
            assert right_env is not None
            z_values.append(
                torch.real(
                    torch.einsum(
                        "bij,bipr,pq,bjqs,brs->b",
                        left_envs[wire],
                        torch.conj(tensor),
                        z_op,
                        tensor,
                        right_env,
                    )
                )
            )
            if wire + 1 < self.n_wires:
                next_tensor = self.tensors[wire + 1]
                after_pair = right_envs[wire + 2]
                assert after_pair is not None
                inserted = torch.einsum(
                    "bij,bipr,pq,bjqs->brs",
                    left_envs[wire],
                    torch.conj(tensor),
                    z_op,
                    tensor,
                )
                zz_values.append(
                    torch.real(
                        torch.einsum(
                            "bij,bipr,pq,bjqs,brs->b",
                            inserted,
                            torch.conj(next_tensor),
                            z_op,
                            next_tensor,
                            after_pair,
                        )
                    )
                )
        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
        z = (
            torch.stack(z_values, dim=-1)
            if z_values
            else torch.empty(self.bsz, 0, dtype=real_dtype, device=self.device)
        )
        zz = (
            torch.stack(zz_values, dim=-1)
            if zz_values
            else torch.empty(self.bsz, 0, dtype=real_dtype, device=self.device)
        )
        return z, zz

    def _expectation_product_ops(self, ops: dict[int, torch.Tensor]) -> torch.Tensor:
        env = torch.ones(
            self.bsz,
            1,
            1,
            dtype=self.dtype,
            device=self.device,
        )
        identity = GATE_MAT_DICT["i"].to(device=self.device, dtype=self.dtype)
        for wire, tensor in enumerate(self.tensors):
            op = ops.get(wire, identity)
            env = torch.einsum(
                "bij,bipr,pq,bjqs->brs",
                env,
                torch.conj(tensor),
                op,
                tensor,
            )
        return torch.real(env[:, 0, 0])

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

        ops: dict[int, torch.Tensor] = {}
        for wire in x or ():
            ops[int(wire)] = GATE_MAT_DICT["x"].to(device=self.device, dtype=self.dtype)
        for wire in y or ():
            ops[int(wire)] = GATE_MAT_DICT["y"].to(device=self.device, dtype=self.dtype)
        for wire in z or ():
            ops[int(wire)] = GATE_MAT_DICT["z"].to(device=self.device, dtype=self.dtype)
        return self._expectation_product_ops(ops)

    def sample(
        self,
        shots: int = 1,
        *,
        generator: torch.Generator | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        samples = self._sample_indices(shots, generator=generator)
        if format == "index":
            return samples
        if format != "bits":
            raise ValueError("sample format must be 'bits' or 'index'.")
        return _bits_from_indices(samples, self.n_wires)

    def _sample_indices(
        self,
        shots: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        outputs = torch.zeros(self.bsz, shots, dtype=torch.int64, device=self.device)
        for shot in range(int(shots)):
            work = self.copy()
            for wire in range(self.n_wires):
                probs = work._wire_probabilities(wire)
                bit = torch.multinomial(
                    probs,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(-1)
                outputs[:, shot] = (outputs[:, shot] << 1) | bit
                work._project_wire(wire, bit)
                work._normalize()
        return outputs

    def _wire_probabilities(self, wire: int) -> torch.Tensor:
        z_value = self._expectation_product_ops(
            {int(wire): GATE_MAT_DICT["z"].to(device=self.device, dtype=self.dtype)}
        )
        probs = torch.stack(((1 + z_value) / 2, (1 - z_value) / 2), dim=-1)
        probs = torch.clamp(probs, min=0)
        norm = probs.sum(dim=-1, keepdim=True)
        return probs / torch.clamp(norm, min=1e-12)

    def _project_wire(self, wire: int, bits: torch.Tensor) -> None:
        tensor = self.tensors[int(wire)].clone()
        mask = torch.zeros(self.bsz, 2, dtype=self.dtype, device=self.device)
        mask.scatter_(1, bits.reshape(-1, 1), 1)
        self.tensors[int(wire)] = tensor * mask[:, None, :, None]

    def _normalize(self) -> None:
        norm = torch.sqrt(torch.clamp(self._expectation_product_ops({}), min=1e-24))
        self.tensors[0] = self.tensors[0] / norm.reshape(-1, 1, 1, 1).to(self.dtype)

    def counts(
        self,
        shots: int,
        *,
        generator: torch.Generator | None = None,
        format: str = "bin",
    ) -> list[dict[str | int, int]]:
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

    def apply_instruction(
        self,
        instruction: Instruction,
        parameter_bindings: tuple[torch.Tensor, ...] | None = None,
    ) -> "MPSState":
        if instruction.metadata.get("is_channel"):
            self.apply_channel_trajectory(instruction.matrix, instruction.wires)
            return self
        wires = tuple(instruction.wires)
        slots = getattr(instruction, "parameter_slots", ())
        direct_theta = (
            parameter_bindings[slots[0]]
            if parameter_bindings is not None and slots and slots[0] >= 0
            else None
        )
        if (
            len(wires) == 1
            and instruction.matrix is None
            and instruction.name in {"rx", "ry", "rz"}
            and (direct_theta is not None or "theta" in instruction.params)
        ):
            self.apply_parametric_one(
                instruction.name,
                (
                    direct_theta
                    if direct_theta is not None
                    else instruction.params["theta"]
                ),
                wires[0],
            )
            return self
        matrix = gate_matrix(
            instruction,
            bsz=self.bsz,
            device=self.device,
            dtype=self.dtype,
            parameter_bindings=parameter_bindings,
        )
        if len(wires) == 1:
            self.apply_one(matrix, wires[0])
        elif len(wires) == 2 and abs(wires[0] - wires[1]) == 1:
            self.apply_two(matrix, min(wires), reverse=wires[0] > wires[1])
        elif len(wires) == 2:
            self.apply_two_remote(matrix, wires)
        else:
            dense = _apply_matrix(self.to_statevector(), matrix, wires, self.n_wires)
            rebuilt = type(self).from_statevector(
                dense,
                self.n_wires,
                config=self.config,
            )
            self.tensors = rebuilt.tensors
            self.truncation_errors.extend(rebuilt.truncation_errors)
            self.truncation_records.extend(rebuilt.truncation_records)
            self.orthogonality_center = rebuilt.orthogonality_center
        return self

    def apply_one(self, matrix: torch.Tensor, wire: int) -> None:
        tensor = self.tensors[int(wire)]
        equation = "pq,blqr->blpr" if matrix.ndim == 2 else "bpq,blqr->blpr"
        self.tensors[int(wire)] = complex_einsum_pair(equation, matrix, tensor)

    def apply_parametric_one(self, name: str, theta: Any, wire: int) -> None:
        tensor = self.tensors[int(wire)]
        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
        angle = torch.as_tensor(theta, dtype=real_dtype, device=self.device)
        if angle.ndim == 0:
            angle = angle.reshape(1)
        angle = angle.reshape(-1)
        if angle.numel() == 1 and self.bsz > 1:
            angle = angle.expand(self.bsz)
        if angle.numel() != self.bsz:
            raise ValueError(
                "Parameterized MPS gate batch dimension must match MPS batch size."
            )

        view_shape = (self.bsz, 1, 1)
        if name == "rz":
            phase0 = torch.exp((-0.5j * angle).to(self.dtype)).reshape(view_shape)
            phase1 = torch.exp((0.5j * angle).to(self.dtype)).reshape(view_shape)
            out0 = tensor[:, :, 0, :] * phase0
            out1 = tensor[:, :, 1, :] * phase1
        else:
            half = angle / 2
            cos = torch.cos(half).to(self.dtype).reshape(view_shape)
            sin = torch.sin(half).to(self.dtype).reshape(view_shape)
            zero = tensor[:, :, 0, :]
            one = tensor[:, :, 1, :]
            if name == "rx":
                factor = torch.as_tensor(-1j, dtype=self.dtype, device=self.device)
                out0 = cos * zero + factor * sin * one
                out1 = factor * sin * zero + cos * one
            elif name == "ry":
                out0 = cos * zero - sin * one
                out1 = sin * zero + cos * one
            else:
                raise ValueError(f"Unsupported parametric one-qubit gate {name!r}.")
        self.tensors[int(wire)] = torch.stack((out0, out1), dim=2)

    def apply_channel_trajectory(
        self,
        kraus_ops: Sequence[torch.Tensor],
        wires: Sequence[int],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        wires = tuple(int(wire) for wire in wires)
        if len(wires) != 1:
            dense = self.to_statevector()
            ops = [op.to(device=self.device, dtype=self.dtype) for op in kraus_ops]
            branches = tuple(
                _apply_matrix(dense, op, wires, self.n_wires) for op in ops
            )
            probabilities = torch.stack(
                [torch.sum(torch.abs(branch) ** 2, dim=-1) for branch in branches],
                dim=-1,
            )
            probabilities = torch.clamp(probabilities, min=0)
            totals = probabilities.sum(dim=-1, keepdim=True)
            if bool(torch.any(totals <= 1e-12)):
                raise RuntimeError("Kraus branch probabilities have zero total weight")
            choices = torch.multinomial(
                probabilities / totals,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            state = torch.stack(
                [branches[int(choices[b].item())][b] for b in range(self.bsz)],
                dim=0,
            )
            state = state / torch.clamp(
                torch.linalg.vector_norm(state, dim=-1, keepdim=True), min=1e-12
            )
            rebuilt = type(self).from_statevector(
                state,
                self.n_wires,
                config=self.config,
            )
            self.tensors = rebuilt.tensors
            self.truncation_errors.extend(rebuilt.truncation_errors)
            self.truncation_records.extend(rebuilt.truncation_records)
            self.orthogonality_center = rebuilt.orthogonality_center
            return

        wire = wires[0]
        ops = [op.to(device=self.device, dtype=self.dtype) for op in kraus_ops]
        branch_probs = []
        for op in ops:
            branch_probs.append(
                torch.clamp(self._local_effect_probability(op, wire), min=0)
            )
        probs = torch.stack(branch_probs, dim=-1)
        probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
        choices = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        for batch in range(self.bsz):
            op = ops[int(choices[batch].item())]
            tensor = self.tensors[wire][batch : batch + 1]
            self.tensors[wire][batch : batch + 1] = torch.einsum(
                "pq,blqr->blpr",
                op,
                tensor,
            )
        self._normalize()

    def _local_effect_probability(self, op: torch.Tensor, wire: int) -> torch.Tensor:
        effect = torch.conj(op).transpose(-1, -2) @ op
        return self._expectation_product_ops({int(wire): effect})

    def apply_two(
        self, matrix: torch.Tensor, left_wire: int, *, reverse: bool = False
    ) -> None:
        left = self.tensors[int(left_wire)]
        right = self.tensors[int(left_wire) + 1]
        full_rank = min(int(left.shape[1]) * 2, int(right.shape[3]) * 2)
        fixed_rank = self.config.max_bond
        fixed_rank_enabled = os.getenv(
            "FQ_MPS_FIXED_RANK_QR", "0"
        ).strip().lower() not in {"0", "false", "off", "no"}
        if (
            fixed_rank_enabled
            and fixed_rank is not None
            and int(fixed_rank) < full_rank
            and not reverse
        ):
            from .mps_low_rank import fixed_rank_two_site_range_qr

            left_out, right_out = fixed_rank_two_site_range_qr(
                left, matrix, right, int(fixed_rank)
            )
            self.tensors[int(left_wire)] = left_out
            self.tensors[int(left_wire) + 1] = right_out
            self.orthogonality_center = int(left_wire) + 1
            self.fixed_rank_qr_regions += 1
            self.svd_gradient_method = "fixed_rank_range_qr"
            # Computing the exact discarded Frobenius weight would require
            # materializing the full matrix this route is designed to avoid.
            self.truncation_errors.append(float("nan"))
            self.truncation_records.append(
                MPSTruncationRecord(
                    bond=int(left_wire),
                    kept_rank=int(fixed_rank),
                    original_rank=full_rank,
                    discarded_weight=float("nan"),
                    max_bond=self.config.max_bond,
                    cutoff=self.config.cutoff,
                    source="fixed_rank_range_qr_unmeasured",
                )
            )
            return
        contraction_volume = (
            self.bsz * int(left.shape[1]) * int(left.shape[3]) * int(right.shape[3])
        )
        requires_grad = (
            left.requires_grad or matrix.requires_grad or right.requires_grad
        )
        fused_threshold = 2**18 if requires_grad else 2**12
        triton_enabled = os.getenv(
            "FQ_TRITON_MPS_TWO_SITE", "0"
        ).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if (
            triton_enabled
            and left.is_cuda
            and left.dtype == torch.complex64
            and not reverse
            and contraction_volume >= fused_threshold
        ):
            from .triton_kernels.mps_two_site import fused_mps_two_site

            fused = fused_mps_two_site(left, matrix, right)
            self._split_pair(
                fused.reshape(self.bsz, left.shape[1], 2, 2, right.shape[3]),
                int(left_wire),
            )
            self.triton_two_site_regions += 1
            return
        self.eager_two_site_regions += 1
        theta = torch.einsum("blsm,bmtr->blstr", left, right)
        if reverse:
            theta = theta.transpose(2, 3)
        theta = theta.reshape(self.bsz, left.shape[1], 4, right.shape[3])
        if matrix.ndim == 2:
            theta = torch.einsum("ij,bljr->blir", matrix, theta)
        else:
            theta = torch.einsum("bij,bljr->blir", matrix, theta)
        theta = theta.reshape(self.bsz, left.shape[1], 2, 2, right.shape[3])
        if reverse:
            theta = theta.transpose(2, 3)
        self._split_pair(theta, int(left_wire))

    def apply_two_bucket(
        self, matrices: Sequence[torch.Tensor], left_wires: Sequence[int]
    ) -> None:
        """Apply disjoint adjacent gates in equal-shape spatial buckets."""

        buckets: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for position, wire in enumerate(left_wires):
            left = self.tensors[int(wire)]
            right = self.tensors[int(wire) + 1]
            key = (tuple(left.shape), tuple(right.shape))
            buckets.setdefault(key, []).append(position)
        for positions in buckets.values():
            if len(positions) < 2:
                position = positions[0]
                self.apply_two(matrices[position], int(left_wires[position]))
                continue
            self.spatial_two_site_bucket_count += 1
            self.spatial_two_site_bucketed_gate_count += len(positions)
            wires = [int(left_wires[position]) for position in positions]
            left = torch.stack([self.tensors[wire] for wire in wires])
            right = torch.stack([self.tensors[wire + 1] for wire in wires])
            gates = torch.stack(
                [
                    (
                        matrices[position].expand(self.bsz, -1, -1)
                        if matrices[position].ndim == 2
                        else matrices[position]
                    )
                    for position in positions
                ]
            )
            bond_count, batch, left_dim, _, middle_dim = left.shape
            right_dim = int(right.shape[-1])
            flat_left = left.reshape(bond_count * batch, left_dim, 2, middle_dim)
            flat_right = right.reshape(bond_count * batch, middle_dim, 2, right_dim)
            flat_gates = gates.reshape(bond_count * batch, 4, 4)
            volume = bond_count * batch * left_dim * middle_dim * right_dim
            use_triton = (
                os.getenv("FQ_TRITON_MPS_TWO_SITE", "0").strip().lower()
                not in {"0", "false", "off", "no"}
                and flat_left.is_cuda
                and flat_left.dtype == torch.complex64
                and volume >= 2**18
            )
            if use_triton:
                from .triton_kernels.mps_two_site import fused_mps_two_site

                matrix = fused_mps_two_site(flat_left, flat_gates, flat_right)
                self.triton_two_site_regions += bond_count
            else:
                theta = torch.einsum("kblsm,kbmtr->kblstr", left, right).reshape(
                    bond_count, batch, left_dim, 4, right_dim
                )
                matrix = torch.einsum("kbij,kbljr->kblir", gates, theta).reshape(
                    bond_count * batch, left_dim * 2, 2 * right_dim
                )
                self.eager_two_site_regions += bond_count
            split = _split_pair_matrix_bucket(
                matrix.reshape(bond_count, batch, left_dim * 2, 2 * right_dim),
                left_dim=left_dim,
                right_dim=right_dim,
                config=self.config,
            )
            for wire, (left_out, right_out, split_info) in zip(wires, split):
                self.tensors[wire] = left_out
                self.tensors[wire + 1] = right_out
                self.orthogonality_center = wire + 1
                if split_info["method"] == "svd":
                    self.svd_gradient_method = str(
                        split_info.get("gradient_method", "exact")
                    )
                    self.truncation_records.append(
                        MPSTruncationRecord(
                            bond=wire,
                            kept_rank=int(split_info["rank"]),
                            original_rank=int(split_info["original_rank"]),
                            discarded_weight=float(split_info["discarded_weight"]),
                            max_bond=self.config.max_bond,
                            cutoff=self.config.cutoff,
                            source="two_site_bucket",
                        )
                    )
                if split_info["discarded_weight"] > 0:
                    self.truncation_errors.append(float(split_info["discarded_weight"]))

    def apply_two_remote(self, matrix: torch.Tensor, wires: Sequence[int]) -> None:
        first, second = int(wires[0]), int(wires[1])
        if first == second:
            raise ValueError("A two-wire gate requires distinct wires.")
        left = min(first, second)
        right = max(first, second)
        for wire in range(right - 1, left, -1):
            self.apply_swap(wire)
        self.apply_two(matrix, left, reverse=first > second)
        for wire in range(left + 1, right):
            self.apply_swap(wire)

    def apply_swap(self, left_wire: int) -> None:
        self.apply_two(
            GATE_MAT_DICT["swap"].to(device=self.device, dtype=self.dtype), left_wire
        )
        self.local_swap_count += 1

    def _split_pair(self, theta: torch.Tensor, left_wire: int) -> None:
        bsz, left_dim, _, _, right_dim = theta.shape
        matrix = theta.reshape(bsz, left_dim * 2, 2 * right_dim)
        left_tensor, right_tensor, split_info = _split_pair_matrix(
            matrix,
            left_dim=left_dim,
            right_dim=right_dim,
            config=self.config,
        )
        self.tensors[left_wire] = left_tensor
        self.tensors[left_wire + 1] = right_tensor
        self.orthogonality_center = int(left_wire) + 1
        if split_info["method"] == "svd":
            self.svd_gradient_method = str(split_info.get("gradient_method", "exact"))
            self.truncation_records.append(
                MPSTruncationRecord(
                    bond=int(left_wire),
                    kept_rank=int(split_info["rank"]),
                    original_rank=int(split_info["original_rank"]),
                    discarded_weight=float(split_info["discarded_weight"]),
                    max_bond=self.config.max_bond,
                    cutoff=self.config.cutoff,
                    source="two_site",
                )
            )
        if split_info["discarded_weight"] > 0:
            self.truncation_errors.append(float(split_info["discarded_weight"]))
