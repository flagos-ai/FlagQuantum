"""Experimental dense-island state representation.

Phase 1 supports immediate merge/split for gates spanning adjacent islands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..circuit import _apply_matrix


@dataclass(frozen=True)
class DenseIslandPlan:
    n_wires: int
    intervals: tuple[tuple[int, int], ...]
    max_dense_width: int
    max_bond: int
    compression_interval: int = 1

    @classmethod
    def equal_width(
        cls,
        n_wires: int,
        *,
        island_width: int,
        max_bond: int,
        compression_interval: int = 1,
    ) -> "DenseIslandPlan":
        if n_wires < 1 or island_width < 1 or max_bond < 1:
            raise ValueError("n_wires, island_width, and max_bond must be positive")
        intervals = tuple(
            (start, min(start + island_width, n_wires))
            for start in range(0, n_wires, island_width)
        )
        return cls(
            n_wires=int(n_wires),
            intervals=intervals,
            max_dense_width=int(island_width),
            max_bond=int(max_bond),
            compression_interval=int(compression_interval),
        )

    def island_for_wire(self, wire: int) -> int:
        for index, (start, stop) in enumerate(self.intervals):
            if start <= int(wire) < stop:
                return index
        raise IndexError(f"wire {wire} is outside the plan")


class DenseIslandState:
    """Chain of tensors shaped ``[batch,left,2**width,right]``."""

    def __init__(self, plan: DenseIslandPlan, tensors: Sequence[torch.Tensor]):
        if len(tensors) != len(plan.intervals):
            raise ValueError("one tensor is required per dense island")
        self.plan = plan
        self.tensors = list(tensors)
        self.local_gate_count = 0
        self.cross_island_gate_count = 0
        self.merge_count = 0
        self.split_count = 0
        self.truncation_errors: list[float] = []
        self.truncation_records: list[dict[str, object]] = []
        self.gradient_method = "exact_dense"

    @classmethod
    def zero(
        cls,
        plan: DenseIslandPlan,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> "DenseIslandState":
        tensors = []
        for start, stop in plan.intervals:
            tensor = torch.zeros(
                batch_size, 1, 2 ** (stop - start), 1, device=device, dtype=dtype
            )
            tensor[:, 0, 0, 0] = 1
            tensors.append(tensor)
        return cls(plan, tensors)

    @property
    def batch_size(self) -> int:
        return int(self.tensors[0].shape[0])

    def apply_local(self, matrix: torch.Tensor, wires: Sequence[int]) -> None:
        wires = tuple(int(wire) for wire in wires)
        islands = {self.plan.island_for_wire(wire) for wire in wires}
        if len(islands) != 1:
            self.cross_island_gate_count += 1
            self._apply_cross_island(matrix, wires, tuple(sorted(islands)))
            return
        island = islands.pop()
        start, stop = self.plan.intervals[island]
        tensor = self.tensors[island]
        batch, left_bond, physical, right_bond = tensor.shape
        width = stop - start
        local_wires = tuple(wire - start for wire in wires)
        packed = tensor.permute(0, 1, 3, 2).reshape(
            batch * left_bond * right_bond, physical
        )
        if matrix.ndim == 3:
            if int(matrix.shape[0]) != batch:
                raise ValueError("batched gate does not match island batch")
            matrix = (
                matrix[:, None, None]
                .expand(batch, left_bond, right_bond, -1, -1)
                .reshape(
                    batch * left_bond * right_bond, matrix.shape[-2], matrix.shape[-1]
                )
            )
        updated = _apply_matrix(packed, matrix, local_wires, width)
        self.tensors[island] = updated.reshape(
            batch, left_bond, right_bond, physical
        ).permute(0, 1, 3, 2)
        self.local_gate_count += 1

    def _apply_cross_island(
        self,
        matrix: torch.Tensor,
        wires: tuple[int, ...],
        islands: tuple[int, ...],
    ) -> None:
        if len(islands) != 2 or islands[1] != islands[0] + 1:
            raise NotImplementedError("only adjacent dense islands can be merged")
        left_index, right_index = islands
        left_start, left_stop = self.plan.intervals[left_index]
        right_start, right_stop = self.plan.intervals[right_index]
        if left_stop != right_start:
            raise ValueError("dense-island intervals are not contiguous")
        left = self.tensors[left_index]
        right = self.tensors[right_index]
        if int(left.shape[-1]) != int(right.shape[1]):
            raise ValueError("dense-island bond dimensions do not align")
        batch, outer_left, left_physical, _ = left.shape
        right_physical, outer_right = int(right.shape[2]), int(right.shape[3])
        merged = torch.einsum("blpm,bmqr->blpqr", left, right).reshape(
            batch, outer_left, left_physical * right_physical, outer_right
        )
        width = right_stop - left_start
        packed = merged.permute(0, 1, 3, 2).reshape(
            batch * outer_left * outer_right, left_physical * right_physical
        )
        if matrix.ndim == 3:
            if int(matrix.shape[0]) != batch:
                raise ValueError("batched gate does not match island batch")
            matrix = (
                matrix[:, None, None]
                .expand(batch, outer_left, outer_right, -1, -1)
                .reshape(
                    batch * outer_left * outer_right,
                    matrix.shape[-2],
                    matrix.shape[-1],
                )
            )
        updated = _apply_matrix(
            packed,
            matrix,
            tuple(wire - left_start for wire in wires),
            width,
        ).reshape(batch, outer_left, outer_right, left_physical, right_physical)
        updated = updated.permute(0, 1, 3, 4, 2)
        self.merge_count += 1
        self._split_adjacent(
            updated,
            left_index=left_index,
            left_physical=left_physical,
            right_physical=right_physical,
        )

    def _split_adjacent(
        self,
        merged: torch.Tensor,
        *,
        left_index: int,
        left_physical: int,
        right_physical: int,
    ) -> None:
        batch, outer_left, _, _, outer_right = merged.shape
        pair_matrix = merged.reshape(
            batch, outer_left * left_physical, right_physical * outer_right
        )
        rows, columns = int(pair_matrix.shape[-2]), int(pair_matrix.shape[-1])
        full_rank = min(rows, columns)
        rank = min(int(self.plan.max_bond), full_rank)
        discarded_weight = 0.0
        if rank == full_rank:
            if rows <= columns:
                eye = torch.eye(
                    rows, dtype=pair_matrix.dtype, device=pair_matrix.device
                ).expand(batch, rows, rows)
                left_out = eye.reshape(batch, outer_left, left_physical, rows)
                right_out = pair_matrix.reshape(
                    batch, rows, right_physical, outer_right
                )
            else:
                left_out = pair_matrix.reshape(
                    batch, outer_left, left_physical, columns
                )
                eye = torch.eye(
                    columns, dtype=pair_matrix.dtype, device=pair_matrix.device
                ).expand(batch, columns, columns)
                right_out = eye.reshape(batch, columns, right_physical, outer_right)
            method = "exact_autograd"
        else:
            left_parts = []
            right_parts = []
            for item in pair_matrix:
                u, singular_values, vh = torch.linalg.svd(item, full_matrices=False)
                discarded = singular_values[rank:]
                discarded_weight = max(
                    discarded_weight,
                    float(torch.sum(discarded.square()).detach().cpu()),
                )
                retained = u[:, :rank]
                if pair_matrix.requires_grad:
                    retained = retained.detach()
                    projected = torch.conj(retained).transpose(-2, -1) @ item
                    method = "retained_subspace_adjoint"
                else:
                    projected = singular_values[:rank, None] * vh[:rank]
                    method = "truncated_svd"
                left_parts.append(retained.reshape(outer_left, left_physical, rank))
                right_parts.append(projected.reshape(rank, right_physical, outer_right))
            left_out = torch.stack(left_parts)
            right_out = torch.stack(right_parts)
        self.tensors[left_index] = left_out
        self.tensors[left_index + 1] = right_out
        self.split_count += 1
        if method == "retained_subspace_adjoint":
            self.gradient_method = method
        elif self.gradient_method != "retained_subspace_adjoint":
            self.gradient_method = method
        if discarded_weight > 0:
            self.truncation_errors.append(discarded_weight)
        self.truncation_records.append(
            {
                "boundary": left_index,
                "kept_rank": rank,
                "original_rank": full_rank,
                "discarded_weight": discarded_weight,
                "method": method,
            }
        )

    def to_statevector(self) -> torch.Tensor:
        merged = self.tensors[0]
        for tensor in self.tensors[1:]:
            if int(merged.shape[-1]) != int(tensor.shape[1]):
                raise ValueError("dense-island bond dimensions do not align")
            merged = torch.einsum("blpr,brqs->blpqs", merged, tensor).reshape(
                merged.shape[0], merged.shape[1], -1, tensor.shape[-1]
            )
        if int(merged.shape[1]) != 1 or int(merged.shape[-1]) != 1:
            raise ValueError("statevector conversion requires closed boundary bonds")
        return merged[:, 0, :, 0]

    @staticmethod
    def _environment_step(
        environment: torch.Tensor,
        tensor: torch.Tensor,
        diagonal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ket = tensor if diagonal is None else tensor * diagonal[None, None, :, None]
        return torch.einsum("bij,bipr,bjps->brs", environment, torch.conj(tensor), ket)

    @staticmethod
    def _local_z_diagonal(
        width: int, local_wire: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        indices = torch.arange(2**width, device=device)
        bits = (indices >> (width - 1 - int(local_wire))) & 1
        return (1 - 2 * bits).to(dtype)

    def expectation_z_zz_chain(
        self,
        *,
        z_weights: Sequence[float] | torch.Tensor,
        zz_weights: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate weighted Z and nearest-neighbour ZZ in shared environments."""

        if len(z_weights) != self.plan.n_wires:
            raise ValueError("z_weights must contain one value per wire")
        if len(zz_weights) != self.plan.n_wires - 1:
            raise ValueError("zz_weights must contain one value per adjacent bond")
        device = self.tensors[0].device
        real_dtype = self.tensors[0].real.dtype
        z_weights = torch.as_tensor(z_weights, device=device, dtype=real_dtype)
        zz_weights = torch.as_tensor(zz_weights, device=device, dtype=real_dtype)
        boundary = torch.ones(
            self.batch_size, 1, 1, device=device, dtype=self.tensors[0].dtype
        )
        left_environments = [boundary]
        for tensor in self.tensors:
            left_environments.append(
                self._environment_step(left_environments[-1], tensor)
            )
        right_environments: list[torch.Tensor] = [boundary] * (len(self.tensors) + 1)
        right_environments[-1] = boundary
        for index in range(len(self.tensors) - 1, -1, -1):
            tensor = self.tensors[index]
            right_environments[index] = torch.einsum(
                "bipr,bjps,brs->bij",
                torch.conj(tensor),
                tensor,
                right_environments[index + 1],
            )

        energy = torch.zeros(self.batch_size, device=device, dtype=real_dtype)
        diagonals: list[list[torch.Tensor]] = []
        for island, ((start, stop), tensor) in enumerate(
            zip(self.plan.intervals, self.tensors)
        ):
            width = stop - start
            local_diagonals = [
                self._local_z_diagonal(width, wire, device=device, dtype=real_dtype)
                for wire in range(width)
            ]
            diagonals.append(local_diagonals)
            combined = torch.zeros(2**width, device=device, dtype=real_dtype)
            for local_wire, diagonal in enumerate(local_diagonals):
                combined = combined + z_weights[start + local_wire] * diagonal
            for local_wire in range(width - 1):
                combined = combined + zz_weights[start + local_wire] * (
                    local_diagonals[local_wire] * local_diagonals[local_wire + 1]
                )
            inserted = self._environment_step(
                left_environments[island], tensor, combined
            )
            energy = energy + torch.real(
                torch.einsum("bij,bij->b", inserted, right_environments[island + 1])
            )

        for island in range(len(self.tensors) - 1):
            boundary_wire = self.plan.intervals[island][1] - 1
            inserted = self._environment_step(
                left_environments[island],
                self.tensors[island],
                diagonals[island][-1],
            )
            inserted = self._environment_step(
                inserted,
                self.tensors[island + 1],
                diagonals[island + 1][0],
            )
            energy = energy + zz_weights[boundary_wire] * torch.real(
                torch.einsum("bij,bij->b", inserted, right_environments[island + 2])
            )
        return energy

    def summary(self) -> dict[str, object]:
        return {
            "state_mode": "dense_island_mps_experimental",
            "n_wires": self.plan.n_wires,
            "intervals": self.plan.intervals,
            "max_dense_width": self.plan.max_dense_width,
            "max_bond": self.plan.max_bond,
            "compression_interval": self.plan.compression_interval,
            "local_gate_count": self.local_gate_count,
            "cross_island_gate_count": self.cross_island_gate_count,
            "merge_count": self.merge_count,
            "split_count": self.split_count,
            "truncation_steps": len(self.truncation_errors),
            "truncation_error": float(sum(self.truncation_errors)),
            "gradient_method": self.gradient_method,
            "truncation_records": tuple(self.truncation_records),
            "hamiltonian_environment": "shared_block_z_zz",
        }


__all__ = ["DenseIslandPlan", "DenseIslandState"]
