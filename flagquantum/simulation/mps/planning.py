"""Adaptive bond planning and state diagnostics for MPS numerics."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Any

import torch

from .models import (
    MPSAdaptiveBondPlan,
    MPSBondProfile,
    MPSConfig,
    MPSLocalRefinementPlan,
    MPSTruncationRecord,
)


class MPSPlanningMixin(ABC):
    """State diagnostics and planning over a concrete MPS numerical implementation."""

    config: MPSConfig
    truncation_errors: list[float]
    truncation_records: list[MPSTruncationRecord]
    orthogonality_center: int | None
    local_swap_count: int
    triton_two_site_regions: int
    eager_two_site_regions: int
    svd_gradient_method: str
    fixed_rank_qr_regions: int
    spatial_two_site_bucket_count: int
    spatial_two_site_bucketed_gate_count: int

    @property
    @abstractmethod
    def n_wires(self) -> int: ...

    @property
    @abstractmethod
    def bsz(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def bond_dims(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def max_bond(self) -> int: ...

    @property
    @abstractmethod
    def parameter_count(self) -> int: ...

    @abstractmethod
    def state_norm(self) -> torch.Tensor: ...

    @abstractmethod
    def left_canonical_residual(self) -> float: ...

    @abstractmethod
    def right_canonical_residual(self) -> float: ...

    @abstractmethod
    def mixed_canonical_residual(self, center: int | None = None) -> float: ...

    @torch.no_grad()
    def bond_profile(self) -> MPSBondProfile:
        total_error = float(sum(self.truncation_errors))
        norms = self.state_norm().detach().cpu()
        truncation_by_bond = self.truncation_error_by_bond()
        bond_dims = self.bond_dims
        left_residual = self.left_canonical_residual()
        right_residual = self.right_canonical_residual()
        mixed_residual = (
            max(left_residual, right_residual)
            if self.orthogonality_center is None
            else self.mixed_canonical_residual()
        )
        return MPSBondProfile(
            n_wires=self.n_wires,
            batch_size=self.bsz,
            bond_dims=bond_dims,
            max_bond=self.max_bond,
            mean_bond=(float(sum(bond_dims) / len(bond_dims)) if bond_dims else 1.0),
            parameter_count=self.parameter_count,
            state_norm_min=float(torch.min(norms).item()),
            state_norm_max=float(torch.max(norms).item()),
            orthogonality_center=self.orthogonality_center,
            left_canonical_residual=left_residual,
            right_canonical_residual=right_residual,
            mixed_canonical_residual=mixed_residual,
            truncation_steps=len(self.truncation_errors),
            truncation_error=total_error,
            max_truncation_error=max(self.truncation_errors, default=0.0),
            truncation_by_bond=tuple(sorted(truncation_by_bond.items())),
            truncation_records=tuple(self.truncation_records),
            error_budget_satisfied=None,
        )

    def truncation_error_by_bond(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for record in self.truncation_records:
            out[record.bond] = out.get(record.bond, 0.0) + float(
                record.discarded_weight
            )
        return out

    def truncation_error_within_budget(self, budget: float) -> bool:
        return float(sum(self.truncation_errors)) <= float(budget)

    def adaptive_bond_plan(
        self,
        *,
        global_error_budget: float | None = None,
        growth_factor: float = 2.0,
        min_increment: int = 1,
    ) -> MPSAdaptiveBondPlan:
        if not math.isfinite(growth_factor):
            raise ValueError("growth_factor must be finite.")
        if growth_factor < 1:
            raise ValueError("growth_factor must be >= 1.")
        if type(min_increment) is not int:
            raise ValueError("min_increment must be an integer.")
        if min_increment < 0:
            raise ValueError("min_increment must be >= 0.")
        observed_error = float(sum(self.truncation_errors))
        by_bond = self.truncation_error_by_bond()
        if global_error_budget is None:
            threshold = 0.0
            budget_satisfied = observed_error == 0.0
        else:
            budget = float(global_error_budget)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError("global_error_budget must be finite and non-negative.")
            budget_satisfied = observed_error <= budget
            threshold = budget / max(1, len(by_bond)) if budget > 0 else 0.0

        hot_bonds = tuple(
            sorted(bond for bond, error in by_bond.items() if error > threshold)
        )
        suggestions: list[tuple[int, int]] = []
        current_limit = self.config.max_bond or self.max_bond
        bond_dims = self.bond_dims if hot_bonds else ()
        for bond in hot_bonds:
            current_rank = bond_dims[bond] if bond < len(bond_dims) else self.max_bond
            grown = max(
                current_rank + min_increment,
                math.ceil(current_rank * growth_factor),
            )
            suggestions.append((bond, max(grown, current_limit)))
        suggested_max_bond = max(
            (value for _, value in suggestions), default=current_limit
        )
        return MPSAdaptiveBondPlan(
            current_max_bond=int(current_limit),
            suggested_max_bond=int(suggested_max_bond),
            global_error_budget=global_error_budget,
            observed_error=observed_error,
            budget_satisfied=budget_satisfied,
            hot_bonds=hot_bonds,
            per_bond_suggestions=tuple(suggestions),
        )

    def local_refinement_plan(
        self,
        *,
        global_error_budget: float | None = None,
        growth_factor: float = 2.0,
        window_radius: int = 1,
    ) -> MPSLocalRefinementPlan:
        if type(window_radius) is not int:
            raise ValueError("window_radius must be an integer.")
        if window_radius < 0:
            raise ValueError("window_radius must be >= 0.")
        adaptive = self.adaptive_bond_plan(
            global_error_budget=global_error_budget,
            growth_factor=growth_factor,
        )
        windows = []
        for bond in adaptive.hot_bonds:
            left = max(0, bond - window_radius)
            right = min(self.n_wires - 1, bond + 1 + window_radius)
            windows.append((left, right))
        merged: list[tuple[int, int]] = []
        for left, right in sorted(windows):
            if not merged or left > merged[-1][1] + 1:
                merged.append((left, right))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        return MPSLocalRefinementPlan(
            windows=tuple(merged),
            hot_bonds=adaptive.hot_bonds,
            suggested_max_bond=adaptive.suggested_max_bond,
            observed_error=adaptive.observed_error,
        )

    def summary(self) -> dict[str, Any]:
        profile = self.bond_profile()
        refinement = self.local_refinement_plan()
        return {
            "state_mode": "mps",
            "n_wires": profile.n_wires,
            "batch_size": profile.batch_size,
            "bond_dims": profile.bond_dims,
            "max_bond": profile.max_bond,
            "mean_bond": profile.mean_bond,
            "parameter_count": profile.parameter_count,
            "state_norm_min": profile.state_norm_min,
            "state_norm_max": profile.state_norm_max,
            "orthogonality_center": profile.orthogonality_center,
            "left_canonical_residual": profile.left_canonical_residual,
            "right_canonical_residual": profile.right_canonical_residual,
            "mixed_canonical_residual": profile.mixed_canonical_residual,
            "local_swap_count": self.local_swap_count,
            "truncation_steps": profile.truncation_steps,
            "truncation_error": profile.truncation_error,
            "max_truncation_error": profile.max_truncation_error,
            "truncation_by_bond": profile.truncation_by_bond,
            "svd_gradient_method": self.svd_gradient_method,
            "adaptive_suggested_max_bond": refinement.suggested_max_bond,
            "adaptive_hot_bonds": refinement.hot_bonds,
            "local_refinement_windows": refinement.windows,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "triton_mps_two_site_enabled": os.getenv("FQ_TRITON_MPS_TWO_SITE", "0")
            .strip()
            .lower()
            not in {"0", "false", "off", "no"},
            "triton_mps_two_site_regions": self.triton_two_site_regions,
            "eager_mps_two_site_regions": self.eager_two_site_regions,
            "fixed_rank_qr_regions": self.fixed_rank_qr_regions,
            "spatial_two_site_bucket_count": self.spatial_two_site_bucket_count,
            "spatial_two_site_bucketed_gate_count": (
                self.spatial_two_site_bucketed_gate_count
            ),
        }
