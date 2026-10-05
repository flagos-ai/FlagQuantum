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
from .one_site_dispatch import _mps_one_site_rollout_enabled


class MPSPlanningMixin(ABC):
    """State diagnostics and planning over a concrete MPS numerical implementation."""

    config: MPSConfig
    truncation_errors: list[float]
    truncation_records: list[MPSTruncationRecord]
    orthogonality_center: int | None
    local_swap_count: int
    triton_one_site_regions: int
    triton_two_site_regions: int
    eager_two_site_regions: int
    svd_gradient_method: str
    fixed_rank_qr_regions: int
    spatial_two_site_bucket_count: int
    spatial_two_site_bucketed_gate_count: int

    @property
    @abstractmethod
    def n_qubits(self) -> int: ...

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
        # `max` over a sequence holding `nan` answers with whichever element it
        # reached first, so a measured bond recorded earlier could hide a later
        # unmeasured one and report a finite maximum. "The largest recorded
        # error" over a list where one entry was never measured is itself
        # unmeasured, so any non-finite entry makes the maximum non-finite.
        max_error = max(self.truncation_errors, default=0.0)
        if any(not math.isfinite(error) for error in self.truncation_errors):
            max_error = float("nan")
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
            n_qubits=self.n_qubits,
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
            max_truncation_error=max_error,
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
        """Report which bonds the recorded truncation error says to grow.

        The global error budget is divided evenly across the bonds that have error,
        and every bond above its share is named as hot. With no budget the
        threshold is zero, so only exact bonds are cold -- which is the honest
        reading of "no budget was given" rather than an implicit one being picked.

        A bond whose recorded discarded weight is not finite was never measured;
        the fixed-rank range QR route records `nan` because the exact Frobenius
        weight would need the matrix that route exists to avoid. `nan` compares
        false against every threshold, so such a bond would fall through the
        comparison and read as cold, which would report "do not grow" for a bond
        nothing is known about. Unmeasured bonds are therefore treated as hot and
        listed in `unmeasured_bonds`, and the plan cannot satisfy its budget while
        any of them remains.

        The plan is derived from errors already recorded by the run; it does not
        measure anything itself, so calling it before applying gates reports a plan
        over no evidence.

        Raises:
            ValueError: if `growth_factor` is not finite or below one, if
                `min_increment` is not a non-negative integer, or if the budget is
                not a finite non-negative number.
        """
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
        unmeasured_bonds = tuple(
            sorted(bond for bond, error in by_bond.items() if not math.isfinite(error))
        )
        measured_by_bond = {
            bond: error for bond, error in by_bond.items() if math.isfinite(error)
        }
        if global_error_budget is None:
            threshold = 0.0
            budget_satisfied = observed_error == 0.0 and not unmeasured_bonds
        else:
            budget = float(global_error_budget)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError("global_error_budget must be finite and non-negative.")
            budget_satisfied = (
                math.isfinite(observed_error)
                and observed_error <= budget
                and not unmeasured_bonds
            )
            threshold = budget / max(1, len(by_bond)) if budget > 0 else 0.0

        hot_bonds = tuple(
            sorted(
                [bond for bond, error in measured_by_bond.items() if error > threshold]
                + list(unmeasured_bonds)
            )
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
            unmeasured_bonds=unmeasured_bonds,
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
            right = min(self.n_qubits - 1, bond + 1 + window_radius)
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
            unmeasured_bonds=adaptive.unmeasured_bonds,
        )

    def summary(self) -> dict[str, Any]:
        profile = self.bond_profile()
        refinement = self.local_refinement_plan()
        return {
            "state_mode": "mps",
            "n_qubits": profile.n_qubits,
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
            "adaptive_unmeasured_bonds": refinement.unmeasured_bonds,
            "local_refinement_windows": refinement.windows,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "triton_mps_one_site_enabled": _mps_one_site_rollout_enabled(),
            "triton_mps_one_site_regions": self.triton_one_site_regions,
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
