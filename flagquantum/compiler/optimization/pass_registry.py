"""Built-in optimization pass bindings and analysis declarations."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ...core.ir import CircuitIR
from .canonicalization import (
    _ROTATION_PARAM,
    _SELF_INVERSE,
    merge_adjacent_rotations,
    merge_self_inverse,
    remove_identity_gates,
)
from .commutation import CommutationAnalysis, analyze_commutation
from .pass_manager import AnalysisSpec, PassFunction, PassRegistry, PassSpec
from .zero_state_reset import remove_zero_state_resets


def _self_inverse_opcodes(_program: CircuitIR) -> frozenset[str]:
    """Every opcode whose declared adjoint is itself, as the pipeline uses the term."""

    return _SELF_INVERSE


def _rotation_parameters(_program: CircuitIR) -> Mapping[str, str]:
    """Every opcode whose single angle parameter its adjoint negates, to that name."""

    return _ROTATION_PARAM


def _commutation_blocks(program: CircuitIR) -> CommutationAnalysis:
    """The commuting blocks of ``program``, one ordered partition per qubit."""

    return analyze_commutation(program)


_BUILTIN_ANALYSES: Mapping[str, AnalysisSpec] = MappingProxyType(
    {
        "self_inverse_opcodes": AnalysisSpec(
            _self_inverse_opcodes, program_independent=True
        ),
        "rotation_parameters": AnalysisSpec(
            _rotation_parameters, program_independent=True
        ),
        # A function of the program, so it is recomputed from the program a round
        # actually produced and no pass may claim to preserve it.
        "commutation_blocks": AnalysisSpec(_commutation_blocks),
    }
)


def _remove_diagonal_gates_before_measure(ir: CircuitIR) -> CircuitIR:
    from .diagonal_before_measure import (
        remove_diagonal_gates_before_measure,
    )

    return remove_diagonal_gates_before_measure(ir)


def _merge_inverse_pairs(ir: CircuitIR) -> CircuitIR:
    from .inverse_cancellation import merge_inverse_pairs

    return merge_inverse_pairs(ir)


def _cancel_commuting_self_inverse(
    ir: CircuitIR, analyses: Mapping[str, object] | None = None
) -> CircuitIR:
    from .commutation_cancellation import cancel_commuting_self_inverse

    return cancel_commuting_self_inverse(ir, analyses)


def _merge_commuting_rotations(ir: CircuitIR) -> CircuitIR:
    from .commutation_cancellation import merge_commuting_rotations

    return merge_commuting_rotations(ir)


def _collapse_one_qubit_runs(ir: CircuitIR) -> CircuitIR:
    from .one_qubit_optimization import collapse_one_qubit_runs

    return collapse_one_qubit_runs(ir)


def _collapse_two_qubit_blocks(ir: CircuitIR) -> CircuitIR:
    from .two_qubit_optimization import collapse_two_qubit_blocks

    return collapse_two_qubit_blocks(ir)


BUILTIN_PASSES: Mapping[str, PassFunction] = MappingProxyType(
    {
        "remove_zero_state_resets": remove_zero_state_resets,
        "remove_diagonal_gates_before_measure": (_remove_diagonal_gates_before_measure),
        "remove_identity_gates": remove_identity_gates,
        "merge_self_inverse": merge_self_inverse,
        "merge_inverse_pairs": _merge_inverse_pairs,
        "merge_adjacent_rotations": merge_adjacent_rotations,
        "merge_commuting_rotations": _merge_commuting_rotations,
        "cancel_commuting_self_inverse": _cancel_commuting_self_inverse,
        "collapse_one_qubit_runs": _collapse_one_qubit_runs,
        "collapse_two_qubit_blocks": _collapse_two_qubit_blocks,
    }
)


_BUILTIN_PASS_SPECS: Mapping[str, PassSpec] = MappingProxyType(
    {
        "remove_zero_state_resets": PassSpec(remove_zero_state_resets),
        "remove_diagonal_gates_before_measure": PassSpec(
            _remove_diagonal_gates_before_measure
        ),
        "remove_identity_gates": PassSpec(
            remove_identity_gates,
            uses=("rotation_parameters",),
            preserves=("rotation_parameters",),
        ),
        "merge_self_inverse": PassSpec(
            merge_self_inverse,
            uses=("self_inverse_opcodes",),
            preserves=("self_inverse_opcodes",),
        ),
        "merge_inverse_pairs": PassSpec(_merge_inverse_pairs),
        "merge_adjacent_rotations": PassSpec(
            merge_adjacent_rotations,
            uses=("rotation_parameters",),
            preserves=("rotation_parameters",),
        ),
        "merge_commuting_rotations": PassSpec(_merge_commuting_rotations),
        "cancel_commuting_self_inverse": PassSpec(
            _cancel_commuting_self_inverse,
            uses=("self_inverse_opcodes", "commutation_blocks"),
            preserves=("self_inverse_opcodes",),
        ),
        "collapse_one_qubit_runs": PassSpec(_collapse_one_qubit_runs),
        "collapse_two_qubit_blocks": PassSpec(_collapse_two_qubit_blocks),
    }
)


def default_pass_registry() -> PassRegistry:
    """The registry that resolves exactly the built-in optimization passes."""

    return PassRegistry(BUILTIN_PASSES, _BUILTIN_PASS_SPECS, _BUILTIN_ANALYSES)
