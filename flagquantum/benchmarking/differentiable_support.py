"""Capability metadata for differentiable simulator benchmark engines."""

from __future__ import annotations

from collections.abc import Sequence


def build_support_matrix(engines: Sequence[str]) -> dict[str, dict[str, object]]:
    """Return the declared differentiation contract for every benchmark engine."""

    contracts = {
        "flagquantum_native": (
            "native PyTorch reverse-mode autograd",
            "backpropagation through exact statevector",
        ),
        "pennylane_default_qubit": (
            "default.qubit backprop through the PyTorch interface",
            "torch_reverse_mode_autograd",
        ),
        "flagquantum_adjoint": (
            "native reversible statevector adjoint",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_python_fallback": (
            "same direct-layout adjoint with native operator disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_gather_rollback": (
            "rollback to per-gate full-state gather indices",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_forward_cx_rollback": (
            "same native adjoint with forward CX gather disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_observable_cache_rollback": (
            "same native adjoint with observable-weight cache disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_rotation_tile_rollback": (
            "same native adjoint with legacy two-wire rotation tiles",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_euler_triple_rollback": (
            "same native adjoint with Euler fast path disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_observable_boundary_rollback": (
            "same native adjoint with eager observable boundary",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_observable_rotation_rollback": (
            "same native adjoint with observable seeding separate from the first "
            "reverse rotation tile",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_shared_rzz_rollback": (
            "same native adjoint with shared-RZZ fusion disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_forward_rzz_rotation_rollback": (
            "same native adjoint with the backward RZZ/H boundary fusion, forward "
            "shared-RZZ fusion, and specialized rotation arithmetic disabled",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_forward_wide_tile_rollback": (
            "same native adjoint with legacy six-wire forward rotation tiles",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_flat_pair_simd_rollback": (
            "same native adjoint with nested rotation-pair traversal",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_cx_rotation_fusion_rollback": (
            "same native adjoint with separate CX gather and rotation traversal",
            "statevector_adjoint",
        ),
        "flagquantum_adjoint_terminal_no_restore_rollback": (
            "same native adjoint with earliest-layer state restoration enabled",
            "statevector_adjoint",
        ),
        "pennylane_lightning_adjoint": (
            "lightning.qubit adjoint through the PyTorch interface",
            "statevector_adjoint",
        ),
    }
    support = {
        name: {
            "included": name in engines,
            "contract": contract,
            "gradient_method": gradient_method,
        }
        for name, (contract, gradient_method) in contracts.items()
    }
    support.update(
        {
            "qiskit_aer": {
                "included": False,
                "reason": (
                    "FlagQuantum's Aer bridge has no native PyTorch gradient contract"
                ),
            },
            "cirq_simulator": {
                "included": False,
                "reason": (
                    "FlagQuantum's Cirq bridge has no native PyTorch gradient contract"
                ),
            },
        }
    )
    return support
