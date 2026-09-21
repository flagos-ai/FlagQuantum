"""Runtime-owned memory estimates used by execution planning."""

from __future__ import annotations


def estimate_state_bytes(n_wires: int, bsz: int = 1, complex_bytes: int = 8) -> int:
    """Estimate dense statevector memory."""

    return int(bsz) * (1 << int(n_wires)) * int(complex_bytes)


def estimate_density_bytes(n_wires: int, bsz: int = 1, complex_bytes: int = 8) -> int:
    """Estimate dense density-matrix memory."""

    dim = 1 << int(n_wires)
    return int(bsz) * dim * dim * int(complex_bytes)


def estimate_mps_bytes(
    n_wires: int,
    *,
    bsz: int = 1,
    max_bond: int | None = None,
    complex_bytes: int = 8,
) -> int:
    """Estimate MPS memory for a fixed maximum bond dimension."""

    n_wires = int(n_wires)
    if max_bond is None:
        max_bond = 2 ** max(0, n_wires // 2)
    return int(bsz) * n_wires * 2 * int(max_bond) * int(max_bond) * int(complex_bytes)


def estimate_tensor_network_bytes(
    n_wires: int, bsz: int = 1, complex_bytes: int = 8
) -> int:
    """Estimate the materialized output state of a tensor-network contraction.

    A contraction that returns every amplitude materializes the same dense
    state a statevector run does, so this is :func:`estimate_state_bytes`. It
    is the size of the *result*, not the residency of the contraction that
    produced it; use :func:`estimate_tensor_network_working_set_bytes` for the
    working set, and the executor's own ``peak_size`` for a measurement.
    """

    return estimate_state_bytes(n_wires, bsz=bsz, complex_bytes=complex_bytes)


def estimate_tensor_network_working_set_bytes(
    n_wires: int,
    *,
    contraction_width: int | None = None,
    complex_bytes: int = 8,
    target_count: int = 1,
    require_gradients: bool = False,
) -> int:
    """Estimate tensor-network contraction residency from the interaction width.

    This is the *uncalibrated* preflight proxy the Runtime already reasons with,
    given one definition so selection, the selection context, and the execution
    plan cannot drift apart:

    * ``contraction_width`` is the min-fill interaction width of the program
      (``interaction_width``). Omit it to assume the width of a fully connected
      program, which is the conservative default.
    * The width term is ``2 ** width`` elements, and ``n_wires`` is retained as
      the safety margin of the pre-existing proxy. That margin is an inherited
      heuristic, not a measured leg count.
    * ``target_count`` scales a batched amplitude or observable target, which
      contracts one additional output leg per requested target.
    * ``require_gradients`` reserves two more legs for the reverse-mode
      cotangent, matching what the adjoint path holds at once.
    * The exponent saturates at 62 bits so the value stays representable.

    This is **not** an upper bound on the chosen contraction order's peak, and it is
    a floor rather than a capacity guarantee. A bonded network holds far more than
    ``width`` legs at once: on an 8-wire circuit with 24 two-qubit gates and width 3
    this returns 512 bytes while the executor measures 32768 with ``memory_greedy``
    and 8192 with ``quality_multistart``. The measured value is
    ``TensorNetworkContractionProfile.peak_size``; only that is evidence about a
    specific run, and
    ``flagquantum.simulation.tensor_network.local.tensor_network_contraction_peak_bytes``
    reports it for one named order without allocating.

    What bounds a run is capacity rather than this estimate. A tensor-network run
    that declares ``memory_limit_bytes`` is executed against it as a hard peak
    budget, so the residency of an admitted run cannot exceed the declared limit;
    with no declared limit there is no ceiling and this estimate is not one. A
    batch axis is not modelled here; the estimate is per batch item.
    """

    n_wires = max(1, int(n_wires))
    width = n_wires if contraction_width is None else int(contraction_width)
    width = max(0, width) + (2 if require_gradients else 0)
    working = (1 << min(width, 62)) * int(complex_bytes) * n_wires
    return max(1, working * max(1, int(target_count)))


__all__ = [
    "estimate_density_bytes",
    "estimate_mps_bytes",
    "estimate_state_bytes",
    "estimate_tensor_network_bytes",
    "estimate_tensor_network_working_set_bytes",
]
