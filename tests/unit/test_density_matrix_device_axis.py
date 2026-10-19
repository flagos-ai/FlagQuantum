"""The density-matrix route must honour or refuse a request, never ignore it.

``density_matrix_from_ir`` accepts a ``Circuit`` or a ``CircuitIR`` and the same
``bsz``, ``device``, and ``dtype`` keywords for both. Only the ``CircuitIR``
branch ever read them: the ``Circuit`` branch took all three from the circuit's
own state, so ``density_matrix_from_ir(circuit, bsz=3)`` returned one batch row,
``dtype=torch.complex128`` returned ``complex64``, and a circuit living on
``meta`` returned an unrealized ``meta`` tensor while ``device="cpu"`` was
discarded. A caller could not tell an honoured precision from a dropped one, and
a device the catalog does not declare was accepted as quietly as ``cpu``.

These tests pin the contract that replaced it: a request is honoured or refused
with a named error. A genuine second execution device cannot be exercised on a
CPU-only host, so the honoured direction is proven through the batch-size and
precision axes and through the ``meta`` transfer that torch itself refuses.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.density_matrix as density_matrix_module
from flagquantum.kernels.catalog import KERNEL_DEVICES
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit

N_WIRES = 2


def _circuit() -> fq.Circuit:
    return fq.Circuit(N_WIRES).h(0).cx(0, 1)


def _ir() -> object:
    return _circuit().to_ir()


def test_a_requested_batch_size_is_honoured_on_both_input_forms() -> None:
    circuit_ir = _ir()

    assert tuple(density_matrix_from_ir(circuit_ir, bsz=3).shape) == (3, 4, 4)
    # A Circuit already carries its own batch of rows, so the matching request
    # is accepted and a fresh one would not be that circuit's state.
    circuit = fq.Circuit(N_WIRES, bsz=4).x(0)
    assert tuple(density_matrix_from_ir(circuit).shape) == (4, 4, 4)
    assert tuple(density_matrix_from_ir(circuit, bsz=4).shape) == (4, 4, 4)


def test_a_requested_precision_is_honoured_on_both_input_forms() -> None:
    circuit = _circuit()

    assert density_matrix_from_ir(circuit, dtype=torch.complex128).dtype == (
        torch.complex128
    )
    assert density_matrix_from_ir(_ir(), dtype=torch.complex128).dtype == (
        torch.complex128
    )


def test_an_omitted_request_inherits_what_the_input_declares() -> None:
    circuit = fq.Circuit(N_WIRES, bsz=2, dtype=torch.complex128).x(0)

    rho = density_matrix_from_ir(circuit)

    assert tuple(rho.shape) == (2, 4, 4)
    assert rho.dtype == torch.complex128
    assert rho.device.type == "cpu"


def test_an_omitted_device_request_does_not_substitute_cpu() -> None:
    """The inherited device must come from the input, not from a default.

    A circuit on ``meta`` is the only way to observe this on a CPU-only host: if
    the omitted request resolved to ``cpu`` instead of to the state's own device,
    the transfer would be attempted and torch would refuse it.
    """

    circuit = fq.Circuit(N_WIRES, device="meta").h(0).cx(0, 1)

    rho = density_matrix_from_ir(circuit)

    assert rho.device.type == "meta"


def test_a_batch_size_that_contradicts_the_circuit_state_is_refused() -> None:
    with pytest.raises(ValueError, match="batch size mismatch"):
        density_matrix_from_ir(_circuit(), bsz=2)


@pytest.mark.parametrize("bsz", [0, -1])
def test_a_non_positive_batch_size_is_refused(bsz: int) -> None:
    with pytest.raises(ValueError, match=f"batch size must be positive, got {bsz}"):
        density_matrix_from_ir(_ir(), bsz=bsz)


@pytest.mark.parametrize("spelling", ["gpu", "meta", "mps", "CUDA"])
@pytest.mark.parametrize("input_kind", ["circuit", "circuit_ir"])
def test_an_undeclared_device_is_refused_for_both_input_forms(
    spelling: str, input_kind: str
) -> None:
    """`CUDA` stays refused: the catalog matches device fields by exact spelling."""

    argument = _circuit() if input_kind == "circuit" else _ir()
    declared = ", ".join(sorted(KERNEL_DEVICES))

    with pytest.raises(
        ValueError,
        match=f"undeclared density-matrix device: {spelling}; "
        f"declared devices are {declared}",
    ):
        density_matrix_from_ir(argument, device=spelling)


@pytest.mark.parametrize("input_kind", ["circuit", "circuit_ir"])
def test_a_declared_device_is_accepted(input_kind: str) -> None:
    argument = _circuit() if input_kind == "circuit" else _ir()

    rho = density_matrix_from_ir(argument, device="cpu")

    assert rho.device.type == "cpu"
    assert rho.device.type in KERNEL_DEVICES


def test_a_requested_device_is_reached_instead_of_being_discarded() -> None:
    """A discarded request returned a tensor on the input's device without a word.

    The circuit here lives on ``meta`` and the request is ``cpu``, which is the
    only declared device this host can build. torch refuses to copy out of a
    ``meta`` tensor, so the refusal is what proves the transfer was attempted:
    the previous behaviour returned the ``meta`` tensor and never reached
    ``.to`` at all.
    """

    circuit = fq.Circuit(N_WIRES, device="meta").h(0)

    with pytest.raises(NotImplementedError, match="Cannot copy out of meta tensor"):
        density_matrix_from_ir(circuit, device="cpu")


def test_a_real_dtype_is_refused_because_phase_would_be_discarded() -> None:
    with pytest.raises(ValueError, match="density matrix dtype must be complex"):
        density_matrix_from_ir(_ir(), dtype=torch.float32)


def test_the_simulator_shares_the_device_axis_declared_by_the_catalog() -> None:
    """One declared axis, not a second copy of the device vocabulary."""

    assert density_matrix_module.KERNEL_DEVICES is KERNEL_DEVICES


def test_the_runtime_entry_forwards_all_three_requests() -> None:
    """``noisy_density_matrix`` lowered the circuit before the branch was read.

    Lowering yields a ``CircuitIR``, so the runtime entry always reached the
    branch that honoured the keywords; the forwarded values must keep meaning
    what they say after the two branches were made to agree.
    """

    from flagquantum.noise import NoiseModel, depolarizing_channel
    from flagquantum.runtime.noise_registry import noisy_density_matrix

    circuit = _circuit()
    model = NoiseModel().add(("h",), depolarizing_channel(0.01))

    assert tuple(noisy_density_matrix(circuit, model, bsz=3).shape) == (3, 4, 4)
    assert noisy_density_matrix(circuit, model, dtype=torch.complex128).dtype == (
        torch.complex128
    )
    with pytest.raises(ValueError, match="undeclared density-matrix device: gpu"):
        noisy_density_matrix(circuit, model, device="gpu")
