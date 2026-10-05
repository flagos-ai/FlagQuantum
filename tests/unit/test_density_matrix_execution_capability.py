"""The density-matrix execution registry entry, pinned to the engine it describes.

Four properties are asserted here rather than described.

**The registry entry names symbols that exist.**  A ``public_apis`` list is the
part of a capability claim a reader is meant to call, so every name is resolved
here through its module and ``getattr``.  The central maturity checker only
requires the list to be non-empty, so a typo, a rename, or a deletion would
otherwise leave a claim pointing at nothing.

**The route the entry declares is the route the runtime selects.**  The entry
says ``density_matrix`` is reached by asking for it and by ``auto`` on a program
that carries a channel, and that ``statevector`` and ``mps`` refuse such a
program by name.  Each of those four is executed here, so a change in mode
selection fails a test rather than silently widening or narrowing the claim.

**The engine's wire convention is the one the entry publishes.**  Wire 0 is the
highest-order index, which is the single fact most likely to be misread from the
code, so it is asserted against ``torch.kron`` rather than against the engine's
own output.

**The claim stops where the measurement stops.**  The entry claims CPU hardware,
a 13-wire measured ceiling, an unenforced memory limit, and a lower readout
ceiling than storage ceiling.  The tests below reproduce each boundary, so
relaxing the prose without relaxing the engine, or the reverse, fails here.

None of the numbers below is a performance or scalability claim.  Every assertion
is either exact or an explicitly stated tolerance.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest
import tomllib
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATURITY = ROOT / "capability-maturity.toml"
PARITY = ROOT / "contracts/cudaq-parity-matrix.toml"
CAPABILITY = "density_matrix_execution"
PARITY_ROW = "backend_density_matrix"

#: The storage ceiling the entry claims, in wires, and its measured cost.
MEASURED_CEILING_WIRES = 13
MEASURED_CEILING_BYTES = 2**MEASURED_CEILING_WIRES * 2**MEASURED_CEILING_WIRES * 16

#: The readout ceiling the entry claims, in wires, before a marginal probability
#: is refused until ``max_marginal_wires`` is raised.
READOUT_CEILING_WIRES = 8

#: The precision at which the entry claims a bit-for-bit gradient reproduction.
EXACT_GRADIENT_PRECISION = "complex128"


def _maturity_contract() -> dict:
    with MATURITY.open("rb") as handle:
        return tomllib.load(handle)


def _capability() -> dict:
    return _maturity_contract()["capabilities"][CAPABILITY]


def _parity_rows() -> dict[str, dict]:
    """Return the parity contract's rows keyed by capability id."""

    with PARITY.open("rb") as handle:
        contract = tomllib.load(handle)
    return {
        capability["id"]: capability
        for domain in contract["domains"]
        for capability in domain["capabilities"]
    }


def _resolve(dotted: str) -> object:
    """Resolve one dotted ``module.attribute`` name, failing if it is absent."""

    module_name, _, attribute = dotted.rpartition(".")
    assert module_name, f"not a resolvable public API name: {dotted}"
    module = importlib.import_module(module_name)
    assert hasattr(module, attribute), f"{dotted} is not defined in {module_name}"
    return getattr(module, attribute)


def _channel_circuit() -> object:
    """Return a two-wire Bell pair carrying one inline depolarizing channel."""

    return fq.Circuit(2).h(0).cx(0, 1).depolarizing(0, 0.2)


def test_the_entry_registers_the_route_the_engine_executes() -> None:
    """The registered level and semantics are the ones the engine supports."""

    capability = _capability()
    assert capability["category"] == "simulation_and_training"
    assert capability["level"] == "development_evidence"
    assert capability["runtime_modes"] == ["density_matrix"]
    assert capability["hardware"] == ["cpu"]
    assert capability["gradient_support"] == "exact"
    assert capability["distribution_semantics"] == "single_device_fast_path"
    assert capability["owner"] == "simulation maintainers"
    assert (
        capability["focused_tests"]
        == "tests/unit/test_density_matrix_execution_capability.py"
    )


def test_every_declared_public_api_resolves() -> None:
    """A claim a reader may call must name something a reader can call."""

    names = _capability()["public_apis"]
    assert names
    for name in names:
        assert name.startswith("flagquantum."), name
        assert callable(_resolve(name)), f"{name} resolves to a non-callable"


def test_the_maturity_entry_names_only_paths_that_exist() -> None:
    capability = _capability()
    for field in (
        "quick_start",
        "documentation",
        "focused_tests",
        "development_artifact",
    ):
        value = capability.get(field)
        assert isinstance(value, str) and value, field
        assert (ROOT / value).exists(), f"{field} does not exist: {value}"


def test_the_entry_declares_the_evidence_its_level_requires() -> None:
    """A level is a promise about evidence; the entry must carry all of it."""

    document = _maturity_contract()
    capability = document["capabilities"][CAPABILITY]
    required = set(document["levels"][capability["level"]]["required_evidence"])
    assert "limitations" in required
    assert "focused_tests" in required
    assert "development_artifact" in required
    assert isinstance(capability["limitations"], str) and capability["limitations"]


def test_every_boundary_the_engine_has_is_disclosed() -> None:
    """Each disclosure below is a boundary a reader would otherwise misread.

    Dropping one silently widens the claim, so the entry is asserted to keep
    naming every boundary this file also pins by execution.
    """

    limitations = _capability()["limitations"]
    for disclosure in (
        f"{MEASURED_CEILING_WIRES} wires",
        "4^n",
        "max_marginal_wires",
        "highest-order index",
        "unverified",
        "not sharded",
        "memory_limit_bytes",
        "is not enforced",
        "require_gradients",
        "gradients_not_available",
        "mid-circuit measurement",
        "DynamicCircuit",
        EXACT_GRADIENT_PRECISION,
    ):
        assert disclosure in limitations, disclosure


def test_asking_for_the_mode_selects_the_density_matrix_route() -> None:
    program = _channel_circuit()
    result = fq.run(
        program,
        options=fq.ExecutionOptions(mode="density_matrix"),
        outputs=fq.probabilities(),
    )
    assert result.plan.state_mode == "density_matrix"
    assert result.probabilities.shape == (1, 4)


def test_auto_selects_the_route_exactly_when_the_program_carries_a_channel() -> None:
    """``auto`` is a decision about the program, not about the caller's intent."""

    noisy = fq.run(
        _channel_circuit(),
        options=fq.ExecutionOptions(mode="auto"),
        outputs=fq.probabilities(),
    )
    clean = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        options=fq.ExecutionOptions(mode="auto"),
        outputs=fq.probabilities(),
    )
    assert noisy.plan.state_mode == "density_matrix"
    assert clean.plan.state_mode == "statevector"


@pytest.mark.parametrize("mode", ["statevector", "mps"])
def test_a_carrier_of_a_channel_is_refused_by_name_on_other_routes(mode: str) -> None:
    """A channel is refused, never approximated onto an amplitude store."""

    with pytest.raises(Exception) as caught:
        fq.run(
            _channel_circuit(),
            options=fq.ExecutionOptions(mode=mode),
            outputs=fq.probabilities(),
        )
    message = str(caught.value)
    assert "mode='auto' or mode='density_matrix'" in message, message


def test_a_channel_free_program_uses_the_same_engine_as_the_runtime() -> None:
    """The outer-product shortcut and the executor must agree where both apply."""

    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    program = fq.Circuit(2).h(0).cx(0, 1)
    assert torch.allclose(program.density_matrix(), density_matrix_from_ir(program))


def test_a_channel_carrying_program_is_routed_rather_than_read_off_a_statevector() -> (
    None
):
    """An outer product is rank one and cannot be the mixed state a channel makes."""

    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    program = _channel_circuit()
    routed = program.density_matrix()
    assert torch.allclose(routed, density_matrix_from_ir(program))
    rho = routed[0]
    assert torch.allclose(torch.trace(rho), torch.ones((), dtype=rho.dtype), atol=1e-6)
    purity = torch.trace(rho @ rho).real
    assert float(purity) < 1.0 - 1e-3


def test_wire_zero_is_the_high_order_index() -> None:
    """The published convention, asserted against ``torch.kron`` and not itself."""

    from flagquantum.simulation.density_matrix import expand_operator

    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    eye = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(expand_operator(x, [0], 2), torch.kron(x, eye))
    assert not torch.allclose(expand_operator(x, [0], 2), torch.kron(eye, x))
    assert torch.allclose(expand_operator(x, [1], 2), torch.kron(eye, x))


def test_the_exact_route_carries_a_gradient_in_both_precisions() -> None:
    """A claimed exact graph is asserted at the exact precision it is claimed at."""

    theta = 0.7
    for precision in ("complex64", EXACT_GRADIENT_PRECISION):
        parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        run = fq.run(
            fq.Circuit(1).ry(0, parameter),
            options=fq.ExecutionOptions(mode="density_matrix", precision=precision),
            outputs=fq.expectation(fq.Z(0)),
        )
        (gradient,) = torch.autograd.grad(run.expectations[0].sum(), parameter)
        difference = abs(float(gradient) + math.sin(theta))
        if precision == EXACT_GRADIENT_PRECISION:
            assert difference == 0.0
        else:
            assert difference < 1e-6


def test_the_measured_storage_ceiling_is_the_one_the_entry_claims() -> None:
    """The entry names a ceiling; the arithmetic behind it is pinned here.

    The 13-wire state is not allocated a second time: the payload is over a
    gigabyte, and the entry's own disclosure is the thing under test. The
    engine's cost model is checked at a size that costs nothing to build.
    """

    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    capability = _capability()
    assert f"{MEASURED_CEILING_WIRES} wires" in capability["limitations"]
    assert 4**MEASURED_CEILING_WIRES * 16 == MEASURED_CEILING_BYTES

    wires = 4
    rho = density_matrix_from_ir(fq.Circuit(wires).h(0), dtype=torch.complex128)
    assert rho.shape == (1, 2**wires, 2**wires)
    assert rho.dtype is torch.complex128
    assert rho.nelement() * rho.element_size() == 4**wires * 16


def test_the_readout_ceiling_is_lower_than_the_storage_ceiling() -> None:
    """Storing a state and reading a marginal off it are two different ceilings."""

    from flagquantum.simulation.density_matrix import density_matrix_from_ir

    program = fq.Circuit(READOUT_CEILING_WIRES + 1).h(0)
    assert density_matrix_from_ir(program).shape == (
        1,
        2 ** (READOUT_CEILING_WIRES + 1),
        2 ** (READOUT_CEILING_WIRES + 1),
    )
    with pytest.raises(ValueError) as caught:
        fq.run(
            program,
            options=fq.ExecutionOptions(mode="density_matrix"),
            outputs=fq.probabilities(),
        )
    assert "max_marginal_wires" in str(caught.value)


def test_the_memory_limit_is_recorded_and_not_enforced() -> None:
    """The entry discloses an unenforced limit; the run must still show both halves."""

    program = fq.Circuit(4).h(0).cx(0, 1)
    limit = 1
    assert program.state() is not None
    result = fq.run(
        program,
        options=fq.ExecutionOptions(mode="density_matrix", memory_limit_bytes=limit),
        outputs=fq.probabilities(),
    )
    decision = result.plan.to_dict()["decision"]
    assert decision["memory_limit_bytes"] == limit
    # The dense state is far larger than the declared limit, and it ran anyway.
    assert result.plan.state_bytes > limit
    assert result.probabilities.shape == (1, 16)


def test_a_declared_gradient_requirement_is_not_a_post_condition_here() -> None:
    """The entry says the post-condition lives one layer up; this pins that split."""

    result = fq.run(
        fq.Circuit(1).ry(0, 0.4),
        options=fq.ExecutionOptions(mode="density_matrix", require_gradients=True),
        outputs=fq.expectation(fq.Z(0)),
    )
    assert result.expectations[0].requires_grad is False


def test_a_mid_circuit_measurement_is_not_expressible_on_this_route() -> None:
    """The entry names the absence, and the operator vocabulary is where it lives."""

    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

    opcodes = set(OPERATOR_SCHEMAS)
    assert "measure" not in opcodes
    assert "reset" not in opcodes
    assert "measure" not in dir(fq.Circuit)
    assert "reset" not in dir(fq.Circuit)


def test_the_entry_does_not_claim_an_unverified_accelerator() -> None:
    """Hardware evidence is a measurement; the entry may not outrun this checkout."""

    capability = _capability()
    assert capability["hardware"] == ["cpu"]
    assert "unverified" in capability["limitations"]
    assert not torch.cuda.is_available()


def test_the_run_route_refuses_an_undeclared_device_at_admission() -> None:
    """The entry says admission, not the engine, refuses these devices."""

    for device in ("meta", "mps"):
        with pytest.raises(Exception) as caught:
            fq.run(
                fq.Circuit(1).h(0),
                options=fq.ExecutionOptions(mode="density_matrix", device=device),
                outputs=fq.probabilities(),
            )
        assert "backend_unavailable" in str(caught.value)


def test_the_parity_row_cites_this_capability() -> None:
    """The row cites the entry that establishes the FlagQuantum side."""

    row = _parity_rows()[PARITY_ROW]
    assert row["status"] == "partial"
    assert row["maturity_ref"] == CAPABILITY
    assert row["priority"] == "now"
    assert row["dependency_class"] == "none"
    assert "maturity_refs" not in row
    for evidence in row["evidence"]:
        assert not evidence.startswith("search:"), evidence
        assert (ROOT / evidence).exists(), evidence


def test_the_parity_row_no_longer_says_the_capability_is_unregistered() -> None:
    """A closed row whose reason still says "no entry" is a stale claim."""

    reason = _parity_rows()[PARITY_ROW]["reason"]
    assert "no capability-maturity.toml entry" not in reason
    assert CAPABILITY in reason
    # The row is partial because the route cannot express a mid-circuit
    # measurement, not because the engine is absent.
    assert "mid-circuit measurement" in reason


def test_the_parity_baseline_gap_claims_no_measurement_it_did_not_take() -> None:
    """A narrower scope is prose; a measured comparison is not claimed here."""

    gap = _parity_rows()[PARITY_ROW].get("baseline_gap")
    assert isinstance(gap, str) and gap
    assert "No measured comparison" in gap
    assert str(MEASURED_CEILING_WIRES) in gap
    assert str(READOUT_CEILING_WIRES) in gap
