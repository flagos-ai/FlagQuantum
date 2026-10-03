from __future__ import annotations

import json
import random
import time
from dataclasses import FrozenInstanceError

import pytest

import flagquantum as fq
from flagquantum.compiler import ESTIMATE_BASIS, ResourceEstimate, estimate_resources
from flagquantum.compiler.pipeline import schedule_layers
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit

_MEASURED_QUANTITY_MARKERS = (
    "allocat",
    "byte",
    "communicat",
    "duration",
    "elapsed",
    "fidelity",
    "memory",
    "seconds",
    "time",
    "wall",
)
"""Substrings a field name would carry if it held a measured quantity.

A static estimate reports what a program would apply.  A field named for elapsed
time, resident memory, transferred bytes, or achieved fidelity would report what a
run did cost instead, and a caller reading it as an estimate would be reading
something the estimator never measured.
"""


def _estimate(program: object) -> ResourceEstimate:
    return estimate_resources(program)


def test_a_bell_pair_reports_its_two_operations_and_their_layer_count() -> None:
    estimate = _estimate(fq.Circuit(2).h(0).cx(0, 1))

    assert estimate.n_operations == 2
    assert estimate.operation_counts == {"h": 1, "cx": 1}
    assert estimate.depth == 2


def test_operations_on_different_wires_share_one_layer() -> None:
    """A count of instructions is not a depth: three independent gates are one layer."""

    estimate = _estimate(fq.Circuit(3).h(0).h(1).h(2))

    assert estimate.n_operations == 3
    assert estimate.depth == 1


def test_depth_follows_dependencies_and_not_per_wire_operation_counts() -> None:
    """Each wire here carries two operations, yet the chain is three layers deep.

    The last ``h`` on wire 1 cannot start before the ``cx`` on wire 0 finishes, and
    the ``cx`` cannot start before the first ``h`` does, so the deepest single-wire
    operation count understates the dependency chain by one layer.
    """

    estimate = _estimate(fq.Circuit(2).h(0).cx(0, 1).h(1))

    assert estimate.per_wire_depth == (2, 3)
    assert estimate.depth == 3


def test_a_repeated_operation_is_counted_once_per_occurrence() -> None:
    """A count of distinct opcodes would report one here, not three."""

    estimate = _estimate(fq.Circuit(1).h(0).h(0).h(0))

    assert estimate.operation_counts == {"h": 3}
    assert estimate.n_operations == 3


def test_a_wire_that_no_operation_touches_is_unused_but_still_allocated() -> None:
    """The peak is the declared register, which no operation can raise.

    ``CircuitIR`` allocates its whole register before the first operation, so a
    program that touches one wire of four has a peak of four and a usage of one.
    """

    estimate = _estimate(fq.Circuit(4).h(0))

    assert estimate.n_qubits == 4
    assert estimate.used_wires == 1
    assert estimate.per_wire_depth == (1, 0, 0, 0)


def test_the_operation_width_is_not_the_declared_register_width() -> None:
    one_wire = _estimate(fq.Circuit(3).h(0))
    three_wire = _estimate(fq.Circuit(3).ccx(0, 1, 2))
    two_wire = _estimate(fq.Circuit(4).cx(0, 1))

    assert one_wire.max_operation_width == 1
    assert three_wire.max_operation_width == 3
    assert two_wire.max_operation_width == 2
    assert two_wire.n_qubits == 4


def test_per_wire_depth_records_the_last_layer_that_touches_the_wire() -> None:
    """Wire 0 is idle in the second layer, so it stays one layer deep, not two.

    Counting the layers in which a wire appears would report two for both wires and
    would overstate how much of the chain that wire is on the critical path of.  The
    second program separates the two rules: wire 0 is idle in the second layer,
    which only wire 1 uses, and the third layer's ``cx`` cannot start before wire 1
    is free, so wire 0 comes back in layer three.  Wire 0 is therefore two layers
    deep by a count of the layers it appears in and three by the last layer that
    touches it.
    """

    idle_last = _estimate(fq.Circuit(2).cx(0, 1).h(1))
    skipped_layer = _estimate(fq.Circuit(2).cx(0, 1).h(1).cx(0, 1))

    assert idle_last.per_wire_depth == (1, 2)
    assert idle_last.depth == 2
    assert skipped_layer.per_wire_depth == (3, 3)
    assert skipped_layer.depth == 3


def test_parallel_t_operations_share_one_t_layer() -> None:
    """Two T gates on different wires are one T layer, though there are two of them."""

    estimate = _estimate(fq.Circuit(2).t(0).t(1))

    assert estimate.t_count == 2
    assert estimate.t_depth == 1
    assert estimate.depth == 1


def test_t_depth_serializes_t_operations_on_one_wire() -> None:
    estimate = _estimate(fq.Circuit(1).t(0).t(0))

    assert estimate.t_count == 2
    assert estimate.t_depth == 2


def test_t_depth_covers_the_inverse_t_gate_and_its_alias() -> None:
    explicit = _estimate(fq.Circuit(1).t(0).tdg(0))
    aliased = _estimate(fq.Circuit(1).td(0))

    assert explicit.t_count == 2
    assert explicit.t_depth == 2
    assert aliased.operation_counts == {"tdg": 1}
    assert aliased.t_count == 1


def test_a_non_t_operation_between_two_t_layers_separates_them() -> None:
    estimate = _estimate(fq.Circuit(1).t(0).h(0).t(0))

    assert estimate.depth == 3
    assert estimate.t_count == 2
    assert estimate.t_depth == 2


def test_a_program_with_no_t_operation_has_no_t_depth() -> None:
    estimate = _estimate(fq.Circuit(1).h(0).h(0))

    assert estimate.depth == 2
    assert estimate.t_count == 0
    assert estimate.t_depth == 0


def test_a_program_that_uses_two_wires_of_a_wide_register_is_estimated_without_running_it() -> (
    None
):
    """Thirty wires is 2**30 complex64 amplitudes, which no run could hold.

    The estimate returns in well under a second at a width whose statevector would
    be eight gibibytes, so no execution, allocation, or device query produced these
    numbers.
    """

    started = time.perf_counter()
    estimate = _estimate(fq.Circuit(30).h(0).cx(0, 29).t(29))
    elapsed = time.perf_counter() - started

    assert estimate.n_qubits == 30
    assert estimate.used_wires == 2
    assert estimate.depth == 3
    assert elapsed < 2.0


def test_an_empty_program_estimates_to_nothing() -> None:
    """Nothing scheduled is zero layers, so the empty program is not a special case."""

    estimate = _estimate(fq.Circuit(3))

    assert estimate.n_operations == 0
    assert estimate.operation_counts == {}
    assert estimate.depth == 0
    assert estimate.per_wire_depth == (0, 0, 0)
    assert estimate.used_wires == 0
    assert estimate.max_operation_width == 0
    assert estimate.t_depth == 0


def test_an_unbound_parameter_needs_no_substitution() -> None:
    """The operations a program applies do not depend on the value of an angle."""

    estimate = _estimate(fq.Circuit(1).rx(0, fq.Parameter("theta")))

    assert estimate.operation_counts == {"rx": 1}
    assert estimate.depth == 1


def test_the_estimate_states_that_it_is_a_static_count_and_carries_no_measurement() -> (
    None
):
    estimate = _estimate(fq.Circuit(2).h(0).cx(0, 1))
    payload = estimate.to_dict()

    assert ESTIMATE_BASIS == "static_instruction_sequence"
    assert estimate.to_dict()["estimate_basis"] == ESTIMATE_BASIS
    assert payload["kind"] == "flagquantum.resource_estimate"
    measured = sorted(
        key
        for key in payload
        if any(marker in key for marker in _MEASURED_QUANTITY_MARKERS)
    )
    assert measured == []
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_the_record_is_frozen() -> None:
    estimate = _estimate(fq.Circuit(1))

    with pytest.raises(FrozenInstanceError):
        estimate.depth = 7  # type: ignore[misc]


def test_the_estimate_agrees_with_the_compilers_own_schedule() -> None:
    """A second depth rule would drift from the scheduler the compiler uses.

    The estimator reuses ``schedule_layers``, so its depth is the layer count and
    its operation counts sum to the operation count for every program, not just for
    the fixtures above.
    """

    single_wire = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg")
    two_wire = ("cx", "cy", "cz", "swap")
    generator = random.Random(20250915)
    for _ in range(200):
        n_wires = generator.randint(1, 5)
        circuit = fq.Circuit(n_wires)
        for _ in range(generator.randint(0, 12)):
            if n_wires == 1 or generator.random() < 0.5:
                wire = generator.randrange(n_wires)
                circuit = getattr(circuit, generator.choice(single_wire))(wire)
            else:
                control, target = generator.sample(range(n_wires), 2)
                circuit = getattr(circuit, generator.choice(two_wire))(control, target)
        ir = circuit.to_ir()
        estimate = _estimate(ir)

        assert estimate.depth == len(schedule_layers(ir))
        assert estimate.n_operations == len(ir)
        assert sum(estimate.operation_counts.values()) == estimate.n_operations
        assert estimate.used_wires == sum(
            1 for depth in estimate.per_wire_depth if depth
        )


def test_a_noise_channel_is_counted_apart_from_algorithmic_operations() -> None:
    """A lowered noise model injects operations, and reading them as work misleads."""

    from flagquantum.compiler import lower_noise_model
    from flagquantum.noise import NoiseModel, depolarizing_channel

    model = NoiseModel().add("h", depolarizing_channel(0.01))
    noisy = lower_noise_model(fq.Circuit(2).h(0).cx(0, 1).to_ir(), model)
    estimate = _estimate(noisy)

    assert estimate.channel_count == 1
    assert estimate.operation_counts == {"h": 1, "depolarizing": 1, "cx": 1}
    assert estimate.n_operations == 3


def test_operation_counts_use_the_canonical_opcode() -> None:
    """``cnot`` and ``td`` are names for ``cx`` and ``tdg``, and one spelling is reported.

    This is where this estimate departs from CUDA-Q, which prefixes an operation's
    recorded control count onto its name and so reports a recorded two-control
    ``x`` and a written ``ccx`` identically.  No ``Instruction`` field holds a
    control count the opcode does not already name, so the canonical opcode is what
    can be reported without inventing one.
    """

    alias = _estimate(fq.Circuit(2).cnot(0, 1))
    explicit = _estimate(fq.Circuit(2).cx(0, 1))

    assert alias.operation_counts == explicit.operation_counts == {"cx": 1}


def test_a_run_time_dependent_operation_is_refused_rather_than_estimated() -> None:
    """No static count describes a program whose operations depend on outcomes."""

    program = CircuitIR(
        n_wires=1,
        instructions=(Instruction("reset", (0,), metadata={"is_dynamic": True}),),
    )

    with pytest.raises(CapabilityError) as error:
        _estimate(program)

    message = str(error.value)
    assert "instruction 0" in message
    assert "'reset'" in message
    assert "depend on run-time outcomes" in message


@pytest.mark.parametrize(
    "metadata",
    (
        {"conditions": ((0, 1),)},
        {"condition_clauses": (((0, 1),),)},
    ),
    ids=("conditions", "condition_clauses"),
)
def test_a_classically_conditioned_operation_is_refused(metadata: dict) -> None:
    """A conditioned operation may apply zero times, so it has no static count."""

    program = CircuitIR(
        n_wires=1,
        instructions=(Instruction("x", (0,), metadata=metadata),),
    )

    with pytest.raises(CapabilityError) as error:
        _estimate(program)

    message = str(error.value)
    assert "instruction 0" in message
    assert "'x'" in message
    assert "classical condition" in message


def test_a_dependent_operation_later_in_the_program_is_refused_by_its_own_index() -> (
    None
):
    """The refusal names the offending instruction, and nothing is counted before it."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction("h", (0,)),
            Instruction("reset", (0,), metadata={"is_dynamic": True}),
        ),
    )

    with pytest.raises(CapabilityError) as error:
        _estimate(program)

    assert "instruction 1" in str(error.value)


def test_an_input_that_is_not_a_program_is_refused() -> None:
    with pytest.raises(TypeError):
        _estimate([("h", 0)])


def test_the_estimate_is_a_compiler_surface_and_not_a_stable_core_export() -> None:
    """The estimator lives beside the compiler's other expert entry points.

    Adding it to the root namespace would change the 34-name Stable Core export
    set, which needs an approved API change proposal rather than an implementation
    change, so the compiler namespace is the home and the root stays untouched.
    """

    assert fq.compiler.estimate_resources is estimate_resources
    assert not hasattr(fq, "estimate_resources")
