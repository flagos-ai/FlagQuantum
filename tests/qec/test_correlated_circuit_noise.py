"""Exclusive circuit-fault laws checked by direct product injection and enumeration."""

from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from collections.abc import Callable
from itertools import product

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.noise import (
    KrausChannel,
    NoiseModel,
    bit_flip_channel,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
    y_flip_channel,
)
from flagquantum.qec import (
    DemError,
    DetectorErrorModel,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    ZxxzSurfaceCode,
    build_memory_circuit,
    compile_memory_sampler,
    sample_memory_circuit,
)
from flagquantum.qec.circuits.memory import MemoryCircuit
from flagquantum.qec.codes import StabilizerCode
from flagquantum.qec.dem.alternatives import fault_groups
from flagquantum.qec.dem.circuit_noise import _circuit_noise_locations
from flagquantum.qec.execution.signatures import _executed_flip_set, _lowered_program

pytestmark = pytest.mark.unit


def _mixture(labels: tuple[str, ...], weights: tuple[float, ...]) -> KrausChannel:
    matrices = {
        "I": torch.eye(2, dtype=torch.complex128),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
    }
    operators = []
    for label, weight in zip(labels, weights, strict=True):
        matrix = matrices[label[0]]
        for letter in label[1:]:
            matrix = torch.kron(matrix, matrices[letter])
        operators.append(math.sqrt(weight) * matrix)
    return KrausChannel("biased", tuple(operators))


def _injected(
    program: CircuitIR, faults: tuple[tuple[int, tuple[int, ...], str], ...]
) -> CircuitIR:
    gates = {"I": (), "X": ("x",), "Z": ("h", "x", "h"), "Y": ("x", "h", "x", "h")}
    instructions = []
    for index, instruction in enumerate(program.instructions):
        instructions.append(instruction)
        for position, qubits, label in faults:
            if position == index:
                instructions.extend(
                    Instruction(gate, (q,))
                    for q, letter in zip(qubits, label, strict=True)
                    for gate in gates[letter]
                )
    return CircuitIR(
        n_wires=program.n_wires,
        instructions=tuple(instructions),
        metadata=dict(program.metadata),
    )


@pytest.mark.parametrize(
    ("code", "basis"),
    [
        (RepetitionCode(distance=3), "z"),
        (RotatedSurfaceCode(distance=2), "z"),
        (SteaneCode(), "z"),
        (SteaneCode(), "x"),
        (ZxxzSurfaceCode(distance=3), "z"),
        (ZxxzSurfaceCode(distance=3), "x"),
    ],
)
def test_all_biased_locations_match_independent_forced_execution(
    code: StabilizerCode, basis: str
) -> None:
    memory = build_memory_circuit(code, rounds=2, readout_basis=basis)
    labels = ("II", "IX", "XI", "YZ")
    weights = (0.5, 0.125, 0.25, 0.125)
    noise = NoiseModel().add("cx", _mixture(labels, weights))
    program = _lowered_program(memory)
    expected: dict[tuple[int, tuple[int, ...], tuple[int, ...]], float] = defaultdict(
        float
    )
    location = 0
    for position, instruction in enumerate(program.instructions):
        for _, qubits in noise.channels_for(instruction):
            for label, weight in zip(labels[1:], weights[1:], strict=True):
                d, o = _executed_flip_set(
                    memory, _injected(program, ((position, qubits, label),))
                )
                if d or o:
                    expected[(location, d, o)] += weight
            location += 1
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    actual = {
        (e.error_id, e.detectors, e.observables): e.probability for e in model.errors
    }
    assert actual == pytest.approx(dict(expected))
    assert model == DetectorErrorModel.from_memory_circuit(memory, noise=noise)


def _law(
    model: DetectorErrorModel,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    groups = []
    for indices in fault_groups(model.errors):
        errors = tuple(model.errors[i] for i in indices)
        groups.append(
            (
                *((e.probability, e.detectors, e.observables) for e in errors),
                (1 - math.fsum(e.probability for e in errors), (), ()),
            )
        )
    law: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for outcomes in product(*groups):
        detectors: set[int] = set()
        observables: set[int] = set()
        for _, d, o in outcomes:
            detectors.symmetric_difference_update(d)
            observables.symmetric_difference_update(o)
        law[(tuple(sorted(detectors)), tuple(sorted(observables)))] += math.prod(
            p for p, _, _ in outcomes
        )
    return dict(law)


def _two_location_experiment() -> tuple[MemoryCircuit, NoiseModel]:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    noise = NoiseModel().add(
        "cx", _mixture(("II", "IX", "XI"), (0.5, 0.2, 0.3)), qubits=(0, 3)
    )
    noise.add("cx", _mixture(("II", "XY", "ZX"), (0.625, 0.125, 0.25)), qubits=(2, 4))
    return memory, noise


def test_complete_joint_law_matches_cartesian_circuit_execution() -> None:
    memory, noise = _two_location_experiment()
    program = _lowered_program(memory)
    placements = [
        (i, targets)
        for i, instruction in enumerate(program.instructions)
        for _, targets in noise.channels_for(instruction)
    ]
    assert len(placements) == 2
    expected: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for (a, pa), (b, pb) in product(
        zip(("II", "IX", "XI"), (0.5, 0.2, 0.3), strict=True),
        zip(("II", "XY", "ZX"), (0.625, 0.125, 0.25), strict=True),
    ):
        faults = tuple(
            (i, targets, label)
            for (i, targets), label in zip(placements, (a, b), strict=True)
        )
        expected[_executed_flip_set(memory, _injected(program, faults))] += pa * pb
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert _law(model) == pytest.approx(dict(expected))
    assert sum(_law(model).values()) == pytest.approx(1)
    for d in range(model.num_detectors):
        assert float(model.detector_rates()[d]) == pytest.approx(
            sum(p for (ds, _), p in expected.items() if d in ds)
        )


def test_exclusive_sampler_cannot_create_a_joint_alternative() -> None:
    model = DetectorErrorModel(
        2, 0, (DemError(0.2, (0,), error_id=0), DemError(0.3, (1,), error_id=0))
    )
    sample = model.dem_sampling(shots=20000, seed=47)
    assert not torch.logical_and(sample.detectors[:, 0], sample.detectors[:, 1]).any()
    assert sample == model.dem_sampling(shots=20000, seed=47)
    # Independent Bernoulli substitution has positive mass .2*.3 here.
    independent = DetectorErrorModel(2, 0, (DemError(0.2, (0,)), DemError(0.3, (1,))))
    drawn = independent.dem_sampling(shots=20000, seed=47)
    assert torch.logical_and(drawn.detectors[:, 0], drawn.detectors[:, 1]).any()


@pytest.mark.parametrize("p", [0.0, 0.2, 1.0])
@pytest.mark.parametrize(
    "factory", [y_flip_channel, depolarizing_channel, two_qubit_depolarizing_channel]
)
def test_factories_and_probability_boundaries(
    p: float, factory: Callable[[float], KrausChannel]
) -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=NoiseModel().add("cx", factory(p))
    )
    assert model.num_errors > 0 if p else model.num_errors == 0
    assert all(0 <= float(rate) <= 1 for rate in model.detector_rates())


def test_reversed_targets_and_multiple_rules_keep_independent_locations() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    channel = _mixture(("II", "XI"), (0.75, 0.25))
    noise = (
        NoiseModel()
        .add("cx", channel, qubits=(3, 0))
        .add("cx", bit_flip_channel(0.125))
    )
    locations = _circuit_noise_locations(noise, _lowered_program(memory))
    assert locations[0].qubits == (3, 0)
    assert sum(not loc.independent for loc in locations) == 1
    assert sum(loc.independent for loc in locations) == 8
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert model.error_ids is not None
    with pytest.raises(ValueError, match="alternatives"):
        model.to_stim_text()


def test_circuit_sampling_and_reusable_sampler_match_exact_joint_law() -> None:
    memory, noise = _two_location_experiment()
    if importlib.util.find_spec("stim") is None:
        from flagquantum.simulation.stabilizer import StabilizerDependencyError

        with pytest.raises(StabilizerDependencyError):
            sample_memory_circuit(memory, noise=noise, shots=4, seed=47)
        return
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    law = _law(model)
    shots = 40000
    expected_bound = math.sqrt(
        math.log(2 * (2 ** (model.num_detectors + model.num_observables)) / 1e-5)
        / (2 * shots)
    )
    circuit_sample = sample_memory_circuit(memory, noise=noise, shots=shots, seed=47)
    for (d, o), rate in law.items():
        d_bits = torch.zeros(model.num_detectors, dtype=torch.int8)
        d_bits[list(d)] = 1
        o_bits = torch.zeros(model.num_observables, dtype=torch.int8)
        o_bits[list(o)] = 1
        observed = (
            (
                (circuit_sample.detectors == d_bits).all(dim=1)
                & (circuit_sample.observables == o_bits).all(dim=1)
            )
            .double()
            .mean()
        )
        assert abs(float(observed) - rate) <= expected_bound
    with compile_memory_sampler(memory, noise=noise, seed=47) as first:
        batch = first.sample(shots=64)
    with compile_memory_sampler(memory, noise=noise, seed=47) as second:
        assert batch == second.sample(shots=64)


def test_all_fifteen_depolarizing_products_match_the_injected_dem() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=2)
    channel = two_qubit_depolarizing_channel(0.3, dtype=torch.complex128)
    noise = NoiseModel().add("cx", channel, qubits=(0, 3))
    program = _lowered_program(memory)
    expected: dict[tuple[int, tuple[int, ...], tuple[int, ...]], float] = defaultdict(
        float
    )
    location = 0
    for position, instruction in enumerate(program.instructions):
        for _, qubits in noise.channels_for(instruction):
            for letters in product("IXYZ", repeat=2):
                if letters == ("I", "I"):
                    continue
                signature = _executed_flip_set(
                    memory, _injected(program, ((position, qubits, "".join(letters)),))
                )
                if any(signature):
                    expected[(location, *signature)] += 0.3 / 15
            location += 1
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert {
        (e.error_id, e.detectors, e.observables): e.probability for e in model.errors
    } == pytest.approx(dict(expected))


def test_old_x_z_sampler_instructions_remain_bit_identical() -> None:
    from flagquantum.noise import phase_flip_channel
    from flagquantum.qec.execution.sampling import (
        _gate_bound_locations,
        _measurement_plan,
        _noisy_program,
        _positioned_noise_program,
    )

    memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=2)
    plan = _measurement_plan(memory)
    noise = (
        NoiseModel()
        .add("cx", bit_flip_channel(0.123))
        .add("cx", phase_flip_channel(0.071))
    )
    previous = _noisy_program(plan, _gate_bound_locations(plan, noise))
    current = _positioned_noise_program(memory, plan, noise)
    assert len(previous.instructions) == len(current.instructions)
    for old, new in zip(previous.instructions, current.instructions, strict=True):
        assert (old.name, old.wires, old.params, old.metadata) == (
            new.name,
            new.wires,
            new.params,
            new.metadata,
        )
        if old.matrix is None:
            assert new.matrix is None
        else:
            assert torch.equal(torch.stack(old.matrix), torch.stack(new.matrix))
    if importlib.util.find_spec("stim") is not None:
        from flagquantum.simulation.stabilizer import sample_noisy_measurements

        kwargs = dict(shots=128, terminal_qubits=tuple(plan.terminal_columns), seed=47)
        assert torch.equal(
            sample_noisy_measurements(previous, **kwargs),
            sample_noisy_measurements(current, **kwargs),
        )


def test_silent_products_and_duplicate_branches_do_not_renormalize_visible_mass() -> (
    None
):
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    silent = NoiseModel().add("cx", _mixture(("II", "ZI"), (0.5, 0.5)), qubits=(0, 3))
    assert DetectorErrorModel.from_memory_circuit(memory, noise=silent).errors == ()
    channel = _mixture(("II", "XI", "XI", "ZI"), (0.25, 0.125, 0.125, 0.5))
    noise = NoiseModel().add("cx", channel, qubits=(0, 3))
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert len(model.errors) == 1
    assert model.errors[0].probability == pytest.approx(0.25)
    assert model.errors[0].error_id == 0
    program = _lowered_program(memory)
    position = next(
        i
        for i, inst in enumerate(program.instructions)
        if inst.name == "cx" and inst.wires == (0, 3)
    )
    assert (
        model.errors[0].detectors,
        model.errors[0].observables,
    ) == _executed_flip_set(memory, _injected(program, ((position, (0, 3), "XI"),)))


def test_generic_mixtures_preserve_target_order_numerically() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    channel = _mixture(("II", "XI"), (0.75, 0.25))
    models = []
    program = _lowered_program(memory)
    position = next(
        i
        for i, inst in enumerate(program.instructions)
        if inst.name == "cx" and inst.wires == (0, 3)
    )
    for targets in ((0, 3), (3, 0)):
        model = DetectorErrorModel.from_memory_circuit(
            memory, noise=NoiseModel().add("cx", channel, qubits=targets)
        )
        signature = _executed_flip_set(
            memory, _injected(program, ((position, targets, "XI"),))
        )
        assert len(model.errors) == 1
        assert (model.errors[0].detectors, model.errors[0].observables) == signature
        models.append(model)
    assert models[0] != models[1]
