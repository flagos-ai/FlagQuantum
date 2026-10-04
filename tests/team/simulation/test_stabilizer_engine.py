"""The stabilizer engine's own contract: gate set, bit order, and refusal.

Every test here is about something the engine promises independently of physics:
which gates it accepts, whose wire a sample column belongs to, which seed it
accepts, and what it does when it cannot answer. The differential check that its
samples carry the right distribution lives in `test_stabilizer_sampling.py`,
because that one has to build a dense amplitude store to compare against.

The gate set is checked behaviourally rather than by restating the table. A gate
is Clifford exactly when conjugating a Pauli generator by it yields a single Pauli
string up to a phase, and that property is decided here from dense unitaries, so
the accepted set is derived from mathematics instead of from the module under
test agreeing with a second hand-written list.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import itertools
import sys
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.observables as observables
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.errors import CapabilityError, FlagQuantumError, ValidationError
from flagquantum.simulation.stabilizer import (
    CLIFFORD_GATE_NAMES,
    StabilizerDependencyError,
    sample_stabilizer,
)

# Stim is the backend the engine delegates to and an optional distribution, and
# the gate set below is decided from dense unitaries. The core environment
# installs no stim, so the file skips rather than failing to import; NumPy
# arrives with stim and is imported below the same guard for that reason.
pytest.importorskip("stim")

import numpy as np

pytestmark = pytest.mark.unit

PACKAGE = (
    Path(__file__).resolve().parents[3] / "flagquantum" / "simulation" / "stabilizer"
)

# One generic representative per non-channel opcode the schema declares. The
# angle is not a Clifford angle for any of the parameterized gates, and `ccx` and
# `cswap` have no angle at all, so each entry's classification is unambiguous.
GENERIC_GATES: dict[str, tuple[str, ...]] = {
    "i": ("i", 0),
    "x": ("x", 0),
    "y": ("y", 0),
    "z": ("z", 0),
    "h": ("h", 0),
    "s": ("s", 0),
    "sdg": ("sdg", 0),
    "sx": ("sx", 0),
    "sxdg": ("sxdg", 0),
    "t": ("t", 0),
    "tdg": ("tdg", 0),
    "rx": ("rx", 0, 0.3),
    "ry": ("ry", 0, 0.3),
    "rz": ("rz", 0, 0.3),
    "phase": ("phase", 0, 0.3),
    "u1": ("u1", 0, 0.3),
    "u2": ("u2", 0, 0.3, 0.7),
    "u3": ("u3", 0, 0.3, 0.7, 1.1),
    "cx": ("cx", 0, 1),
    "cy": ("cy", 0, 1),
    "cz": ("cz", 0, 1),
    "swap": ("swap", 0, 1),
    "crx": ("crx", 0, 1, 0.3),
    "cry": ("cry", 0, 1, 0.3),
    "crz": ("crz", 0, 1, 0.3),
    "cphase": ("cphase", 0, 1, 0.3),
    "rxx": ("rxx", 0, 1, 0.3),
    "ryy": ("ryy", 0, 1, 0.3),
    "rzz": ("rzz", 0, 1, 0.3),
    "ccx": ("ccx", 0, 1, 2),
    "cswap": ("cswap", 0, 1, 2),
}

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PHASES = (1.0, -1.0, 1.0j, -1.0j)
_LABELS = {
    width: tuple("".join(chars) for chars in itertools.product("IXYZ", repeat=width))
    for width in (1, 2, 3)
}


def _pauli(label: str) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=complex)
    for character in label:
        matrix = np.kron(matrix, _PAULI_MATRICES[character])
    return matrix


def _gate_unitary(name: str) -> tuple[int, np.ndarray]:
    """Return the gate's wire count and dense unitary, built through `Circuit`."""

    call = GENERIC_GATES[name]
    n_wires = get_operator_schema(name).arity
    columns = [
        getattr(_basis_circuit(n_wires, index), call[0])(*call[1:])
        .state()
        .reshape(-1)
        .numpy()
        for index in range(2**n_wires)
    ]
    return n_wires, np.stack(columns, axis=1)


def _pauli_normalization_witness(unitary: np.ndarray) -> str | None:
    """Return a Pauli generator the gate does not map to a single Pauli string.

    `None` means the gate is Clifford: every non-identity Pauli string on the
    gate's wires conjugates to one Pauli string up to a power of `i`. The first
    generator that does not is returned, so a failure says which one.
    """

    n_wires = int(round(np.log2(unitary.shape[0])))
    labels = _LABELS[n_wires]
    for label in labels:
        if label == "I" * n_wires:
            continue
        image = unitary @ _pauli(label) @ unitary.conj().T
        for other in labels:
            target = _pauli(other)
            if any(np.allclose(image, phase * target) for phase in _PHASES):
                break
        else:
            return label
    return None


def _basis_circuit(n_wires: int, index: int) -> fq.Circuit:
    """Return a circuit in computational basis state `index`, wire 0 most significant."""

    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires):
        if (index >> (n_wires - 1 - wire)) & 1:
            circuit = circuit.x(wire)
    return circuit


def test_every_non_channel_opcode_is_classified_by_pauli_normalization() -> None:
    """The accepted set is exactly the Clifford subset of the operator schema.

    This is the test the gate set has to survive, and it derives the answer
    instead of restating it. A gate added to the engine that does not normalize
    the Pauli group fails here even if it were added to the expected list too; a
    Clifford gate the schema declares but the engine refuses fails here as well,
    because it has no witness.
    """

    declared = {name for name, schema in OPERATOR_SCHEMAS.items() if not schema.channel}
    missing = declared - set(GENERIC_GATES)
    assert not missing, f"no generic gate is recorded for {sorted(missing)}"
    assert not set(GENERIC_GATES) - declared

    accepted = {
        name
        for name in sorted(GENERIC_GATES)
        if _pauli_normalization_witness(_gate_unitary(name)[1]) is None
    }

    assert accepted == CLIFFORD_GATE_NAMES
    assert accepted == {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "cx",
        "cy",
        "cz",
        "swap",
    }


def test_the_two_higher_level_permutations_are_refused_with_a_witness() -> None:
    """`ccx` and `cswap` are the two a reader is most likely to assume belong.

    Both look like the rest of the permutation group. A Toffoli's induced basis
    map is not GF(2)-linear and a Fredkin's reads `b + a(b + c)`, so neither
    normalizes the Pauli group; the witness says which generator proves it. Both
    belong to a higher level of the Clifford hierarchy, which is a different
    simulation regime with a different capacity curve, so translating either into
    this engine would report samples from a circuit it did not execute.
    """

    for name in ("ccx", "cswap"):
        assert _pauli_normalization_witness(_gate_unitary(name)[1]) is not None, name
        call = GENERIC_GATES[name]
        circuit = getattr(fq.Circuit(3), call[0])(*call[1:])
        with pytest.raises(CapabilityError, match=name):
            sample_stabilizer(circuit, shots=2, seed=1)


def test_a_bit_flip_samples_its_own_wire_in_the_output_column_for_that_wire() -> None:
    """Wire identity in the output is the one thing a caller cannot recompute."""

    samples = sample_stabilizer(fq.Circuit(3).x(1), shots=4, seed=1)

    assert samples.dtype == torch.int64
    assert samples.shape == (4, 3)
    assert samples.tolist() == [[0, 1, 0]] * 4


def test_the_requested_wire_order_is_the_output_column_order() -> None:
    """`wires` orders the columns, so it is not a set.

    Measuring `[2, 1, 0]` puts wire 2 in the first column. A caller that read the
    columns as ascending wires would otherwise read a bit-reversed shot.
    """

    circuit = fq.Circuit(3).x(1)

    assert (
        sample_stabilizer(circuit, shots=2, wires=[2, 1, 0], seed=1).tolist()
        == [[0, 1, 0]] * 2
    )
    assert (
        sample_stabilizer(circuit, shots=2, wires=[1, 0], seed=1).tolist()
        == [[1, 0]] * 2
    )
    assert sample_stabilizer(circuit, shots=2, wires=[1], seed=1).tolist() == [[1]] * 2


def test_a_measured_wire_subset_leaves_the_other_wires_unmeasured() -> None:
    """A deterministic circuit makes the subset visible without sampling luck."""

    circuit = fq.Circuit(4).x(0).x(3)

    assert (
        sample_stabilizer(circuit, shots=3, wires=[0, 3], seed=2).tolist()
        == [[1, 1]] * 3
    )


def test_a_repeated_seed_reproduces_the_same_samples() -> None:
    """Determinism holds for a seed on one engine version and one machine.

    Only the reproducible half is asserted, because the module documents that a
    seed does not pin a bit pattern across machines or engine versions.
    """

    circuit = fq.Circuit(4).h(0).cx(0, 1).cx(1, 2).h(3)

    first = sample_stabilizer(circuit, shots=64, seed=20260930)
    assert torch.equal(first, sample_stabilizer(circuit, shots=64, seed=20260930))


def test_both_ends_of_the_engine_seed_range_are_accepted() -> None:
    """The range check is a bound, not a rejection of the boundary values."""

    circuit = fq.Circuit(2).h(0).cx(0, 1)

    for seed in (0, 2**64 - 1):
        assert sample_stabilizer(circuit, shots=4, seed=seed).shape == (4, 2)


@pytest.mark.parametrize(
    "opcode",
    [
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "phase",
        "u1",
        "u2",
        "u3",
        "crx",
        "cry",
        "crz",
        "cphase",
        "rxx",
        "ryy",
        "rzz",
        "ccx",
        "cswap",
    ],
)
def test_a_non_clifford_instruction_is_refused_by_opcode(opcode: str) -> None:
    """The refusal names the opcode, because that is how a caller finds it.

    The schema's non-channel opcodes divide into the accepted set and this list,
    and the two were shown to be exactly the Clifford and non-Clifford parts
    above, so this test is the same classification boundary read through the
    public entry point.
    """

    gate = GENERIC_GATES[opcode]
    circuit = getattr(fq.Circuit(3), gate[0])(*gate[1:])

    with pytest.raises(CapabilityError, match=opcode):
        sample_stabilizer(circuit, shots=2, seed=1)


def test_the_refusal_names_the_gate_set_it_accepts() -> None:
    """A refusal that does not say what would work forces the caller to guess."""

    with pytest.raises(CapabilityError) as info:
        sample_stabilizer(fq.Circuit(2).t(0), shots=2, seed=1)

    message = str(info.value)
    for name in sorted(CLIFFORD_GATE_NAMES):
        assert name in message
    assert "fails closed rather than approximating" in message


def test_a_noise_channel_is_refused_rather_than_sampled_as_a_unitary() -> None:
    """A channel would return shots from a different circuit, so it fails closed."""

    with pytest.raises(CapabilityError, match="noise channel"):
        sample_stabilizer(fq.Circuit(2).h(0).depolarizing(1, 0.1), shots=2, seed=1)


def test_a_program_carrying_lowered_measurement_nodes_is_refused() -> None:
    """Reconciling lowered outputs with a caller's wires is a Runtime decision.

    A lowered node already states which wires to measure and how many shots to
    take. Accepting both that and a `wires=`/`shots=` argument would leave two
    answers to one question and no rule for which wins.
    """

    base = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    nodes = observables.lower_outputs(fq.samples(), n_qubits=2, shots=10)
    assert nodes
    lowered = dataclasses.replace(base, measurements=nodes)
    lowered.validate()

    with pytest.raises(CapabilityError, match="lowered measurement nodes") as info:
        sample_stabilizer(lowered, shots=10, seed=1)

    assert "sample" in str(info.value)


@pytest.mark.parametrize("shots", [0, -1, 4.0, True, "4"])
def test_the_shot_count_must_be_a_positive_integer(shots: object) -> None:
    """A shot count that is not a count would run a different number of shots."""

    with pytest.raises((ValidationError, TypeError)):
        sample_stabilizer(fq.Circuit(2).h(0), shots=shots, seed=1)


def test_the_shot_count_is_required() -> None:
    """Sampling is the whole capability, so there is no sensible default."""

    with pytest.raises(TypeError):
        sample_stabilizer(fq.Circuit(2).h(0), seed=1)


def test_an_explicit_none_shot_count_is_refused_before_it_reaches_the_engine() -> None:
    """The annotation is `int`, and Python does not enforce it.

    `None` is the one value the Core measurement contract admits as "unstated", so
    a caller reaching this entry point from untyped code would otherwise have it
    travel as far as the engine's own sampler call and fail there with a message
    about the sampler rather than about the request.
    """

    with pytest.raises(ValidationError, match="positive shot count"):
        sample_stabilizer(fq.Circuit(2).h(0), shots=None)


def test_omitting_the_seed_samples_without_one() -> None:
    """A seed is optional, and its absence is not seed zero.

    An implementation that defaulted the seed to `0` would make every unseeded
    call reproducible, which is a different promise than the module makes, so the
    unseeded path is exercised rather than only the seeded one.
    """

    circuit = fq.Circuit(3).h(0).cx(0, 1)

    assert sample_stabilizer(circuit, shots=32).shape == (32, 3)
    assert sample_stabilizer(circuit, shots=32, seed=None).shape == (32, 3)


def test_unseeded_calls_do_not_agree_with_each_other() -> None:
    """The shape check above cannot tell an unseeded draw from a seeded one.

    `seed=0` is a legal seed, so defaulting an absent seed to it produces the
    same shapes as drawing from entropy and is invisible to every deterministic
    assertion. What separates the two is that one repeats: four unseeded draws of
    a superposition must not all be the same bit pattern. The failure this pins
    is a real one -- it would make every unseeded call reproducible while the
    documentation says only a seeded one is.

    One run of this test draws 256 independent bits in four groups, so the
    probability that a correct implementation fails it by coincidence is
    `2**-192`.
    """

    circuit = fq.Circuit(1).h(0)
    draws = {
        tuple(
            int(bit)
            for row in sample_stabilizer(circuit, shots=64).tolist()
            for bit in row
        )
        for _ in range(4)
    }

    assert len(draws) > 1, (
        "four unseeded draws of a superposition returned the same 64 outcomes, so "
        "an absent seed is being replaced by a fixed one"
    )


@pytest.mark.parametrize("wires", [[2], [5], [0, 9]])
def test_a_wire_outside_the_circuit_is_refused(wires: list[int]) -> None:
    """A wire the circuit does not have is not measurable, not silently ignored."""

    with pytest.raises(ValidationError, match="outside circuit range"):
        sample_stabilizer(fq.Circuit(2).h(0), shots=2, wires=wires, seed=1)


def test_an_empty_wire_list_is_refused() -> None:
    """Zero measured wires is not a sampling request."""

    with pytest.raises(ValidationError):
        sample_stabilizer(fq.Circuit(2).h(0), shots=2, wires=[], seed=1)


@pytest.mark.parametrize("seed", [1.5, True, "7", -1, 2**64, 2**64 + 1])
def test_the_seed_must_be_an_integer_the_engine_accepts(seed: object) -> None:
    """A seed outside the engine's range is refused, never truncated or wrapped."""

    with pytest.raises(ValidationError, match="seed"):
        sample_stabilizer(fq.Circuit(2).h(0), shots=2, seed=seed)


def test_a_missing_engine_names_the_extra_that_provides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed, and tell the caller the one command that fixes it.

    The blocked import is what a caller without the extra actually sees, so the
    test replaces `__import__` rather than asserting on a message string alone.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object):
        if name == "stim" or name.startswith("stim."):
            raise ImportError("No module named 'stim'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "stim", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(StabilizerDependencyError, match=r"flagquantum\[stim\]") as info:
        sample_stabilizer(fq.Circuit(2).h(0), shots=4, seed=1)

    # A caller that catches either the package's error hierarchy or the standard
    # import failure has to be able to.
    assert isinstance(info.value, CapabilityError)
    assert isinstance(info.value, ImportError)
    assert isinstance(info.value, FlagQuantumError)


def test_exactly_one_module_in_the_package_names_the_engine() -> None:
    """The single import site is what makes the engine replaceable.

    `flagquantum/simulation/stabilizer/README.md` states that `engine.py` owns the
    only `import stim`, and the replacement argument for the whole dependency
    rests on it. A second import site would spread the seam without failing any
    other test, so the package's own source is parsed here: no module may import
    the engine eagerly, and exactly one may name it through the deferred loader.
    """

    sources = sorted(PACKAGE.glob("*.py"))
    assert sources, "the package has no source files"
    deferred: list[str] = []
    eager: list[str] = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                eager += [
                    path.name
                    for alias in node.names
                    if alias.name.split(".")[0] == "stim"
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "stim"
            ):
                eager.append(path.name)
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "stim"
            ):
                deferred.append(path.name)

    # Deferred, so the engine never reaches the `import flagquantum` path...
    assert eager == []
    # ...and named in exactly one module, which is the seam a replacement swaps.
    assert deferred == ["engine.py"]
