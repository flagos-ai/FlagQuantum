"""The stabilizer engine's exact Pauli readout against dense amplitudes.

`test_stabilizer_engine.py` checks the engine's own contract and
`test_stabilizer_sampling.py` checks its sampled law against a dense amplitude
store. This module checks the third question the representation answers: what a
measured Pauli is worth. The readout is exact, so the reference is the dense
statevector path rather than a second statistical run, and every assertion below
compares against a value built from dense unitaries rather than against a second
copy of the engine's own tables.

Two things are decided here from mathematics instead of being restated:

* the conjugation each gate performs, which is derived from the gate's dense
  unitary acting on every Pauli string on its wires, so a wrong image, a wrong
  power of `i`, or a wrong sign cannot pass by agreeing with itself;
* the direction of the walk, which is separated from its opposite by computing
  both `U^dagger P U` and `U P U^dagger` densely and asserting that the readout
  takes the first while the sample contains a case where the two differ.

The readout is also the one entry point in this package that needs no external
engine, and that is asserted rather than assumed, because it is the difference
between a replacement seam and a first-party capability.

Nothing here reads a lowered measurement node, because the readout answers about
the circuit; the Runtime route that does lower one is tested in
`tests/integration/test_stabilizer_execution_mode.py`.
"""

from __future__ import annotations

import builtins
import itertools
import random
import sys

import numpy as np
import pytest

import flagquantum as fq
from flagquantum.core.ir import MeasurementNode
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.simulation.stabilizer import (
    CLIFFORD_GATE_NAMES,
    PauliReadout,
    pauli_readout,
)

pytestmark = pytest.mark.unit

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

# One generic call per Clifford gate. None of them takes a parameter, so the
# argument count is the gate's arity and the wire list is contiguous from zero:
# that keeps the dense reference a plain Kronecker product instead of a
# wire-permutation exercise, which is a different claim and is tested separately.
_GATE_CALLS: dict[str, tuple[object, ...]] = {
    "i": ("i", 0),
    "x": ("x", 0),
    "y": ("y", 0),
    "z": ("z", 0),
    "h": ("h", 0),
    "s": ("s", 0),
    "sdg": ("sdg", 0),
    "sx": ("sx", 0),
    "sxdg": ("sxdg", 0),
    "cx": ("cx", 0, 1),
    "cy": ("cy", 0, 1),
    "cz": ("cz", 0, 1),
    "swap": ("swap", 0, 1),
}

# The dense reference is built through the public dense path, which is
# `complex64`; a tolerance has to absorb that rather than the readout's own
# arithmetic, which is exact integer bookkeeping.
DENSE_TOLERANCE = 1e-5


def _pauli(label: str) -> np.ndarray:
    """Return the tensor-product Pauli named by `label`, first character highest."""

    matrix = np.array([[1.0]], dtype=complex)
    for character in label:
        matrix = np.kron(matrix, _PAULI_MATRICES[character])
    return matrix


def _axes(label: str) -> dict[str, tuple[int, ...]]:
    """Return the three wire tuples that spell `label` through `pauli_readout`."""

    return {
        axis: tuple(
            index for index, character in enumerate(label) if character.lower() == axis
        )
        for axis in ("x", "y", "z")
    }


def _basis_circuit(n_wires: int, index: int) -> fq.Circuit:
    """Return a circuit in computational basis state `index`, wire 0 highest."""

    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires):
        if (index >> (n_wires - 1 - wire)) & 1:
            circuit = circuit.x(wire)
    return circuit


def _gate_unitary(name: str) -> np.ndarray:
    """Return the gate's dense unitary, built by running `Circuit` on basis states.

    Building it from statevectors rather than from a matrix table keeps this
    reference independent of every constant the engine keeps, which is what makes
    a mismatch here evidence about the engine rather than about two tables
    agreeing.
    """

    call = _GATE_CALLS[name]
    arity = len(call) - 1
    columns = [
        getattr(_basis_circuit(arity, index), str(call[0]))(*call[1:])
        .state()
        .reshape(-1)
        .numpy()
        .astype(complex)
        for index in range(2**arity)
    ]
    return np.stack(columns, axis=1)


def _embed(unitary: np.ndarray, wires: tuple[int, ...], n_wires: int) -> np.ndarray:
    """Return the gate unitary acting on `wires` inside an `n_wires` register.

    Wire 0 is the highest Kronecker factor, which is the order the Pauli labels
    in this module and the engine's own letters both use. The wires have to be
    contiguous and ascending, which every gate placement in this module is; a
    general permutation would be a different claim and is not made here.
    """

    assert wires == tuple(range(wires[0], wires[0] + len(wires))), wires
    left = np.eye(2 ** wires[0], dtype=complex)
    right = np.eye(2 ** (n_wires - wires[-1] - 1), dtype=complex)
    return np.kron(np.kron(left, unitary), right)


def _one_gate_circuit(name: str) -> fq.Circuit:
    call = _GATE_CALLS[name]
    circuit = fq.Circuit(len(call) - 1)
    return getattr(circuit, str(call[0]))(*call[1:])


def _random_clifford(n_wires: int, *, seed: int, layers: int = 3) -> fq.Circuit:
    """Return a shallow Clifford circuit on `n_wires` wires.

    Two-wire gates are drawn only between adjacent wires so the dense reference
    stays cheap, and the layers alternate single-wire rotations with an
    entangling pass so the state is not a product state.
    """

    generator = random.Random(seed)
    circuit = fq.Circuit(n_wires)
    for _ in range(layers):
        for wire in range(n_wires):
            getattr(circuit, generator.choice(("h", "s", "sdg", "x", "z", "sx")))(wire)
        for wire in range(n_wires - 1):
            getattr(circuit, generator.choice(("cx", "cz", "swap")))(wire, wire + 1)
    return circuit


def _dense_conjugations(circuit: fq.Circuit, label: str) -> tuple[complex, complex]:
    """Return `<0|U^dagger P U|0>` and `<0|U P U^dagger|0>` for one dense circuit.

    Both directions are built here from the gate unitaries alone, so the pair is
    an independent statement about which one the readout took.
    """

    ir = circuit.to_ir()
    n_wires = ir.n_wires
    unitary = np.eye(2**n_wires, dtype=complex)
    for instruction in ir.instructions:
        wires = tuple(int(wire) for wire in instruction.wires)
        gate = _embed(_gate_unitary(str(instruction.name)), wires, n_wires)
        unitary = gate @ unitary
    pauli = _pauli(label)
    zero = np.zeros(2**n_wires, dtype=complex)
    zero[0] = 1.0
    backwards = (unitary.conj().T @ pauli @ unitary) @ zero
    forwards = (unitary @ pauli @ unitary.conj().T) @ zero
    return complex(zero.conj() @ backwards), complex(zero.conj() @ forwards)


def _random_label(generator: random.Random, n_wires: int) -> str:
    return "".join(generator.choice("IXYZ") for _ in range(n_wires))


@pytest.mark.parametrize("gate", sorted(_GATE_CALLS))
def test_every_gate_conjugates_every_pauli_as_its_dense_unitary_does(
    gate: str,
) -> None:
    """The images and the `i`-powers are decided from the gate, not restated.

    This is the test a wrong image, a wrong sign convention, or a dropped `Y`
    factor has to survive, and it is exhaustive rather than sampled: every Pauli
    string on the gate's wires, compared as a full operator rather than only
    through its diagonal. `i**phase * letters` is rebuilt from the readout's own
    answer, so the comparison uses no table the engine keeps.
    """

    assert gate in CLIFFORD_GATE_NAMES

    circuit = _one_gate_circuit(gate)
    arity = len(_GATE_CALLS[gate]) - 1
    unitary = _gate_unitary(gate)

    for characters in itertools.product("IXYZ", repeat=arity):
        label = "".join(characters)
        readout = pauli_readout(circuit, **_axes(label))
        expected = unitary.conj().T @ _pauli(label) @ unitary

        assert isinstance(readout, PauliReadout)
        assert readout.letters in {
            "".join(letters) for letters in itertools.product("IXYZ", repeat=arity)
        }, (gate, label, readout)
        assert np.allclose((1j**readout.phase) * _pauli(readout.letters), expected), (
            gate,
            label,
            readout,
        )


@pytest.mark.parametrize("n_wires", (2, 3, 4))
def test_the_readout_matches_the_dense_expectation_on_random_cliffords(
    n_wires: int,
) -> None:
    """The differential test: the value has to be the circuit's own expectation.

    `Circuit.expectation_ps` is the dense path, which holds amplitudes rather
    than a tableau, so the two disagree unless the conjugation is right. A single
    wrong case is reported with its circuit and string, because a failure here
    names a specific gate placement rather than a class of them.
    """

    generator = random.Random(90210 + n_wires)
    mismatches: list[tuple[int, str, float, float]] = []
    traceless = 0

    for trial in range(24):
        circuit = _random_clifford(n_wires, seed=1000 + trial)
        for _ in range(12):
            label = _random_label(generator, n_wires)
            axes = _axes(label)
            readout = pauli_readout(circuit, **axes)
            dense = float(circuit.expectation_ps(**axes).reshape(-1)[0])
            if abs(dense) != 1.0:
                traceless += 1
            if abs(dense - float(readout.expectation)) > DENSE_TOLERANCE:
                mismatches.append((trial, label, dense, float(readout.expectation)))

    assert not mismatches, mismatches[:5]
    # The sample has to contain strings the circuit does not stabilize, or the
    # comparison above only ever checks `1` against `1`.
    assert traceless > 0


def test_the_walk_takes_the_heisenberg_direction_and_not_its_opposite() -> None:
    """Both directions are computed densely, so the choice is evidence.

    A forward walk is the single most likely way to get this wrong, and it
    agrees with the right one on every string the circuit stabilizes. The
    assertion on `separating` is what keeps the check non-vacuous: the sample has
    to contain a program and a string where the two directions differ.
    """

    generator = random.Random(4242)
    separating = 0

    for trial in range(16):
        n_wires = 2 + trial % 3
        circuit = _random_clifford(n_wires, seed=7000 + trial)
        for _ in range(8):
            label = _random_label(generator, n_wires)
            backwards, forwards = _dense_conjugations(circuit, label)
            readout = pauli_readout(circuit, **_axes(label))

            assert abs(backwards - float(readout.expectation)) <= DENSE_TOLERANCE, (
                trial,
                label,
                backwards,
                readout,
            )
            if abs(backwards - forwards) > DENSE_TOLERANCE:
                separating += 1

    assert separating > 0, "no sampled case separates the two directions"


def test_a_conjugated_string_with_no_x_or_y_is_read_with_its_sign() -> None:
    """A negative value is where a dropped `i`-power shows up.

    `Y` on both wires of a Bell pair conjugates to `-Z` on one wire, which is
    `-1` on `|00>`. Every `+1` correlation in the suite above would survive an
    engine that lost the phase, so this is the case that pins it.
    """

    bell = fq.Circuit(2).h(0).cx(0, 1)

    readout = pauli_readout(bell, y=(0, 1))

    assert readout.expectation == -1
    assert readout.phase % 2 == 0
    assert float(bell.expectation_ps(y=(0, 1)).reshape(-1)[0]) == pytest.approx(-1.0)


def test_the_identity_string_is_the_identity_whatever_the_circuit_is() -> None:
    """An empty string carries no letter, so no gate can move it."""

    for circuit in (
        fq.Circuit(1),
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2).s(2),
    ):
        readout = pauli_readout(circuit)

        assert readout.letters == "I" * circuit.n_wires
        assert readout.expectation == 1


def test_a_string_whose_expansion_still_carries_x_is_zero() -> None:
    """The traceless branch, stated as the exact value it is."""

    # `X` on `|0>` is traceless, and no Clifford gate can turn an `X` into a
    # diagonal string unless the circuit supplies the superposition itself.
    assert pauli_readout(fq.Circuit(1), x=(0,)).expectation == 0
    assert pauli_readout(fq.Circuit(2).cx(0, 1), x=(0,)).expectation == 0


def test_the_readout_reaches_wire_counts_no_amplitude_store_can_hold() -> None:
    """The capacity claim: a 1024-wire correlation is one Pauli and one value.

    The readout is exact and holds one string, so the wire count is bounded by
    memory for the string rather than by memory for amplitudes. This is the
    capability the parity contract records as missing from the sampling route:
    an expectation over the representation, at a width no dense state reaches.
    """

    n_wires = 1024
    circuit = fq.Circuit(n_wires).h(0)
    for wire in range(n_wires - 1):
        circuit = circuit.cx(wire, wire + 1)

    parity = pauli_readout(circuit, z=tuple(range(n_wires)))
    assert parity.expectation == 1
    assert len(parity.letters) == n_wires

    # The GHZ correlations are exact rather than statistical. Both strings that
    # stabilize the state read `1` -- `X` on every wire, and `Z` on every wire,
    # which is a stabilizer only at an even width -- while a single `X` and a
    # single `Z` are traceless and two adjacent `X`s are neither.
    assert pauli_readout(circuit, x=tuple(range(n_wires))).expectation == 1
    assert pauli_readout(circuit, z=(0,)).expectation == 0
    assert pauli_readout(circuit, x=(0,)).expectation == 0
    assert pauli_readout(circuit, x=(0, 1)).expectation == 0

    tebibytes = 2**n_wires * 8 / 2**40
    assert tebibytes > 1_000_000


def test_the_readout_needs_no_external_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pauli_readout` is first-party, and this is the assertion that says so.

    The sampling entry points delegate to an optional distribution. The readout
    does not: conjugating a Pauli through a Clifford circuit is integer
    bookkeeping, so making the distribution unimportable has to leave this entry
    point working. That is what puts the readout outside the replacement seam
    rather than inside it.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object):
        if name == "stim" or name.startswith("stim."):
            raise ImportError("No module named 'stim'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "stim", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    bell = fq.Circuit(2).h(0).cx(0, 1)

    assert pauli_readout(bell, z=(0, 1)).expectation == 1
    assert pauli_readout(bell, y=(0, 1)).expectation == -1


def test_the_program_measurement_nodes_are_not_read() -> None:
    """The question is about the circuit, and Runtime strips the nodes it lowers.

    Reading a lowered request would make this entry point a second owner of the
    Runtime contract, so a program carrying one answers the same way as the same
    program without it.
    """

    from dataclasses import replace

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    with_nodes = replace(
        circuit.to_ir(),
        measurements=(MeasurementNode("expectation_z", (0,)),),
    )

    assert pauli_readout(with_nodes, z=(0, 1)) == pauli_readout(circuit, z=(0, 1))


def test_a_non_clifford_gate_is_refused_with_the_gate_named() -> None:
    with pytest.raises(CapabilityError, match=r"instruction 1 't' is not a Clifford"):
        pauli_readout(fq.Circuit(2).h(0).t(1), z=(0,))


def test_a_noise_channel_is_refused() -> None:
    with pytest.raises(CapabilityError, match="is a noise channel"):
        pauli_readout(fq.Circuit(2).h(0).depolarizing(1, 0.1), z=(0,))


@pytest.mark.parametrize(
    ("axis", "wires", "message"),
    [
        ("x", (5,), "is outside a 2-wire program"),
        # The boundary is `n_wires` itself, not `n_wires + 1`: an index equal to
        # the width is one past the last wire, and a check written as `<=` would
        # let it through to an out-of-range write rather than refusing it.
        ("x", (2,), "is outside a 2-wire program"),
        ("z", (-1,), "is outside a 2-wire program"),
        ("y", (True,), "must be integers"),
        ("z", ("a",), "must be integers"),
        # A float is the interesting refusal: `int(0.5)` is `0`, so converting
        # instead of checking the integer protocol would silently read wire 0.
        ("z", (0.5,), "must be integers"),
        ("x", (1.0,), "must be integers"),
    ],
)
def test_a_malformed_string_is_refused(
    axis: str, wires: tuple[object, ...], message: str
) -> None:
    bell = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValidationError, match=message):
        pauli_readout(bell, **{axis: wires})


def test_a_wire_named_on_two_axes_is_refused() -> None:
    """A second letter would silently overwrite the first, which is a typo."""

    bell = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValidationError, match="more than one axis"):
        pauli_readout(bell, x=(0,), z=(0,))
    with pytest.raises(ValidationError, match="more than one axis"):
        pauli_readout(bell, x=(1, 1))
