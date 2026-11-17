#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/clifford_t_repr_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/clifford_t_repr_test.py

from collections.abc import Iterator
from math import pi

import cirq
import numpy as np
import pytest
import torch

from flagquantum import Circuit
from flagquantum.ecosystem import _quirk_rotation
from flagquantum.ecosystem.cirq._rotation_export import to_cirq
from flagquantum.synth import _clifford_t_repr as ctr
from flagquantum.synth import _su2_ct
from flagquantum.synth._clifford_t_operation import CliffordTOperation
from flagquantum.synth._rotation_generation import generate_rotations_iter


def _make_random_su(
    n: int, m: int, random_cliffords: bool = False, seed: int | None = None
) -> Iterator[_su2_ct.SU2CliffordT]:
    rng = np.random.default_rng(seed)
    gates = [_su2_ct.Tx, _su2_ct.Ty, _su2_ct.Tz]
    if random_cliffords:
        gates += [_su2_ct.SSqrt2, _su2_ct.HSqrt2]
    for _ in range(n):
        res = _su2_ct.ISqrt2
        for i in rng.choice(len(gates), m):
            res = res @ gates[i]
        yield res


@pytest.mark.parametrize(
    "g", tuple(_make_random_su(50, 5, random_cliffords=True, seed=0))
)
def test_to_xyz_seq(g: _su2_ct.SU2CliffordT) -> None:
    seq = ctr.to_sequence(g, "xyz")
    assert not any("*" in g for g in seq)
    got = _su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize(
    "g", tuple(_make_random_su(50, 5, random_cliffords=True, seed=0))
)
def test_to_xz_seq(g: _su2_ct.SU2CliffordT) -> None:
    g = g.rescale()
    seq = ctr.to_sequence(g, "xz")
    assert not any("Ty" in g for g in seq)
    first_t = None
    for i, label in enumerate(seq):
        if label.startswith("T"):
            first_t = i
            break
    if first_t is not None:
        ts = "Tx", "Tx*", "Tz", "Tz*"
        assert all(s in ts for s in seq[first_t:-2]), f"{seq=}"
    got = _su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize(
    "g", tuple(_make_random_su(50, 5, random_cliffords=True, seed=0))
)
def test_to_cirq(g: _su2_ct.SU2CliffordT) -> None:
    circuit = cirq.Circuit(to_cirq(g, "xyz"))
    unitary = cirq.unitary(circuit)
    u = g.matrix.astype(complex)
    u = u / np.linalg.det(u) ** 0.5
    assert cirq.protocols.equal_up_to_global_phase(u, unitary)


@pytest.mark.parametrize("fmt", ["xyz", "xz"])
def test_locked_frontier_gate_and_export_order(fmt: str) -> None:
    from qualtran.rotation_synthesis.matrix import _clifford_t_repr as locked
    from qualtran.rotation_synthesis.matrix import _generation as source

    qubit = cirq.NamedQubit("target")
    frontier = generate_rotations_iter(2, with_progress_bar=False)
    reference = source.generate_rotations_iter(2, with_progress_bar=False)
    unscaled_failures = 0
    for native_level, locked_level in zip(frontier, reference, strict=True):
        for native, original in zip(native_level, locked_level, strict=True):
            got = ctr.to_sequence(native, fmt)
            want = locked.to_sequence(original, fmt)
            assert got == want
            if got is None:
                unscaled_failures += 1
                native = native.rescale()
                original = original.rescale()
                got = ctr.to_sequence(native, fmt)
                assert got == locked.to_sequence(original, fmt)
                assert got is not None
            assert _su2_ct.SU2CliffordT.from_sequence(got) == native
            assert to_cirq(native, fmt, qubit) == locked.to_cirq(original, fmt, qubit)
            for allow_phase in (False, True):
                assert _quirk_rotation.to_quirk(
                    native, fmt, allow_phase
                ) == locked.to_quirk(original, fmt, allow_phase)

    assert unscaled_failures == (24 if fmt == "xz" else 0)


def test_export_token_admission_and_negative_clifford() -> None:
    from qualtran.rotation_synthesis.matrix import _clifford_t_repr as locked

    for name in ("I", "X", "Y", "Z", "H", "S", "S*", "Tx", "Tz*", "Ty*"):
        for allow_phase in (True, False):
            assert _quirk_rotation._to_quirk_name(
                name, allow_phase
            ) == locked._to_quirk_name(name, allow_phase)
    with pytest.raises(ValueError, match="not supported"):
        _quirk_rotation._to_quirk_name("CZ")
    with pytest.raises(ValueError, match="not supported"):
        ctr.to_sequence(_su2_ct.ISqrt2, "bad")
    assert ctr.to_sequence(-_su2_ct.ISqrt2) == ("Z", "X", "Z", "X")


# Name, physical opcodes, and normalization in eighth turns of pi.
_BINDING: dict[str, tuple[tuple[str, ...], int]] = {
    "H": (("h",), 4),
    "S": (("s",), -2),
    "Z": (("z",), 4),
    "X": (("x",), 4),
    "Tx": (("h", "t", "h"), -1),
    "Ty": (("sdg", "h", "t", "h", "s"), -1),
    "Tz": (("t",), -1),
    "Tx*": (("h", "tdg", "h"), 1),
    "Tz*": (("tdg",), 1),
}


@pytest.mark.parametrize(
    "word,negated",
    [
        ((), False),
        ((), True),
        (("Tx",), False),
        (("Ty",), False),
        (("Tz",), False),
        (("Tz*",), False),
        (("Tx", "Tx"), False),
        (("Tx", "S", "Ty", "Tz"), True),
    ],
)
@pytest.mark.parametrize("fmt", ["xyz", "xz"])
def test_native_append_full_operator(
    word: tuple[str, ...], negated: bool, fmt: str
) -> None:
    from qualtran.rotation_synthesis.matrix import _clifford_t_repr as locked
    from qualtran.rotation_synthesis.matrix import _su2_ct as source

    original = source.SU2CliffordT.from_sequence(word)
    matrix = _su2_ct.SU2CliffordT.from_sequence(word)
    if fmt == "xz":
        original = original.rescale()
    if negated:
        original, matrix = -original, -matrix

    expected: list[str] = []
    phase = 0
    for label in locked.to_sequence(original, fmt):
        opcodes, correction = _BINDING[label]
        expected.extend(opcodes)
        phase += correction
    if phase % 16:
        expected.extend(("x", "phase", "x", "phase"))

    expected_operator = np.kron(np.eye(2), original.numpy())
    for basis in range(4):
        inputs = torch.zeros(4, dtype=torch.complex128)
        inputs[basis] = 1
        circuit = Circuit(2, inputs=inputs, dtype=torch.complex128, device="cpu")
        circuit.append(CliffordTOperation(matrix, 1, fmt))
        instructions = circuit.to_ir().instructions
        assert [inst.name for inst in instructions] == expected
        assert all(inst.wires == (1,) for inst in instructions)
        if phase % 16:
            assert instructions[-1].params["theta"] == ((phase + 8) % 16 - 8) * pi / 8
        assert (
            sum(inst.name in ("t", "tdg") for inst in instructions)
            == original.num_t_gates()
        )
        np.testing.assert_allclose(
            circuit.state().detach().reshape(-1).numpy(),
            expected_operator[:, basis],
            atol=1e-12,
        )


def test_bound_wire_is_valid_even_for_identity() -> None:
    with pytest.raises(ValueError, match="outside the circuit"):
        Circuit(1).append(CliffordTOperation(_su2_ct.ISqrt2, 1))
