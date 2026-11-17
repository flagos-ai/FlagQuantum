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

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/su2_ct_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/su2_ct_test.py

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum import Circuit
from flagquantum.synth import _rotation_math as mc
from flagquantum.synth import _su2_ct, _zsqrt2, _zw

_GATES = [
    _su2_ct.ISqrt2,
    _su2_ct.SSqrt2,
    _su2_ct.HSqrt2,
    _su2_ct.Tx,
    _su2_ct.Ty,
    _su2_ct.Tz,
]
_SQRT2 = np.sqrt(2)
_LAMBDA = _zsqrt2.ZSqrt2(2, 1)
_LAMBDA_ZW = _zw.ZW.from_pair(_LAMBDA, _zsqrt2.Zero)


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
def test_parametric_form(g: _su2_ct.SU2CliffordT) -> None:
    pf = g.parametric_form()
    got = _su2_ct.SU2CliffordT.from_parametric_form(pf)
    assert got == g


@pytest.mark.parametrize("seq", np.random.default_rng(0).choice(6, size=(10, 5)))
def test_multiply(seq: NDArray[np.integer[Any]]) -> None:
    g = _su2_ct.ISqrt2
    g_numpy = np.eye(2)
    k = 0
    for i in seq:
        g = g @ _GATES[i]
        k += int(i >= 3)  # is a T gate.
        g_numpy = (g_numpy @ _GATES[i].matrix.astype(complex)) / _SQRT2
    assert g.det() == 2 * _LAMBDA**k
    np.testing.assert_allclose(g.matrix.astype(complex) / _SQRT2, g_numpy, atol=1e-9)


@pytest.mark.parametrize(
    "g", tuple(_make_random_su(10, 3, random_cliffords=False, seed=0))
)
def test_adjoint(g: _su2_ct.SU2CliffordT) -> None:
    assert g @ g.adjoint() == _su2_ct.ISqrt2 * _LAMBDA_ZW**3


_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]])
_TX_numpy = (np.eye(2) + (np.eye(2) - 1j * _X) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TY_numpy = (np.eye(2) + (np.eye(2) - 1j * _Y) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TZ_numpy = (np.eye(2) + (np.eye(2) - 1j * _Z) / _SQRT2) / np.sqrt(2 + _SQRT2)


@pytest.mark.parametrize(
    ["g", "g_numpy"],
    [[_su2_ct.Tx, _TX_numpy], [_su2_ct.Ty, _TY_numpy], [_su2_ct.Tz, _TZ_numpy]],
)
def test_t_gates(g: _su2_ct.SU2CliffordT, g_numpy: NDArray[np.complex128]) -> None:
    np.testing.assert_allclose(
        g.matrix.astype(complex) / _SQRT2 / np.sqrt(2 + _SQRT2), g_numpy
    )
    np.testing.assert_allclose(
        g.adjoint().matrix.astype(complex) / _SQRT2 / np.sqrt(2 + _SQRT2),
        g_numpy.T.conjugate(),
    )


@pytest.mark.parametrize(
    "g", tuple(_make_random_su(50, 5, random_cliffords=True, seed=0))
)
@pytest.mark.parametrize("config", [None, mc.NumpyConfig])
def test_rescale(g: _su2_ct.SU2CliffordT, config: mc.MathConfig | None) -> None:
    np.testing.assert_allclose(g.numpy(config), g.rescale().numpy(config))


def test_num_t_gates() -> None:
    from flagquantum.synth._rotation_generation import generate_cliffords

    for clifford in generate_cliffords():
        assert clifford.num_t_gates() == 0

        for t in _su2_ct.Ts:
            assert (clifford @ t).num_t_gates() == 1

    for t1 in _su2_ct.Ts:
        for t2 in _su2_ct.Ts:
            assert (t1 @ t2).num_t_gates() == 2

    for t in _su2_ct.Ts:
        # still two T gates
        assert (t @ t.adjoint()).num_t_gates() == 2

        # We need to call .rescale to remove the common factor and reduce the T count.
        assert (t @ t.adjoint()).rescale().num_t_gates() == 0


_RECIPES: dict[str, tuple[tuple[str, ...], complex]] = {
    "I": ((), 1),
    "H": (("h",), 1j),
    "S": (("s",), np.exp(-1j * np.pi / 4)),
    "X": (("x",), 1j),
    "Y": (("y",), -1j),
    "Z": (("z",), 1j),
    "Tx": (("h", "t", "h"), np.exp(-1j * np.pi / 8)),
    "Ty": (("sdg", "h", "t", "h", "s"), np.exp(-1j * np.pi / 8)),
    "Tz": (("t",), np.exp(-1j * np.pi / 8)),
}
_INVERSE = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}


class _BoundWord:
    def __init__(self, gates: tuple[str, ...]) -> None:
        self.gates = gates

    def append_to(self, circuit: Circuit) -> None:
        for opcode in self.gates:
            circuit.gate(opcode, 0)


@pytest.mark.parametrize(
    "word",
    [
        (),
        *((gate,) for gate in _RECIPES),
        ("S", "Tx", "H", "Ty", "Tz"),
        ("Tx", "Ty*", "Tz**", "S***", "Tx*"),
        ("Tz", "Tz*"),
    ],
)
def test_source_sequences_and_circuit(word: tuple[str, ...]) -> None:
    from qualtran.rotation_synthesis.matrix import _su2_ct as source

    native = _su2_ct.SU2CliffordT.from_sequence(word)
    reference = source.SU2CliffordT.from_sequence(word)
    assert [entry.coords for entry in native.matrix.flat] == [
        entry.coords for entry in reference.matrix.flat
    ]
    assert native.gates == reference.gates
    assert native.num_t_gates() == reference.num_t_gates()
    assert native._key == reference._key
    gates: list[str] = []
    phase = 1 + 0j
    for label in word:
        recipe, factor = _RECIPES[label.rstrip("*")]
        if label.count("*") % 2:
            recipe = tuple(_INVERSE.get(op, op) for op in reversed(recipe))
            factor = factor.conjugate()
        gates.extend(recipe)
        phase *= factor

    columns = []
    for bit in (0, 1):
        circuit = Circuit(1, dtype=torch.complex128, device="cpu")
        if bit:
            circuit.x(0)
        circuit.append(_BoundWord(tuple(gates)))
        instructions = circuit.to_ir().instructions
        assert all(op.wires == (0,) for op in instructions)
        emitted = tuple(op.name for op in instructions)
        assert emitted == ("x",) * bit + tuple(gates)
        assert sum(op in {"t", "tdg"} for op in emitted) == native.num_t_gates()
        columns.append(circuit.state().detach().reshape(-1).numpy())
    physical = phase * np.column_stack(columns)
    np.testing.assert_allclose(physical, reference.numpy(), atol=1e-12)
    np.testing.assert_allclose(physical, native.numpy(mc.NumpyConfig), atol=1e-12)


def test_matrix_helpers_and_exact_phase_selection() -> None:
    import mpmath
    from qualtran.rotation_synthesis._math_config import with_dps
    from qualtran.rotation_synthesis.matrix import _su2_ct as source

    assert _su2_ct._key_map() == source._key_map()
    assert _su2_ct._key_map() is _su2_ct._key_map()
    identity = _su2_ct.ISqrt2
    assert identity + identity == 2 * identity
    assert identity - identity == identity * 0
    assert hash(identity) == hash(_su2_ct.SU2CliffordT(identity.matrix))
    with pytest.raises(AssertionError):
        identity * identity  # type: ignore[operator]
    with pytest.raises(AttributeError):
        _ = identity == 1
    with pytest.raises(AssertionError):
        _su2_ct._gate_from_name("H*X*")
    with pytest.raises(KeyError):
        _su2_ct._gate_from_name("unknown")
    for pick_phase in (False, True):
        for gate in _su2_ct.GATE_MAP.values():
            p, q = gate.matrix[:, 0]
            a, b = (source._zw.ZW(entry.coords) for entry in (p, q))
            got = _su2_ct.SU2CliffordT.from_pair(p, q, pick_phase)
            ref = source.SU2CliffordT.from_pair(a, b, pick_phase)
            assert [v.coords for v in got.matrix.flat] == [
                v.coords for v in ref.matrix.flat
            ]
            with pytest.raises(ValueError):
                _su2_ct.SU2CliffordT.from_pair(_zw.Zero, _zw.Zero, pick_phase)

    with mpmath.workdps(50):
        native = _su2_ct.Tx.numpy(mc.with_dps(50))
        reference = source.Tx.numpy(with_dps(50))
        np.testing.assert_array_equal(native, reference)
