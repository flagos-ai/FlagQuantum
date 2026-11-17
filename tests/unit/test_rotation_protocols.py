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

# Adapted from Qualtran v0.7.0 rotation_synthesis/protocols/clifford_t_synthesis_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/protocols/clifford_t_synthesis_test.py

from collections.abc import Iterator

import mpmath
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from flagquantum.synth import _rotation_math as mc
from flagquantum.synth._rotation_protocols import _clifford_t_synthesis


@pytest.fixture
def config40() -> Iterator[mc.MathConfig]:
    with mpmath.workdps(40):
        yield mc.with_dps(40)


@pytest.fixture
def config50() -> Iterator[mc.MathConfig]:
    with mpmath.workdps(50):
        yield mc.with_dps(50)


def _make_tests(n_points: int, seed: int) -> Iterator[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    for _ in range(n_points):
        theta = rng.random() * 2 * np.pi
        error = np.power(10, -1 - 4 * rng.random())  # 1e-4 <= eps <= 1e-1
        yield float(theta), float(error)


@pytest.mark.parametrize(["theta", "error"], tuple(_make_tests(20, 0)))
def test_diagonal_protocol(theta: float, error: float, config40: mc.MathConfig) -> None:
    res = _clifford_t_synthesis.diagonal_unitary_approx(theta, error, 100, config40)
    assert res is not None
    assert res.n <= (-3.02 * np.log2(error) + 1.77) * 1.3
    assert res.diamond_norm_distance_to_rz(theta, config40) <= error


@pytest.mark.parametrize(["theta", "error"], tuple(_make_tests(20, 0)))
def test_fallback_protocol(theta: float, error: float, config40: mc.MathConfig) -> None:
    res = _clifford_t_synthesis.fallback_protocol(
        theta, error, 0.99, 100, config40, eps_rotation=error
    )
    assert res is not None
    assert res.expected_num_ts(config40) <= (-1.03 * np.log2(error) + 5.75) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, config40)
    assert actual_error <= error


@pytest.mark.parametrize(["theta", "error"], tuple(_make_tests(20, 0)))
def test_mixed_diagonal_protocol(
    theta: float, error: float, config40: mc.MathConfig
) -> None:
    res = _clifford_t_synthesis.mixed_diagonal_protocol(theta, error, 100, config40)
    assert res is not None
    assert res.expected_num_ts(config40) <= (-1.52 * np.log2(error) - 0.01) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, config40)
    assert actual_error <= error


@pytest.mark.parametrize(["theta", "error"], tuple(_make_tests(20, 0)))
def test_mixed_fallback_protocol(
    theta: float, error: float, config40: mc.MathConfig
) -> None:
    res = _clifford_t_synthesis.mixed_fallback_protocol(
        theta, error, 0.99, 100, config40, fallback_min_eps=1e-5, num_valid_points=2
    )
    if res is None:
        return
    assert res.expected_num_ts(config40) <= (-0.53 * np.log2(error) + 5) * 1.3
    actual_error = res.diamond_norm_distance_to_rz(theta, config40)
    assert actual_error <= error


@pytest.mark.parametrize(
    ["u", "eps"],
    tuple(
        zip(stats.unitary_group(2).rvs(10, 0), np.linspace(1e-3, 1e-1, 10), strict=True)
    ),
)
def test_magnitude_approx(
    u: NDArray[np.complex128], eps: float, config50: mc.MathConfig
) -> None:
    config = config50
    res = _clifford_t_synthesis.magnitude_approx(u, eps, 50, config)
    assert res is not None
    assert res.diamond_norm_distance_to_unitary(u, config) <= eps


def _channel_signature(channel: object) -> object:
    # Both the native and locked models expose these exact ring fields.
    name = type(channel).__name__
    if name == "UnitaryChannel":
        return (
            name,
            getattr(channel, "p").coords,
            getattr(channel, "q").coords,
            getattr(channel, "n"),
            getattr(channel, "twirl"),
        )
    if name == "ProjectiveChannel":
        return (
            name,
            _channel_signature(getattr(channel, "rotation")),
            _channel_signature(getattr(channel, "correction")),
        )
    if name == "ProbabilisticChannel":
        return (
            name,
            _channel_signature(getattr(channel, "c1")),
            _channel_signature(getattr(channel, "c2")),
            getattr(channel, "probability"),
        )
    assert channel is None
    return None


def test_locked_protocol_channels_and_resources(config40: mc.MathConfig) -> None:
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis import protocols as source

    from flagquantum.synth import _rotation_protocols as native

    reference_config = source_math.with_dps(40)
    for name, positional in [
        ("diagonal_unitary_approx", (0.9, 0.03)),
        ("fallback_protocol", (0.9, 0.03, 0.99)),
        ("mixed_diagonal_protocol", (0.9, 0.03)),
        ("mixed_fallback_protocol", (0.9, 0.03, 0.99)),
    ]:
        got = getattr(native, name)(*positional, 100, config40)
        expected = getattr(source, name)(*positional, 100, reference_config)
        assert got is not None and expected is not None
        assert _channel_signature(got) == _channel_signature(expected)
        assert got.expected_num_ts(config40) == expected.expected_num_ts(
            reference_config
        )
        assert got.diamond_norm_distance_to_rz(0.9, config40) == (
            expected.diamond_norm_distance_to_rz(0.9, reference_config)
        )
    matrix = stats.unitary_group(2).rvs(1, 19)
    got = native.magnitude_approx(matrix, 0.05, 70, config40)
    expected = source.magnitude_approx(matrix, 0.05, 70, reference_config)
    assert got is not None and expected is not None
    assert _channel_signature(got) == _channel_signature(expected)


def test_locked_bounded_search_exhaustion(config40: mc.MathConfig) -> None:
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis import protocols as source

    from flagquantum.synth import _rotation_protocols as native

    assert set(native.__all__) == {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and callable(value)
    }
    for owner, config in ((native, config40), (source, source_math.with_dps(40))):
        assert owner.diagonal_unitary_approx(0.37, 0.001, 0, config) is None
        with pytest.raises(TypeError, match="not iterable"):
            owner.fallback_protocol(0.37, 0.001, 0.99, 0, config)
        with pytest.raises(TypeError, match="cannot unpack"):
            owner.mixed_diagonal_protocol(0.37, 0.001, 0, config)
        with pytest.raises(ValueError, match="no candidates for an under rotation"):
            owner.mixed_fallback_protocol(0.37, 0.001, 0.99, 0, config)


def test_locked_ordered_regions_and_collectors(config40: mc.MathConfig) -> None:
    import itertools

    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis import protocols as source
    from qualtran.rotation_synthesis import rings as source_rings

    from flagquantum.synth import _rings as rings
    from flagquantum.synth import _rotation_protocols as native

    ref_config = source_math.with_dps(40)
    for name, parameters in [
        ("Diagonal", (0.37, 0.08, 10)),
        ("MixedDiagonal", (0.37, 0.08, 10)),
        ("Fallback", (0.37, 0.9, 0.08, 10)),
    ]:
        model = getattr(native, name)(*parameters)
        locked = getattr(source, name)(*parameters)
        actual_stream = itertools.islice(model.get_points(config40, verbose=False), 24)
        locked_stream = itertools.islice(
            locked.get_points(ref_config, verbose=False), 24
        )
        count = 0
        for (n, point), (rn, want) in zip(actual_stream, locked_stream, strict=True):
            count += 1
            assert (n, point.coords) == (rn, want.coords)
            assert model.make_real_bound_fn(n, config40)(
                point
            ) == locked.make_real_bound_fn(rn, ref_config)(want)
        assert count > 0

    collector = native.SplitRegionCollector((1, 0, 0), target_count=2)
    expected = source.SplitRegionCollector((1, 0, 0), target_count=2)
    assert collector.status() == expected.status()
    for coordinate in (0, 1, 1, 0, -1):
        collector.add_point(
            rings.ZW((coordinate, 0, 0, 0)), rings.ZW((0, 0, 0, 0)), 0, config40
        )
        expected.add_point(
            source_rings.ZW((coordinate, 0, 0, 0)),
            source_rings.ZW((0, 0, 0, 0)),
            0,
            ref_config,
        )
        assert collector.status() == expected.status()
        assert collector.is_done() == expected.is_done()
    for got, want in zip(collector.result(), expected.result(), strict=True):
        assert [_channel_signature(item) for item in got] == [
            _channel_signature(item) for item in want
        ]


def test_synthesized_projective_append_and_full_branch_parity(
    config40: mc.MathConfig,
) -> None:
    import torch
    from qualtran.rotation_synthesis import _math_config as source_math
    from qualtran.rotation_synthesis import protocols as source

    from flagquantum import Circuit
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.synth._rotation_channel_operation import (
        ProjectiveChannelOperation,
        UnitaryChannelOperation,
    )
    from tests.unit.measurement_checks import forced_branch

    expected = source.fallback_protocol(0.9, 0.03, 0.99, 100, source_math.with_dps(40))
    got = _clifford_t_synthesis.fallback_protocol(0.9, 0.03, 0.99, 100, config40)
    assert got is not None and expected is not None
    assert _channel_signature(got) == _channel_signature(expected)
    rotation = expected.rotation.to_matrix().numpy()
    correction = expected.correction.to_matrix().numpy()
    diagonal = np.diag(np.diag(rotation))
    branches = (diagonal, correction @ (rotation - diagonal))

    dynamic = DynamicCircuit(2, dtype=torch.complex128, device="cpu")
    dynamic.append(ProjectiveChannelOperation(got, 0, 1, 3, "xyz"))
    instructions = dynamic.to_ir().instructions
    boundary = next(i for i, g in enumerate(instructions) if g.name == "measure")
    assert instructions[boundary].wires == (1,)
    assert instructions[boundary].metadata["classical_bit"] == 3
    assert all(
        g.metadata["conditions"] == ((3, 1),) for g in instructions[boundary + 1 :]
    )
    assert [(g.name, g.wires) for g in instructions if g.name == "cx"] == [
        ("cx", (0, 1)),
        ("cx", (0, 1)),
    ]
    rotation_t = sum(g.name in ("t", "tdg") for g in instructions[:boundary])
    correction_t = sum(g.name in ("t", "tdg") for g in instructions[boundary + 1 :])
    assert (
        (rotation_t, correction_t)
        == (expected.rotation.n, expected.correction.n)
        == (9, 9)
    )
    assert float(got.expected_num_ts(config40)) == pytest.approx(9.014133524538984)

    for col in (0, 1):
        initial = torch.zeros(4, dtype=torch.complex128)
        initial[2 * col] = 1
        for outcome, operator in enumerate(branches):
            branch = forced_branch(initial, instructions, (outcome,)).detach().numpy()
            np.testing.assert_allclose(branch[::2], operator[:, col], atol=1e-12)
            np.testing.assert_array_equal(branch[1::2], 0)
        initial_single = torch.zeros(2, dtype=torch.complex128)
        initial_single[col] = 1
        circuit = Circuit(
            1, inputs=initial_single, dtype=torch.complex128, device="cpu"
        )
        circuit.append(UnitaryChannelOperation(got.rotation, 0, "xyz"))
        np.testing.assert_allclose(
            circuit.state().detach().flatten().numpy(), rotation[:, col], atol=1e-12
        )


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("Diagonal", (0.37, 0.08, 10)),
        ("MixedDiagonal", (0.37, 0.08, 10)),
        ("Fallback", (0.37, 0.9, 0.08, 10)),
    ],
)
def test_target_region_plot_matches_locked_patch_geometry(
    name: str, parameters: tuple[float, ...]
) -> None:
    from matplotlib.figure import Figure
    from qualtran.rotation_synthesis import protocols as source

    from flagquantum.synth import _rotation_protocols as native

    axes, locked_axes = Figure().subplots(), Figure().subplots()
    assert getattr(native, name)(*parameters).plot(1, axes, show_legend=False) is axes
    getattr(source, name)(*parameters).plot(1, locked_axes, show_legend=False)
    for patch, locked in zip(axes.patches, locked_axes.patches, strict=True):
        assert type(patch) is type(locked)
        vertices = patch.get_patch_transform().transform(patch.get_path().vertices)
        locked_vertices = locked.get_patch_transform().transform(
            locked.get_path().vertices
        )
        np.testing.assert_array_equal(vertices, locked_vertices)


def test_rotation_root_exports_map_to_native_owners() -> None:
    from qualtran import rotation_synthesis as source

    from flagquantum.ecosystem._quirk_rotation import to_quirk
    from flagquantum.ecosystem.cirq._rotation_export import to_cirq
    from flagquantum.synth import _rotation_protocols as native
    from flagquantum.synth._clifford_t_repr import to_sequence

    native_exports = {
        "NumpyConfig": mc.NumpyConfig,
        "with_dps": mc.with_dps,
        "to_cirq": to_cirq,
        "to_quirk": to_quirk,
        "to_sequence": to_sequence,
        "diagonal_unitary_approx": native.diagonal_unitary_approx,
        "fallback_protocol": native.fallback_protocol,
        "magnitude_approx": native.magnitude_approx,
        "mixed_diagonal_protocol": native.mixed_diagonal_protocol,
        "mixed_fallback_protocol": native.mixed_fallback_protocol,
    }
    assert set(native_exports) == {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and (callable(value) or name == "NumpyConfig")
    }
    for name, implementation in native_exports.items():
        if name == "NumpyConfig":
            assert isinstance(implementation, mc.MathConfig)
            assert implementation.name == source.NumpyConfig.name
        else:
            assert (
                callable(implementation) and getattr(implementation, "__name__") == name
            )
