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

# Adapted from Qualtran v0.7.0 rotation_synthesis/channels/channel_test.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/channels/channel_test.py

from collections.abc import Iterator, Sequence

import cirq
import numpy as np
import pytest
import torch

from flagquantum import Circuit
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._quirk_rotation_channels import (
    export_projective_channel_quirk,
    export_unitary_channel_quirk,
)
from flagquantum.ecosystem.cirq._rotation_channels import (
    export_projective_channel_cirq,
    export_unitary_channel_cirq,
)
from flagquantum.ecosystem.cirq._rotation_export import to_cirq
from flagquantum.synth import _rotation_channels as ch
from flagquantum.synth import _rotation_math as mc
from flagquantum.synth import _su2_ct
from flagquantum.synth._rotation_channel_operation import (
    ProjectiveChannelOperation,
    UnitaryChannelOperation,
)
from tests.unit.measurement_checks import forced_branch


def _make_gates(n_seqs: int, n_gates: int, seed: int) -> Iterator[list[str]]:
    gates = tuple(_su2_ct.GATE_MAP.keys())
    rng = np.random.default_rng(seed)
    for _ in range(n_seqs):
        yield [gates[i] for i in rng.choice(len(gates), n_gates)]


@pytest.mark.parametrize("gates", tuple(_make_gates(10, 4, 0)))
def test_unitary_from_sequence(gates: Sequence[str]) -> None:
    u = ch.UnitaryChannel.from_sequence(gates)
    assert u.to_matrix() == _su2_ct.SU2CliffordT.from_sequence(gates)


@pytest.mark.parametrize(
    ["gates", "theta", "distance"],
    [
        [["I", "Z", "I", "Tz"], 4.179974975284761, 0.2785216431390912],
        [["I", "S", "Tz"], 1.788395114150811, 0.34841379843764864],
        [["I", "S", "Tz"], 2.147407592736323, 0.36575434272229357],
        [["I", "Z", "S", "Tz"], 0.5569130698464282, 0.3269538876258962],
        [["I", "S"], 5.528036816451128, 0.06049011917942428],
    ],
)
def test_diamond_distance_for_unitary(
    gates: Sequence[str], theta: float, distance: float
) -> None:
    c = ch.UnitaryChannel.from_sequence(gates)
    np.testing.assert_allclose(
        c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance
    )
    u = np.zeros((2, 2), complex)
    u[1, 1] = np.exp(-1j * theta)
    u[0, 0] = u[1, 1].conjugate()
    np.testing.assert_allclose(
        c.diamond_norm_distance_to_unitary(u, mc.NumpyConfig), distance
    )


@pytest.mark.parametrize(
    ["proj", "correction", "theta", "distance"],
    [
        [
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            ["I", "Z", "S", "H", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx", "Ty", "Tz", "Tx"],
            4.058393950738398,
            0.12265553387764809,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 0.7154420837719515, 0.13979806891313962],
        [
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            ["I", "Z", "S", "H", "Tz", "Tx", "Ty", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx"],
            4.041343072659278,
            0.15656986343894638,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 0.8609971423751847, 0.1510539778688528],
        [
            ["I", "H", "S", "Ty", "Tx", "Tz", "Ty", "Tx"],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            2.910895329185765,
            0.18545310121849135,
        ],
        [
            ["I", "Y", "H", "S", "H", "Ty", "Tx", "Ty", "Tz", "Tx"],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            0.5337113492446506,
            0.22683760663945227,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 4.299933159455965, 0.7287141780613403],
        [
            ["I", "H", "S", "Tx", "Ty", "Tz", "Ty", "Tz", "Ty", "Tx"],
            ["I", "S", "H", "Ty", "Tz", "Ty", "Tx", "Ty", "Tz", "Ty", "Tx", "Ty"],
            2.4629437050180374,
            0.17169223895328464,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 3.9026689436493815, 0.04863895092095243],
    ],
)
def test_diamond_distance_for_fallback(
    proj: Sequence[str], correction: Sequence[str], theta: float, distance: float
) -> None:
    c = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj),
        ch.UnitaryChannel.from_sequence(correction),
    )
    np.testing.assert_allclose(
        c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance
    )


@pytest.mark.parametrize(
    ["seq1", "seq2", "theta", "distance"],
    [
        [
            [
                "I",
                "I",
                "Ty",
                "Tx",
                "Ty",
                "Tz",
                "Ty",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
            ],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            4.101555349367053,
            0.007986955439257544,
        ],
        [
            [
                "I",
                "Z",
                "H",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
            ],
            ["I", "Z", "H", "Ty", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz", "Tx", "Ty"],
            3.959743679319142,
            0.0023654253421858873,
        ],
        [
            [
                "I",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
            ],
            [
                "I",
                "H",
                "Ty",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Ty",
                "Tz",
            ],
            5.18570635613891,
            0.004288062424341757,
        ],
        [
            [
                "I",
                "S",
                "H",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
            ],
            ["I", "S", "H", "Tz", "Tx", "Ty", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx", "Tz"],
            4.442270210803678,
            0.002937255469622333,
        ],
        [
            ["I", "X", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz"],
            ["I", "S", "H", "Tz", "Ty", "Tz", "Ty", "Tx", "Ty", "Tx"],
            2.160269652544976,
            0.006305821840210275,
        ],
        [
            ["I", "Y", "I", "Tz", "Tx", "Tz", "Ty", "Tx", "Ty", "Tz", "Ty"],
            [
                "I",
                "Z",
                "H",
                "S",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
                "Tz",
            ],
            4.169354523270907,
            0.006945248214462576,
        ],
        [
            [
                "I",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tz",
                "Tx",
            ],
            [
                "I",
                "X",
                "H",
                "S",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
            ],
            1.9218237240459548,
            0.005494560413584842,
        ],
        [
            [
                "I",
                "X",
                "H",
                "S",
                "Tz",
                "Ty",
                "Tz",
                "Ty",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Ty",
            ],
            [
                "I",
                "H",
                "S",
                "H",
                "Tz",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Ty",
                "Tx",
                "Ty",
                "Tx",
            ],
            0.0203606083765623,
            0.0003712210316894235,
        ],
        [
            [
                "I",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
            ],
            [
                "I",
                "H",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
            ],
            2.007902368169852,
            0.0036763097991141785,
        ],
        [
            [
                "I",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
            ],
            [
                "I",
                "H",
                "Ty",
                "Tx",
                "Ty",
                "Tz",
                "Tx",
                "Tz",
                "Tx",
                "Ty",
                "Tz",
                "Ty",
                "Tz",
            ],
            5.179866016921518,
            0.004542822638048746,
        ],
    ],
)
def test_diamond_distance_for_mixed_diagonal(
    seq1: Sequence[str], seq2: Sequence[str], theta: float, distance: float
) -> None:
    c = ch.ProbabilisticChannel.from_unitary_channels(
        ch.UnitaryChannel.from_sequence(seq1, True),
        ch.UnitaryChannel.from_sequence(seq2, True),
        theta,
        mc.NumpyConfig,
    )
    np.testing.assert_allclose(
        c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance
    )


@pytest.mark.parametrize(
    ["proj_1", "seq_11", "seq_12", "proj_2", "seq_21", "seq_22", "theta", "distance"],
    [
        [
            ["I", "Z", "H", "S", "Tz", "Tx", "Ty", "Tx"],
            ["I", "Z", "I"],
            ["I", "S"],
            ["I", "S"],
            ["I", "Z", "I"],
            ["I", "S"],
            2.2336312299454693,
            0.033899366186582215,
        ],
        [
            ["I", "Z", "S"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            ["I", "Y", "H", "S", "H", "Tx", "Tz", "Ty", "Tx"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            3.978274738516034,
            0.021338488294120432,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.60506064948477,
            0.03282184847099254,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.552288085883322,
            0.022324504375577123,
        ],
        [
            ["I", "Y", "H", "S", "H", "Tx", "Tz", "Ty", "Tx", "Tz"],
            ["I", "I"],
            ["I", "Z", "S", "Tz"],
            ["I", "Z", "S"],
            ["I", "I"],
            ["I", "Z", "S"],
            0.7752283252787129,
            0.002603637233480823,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "S", "H", "Tz", "Tx", "Tz", "Tx", "Tz"],
            ["I", "Z", "I", "Tz"],
            ["I", "Z", "I"],
            2.369467427334453,
            0.0033169663527450343,
        ],
        [
            ["I", "Z", "S"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            ["I", "Y", "H", "S", "H", "Tx", "Ty", "Tz", "Tx"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            0.8840386497791466,
            0.031826887148390826,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Ty", "Tx", "Ty"],
            ["I", "S"],
            ["I", "I"],
            2.4793681001683243,
            0.03392425218025496,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.594916529653181,
            0.03162395060972892,
        ],
    ],
)
def test_diamond_distance_for_mixed_fallback(
    proj_1: Sequence[str],
    seq_11: Sequence[str],
    seq_12: Sequence[str],
    proj_2: Sequence[str],
    seq_21: Sequence[str],
    seq_22: Sequence[str],
    theta: float,
    distance: float,
) -> None:
    p1 = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj_1),
        correction=ch.ProbabilisticChannel.from_unitary_channels(
            ch.UnitaryChannel.from_sequence(seq_11, True),
            ch.UnitaryChannel.from_sequence(seq_12, True),
            theta,
            mc.NumpyConfig,
        ),
    )
    p2 = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj_2),
        correction=ch.ProbabilisticChannel.from_unitary_channels(
            ch.UnitaryChannel.from_sequence(seq_21, True),
            ch.UnitaryChannel.from_sequence(seq_22, True),
            theta,
            mc.NumpyConfig,
        ),
    )
    c = ch.ProbabilisticChannel.from_projective_channels(p1, p2, theta, mc.NumpyConfig)
    np.testing.assert_allclose(
        float(c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig)), distance, atol=3e-5
    )


@pytest.mark.parametrize(
    "gates",
    [
        ["I", "Z", "I", "Tz"],
        ["I", "S", "Tz"],
        ["I", "S", "Tz"],
        ["I", "Z", "S", "Tz"],
        ["I", "S"],
    ],
)
@pytest.mark.parametrize("fmt", ["xz", "xyz"])
def test_unitary_to_cirq(gates: Sequence[str], fmt: str) -> None:
    u = ch.UnitaryChannel.from_sequence(gates)
    assert export_unitary_channel_cirq(u, fmt) == cirq.Circuit(
        to_cirq(u.to_matrix(), fmt)
    )


@pytest.mark.parametrize(
    "gates1",
    [
        ["I", "Z", "I", "Tz"],
        ["I", "S", "Tz"],
        ["I", "S", "Tz"],
        ["I", "Z", "S", "Tz"],
        ["I", "S"],
    ],
)
@pytest.mark.parametrize(
    "gates2",
    [
        ["I", "Z", "I", "Tz"],
        ["I", "S", "Tz"],
        ["I", "S", "Tz"],
        ["I", "Z", "S", "Tz"],
        ["I", "S"],
    ],
)
@pytest.mark.parametrize("fmt", ["xz", "xyz"])
def test_fallback_to_cirq(
    gates1: Sequence[str], gates2: Sequence[str], fmt: str
) -> None:
    c = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(gates1), ch.UnitaryChannel.from_sequence(gates2)
    )
    q0, q1 = cirq.LineQubit.range(2)
    assert export_projective_channel_cirq(c, fmt) == cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(to_cirq(c.rotation.to_matrix(), fmt, q0))
        ),
        cirq.CNOT(q0, q1),
        cirq.measure(q1, key="m"),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(to_cirq(c.correction.to_matrix(), fmt, q0))  # type: ignore[attr-defined]
        ).with_classical_controls("m"),
    )


@pytest.mark.parametrize("fmt", ["xyz", "xz"])
@pytest.mark.parametrize(
    "rotation_word,correction_word",
    [(("Tx",), ("Tz",)), (("H",), ("S", "Tz")), (("Tz",), ("H",))],
)
def test_locked_projective_branches(
    fmt: str, rotation_word: tuple[str, ...], correction_word: tuple[str, ...]
) -> None:
    from qualtran.rotation_synthesis import channels as source

    locked = source.ProjectiveChannel(
        source.UnitaryChannel.from_sequence(rotation_word),
        source.UnitaryChannel.from_sequence(correction_word),
    )
    native = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(rotation_word),
        ch.UnitaryChannel.from_sequence(correction_word),
    )
    target, ancilla = cirq.NamedQubit("target"), cirq.NamedQubit("ancilla")
    assert export_projective_channel_cirq(
        native, fmt, (target, ancilla)
    ) == locked.to_cirq(fmt, (target, ancilla))
    for phase in (False, True):
        assert export_projective_channel_quirk(native, fmt, phase) == locked.to_quirk(
            fmt, phase
        )
        assert export_unitary_channel_quirk(
            native.rotation, fmt, phase
        ) == locked.rotation.to_quirk(fmt, phase)

    unitary = locked.rotation.to_matrix().numpy()
    correction = locked.correction.to_matrix().numpy()
    # The source circuit applies C directly on outcome 1. Its docstring diagram
    # also shows a Y, which is absent in both exporters.
    branches = (
        np.diag(np.diag(unitary)),
        correction @ (unitary - np.diag(np.diag(unitary))),
    )
    success = float(native.success_probability(mc.NumpyConfig))
    circuit = DynamicCircuit(2, dtype=torch.complex128, device="cpu")
    circuit.append(ProjectiveChannelOperation(native, 1, 0, 5, fmt))
    instructions = circuit.to_ir().instructions
    assert [
        (inst.name, inst.wires)
        for inst in instructions
        if inst.name in ("cx", "measure")
    ] == [
        ("cx", (1, 0)),
        ("cx", (1, 0)),
        ("measure", (0,)),
    ]
    measurement = next(
        index for index, gate in enumerate(instructions) if gate.name == "measure"
    )
    assert instructions[measurement].metadata["classical_bit"] == 5
    assert all(
        inst.metadata["conditions"] == ((5, 1),)
        for inst in instructions[measurement + 1 :]
    )
    assert (instructions[-1].name, instructions[-1].wires) == ("x", (0,))
    initial_t = sum(inst.name in ("t", "tdg") for inst in instructions[:measurement])
    fallback_t = sum(
        inst.name in ("t", "tdg") for inst in instructions[measurement + 1 :]
    )
    assert initial_t + (1 - success) * fallback_t == pytest.approx(
        float(native.expected_num_ts(mc.NumpyConfig))
    )

    for bit in (0, 1):
        initial = torch.zeros(4, dtype=torch.complex128)
        initial[bit] = 1
        for outcome in (0, 1):
            branch = forced_branch(initial, instructions, (outcome,)).detach().numpy()
            np.testing.assert_allclose(
                branch[:2], branches[outcome][:, bit], atol=1e-12
            )
            np.testing.assert_array_equal(branch[2:], 0)
            assert np.linalg.norm(branch) ** 2 == pytest.approx(
                (success, 1 - success)[outcome]
            )


def test_static_binding_and_channel_admission() -> None:
    channel = ch.UnitaryChannel.from_sequence(("Tz",))
    circuit = Circuit(1, dtype=torch.complex128, device="cpu").h(0)
    circuit.append(UnitaryChannelOperation(channel, 0))
    state = circuit.state().detach().flatten().numpy()
    np.testing.assert_allclose(
        state, channel.to_matrix().numpy() @ (np.ones(2) / np.sqrt(2))
    )
    twirled = ch.UnitaryChannel.from_sequence(("Tz",), twirl=True)
    with pytest.raises(ValueError, match="twirl"):
        Circuit(1).append(UnitaryChannelOperation(twirled, 0))
    with pytest.raises(ValueError, match="twirl"):
        export_unitary_channel_cirq(twirled)
    with pytest.raises(ValueError, match="twirl"):
        export_unitary_channel_quirk(twirled)
    projective = ch.ProjectiveChannel(channel, channel)
    with pytest.raises(TypeError, match="DynamicCircuit"):
        Circuit(2).append(ProjectiveChannelOperation(projective, 0, 1, 0))
    with pytest.raises(ValueError, match="distinct"):
        DynamicCircuit(2).append(ProjectiveChannelOperation(projective, 0, 0, 0))
    invalid = ch.ProjectiveChannel(channel, ch.ProbabilisticChannel(channel, channel))
    with pytest.raises(ValueError, match="unitary correction"):
        DynamicCircuit(2).append(ProjectiveChannelOperation(invalid, 0, 1, 0))
    with pytest.raises(ValueError, match="unitary correction"):
        export_projective_channel_cirq(invalid)
    with pytest.raises(ValueError, match="correction"):
        export_projective_channel_quirk(invalid)
    mixed = ch.ProbabilisticChannel(channel, projective, 2.0)
    assert mixed.probability == 1
    assert mixed.expected_num_ts(mc.NumpyConfig) == channel.n
    with pytest.raises(NotImplementedError):
        mixed.diamond_norm_distance_to_rz(0, mc.NumpyConfig)


def test_channel_exports_cover_source_package() -> None:
    from qualtran.rotation_synthesis import channels as source

    declared = {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and isinstance(value, type)
    }
    assert declared == {
        cls.__name__
        for cls in (
            ch.Channel,
            ch.UnitaryChannel,
            ch.ProjectiveChannel,
            ch.ProbabilisticChannel,
        )
    }
