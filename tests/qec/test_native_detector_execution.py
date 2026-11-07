"""Native circuit detector sampling against independent fault and SDK oracles."""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.qec.execution import detector_sampling as public
from flagquantum.qec.execution.context import MeasurementMap
from flagquantum.simulation.stabilizer import _detector_sampling as numerical

pytestmark = pytest.mark.unit
DETECTORS = MeasurementMap(3, ((0, 1), (1, 2)))
OBSERVABLES = MeasurementMap(3, ((0,),))


def program(probability: float = 0.1) -> DynamicCircuit:
    from flagquantum.core.runtime_config import runtime_config

    with runtime_config(complex_dtype="complex128"):
        c = DynamicCircuit(3).x(0).x(1).x(2)
        for q in range(3):
            c.bit_flip(q, torch.tensor(probability, dtype=torch.float64))
        return (
            c.measure(0, classical_bit=0)
            .measure(1, classical_bit=1)
            .measure(2, classical_bit=2)
        )


def array(result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    assert isinstance(result, torch.Tensor)
    return result


class FakeStream:
    def __init__(self, detectors: int = 2, observables: int = 1) -> None:
        self.calls = 0
        self.detectors = detectors
        self.observables = observables
        self.fail = False

    def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("retained backend failure")
        return (
            torch.ones((shots, self.detectors), dtype=torch.bool),
            torch.ones((shots, self.observables), dtype=torch.bool),
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeStream:
    stream = FakeStream()
    monkeypatch.setattr(public, "_prepare_detector_sampler", lambda *a, **k: stream)
    return stream


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"shots": -1}, ValidationError),
        ({"shots": True}, ValidationError),
        ({"shots": 1.0}, ValidationError),
        ({"shots": 1, "bit_packed": 1}, TypeError),
        (
            {"shots": 1, "separate_observables": True, "append_observables": True},
            ValueError,
        ),
        ({"shots": 30_000_000}, CapabilityError),
    ],
)
def test_invalid_request_does_not_draw(
    fake: FakeStream, kwargs: dict[str, object], exception: type[Exception]
) -> None:
    sampler = program().compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES
    )
    with pytest.raises(exception):
        getattr(sampler, "sample")(**kwargs)
    assert fake.calls == 0
    assert array(sampler.sample(1)).tolist() == [[True, True]]


def test_zero_shots_layout_ownership_and_poison(fake: FakeStream) -> None:
    sampler = program().compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES
    )
    empty = sampler.sample(0, separate_observables=True, bit_packed=True)
    assert isinstance(empty, tuple) and [tuple(v.shape) for v in empty] == [
        (0, 1),
        (0, 1),
    ]
    assert fake.calls == 0
    assert array(
        sampler.sample(1, append_observables=True, bit_packed=True)
    ).tolist() == [[7]]
    first = sampler.sample(1)
    assert isinstance(first, torch.Tensor)
    first.zero_()
    assert array(sampler.sample(1)).all()
    fake.fail = True
    with pytest.raises(RuntimeError, match="retained backend"):
        sampler.sample(1)
    fake.fail = False
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="serialized"):
            operation(sampler)


def test_packed_append_crosses_parity_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        public, "_prepare_detector_sampler", lambda *a, **k: FakeStream(7, 2)
    )
    c = DynamicCircuit(1).measure(0)
    sampler = c.compile_detector_sampler(
        detector_map=MeasurementMap(1, ((0,),) * 7),
        observable_map=MeasurementMap(1, ((0,),) * 2),
    )
    assert array(
        sampler.sample(1, append_observables=True, bit_packed=True)
    ).tolist() == [[255, 1]]
    pair = sampler.sample(1, separate_observables=True, bit_packed=True)
    assert isinstance(pair, tuple) and [v.tolist() for v in pair] == [[[127]], [[3]]]


@pytest.mark.parametrize(
    "kind", ["t", "conditional_h", "amplitude", "placed", "no_measure"]
)
def test_refusal_precedes_preparation(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("engine preparation must not occur")

    monkeypatch.setattr(numerical, "_SdkDetectorSampler", forbidden)
    c = DynamicCircuit(1)
    if kind == "t":
        c.t(0)
    if kind == "conditional_h":
        c.measure(0).conditional("h", 0, classical_bit=0)
    if kind == "amplitude":
        c.amplitude_damping(0, 0.1)
    if kind == "placed":
        from dataclasses import replace

        ir = replace(c.to_ir(), metadata={"layout": {}})
        monkeypatch.setattr(c, "to_ir", lambda: ir)
    if kind != "no_measure":
        c.measure(0)
    width = 2 if kind == "conditional_h" else (0 if kind == "no_measure" else 1)
    with pytest.raises((CapabilityError, ValueError)):
        c.compile_detector_sampler(
            detector_map=MeasurementMap(width, ((0,),) if width else ()),
            observable_map=MeasurementMap(width, ()),
        )


def test_analysis_budget_precedes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numerical, "_MAX_ANALYSIS_CELLS", 1)
    monkeypatch.setattr(
        numerical, "_stim", lambda: (_ for _ in ()).throw(AssertionError("not reached"))
    )
    with pytest.raises(CapabilityError, match="support cells"):
        program().detector_error_model(
            detector_map=DETECTORS, observable_map=OBSERVABLES
        )


def test_invalid_maps_seed_and_owner(
    fake: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="width"):
        program().compile_detector_sampler(
            detector_map=MeasurementMap(1, ((0,),)), observable_map=OBSERVABLES
        )
    with pytest.raises(ValidationError):
        program().compile_detector_sampler(
            detector_map=DETECTORS, observable_map=OBSERVABLES, seed=True
        )
    sampler = program().compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES
    )
    monkeypatch.setattr(os, "getpid", lambda: -1)
    with pytest.raises(RuntimeError, match="creating thread and process"):
        sampler.sample(1)
    assert fake.calls == 0


@pytest.mark.integration
def test_independent_dem_and_eight_forced_faults() -> None:
    stim = pytest.importorskip("stim")
    from flagquantum.qec import PyMatchingDecoder

    pytest.importorskip("pymatching")
    reference = stim.Circuit(
        "X 0 1 2\nX_ERROR(0.1) 0 1 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    )
    model = program().detector_error_model(
        detector_map=DETECTORS, observable_map=OBSERVABLES
    )
    from flagquantum.qec import DetectorErrorModel

    reference_text = (
        str(reference.detector_error_model())
        + "\ndetector D0\ndetector D1\nlogical_observable L0"
    )
    oracle = DetectorErrorModel.from_stim_text(reference_text)
    assert (model.num_detectors, model.num_observables) == (2, 1)
    assert [(e.detectors, e.observables) for e in model.errors] == [
        (e.detectors, e.observables) for e in oracle.errors
    ]
    assert [e.probability for e in model.errors] == pytest.approx(
        [e.probability for e in oracle.errors], rel=0, abs=1e-12
    )
    decoder = PyMatchingDecoder.from_detector_error_model(model)
    exact_failure = 0.0
    for mask in range(8):
        c = DynamicCircuit(3).x(0).x(1).x(2)
        for q in range(3):
            c.bit_flip(q, float((mask >> q) & 1))
        c.measure(0, classical_bit=0).measure(1, classical_bit=1).measure(
            2, classical_bit=2
        )
        result = c.compile_detector_sampler(
            detector_map=DETECTORS, observable_map=OBSERVABLES, seed=7
        ).sample(1, separate_observables=True)
        assert isinstance(result, tuple)
        det, obs = result
        a, b, z = ((mask >> q) & 1 for q in range(3))
        assert det.tolist() == [[bool(a ^ b), bool(b ^ z)]] and obs.tolist() == [
            [bool(a)]
        ]
        prediction = decoder.decode_batch(det)
        failed = bool((prediction ^ obs).any())
        assert failed == (mask.bit_count() >= 2)
        exact_failure += (
            failed * 0.1 ** mask.bit_count() * 0.9 ** (3 - mask.bit_count())
        )
    assert exact_failure == pytest.approx(0.028, rel=0, abs=1e-15)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["repeated", "noisy", "gauge", "disjoint"])
def test_sdk_reference_and_exact_analysis_boundary(kind: str) -> None:
    stim = pytest.importorskip("stim")
    if kind == "repeated":
        c = (
            DynamicCircuit(3)
            .h(0)
            .measure(0, classical_bit=0)
            .conditional("x", 1, classical_bit=0)
            .reset(0)
            .conditional("x", 0, classical_bit=0, equals=0)
            .measure(0, classical_bit=0)
            .conditional("x", 2, classical_bit=0)
            .measure(1, classical_bit=1)
            .measure(2, classical_bit=2)
        )
        d = MeasurementMap(4, ((0, 1), (0, 2), (1, 3)))
        o = MeasurementMap(4, ((2, 3),))
        sampler = c.compile_detector_sampler(detector_map=d, observable_map=o, seed=7)
        bits = sampler.sample(35, append_observables=True)
        assert (
            isinstance(bits, torch.Tensor) and bits.shape == (35, 4) and not bits.any()
        )
    elif kind == "gauge":
        with pytest.raises(CapabilityError, match="nondeterministic"):
            DynamicCircuit(1).h(0).measure(0).compile_detector_sampler(
                detector_map=MeasurementMap(1, ((0,),)),
                observable_map=MeasurementMap(1, ()),
            )
    elif kind == "disjoint":
        from dataclasses import replace

        from flagquantum.core.ir import Instruction

        identity = torch.eye(2, dtype=torch.complex128)
        x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        kraus = (
            (0.7**0.5) * torch.eye(4, dtype=torch.complex128),
            (0.1**0.5) * torch.kron(x, identity),
            (0.2**0.5) * torch.kron(identity, x),
        )
        c = DynamicCircuit(2).measure(0, classical_bit=0).measure(1, classical_bit=1)
        ir = c.to_ir()
        ir = replace(
            ir,
            instructions=(
                Instruction(
                    "frame_pair", (0, 1), matrix=kraus, metadata={"is_channel": True}
                ),
                *ir.instructions,
            ),
        )
        stream = numerical._prepare_detector_sampler(
            ir, detector_columns=((0,), (1,)), observable_columns=(), seed=7
        )
        assert stream.sample(3)[0].shape == (3, 2)
        with pytest.raises(CapabilityError, match="exact DEM"):
            numerical._exact_detector_error_model_text(
                ir, detector_columns=((0,), (1,)), observable_columns=()
            )
    else:
        sampler = program().compile_detector_sampler(
            detector_map=DETECTORS, observable_map=OBSERVABLES, seed=7
        )
        ref = stim.Circuit(
            numerical._source_text(program().to_ir(), DETECTORS.rows, OBSERVABLES.rows)
        ).compile_detector_sampler(seed=7)
        for shots in (16, 16, 3):
            actual = sampler.sample(shots, separate_observables=True)
            expected = ref.sample(shots, separate_observables=True)
            assert isinstance(actual, tuple)
            assert all(
                torch.equal(x, torch.from_numpy(y))
                for x, y in zip(actual, expected, strict=True)
            )


@pytest.mark.parametrize("malformed", ["dtype", "shape", "noncontiguous"])
def test_invalid_backend_poison(
    monkeypatch: pytest.MonkeyPatch, malformed: str
) -> None:
    class Broken:
        def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
            detectors = torch.zeros((shots, 2), dtype=torch.bool)
            if malformed == "dtype":
                detectors = detectors.to(torch.uint8)
            elif malformed == "shape":
                detectors = detectors[:, :1]
            else:
                detectors = torch.zeros((shots, 4), dtype=torch.bool)[:, ::2]
            return detectors, torch.zeros((shots, 1), dtype=torch.bool)

    monkeypatch.setattr(public, "_prepare_detector_sampler", lambda *a, **k: Broken())
    sampler = program().compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES
    )
    with pytest.raises(CapabilityError, match="invalid sample"):
        sampler.sample(2)
    with pytest.raises(RuntimeError, match="invalidated"):
        sampler.sample(1)


def test_frame_bound_and_zero_observable(
    fake: FakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = DynamicCircuit(4096).measure(0)
    sampler = c.compile_detector_sampler(
        detector_map=MeasurementMap(1, ((0,),)), observable_map=MeasurementMap(1, ())
    )
    with pytest.raises(CapabilityError, match="frame estimate"):
        sampler.sample(100000)
    assert fake.calls == 0
    monkeypatch.setattr(
        public, "_prepare_detector_sampler", lambda *a, **k: FakeStream(1, 0)
    )
    small = (
        DynamicCircuit(1)
        .measure(0)
        .compile_detector_sampler(
            detector_map=MeasurementMap(1, ((0,),)),
            observable_map=MeasurementMap(1, ()),
        )
    )
    pair = small.sample(2, separate_observables=True, bit_packed=True)
    assert isinstance(pair, tuple) and pair[1].shape == (2, 0)
    assert array(
        small.sample(2, append_observables=True, bit_packed=True)
    ).tolist() == [
        [1],
        [1],
    ]


def test_embedded_failure_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.simulation.stabilizer import _embedded_sampler

    monkeypatch.setattr(Path, "is_file", lambda self: True)

    def broken(*args: object, **kwargs: object) -> object:
        raise ImportError("retained broken extension")

    monkeypatch.setattr(_embedded_sampler, "_EmbeddedDetectorSampler", broken)
    monkeypatch.setattr(
        numerical, "_SdkDetectorSampler", lambda *a, **k: pytest.fail("SDK fallback")
    )
    with pytest.raises(ImportError, match="retained broken"):
        program().compile_detector_sampler(
            detector_map=DETECTORS, observable_map=OBSERVABLES
        )


@pytest.mark.integration
def test_source_snapshot_and_validation_preserve_rng() -> None:
    pytest.importorskip("stim")
    c = program()
    first = c.compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES, seed=9
    )
    second = c.compile_detector_sampler(
        detector_map=DETECTORS, observable_map=OBSERVABLES, seed=9
    )
    c.t(0)
    with pytest.raises(ValueError):
        first.sample(1, separate_observables=True, append_observables=True)
    assert array(first.sample(0)).shape == (0, 2)
    assert torch.equal(
        array(first.sample(35, append_observables=True)),
        array(second.sample(35, append_observables=True)),
    )


@pytest.mark.integration
def test_noiseless_nonzero_reference_and_zero_error_dimensions() -> None:
    pytest.importorskip("stim")
    c = DynamicCircuit(1).x(0).measure(0)
    d, o = MeasurementMap(1, ((0,),)), MeasurementMap(1, ((),))
    model = c.detector_error_model(detector_map=d, observable_map=o)
    assert model.num_detectors == model.num_observables == 1 and not model.errors
    sampler = c.compile_detector_sampler(detector_map=d, observable_map=o, seed=7)
    actual = sampler.sample(3, append_observables=True, bit_packed=True)
    assert array(actual).tolist() == [[0], [0], [0]]
