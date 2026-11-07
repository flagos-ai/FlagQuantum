"""Consumer conformance for detector bits, independent of numerical engines."""

from typing import Protocol

import pytest
import torch

pytestmark = [pytest.mark.unit]


class DetectorStream(Protocol):
    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def consume(stream: DetectorStream) -> int:
    total = 0
    for shots in (16, 16, 3):
        pair = stream.sample(shots, separate_observables=True)
        assert isinstance(pair, tuple)
        detectors, observables = pair
        assert detectors.shape == (shots, 2) and observables.shape == (shots, 1)
        assert detectors.dtype == observables.dtype == torch.bool
        total += len(detectors)
    return total


class FakeStream:
    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables and not append_observables and not bit_packed
        return torch.zeros((shots, 2), dtype=torch.bool), torch.zeros(
            (shots, 1), dtype=torch.bool
        )


def test_independent_consumer_accepts_contract_fake() -> None:
    assert consume(FakeStream()) == 35


@pytest.mark.integration
def test_same_consumer_accepts_numerical_replacement() -> None:
    pytest.importorskip("stim")
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.qec.execution.context import MeasurementMap

    circuit = DynamicCircuit(2).measure(0, classical_bit=0).measure(1, classical_bit=1)
    sampler = circuit.compile_detector_sampler(
        detector_map=MeasurementMap(2, ((0,), (1,))),
        observable_map=MeasurementMap(2, ((0,),)),
        seed=7,
    )
    assert consume(sampler) == 35
