"""Complete distillation factory interface against the frozen source module."""

from dataclasses import asdict
from functools import lru_cache
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import sympy
import torch

from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.error_model import LogicalErrorModel, SurfaceCodeScheme
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.qec.resources.magic_state_factory import MagicStateFactory
from flagquantum.qec.resources.parallel_factory import ParallelFactory
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

np = pytest.importorskip("numpy")
_source = pytest.importorskip("qualtran.surface_code")
GateCounts = pytest.importorskip("qualtran.resource_counting").GateCounts
FifteenToOne = _source.FifteenToOne
QECScheme = _source.QECScheme
SourceErrorModel = _source.LogicalErrorModel


def test_frozen_complete_source_and_factory_contract() -> None:
    module = import_module("qualtran.surface_code.fifteen_to_one_factory")
    assert module.__file__ is not None
    for name, blob in (
        ("fifteen_to_one_factory.py", "d4c36592c0ced4268ac11791991b34ec7a566c7a"),
        ("fifteen_to_one_factory_test.py", "616820dc041eeeb3dc5452de7dbb30dfd735ad55"),
    ):
        raw = Path(module.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    assert isinstance(FifteenToOneFactory(7, 3, 3), MagicStateFactory)


@pytest.mark.parametrize(
    "distances", ((1, 1, 1), (2, 2, 2), (7, 3, 3), (9, 3, 3), (11, 5, 5), (17, 7, 7))
)
@pytest.mark.parametrize("physical_error", (0.0, 1e-4, 1e-3))
@pytest.mark.parametrize("scaler", (0.03, 0.1))
@pytest.mark.parametrize("threshold", (0.01, 0.02))
def test_complete_factory_exact_numerical_and_large_demand_boundary(
    distances: tuple[int, int, int],
    physical_error: float,
    scaler: float,
    threshold: float,
) -> None:
    native = FifteenToOneFactory(*distances)
    original = FifteenToOne(*distances)
    model = LogicalErrorModel(physical_error, SurfaceCodeScheme(scaler, threshold))
    reference = SourceErrorModel(
        physical_error=physical_error,
        qec_scheme=QECScheme(error_rate_scaler=scaler, error_rate_threshold=threshold),
    )
    np.testing.assert_array_equal(
        native._final_state(model).numpy(), original._final_state(reference)
    )
    assert native.p_fail(model) == original.p_fail(reference)
    assert native.p_out(model) == original.p_out(reference)
    for demand in (0, 1, 3, 2_500_000_000_091, 10**14, 2**53 + 1, 10**18):
        gates = GateCost(t=demand)
        source_gates = GateCounts(t=demand)
        assert native.n_cycles(gates, model) == int(
            original.n_cycles(source_gates, reference)
        )
        assert native.factory_error(gates, model) == original.factory_error(
            source_gates, reference
        )


@pytest.mark.parametrize("distances", ((7, 3, 3), (9, 3, 3), (11, 5, 5), (17, 7, 7)))
@pytest.mark.parametrize("threshold,scaler", ((0.01, 0.1), (0.02, 0.03)))
def test_complete_bound_error_model_and_parallel_composition(
    distances: tuple[int, int, int], threshold: float, scaler: float
) -> None:
    from qualtran.surface_code import MultiFactory

    actual, expected = FifteenToOneFactory(*distances), FifteenToOne(*distances)
    model = LogicalErrorModel(1e-4, SurfaceCodeScheme(scaler, threshold))
    reference = SourceErrorModel(1e-4, QECScheme(scaler, threshold))
    np.testing.assert_allclose(
        actual._final_state(model).numpy(),
        expected._final_state(reference),
        rtol=0,
        atol=2e-15,
    )
    assert actual.p_fail(model) == pytest.approx(
        expected.p_fail(reference), rel=0, abs=2e-14
    )
    assert actual.p_out(model) == pytest.approx(
        expected.p_out(reference), rel=0, abs=2e-14
    )
    for gates in (
        GateCost(),
        GateCost(t=1),
        GateCost(t=3, toffoli=2, cswap=7, and_bloq=5, rotation=11),
    ):
        original_gates = GateCounts(**asdict(gates))
        assert actual.n_physical_qubits() == expected.n_physical_qubits()
        assert actual.n_cycles(gates, model) == expected.n_cycles(
            original_gates, reference
        )
        assert actual.factory_error(gates, model) == pytest.approx(
            expected.factory_error(original_gates, reference), rel=0, abs=4e-12
        )
        assert ParallelFactory(actual, 3).n_cycles(gates, model) == MultiFactory(
            expected, 3
        ).n_cycles(original_gates, reference)
    count = sympy.Symbol("count", integer=True, positive=True)
    actual_symbolic = actual.factory_error(QECGateCost(t=count), model)
    expected_symbolic = expected.factory_error(GateCounts(t=count), reference)
    assert float((actual_symbolic - expected_symbolic).subs(count, 1)) == pytest.approx(
        0, rel=0, abs=2e-14
    )
    with pytest.raises(TypeError):
        expected.n_cycles(GateCounts(t=count), reference)
    with pytest.raises(TypeError):
        actual.n_cycles(QECGateCost(t=count), model)


def test_density_execution_is_reused_across_factory_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.qec.resources.fifteen_to_one as owner

    execute = density_matrix_from_ir
    calls: list[int] = []

    def counted(ir: CircuitIR, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        calls.append(len(ir.instructions))
        return execute(ir, device=device, dtype=dtype)

    monkeypatch.setattr(owner, "density_matrix_from_ir", counted)
    factory = FifteenToOneFactory(7, 3, 3)
    factory._final_state.cache_clear()
    factory._postselection.cache_clear()
    model = LogicalErrorModel(1e-4, SurfaceCodeScheme.gidney_fowler())
    factory.p_fail(model)
    factory.p_out(model)
    factory.n_cycles(GateCost(t=3), model)
    factory.factory_error(GateCost(t=3), model)
    factory.estimate(GateCost(t=3), physical_error=1e-4, error_rate_scaler=0.1)
    assert calls == [100]
    assert factory._final_state.cache_info().maxsize == 8
    assert factory._postselection.cache_info().maxsize == 8


def test_actual_consumer_accepts_a_source_density_implementation() -> None:
    calls: list[int] = []

    class ReferenceFactory(FifteenToOneFactory):
        @lru_cache(8)
        def _final_state(self, logical_error_model: LogicalErrorModel) -> torch.Tensor:
            calls.append(self.x_distance)
            model = SourceErrorModel(
                logical_error_model.physical_error,
                QECScheme(
                    logical_error_model.qec_scheme.error_rate_scaler,
                    logical_error_model.qec_scheme.error_rate_threshold,
                ),
            )
            return torch.from_numpy(
                FifteenToOne(
                    self.x_distance, self.z_distance, self.measurement_distance
                )
                ._final_state(model)
                .copy()
            )

    logical = ModeledResources(GateCost(t=3, toffoli=5, rotation=2), 17)
    expected = estimate_beverland_physical_resources(
        logical, data_distance=7, factory=FifteenToOneFactory(9, 3, 3)
    )
    # Reset the value-keyed cache so the replacement implementation is exercised.
    FifteenToOneFactory._postselection.cache_clear()
    actual = estimate_beverland_physical_resources(
        logical, data_distance=7, factory=ReferenceFactory(9, 3, 3)
    )
    assert calls == [9]
    assert asdict(actual) == pytest.approx(asdict(expected), rel=0, abs=4e-12)
