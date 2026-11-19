"""Complete logical algorithm summaries through existing resource owners."""

from dataclasses import asdict
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
from qualtran import Bloq
from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.surface_code import AlgorithmSummary

from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    FixedTCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler.registers.control.multi_control import (
    source_multi_control_x_cost,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    source_phase_gradient_unitary_cost,
)
from flagquantum.qec.resources.beverland import estimate_beverland_resources
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_blobs_and_summary_default() -> None:
    module = import_module("qualtran.surface_code.algorithm_summary")
    assert module.__file__ is not None
    for name, blob in (
        ("algorithm_summary.py", "a4c71329551fee2c02c24358ff416aec8282172a"),
        ("algorithm_summary_test.py", "673438bf4aa45d920493efd3d005a7e2ff57ad21"),
    ):
        raw = Path(module.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    from qualtran.resource_counting import GateCounts

    source = AlgorithmSummary(n_algo_qubits=7, n_logical_gates=GateCounts(t=3))
    native = estimate_ftqc_resources(ComponentCost("example", 7, GateCost(t=3))).logical
    assert source.n_rotation_layers is None
    assert native.peak_logical_qubits == source.n_algo_qubits
    assert {
        name: count for name, count in asdict(native.gates).items() if count
    } == source.n_logical_gates.asdict()
    assert native.success_probability == 1


@pytest.mark.parametrize(
    "native,source",
    (
        (PhaseCall("t"), basic_gates.TGate(is_adjoint=False)),
        (FixedTCall("toffoli"), basic_gates.Toffoli()),
        (FixedTCall("fredkin"), basic_gates.TwoBitCSwap()),
        (ConjunctionCall((1, 1)), mcmt.And()),
        (
            RotationCall("zpow", 0.1, 1e-11),
            basic_gates.ZPowGate(exponent=0.1, eps=1e-11),
        ),
        (
            source_phase_gradient_unitary_cost(10),
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
        ),
        (source_multi_control_x_cost((1, 1, 1)), mcmt.MultiControlX(cvs=(1, 1, 1))),
    ),
)
def test_all_seven_original_from_operation_cases(
    native: ResourceCall, source: Bloq
) -> None:
    expected = AlgorithmSummary.from_bloq(source)
    width = query_resource_logical_width(native)
    categories = query_qec_gate_cost(native)
    assert isinstance(width, int)
    # The existing concrete component entry consumes the independently queried
    # native width and gate categories, without lowering the operation.
    model = (
        native
        if isinstance(native, ComponentCost)
        else ComponentCost(
            str(native),
            width,
            GateCost(
                **{name: int(count) for name, count in asdict(categories).items()}
            ),
        )
    )
    actual = estimate_ftqc_resources(model)
    assert actual.logical.peak_logical_qubits == expected.n_algo_qubits
    assert {
        name: count for name, count in asdict(actual.logical.gates).items() if count
    } == expected.n_logical_gates.asdict()
    assert actual.emitted is None
    assert actual.surface_code is None


@pytest.mark.parametrize("layers", (None, 0, 1, 4))
def test_rotation_depth_is_preserved_at_existing_consumer_boundary(
    layers: int | None,
) -> None:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import beverland_et_al_model
    from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost

    model = ComponentCost(
        "rotations", 5, GateCost(t=3, toffoli=2, rotation=4, measurement=1)
    )
    logical = estimate_ftqc_resources(model).logical
    native = estimate_beverland_resources(
        logical, error_budget=1e-3, physical_error=1e-4, rotation_layers=layers
    )
    source = AlgorithmSummary(
        n_algo_qubits=5,
        n_logical_gates=GateCounts(**asdict(logical.gates)),
        n_rotation_layers=layers,
    )
    expected = beverland_et_al_model.minimum_time_steps(
        error_budget=1e-3, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    assert native.minimum_time_steps == expected
    assert source.n_rotation_layers == layers
