"""Check ordered DF source graphs, lifetimes and original control wiring."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.double_factorization_block_encoding import (
    DoubleFactorizationBlockEncoding as ExecutableDoubleFactorization,
)
from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
    source_double_factorization_one_body_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.multi_control import multi_control_z_ir
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _assert_ordered_graph(model, source):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert model.qubits == source.signature.n_qubits()
    graph = tuple(source.decompose_bloq().iter_bloqsoqs())
    assert [edge.component.name for edge in model.calls] == [
        type(instance.bloq).__name__ for instance, _, _ in graph
    ]
    pairs = [(model, source)]
    for edge, (instance, _, _) in zip(model.calls, graph, strict=True):
        child = instance.bloq
        assert edge.bystander_qubits == model.qubits - child.signature.n_qubits()
        pairs.append((edge.component, child))
    for candidate, original in pairs:
        estimate = estimate_component_resources(candidate)
        assert {
            key: count for key, count in asdict(estimate.gates).items() if count
        } == (get_cost_value(original, QECGatesCost()).asdict())
        assert estimate.peak_logical_qubits == get_cost_value(original, QubitCount())
    return graph


@pytest.mark.parametrize("spin", [6, 10, 16])
def test_one_body_preserves_source_decomposition_width_and_declared_gate_cost(spin):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationOneBody,
    )

    source = DoubleFactorizationOneBody(50, spin, 250, 12, num_bits_rot=8)
    model = source_double_factorization_one_body_cost(spin, 250, 12, 8)
    graph = _assert_ordered_graph(model, source)
    operation, inputs, _ = graph[4]
    assert operation.bloq.bitsize == spin // 2
    declared = next(
        child for child in source.bloq_counts() if isinstance(child, ApplyControlledZs)
    )
    assert declared.bitsize == 5
    assert [wire.reg.name for wire in inputs["ctrls"]] == ["succ_l", "succ_p"]
    assert model.calls[4].component.gates.toffoli == 1


@pytest.mark.parametrize("spin", [6, 10, 16])
def test_full_source_graph_retains_actual_reflection_ports(spin):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
    )

    source = DoubleFactorizationBlockEncoding(
        num_aux=50,
        num_spin_orb=spin,
        num_eig=250,
        num_bits_state_prep=12,
        num_bits_rot=8,
        num_bits_rot_aa_outer=9,
        num_bits_rot_aa_inner=7,
    )
    model = source_double_factorization_block_encoding_cost(50, spin, 250, 12, 8, 7)
    graph = _assert_ordered_graph(model, source)
    _, inputs, _ = graph[3]
    assert set(inputs) == {"control", "reg0_", "reg1_"}
    assert inputs["control"].reg.name == "succ_l"
    with pytest.raises(NotImplementedError):
        _ = source.alpha
    with pytest.raises(NotImplementedError):
        _ = source.epsilon


def test_df_source_estimate_from_operation_without_emitting_ir(monkeypatch):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    def unexpected_emit(*args, **kwargs):
        pytest.fail("Source resource estimation must not emit instructions")

    monkeypatch.setattr(ExecutableDoubleFactorization, "_emit", unexpected_emit)
    encoding = ExecutableDoubleFactorization.from_eigenvalues(
        (1.0, 2.0, 3.0),
        ((1.0, -1.0, 2.0),) * 24,
        ((0, 0),) * 75,
        outer_bits=2,
        inner_bits=2,
        rank_rotation_bits=3,
        basis_rotation_bits=4,
    )
    source = DoubleFactorizationBlockEncoding(
        num_aux=24,
        num_spin_orb=6,
        num_eig=72,
        num_bits_state_prep=2,
        num_bits_rot=4,
        num_bits_rot_aa_inner=3,
    )
    report = estimate_ftqc_resources(encoding.source_resource_model())
    assert report.emitted is None
    assert report.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert {
        key: count for key, count in asdict(report.logical.gates).items() if count
    } == get_cost_value(source, QECGatesCost()).asdict()


def test_df_source_estimate_rejects_distinct_alias_precisions():
    encoding = ExecutableDoubleFactorization.from_eigenvalues(
        (1.0,), (), ((),), outer_bits=1, inner_bits=2
    )
    with pytest.raises(ValueError, match="requires shared alias precision"):
        encoding.source_resource_model()


@pytest.mark.parametrize("spatial,controls", [(3, 2), (5, 2), (3, 3)])
def test_df_number_phase_uses_locked_predicate_and_cleans_every_branch(
    spatial, controls
):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    flags = tuple(range(controls))
    system = tuple(range(controls, controls + spatial))
    first_work = controls + spatial
    work = tuple(range(first_work, first_work + controls - 1))
    slots = tuple(range(len(work)))
    width = first_work + len(work)
    values = (1,) * controls
    source = ApplyControlledZs(values, spatial)
    stream = multi_control_z_ir(flags, values, system[0], work, slots)
    reference = reference_arithmetic_gates(
        source,
        {"ctrls": flags, "system": system},
        work,
        dict(zip(work, slots, strict=True)),
    )
    assert_source(stream, reference)
    assert (
        sum(gate.name in {"t", "tdg"} for gate in stream)
        == get_cost_value(source, QECGatesCost()).total_t_count()
    )
    events = sum(gate.name == "measure" for gate in stream)
    assert events == len(work)
    amplitude = complex(1, 1) / sqrt(2 ** (first_work + 1))
    initial = {
        index << len(work): amplitude * (-1j) ** (index % 4)
        for index in range(1 << first_work)
    }
    active = (1 << (controls + 1)) - 1
    expected = {
        index: -value if index >> (width - controls - 1) == active else value
        for index, value in initial.items()
    }
    for history in product((0, 1), repeat=events):
        assert _branch(
            stream, width, history, initial, normalize_measurements=True
        ) == pytest.approx(expected, abs=2e-12)
