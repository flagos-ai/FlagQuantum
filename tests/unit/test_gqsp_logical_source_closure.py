"""Preserve successful GQSP source graphs above genuinely opaque source leaves."""

from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import cirq as cirq

    import_module("numpy")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_thc_selection_factories import (
    thc_selection_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from flagquantum.ecosystem._generalized_qsp_component_library import (
    generalized_qsp_library_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_generalized_qsp_component_library import (
    _assert_source_call_operands,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_majorana_factories import pinned_majorana as pinned_majorana
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_gqsp_source() -> None:
    from qualtran.bloqs.qsp import generalized_qsp

    for filename, digest in (
        ("generalized_qsp.py", "5848dcb4785697563fca6173e499015e504b233b"),
        ("generalized_qsp_test.py", "c925da2a851888c4ceb3c8dbc1893c93e3c7602d"),
    ):
        raw = (Path(generalized_qsp.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def opaque_gqsp(power: int) -> tuple[Any, dict[str, ComponentFactory]]:
    from qualtran.bloqs.chemistry.thc.select_bloq import THCRotations
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP

    source = GeneralizedQSP(
        THCRotations(2, 4, 8),
        P=(0, 0, 0.6),
        Q=(0.8j, 0, 0),
        negative_power=power,
        precision=1e-7,
    )

    def declaration(name: str, original: Any) -> ProgramRecord:
        return ProgramRecord(
            (FunctionRecord(name, (), registers=tuple(_signature(original.signature))),)
        )

    forward = declaration("opaque_forward", source.U)
    inverse = declaration("opaque_inverse", source.U.adjoint())
    positive = (
        declaration("positive_forward", source.U.controlled()),
        declaration("positive_inverse", source.U.adjoint().controlled()),
    )

    def ordinary(child: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert not child.blocks and child.registers == forward.functions[0].registers
        return forward, inverse

    def controlled(child: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
        assert not child.blocks and child.registers == forward.functions[0].registers
        return positive

    return source, {
        **thc_selection_factories(),
        **generalized_qsp_library_factories(
            child_owner=ordinary, positive_child_owner=controlled
        ),
    }


@pytest.mark.parametrize("power", (0, 1, 2, 5))
def test_actual_opaque_source_child_preserves_available_calls_and_binding_refusal(
    power: int,
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    from flagquantum.compiler.formats.legacy_su2_factories import _su2_factory

    source, factories = opaque_gqsp(power)
    source_graph = source.decompose_bloq()
    assert len(source_graph.bloq_instances) == 5 + max(0, power - 2)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, factories)
    assert verify_program_record(record) == ()
    _assert_source_call_operands(source, record)
    # SU2's four-call owner already has complete complex/branch evidence. Stop
    # there rather than treating its scalar carrier as a source decomposition.
    operations = record.functions[0].blocks[0].operations
    roots = [operation.callee for operation in operations[:5:2]]
    assert all(name is not None for name in roots)
    terminals = atomic_leaves(record) | {name for name in roots if name is not None}
    for name, rotation in zip(roots, source.signal_rotations, strict=True):
        assert name is not None
        expected = _su2_factory(
            "main",
            None,
            {
                key: float(getattr(rotation, key))
                for key in ("theta", "phi", "lambd", "global_shift", "eps")
            },
        )
        assert matches_component_definition(record, name, expected)
    pairs = compare_layers(record, source, terminals)
    functions = {function.name: function for function in record.functions}
    for name, original in pairs.items():
        if not functions[name].blocks:
            with pytest.raises((DecomposeNotImplementedError, DecomposeTypeError)):
                original.decompose_bloq()
    _assert_declared_model(source, record, power)
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_used_workspace(record, allow_workspace_outputs=True)
    import sys

    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert resolve_component_program(library, 0, factories) == record


def _assert_declared_model(source: Any, record: ProgramRecord, power: int) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.generalized_qsp_model import (
        source_generalized_qsp_graph,
    )
    from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
        thc_rotations_cost,
    )
    from flagquantum.compiler._component_trace.compound_control_resource import (
        compound_control_key,
    )
    from flagquantum.compiler._component_trace.program_logical_width import (
        query_program_logical_width,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import (
        QECGateCost,
        query_qec_gate_cost,
    )
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )

    forward = estimate_component_resources(thc_rotations_cost(2, 4, 8)).gates
    inverse = estimate_component_resources(
        thc_rotations_cost(2, 4, 8, adjoint=True)
    ).gates
    controlled_key = compound_control_key(FixedTCall("toffoli"), (1,))
    assert controlled_key is not None
    controlled_toffoli = query_qec_gate_cost(controlled_key)
    costs = {
        "rotation": QECGateCost(rotation=3),
        "negative": controlled_toffoli * forward.toffoli + QECGateCost(clifford=2),
        "positive_inverse": controlled_toffoli * inverse.toffoli,
        "inverse": QECGateCost(toffoli=inverse.toffoli),
    }
    graph = source_generalized_qsp_graph(
        3,
        negative_power=power,
        root="gqsp",
        signal_rotation="rotation",
        zero_control_unitary="negative",
        unitary_adjoint="inverse",
        controlled_adjoint="positive_inverse",
    )
    model = QECGateCost()
    for key, count in graph.leaf_counts.items():
        model = model + costs[key] * count
    assert model.asdict() == get_cost_value(source, QECGatesCost()).asdict()
    assert (
        query_program_logical_width(record)
        == get_cost_value(source, QubitCount())
        == 14
    )
    # These are declared categories and logical peaks. Opaque U prevents a
    # whole physical primitive inventory or numerical cleanup certificate.


def test_unknown_signature_still_fails_without_substituting_identity() -> None:
    from flagquantum.compiler.registers.phase.generalized_qsp_component import (
        generalized_qsp_component_program,
    )

    source, _ = opaque_gqsp(0)
    signature = tuple(_signature(source.U.signature))
    opaque = ProgramRecord((FunctionRecord("opaque", (), registers=signature),))
    wrong = replace(opaque.functions[0], registers=None)
    with pytest.raises(ValueError, match="matching through"):
        generalized_qsp_component_program(
            ProgramRecord((wrong,)), opaque, opaque, opaque, (opaque, opaque)
        )
