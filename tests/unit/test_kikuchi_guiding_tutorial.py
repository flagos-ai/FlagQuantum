"""Tutorial word graphs retain the frozen root and one-level expansion."""

import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import sympy

from flagquantum.algorithms.kikuchi_guiding_resource_graph import (
    source_guiding_call_graph,
)
from flagquantum.algorithms.kikuchi_guiding_state import (
    GuidingState,
    ProbabilisticUncompute,
    ScopeGuidingState,
)
from flagquantum.algorithms.kxor_instance import KXorInstance
from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.compiler.registers.component_expansion import (
    expand_component_calls_once,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.test_component_expansion import _labelled_trace
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def tutorial_instances(symbolic: bool) -> tuple[GuidingState, Any]:
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        GuidingState as OriginalGuidingState,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        KXorInstance as OriginalInstance,
    )

    if symbolic:
        n, m, k = sympy.symbols("n m k", positive=True, integer=True)
        return (
            GuidingState(KXorInstance.symbolic(n, m, k), 3 * k),
            OriginalGuidingState(OriginalInstance.symbolic(n, m, k), 3 * k),
        )
    arguments: dict[str, Any] = dict(n=20, m=100, k=4, planted_advantage=0.8)
    return (
        GuidingState(
            KXorInstance.random_instance(**arguments, rng=np.random.default_rng(100)),
            12,
        ),
        OriginalGuidingState(
            OriginalInstance.random_instance(
                **arguments, rng=np.random.default_rng(100)
            ),
            12,
        ),
    )


@pytest.mark.parametrize("symbolic", (False, True))
def test_default_tutorial_root_retains_exact_source_word_connections(
    symbolic: bool,
) -> None:
    native, original = tutorial_instances(symbolic)
    graph = native.program()
    assert not verify_program_record(graph)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq())
    )
    assert len(graph.functions[0].blocks[0].operations) == 12


def test_concrete_tutorial_flatten_once_keeps_source_atomic_partitions() -> None:
    native, original = tutorial_instances(False)
    graph = native.program()
    functions = {function.name: function for function in graph.functions}
    flattened, labels = expand_component_calls_once(
        graph,
        predicate=lambda _, operation: all(
            register.side == RegisterSide.THRU
            for register in functions[operation.callee or ""].registers or ()
        ),
    )
    assert not verify_program_record(flattened)
    assert len(flattened.functions[0].blocks[0].operations) == 21
    assert _named_inputs(_labelled_trace(flattened, labels)) == _named_inputs(
        _source_trace(SourceCalls, original.decompose_bloq().flatten_once())
    )


def test_default_concrete_duplicate_dependency_has_the_actual_source_body() -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    native, _ = tutorial_instances(False)
    graph = native.program()
    candidates = [
        function
        for function in graph.functions
        if tuple(register.name for register in function.registers or ())
        == ("xs", "flag")
        and function.blocks
        and function.registers is not None
        and function.registers[0].shape == (12,)
    ]
    assert len(candidates) == 1
    duplicate = candidates[0]
    dependency = ProgramRecord(
        (
            replace(duplicate, name="main"),
            *(f for f in graph.functions if f.name != "main"),
        )
    )
    compare_layers(dependency, HasDuplicates(12, QUInt(5)), atomic_leaves(dependency))


def native_model_key(node: Any) -> Any:
    if isinstance(node, GuidingState):
        return ("guide", node.ell, node.inst.index_bitsize)
    if isinstance(node, ScopeGuidingState):
        return ("scope", node.target_bitsize, node.phasegrad_bitsize)
    if isinstance(node, ProbabilisticUncompute):
        return ("uncompute", node.bitsize)
    if isinstance(node, SparseRotationPreparation):
        return (
            "sparse",
            node.size,
            node.indices,
            node.coefficients,
            node.phase_width,
            node.target_width,
        )
    if isinstance(node, RotationPreparation):
        return ("dense", node.coefficients, node.phase_width)
    assert isinstance(node, tuple)
    return node


def source_model_key(node: Any) -> Any:
    name = type(node).__name__
    if name == "GuidingState":
        return ("guide", node.ell, node.inst.index_bitsize)
    if name == "SimpleGuidingState":
        return ("scope", node.target_bitsize, node.phasegrad_bitsize)
    if name == "ProbabilisticUncompute":
        return ("uncompute", node.bitsize)
    if name == "SparseStatePreparationViaRotations":
        return (
            "sparse",
            node.N,
            node.sparse_indices,
            node.nonzero_coeffs,
            node.phase_bitsize,
            node.target_bitsize,
        )
    if name == "StatePreparationViaRotations":
        assert node.control_bitsize == 0 and not node.uncompute
        return ("dense", node.state_coefficients, node.phase_bitsize)
    if name == "Permutation":
        return ("permutation", node.N, node.cycles, node.bitsize)
    if name in {"SortInPlace", "HasDuplicates"}:
        return (
            "sort_in_place" if name == "SortInPlace" else "duplicates",
            node.l,
            node.dtype.bitsize,
        )
    if name == "Comparator":
        return ("comparator", node.bitsize)
    if name == "GreaterThan":
        return ("greater_than", node.a_bitsize, node.b_bitsize)
    if name == "Adjoint":
        key = source_model_key(node.subbloq)
        if key[0] == "ctrl_spec_and":
            return (*key[:-1], True)
        assert key[0] in {"comparator", "greater_than"}
        return (key[0] + "_adjoint", *key[1:])
    if name in {"Allocate", "Free"}:
        assert node.dtype.num_qubits == 1 and not node.dirty
        return ("allocate_bit" if name == "Allocate" else "free_bit",)
    if name == "CSwap":
        return ("controlled_swap", node.bitsize)
    if name == "Xor":
        return ("xor", node.dtype.bitsize)
    if name == "LinearDepthHalfLessThan":
        return ("half_less_than", node.dtype.bitsize, node.uncompute)
    if name == "BitwiseNot":
        assert type(node.dtype).__name__ == "QUInt"
        return ("bitwise_not", node.dtype.bitsize)
    if name == "OutOfPlaceAdder":
        return (
            "out_of_place_add",
            node.bitsize,
            node.is_adjoint,
            node.include_most_significant_bit,
        )
    if name == "MultiControlX":
        return ("multi_control_x", tuple(node.concrete_cvs))
    if name == "OnEach":
        assert type(node.gate).__name__ == "Hadamard"
        return ("on_each_h", node.n)
    if name in {"ControlledViaAnd", "CtrlSpecAnd"}:
        spec = node.ctrl_spec
        assert len(spec.qdtypes) == len(spec.cvs) == 1
        dtype, values = spec.qdtypes[0], np.asarray(spec.cvs[0])
        word = type(dtype).__name__ == "QAny"
        if word:
            assert values.shape == ()
            width, activation = int(dtype.num_qubits), int(values)
            pattern = tuple((activation >> (width - i - 1)) & 1 for i in range(width))
        else:
            assert type(dtype).__name__ == "QBit" and values.ndim == 1
            pattern = tuple(map(int, values))
        if name == "ControlledViaAnd":
            assert type(node.subbloq).__name__ == "XGate"
            return ("controlled_x", pattern, word)
        return ("ctrl_spec_and", pattern, word, False)
    primitive = {"XGate": "x", "CNOT": "cx", "Hadamard": "h"}
    assert name in primitive, repr(node)
    return (primitive[name],)


@pytest.mark.parametrize("depth", (0, 1, 2, 3))
def test_concrete_tutorial_model_graph_retains_every_call_parameter_and_weight(
    depth: int,
) -> None:
    native, original = tutorial_instances(False)
    actual = source_guiding_call_graph(native, max_depth=depth)
    expected, expected_leaves = original.call_graph(max_depth=depth)
    assert native_model_key(actual.root) == source_model_key(original)
    assert {
        (native_model_key(a), native_model_key(b)): count
        for (a, b), count in actual.direct_calls.items()
    } == {
        (source_model_key(a), source_model_key(b)): data["n"]
        for a, b, data in expected.edges(data=True)
    }
    assert {
        native_model_key(node): count for node, count in actual.leaf_counts.items()
    } == {source_model_key(node): count for node, count in expected_leaves.items()}


def test_complete_native_tutorial_cells_render_graphs_and_preserve_costs_without_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        GuidingState as OriginalGuide,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        KXorInstance as OriginalInstance,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    from tests.unit.test_addition_file_audit import execute

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/optimization/k_xor_sat/kikuchi_guiding_state_tutorial.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "34e5a813e1b5eaa1f5a9f5774f145be5833c789d"
    )
    original = json.loads(raw)
    notebook = json.loads(
        (root / "examples/notebooks/kikuchi_guiding_state_tutorial.ipynb").read_text()
    )
    assert len(original["cells"]) == 17 and len(notebook["cells"]) == 18
    assert [cell["cell_type"] for cell in notebook["cells"][:-1]] == [
        cell["cell_type"] for cell in original["cells"]
    ]
    assert [
        cell["metadata"]["frozen_source_cell"] for cell in notebook["cells"][:-1]
    ] == list(range(17))
    images: list[object] = []
    monkeypatch.setattr(IPython.display, "display", images.append)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion", "qiskit"}:
            raise AssertionError(
                "Native tutorial must use domain-owned implementations"
            )
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        namespace = execute(notebook)
    assert len(images) == 6
    assert all(
        isinstance(image, IPython.display.SVG) and "<svg" in image.data
        for image in images
    )
    native, source = tutorial_instances(False)
    assert isinstance(native.inst.constraints, tuple)
    assert [(row.S, row.b) for row in native.inst.constraints] == [
        (row.S, row.b) for row in source.inst.constraints
    ]
    original_symbolic = OriginalGuide(
        OriginalInstance.symbolic(namespace["n"], namespace["m"], namespace["k"]),
        namespace["c"] * namespace["k"],
    )
    for guide, expected in (
        (namespace["guiding_state_concrete"], source),
        (namespace["guiding_state"], original_symbolic),
    ):
        actual_cost = guide.resource_gates()
        source_cost = get_cost_value(expected, QECGatesCost())
        for category in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert (
                sympy.simplify(
                    getattr(actual_cost, category) - getattr(source_cost, category)
                )
                == 0
            )
    expected_total = sympy.sympify(
        get_cost_value(original_symbolic, QECGatesCost()).total_t_count()
    )
    assert sympy.simplify(namespace["raw_t_cost"] - expected_total) == 0
    expected_total = (
        expected_total.subs(namespace["klogn_long"], namespace["klogn"])
        .simplify()
        .subs(namespace["c"] * namespace["k"], namespace["l_symb"])
    )
    assert sympy.simplify(namespace["t_cost"] - expected_total) == 0
    assert namespace["declared_order"] == sympy.Order(
        expected_total, *((symbol, sympy.oo) for symbol in expected_total.free_symbols)
    )
    actual = source_guiding_call_graph(namespace["guiding_state"], max_depth=1)
    expected_graph, _ = original_symbolic.call_graph(max_depth=1)
    assert {
        (native_model_key(a), native_model_key(b)): count
        for (a, b), count in actual.direct_calls.items()
    } == {
        (source_model_key(a), source_model_key(b)): data["n"]
        for a, b, data in expected_graph.edges(data=True)
    }
    assert namespace["measured_primitives"]["measure"] == 1
    assert namespace["uncompute_model"].and_bloq == 1


def test_complete_original_guiding_test_roles_and_all_raw_protocol_outcomes() -> None:
    import ast

    import qualtran.testing as source_testing
    from qualtran._infra.bloq_example import BloqExample
    from qualtran.bloqs.optimization.k_xor_sat import kikuchi_guiding_state as source

    path = Path(source.__file__).with_name("kikuchi_guiding_state_test.py")
    definitions = {
        node.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    assert definitions == {
        "test_examples",
        "test_t_cost_simple",
        "test_t_cost_simple_symb",
        "test_t_cost",
        "test_t_cost_symb_c",
        "test_notebook",
        "test_tutorial",
    }
    examples: tuple[BloqExample[Any], ...] = (
        source._simple_guiding_state,
        source._simple_guiding_state_symb,
        source._guiding_state,
        source._guiding_state_symb,
        source._guiding_state_symb_c,
    )
    checks = (
        source_testing.assert_bloq_example_make,
        source_testing.assert_bloq_example_decompose,
        source_testing.assert_equivalent_bloq_example_counts,
        source_testing.assert_bloq_example_serializes,
        source_testing.assert_bloq_example_qtyping,
    )
    expected = (
        ("PASS", "MISSING", "UNVERIFIED", "FAIL", "NA"),
        ("PASS", "NA", "UNVERIFIED", "FAIL", "NA"),
        ("PASS", "PASS", "FAIL", "FAIL", "PASS"),
        ("PASS", "PASS", "FAIL", "FAIL", "PASS"),
        ("PASS", "NA", "UNVERIFIED", "FAIL", "NA"),
    )
    for example, outcomes in zip(examples, expected, strict=True):
        for check, outcome in zip(checks, outcomes, strict=True):
            try:
                check(example)
            except source_testing.BloqCheckException as error:
                assert error.check_result.name == outcome
            else:
                assert outcome == "PASS"
