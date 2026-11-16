"""Verify native logical-width entry points and the pinned teaching fixture."""

from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol

from examples.algorithms.qubit_count_protocol_catalog import (
    interior_width_call_graph,
    interior_width_connections,
    interior_width_model,
    interior_workspace_fixture,
    interior_workspace_signature,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.program_logical_width import (
    query_program_logical_width,
)
from flagquantum.compiler._component_trace.width_schedule import (
    estimate_connection_width,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from tests.unit.many_allocation_fixture import AllocationMode, many_allocation_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/resource_counting"
)
BLOBS = {
    "_qubit_counts.py": "723e06b64ef3b589abb36538e19b79369f1bb8f8",
    "_qubit_counts_test.py": "6f840fc8697cd7f2b649c63540a7b7e263abef3f",
    "qubit_counts.ipynb": "c23db8d556b786164ae87aa9a915119729b1de4a",
}


def test_frozen_qubit_count_family() -> None:
    pytest.importorskip("qualtran")
    for name, expected in BLOBS.items():
        path = REFERENCE / name
        if not path.is_file():
            pytest.skip("Frozen qubit-count assets are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


@pytest.mark.parametrize("width", [1, 3, 10, Symbol("n", positive=True, integer=True)])
def test_symbolic_and_concrete_interior_model_match_source(
    width: SymbolicInt,
) -> None:
    pytest.importorskip("qualtran")
    reference = import_module(
        "qualtran.bloqs.for_testing.interior_alloc"
    ).InteriorAlloc(width)
    costing = import_module("qualtran.resource_counting")
    source_width = import_module("qualtran.resource_counting._qubit_counts")
    cache: dict[str, SymbolicInt] = {}
    expected_cache: dict[Any, SymbolicInt] = {}
    expected = costing.get_cost_value(
        reference, costing.QubitCount(), costs_cache=expected_cache
    )
    model = interior_width_model(width, cache=cache)
    assert (
        interior_workspace_signature(width).n_qubits()
        == reference.signature.n_qubits()
        == 2 * width
    )
    assert model == expected == 3 * width
    assert model == costing.get_cost_value(
        reference.decompose_bloq(), costing.QubitCount()
    )
    steps, edges = interior_width_connections(width, cache.__getitem__)
    assert estimate_connection_width(steps, edges) == model
    zero_steps, incoming = interior_width_connections(width, lambda _: 0)
    assert estimate_connection_width(
        zero_steps, incoming
    ) == source_width._cbloq_max_width(reference.decompose_bloq()._binst_graph)
    for name in ("Allocate", "Free", "Swap", "TwoBitSwap"):
        assert cache[name] == next(
            value
            for node, value in expected_cache.items()
            if type(node).__name__ == name
        )
    direct = interior_width_call_graph(width).direct_calls
    upstream, _ = reference.call_graph()
    assert {
        (
            "Interior allocation" if parent is reference else type(parent).__name__,
            type(child).__name__,
        ): data["n"]
        for parent, child, data in upstream.edges.data()
    } == direct

    if isinstance(width, int):
        program = interior_workspace_fixture(width)
        typed: dict[str, SymbolicInt] = {}
        assert query_program_logical_width(program, cache=typed) == model
        assert typed["swap_word"] == 2 * width
        assert typed["swap_bit"] == 2
        bound = bind_register_program(
            program,
            (tuple(range(width)), tuple(range(width, 2 * width))),
            clean=tuple(range(2 * width, 3 * width)),
        )
        assert bound.circuit.n_wires == model
        assert len(bound.circuit.instructions) == 2 * width


def test_disconnected_and_serial_native_programs_match_source() -> None:
    pytest.importorskip("qualtran")
    fixtures = import_module("qualtran.bloqs.for_testing.with_decomposition")
    costing = import_module("qualtran.resource_counting")
    bit = quantum_bit_type()
    leaf = bit_gate_function("gate", (Register("q", bit),), Circuit(1).t(0).to_ir())
    parallel = ComponentProgramBuilder(Signature(()), (leaf,))
    for _ in range(3):
        body = parallel.call_dict("gate", {"q": parallel.allocate(dtype=bit)})["q"]
        assert isinstance(body, RegisterBinding)
        parallel.free(body)
    serial = ComponentProgramBuilder(Signature((Register("q", bit),)), (leaf,))
    values = serial.inputs
    for _ in range(3):
        values = serial.call_dict("gate", values)
    for program, source in (
        (parallel.finalize({}), fixtures.TestIndependentParallelCombo()),
        (serial.finalize(values), fixtures.TestSerialCombo()),
    ):
        assert (
            query_program_logical_width(program)
            == (costing.get_cost_value(source.decompose_bloq(), costing.QubitCount()))
            == 1
        )


@pytest.mark.parametrize("mode", ["once", "many", "abstracted"])
def test_typed_allocation_schedules_match_pinned_peak(mode: AllocationMode) -> None:
    pytest.importorskip("qualtran")
    reference = import_module("qualtran.bloqs.for_testing.qubit_count_many_alloc")
    costing = import_module("qualtran.resource_counting")
    source = {
        "once": reference.TestManyAllocOnce,
        "many": reference.TestManyAllocMany,
        "abstracted": reference.TestManyAllocAbstracted,
    }[mode](10)
    native = many_allocation_fixture(10, mode)
    assert (
        query_program_logical_width(native)
        == costing.get_cost_value(source, costing.QubitCount())
        == 11
    )


def test_typed_width_counts_callee_peak_and_live_bystanders() -> None:
    bit = quantum_bit_type()
    gate = bit_gate_function("gate", (Register("q", bit),), Circuit(1).t(0).to_ir())
    builder = ComponentProgramBuilder(Signature.build(q=1, spare=1), (gate,))
    acted = builder.call_dict("gate", {"q": builder.inputs["q"]})
    record = builder.finalize({**acted, "spare": builder.inputs["spare"]})
    assert query_program_logical_width(record, "gate") == 1
    cache: dict[str, SymbolicInt] = {}
    assert (
        query_program_logical_width(record, cache=cache, static_costs={"gate": 7}) == 8
    )
    assert cache == {"gate": 7, "main": 8}
