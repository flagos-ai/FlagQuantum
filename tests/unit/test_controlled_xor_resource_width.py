"""Compare controlled register-XOR liveness and caches with frozen source."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    FixedTCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.bits.controlled_xor_width import (
    controlled_xor_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)
from tests.unit.test_addition_resource_width import ENCODINGS

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
p = Symbol("p", nonnegative=True, integer=True)
WIDTHS = (1, 2, 3, 4, 5, 8, Integer(1), Integer(3), n, p)


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/resource_counting/_qubit_counts.py",
        "qualtran/resource_counting/_bloq_counts.py",
        "qualtran/bloqs/arithmetic/bitwise.py",
        "qualtran/_infra/controlled.py",
        "qualtran/bloqs/mcmt/controlled_via_and.py",
        "qualtran/bloqs/basic_gates/toffoli.py",
        "qualtran/bloqs/bookkeeping/split.py",
        "qualtran/bloqs/bookkeeping/join.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran import CtrlSpec, QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import Xor

    dtype = {"unsigned": QUInt, "twos_complement": QInt, "montgomery": QMontgomeryUInt}[
        call.encodings[0]
    ](call.widths[0])
    source = Xor(dtype).controlled(CtrlSpec(cvs=call.control_value))
    return source.adjoint() if call.adjoint else source


def source_key(source: Any, root: ArithmeticCall) -> ResourceCall:
    from qualtran import Controlled
    from qualtran.bloqs.basic_gates import Toffoli, XGate
    from qualtran.bloqs.bookkeeping import Join, Split
    from qualtran.bloqs.mcmt import ControlledViaAnd

    if source == source_for(root):
        return root
    if isinstance(source, (Controlled, ControlledViaAnd)):
        return replace(root, control_value=int(source.ctrl_spec.cvs[0]), adjoint=False)
    if isinstance(source, (Split, Join)):
        return BookkeepingCall(
            "split" if isinstance(source, Split) else "join", source.dtype.num_qubits
        )
    if isinstance(source, Toffoli):
        return FixedTCall("toffoli")
    assert isinstance(source, XGate)
    return CliffordCall("x", 1)


def compare(call: ArithmeticCall, seed: Any = None) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = {} if seed is None else {seed: 20}
    actual: dict[ResourceCall, SymbolicInt] = (
        {} if seed is None else {source_key(seed, call): 20}
    )
    assert query_resource_logical_width(call, cache=actual) == get_cost_value(
        source_for(call), QubitCount(), costs_cache=expected
    )
    normalized = {source_key(key, call): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_cache_and_order(
    width: SymbolicInt, encoding: ArithmeticEncoding, cv: int, adjoint: bool
) -> None:
    compare(
        ArithmeticCall(
            "controlled_xor", (width,), (encoding,), control_value=cv, adjoint=adjoint
        )
    )


@pytest.mark.parametrize("width", [1, 3, n, Integer(3)])
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("seed_name", ["toffoli", "x", "split", "join", "positive"])
def test_live_words_and_control_across_child_peaks(
    width: SymbolicInt, cv: int, adjoint: bool, seed_name: str
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import Toffoli, XGate
    from qualtran.bloqs.bookkeeping import Join, Split

    call = ArithmeticCall("controlled_xor", (width,), control_value=cv, adjoint=adjoint)
    seeds = {
        "toffoli": Toffoli(),
        "x": XGate(),
        "split": Split(QUInt(width if type(width) is int else 3)),
        "join": Join(QUInt(width if type(width) is int else 3)),
        "positive": source_for(replace(call, control_value=1)),
    }
    compare(call, seeds[seed_name])


@pytest.mark.parametrize("width", [1, 3, n, Integer(3)])
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_full_dot_uses_source_width_cache(
    width: SymbolicInt,
    cv: int,
    adjoint: bool,
    depth: int | None,
    aggregation: GateAggregation,
) -> None:
    call = ArithmeticCall("controlled_xor", (width,), control_value=cv, adjoint=adjoint)
    widths = compare(call)
    declarations = arithmetic_call_graph(call)
    assert export_resource_call_graph_dot(
        call, declared_calls=declarations, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        call,
        declared_calls=declarations,
        logical_width=widths.get,
        max_depth=depth,
        aggregation=aggregation,
    )


@pytest.mark.parametrize("width", [0, Integer(0)])
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_zero_failure_partial_cache_and_retry(
    width: SymbolicInt, cv: int, adjoint: bool
) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("controlled_xor", (width,), control_value=cv, adjoint=adjoint)
    source_cache: dict[Any, SymbolicInt] = {}
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount(), costs_cache=source_cache)
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="positive"):
        query_resource_logical_width(call, cache=cache)
    assert cache == {
        source_key(key, call): value for key, value in source_cache.items()
    }
    assert (CliffordCall("x", 1) in cache) == (cv == 0)
    assert query_resource_logical_width(replace(call, widths=(3,)), cache=cache) == 7


def test_owner_boundary_and_decomposition_priority() -> None:
    with pytest.raises(NotImplementedError):
        controlled_xor_width(ArithmeticCall("xor", (3,)), lambda call: 0)
    call = ArithmeticCall("controlled_xor", (3,))
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 7
    )
    assert BookkeepingCall("split", 3) in cache
    assert FixedTCall("toffoli") in cache


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("cv", [0, 1])
def test_append_preserves_complex_control_source_and_spectator(
    width: int, cv: int
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.bits.controlled_register_xor import (
        grouped_controlled_xor_ir,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    count = 2 * width + 2
    data = torch.arange(1 << count, dtype=torch.int64)
    coefficients = data.to(torch.float64)
    state = torch.complex(coefficients + 1, 3 * coefficients - 5)
    state /= state.norm()
    active = (data >> (2 * width + 1)) == cv
    source = (data >> (width + 1)) & ((1 << width) - 1)
    mapped = data ^ (active.to(torch.int64) * (source << 1))
    expected = torch.zeros_like(state)
    expected[mapped] = state
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 1, 2 * width + 1))
    gates = grouped_controlled_xor_ir(
        ((0,),), (ControlRegister(1, (cv,)),), left, right, (), ()
    )
    assert sum(g.name == "ccx" for g in gates) == width
    assert sum(g.name == "x" for g in gates) == 2 * (cv == 0)
    assert all(g.name in {"ccx", "x"} for g in gates)
    op = FieldOperation(count, gates)
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(op).state().flatten(), state, rtol=0, atol=1e-12
    )
