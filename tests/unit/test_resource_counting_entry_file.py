"""Audit the complete frozen resource entry and independent native entry paths."""

from ast import ImportFrom, parse
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import FixedTCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.primitives.controlled_swap import (
    controlled_swap_bit_ir,
)
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.core.ir import ensure_circuit_ir

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REPO = Path(__file__).resolve().parents[2]


def test_every_frozen_resource_entry_has_a_resolvable_native_owner() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    path = REPO.parent / "ftqc-reference-env/qualtran/resource_counting/__init__.py"
    if not path.is_file():
        pytest.skip("Frozen resource entry file is not installed")
    blob = path.read_bytes()
    assert sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest() == (
        "22368f539adb6fd0fd0ac02cff49263b661227bd"
    )
    names = [
        alias.asname or alias.name
        for node in parse(blob).body
        if isinstance(node, ImportFrom)
        for alias in node.names
    ]
    assert len(names) == 22 and len(set(names)) == 21
    assert names.count("GeneralizerT") == 2

    audit = REPO / "docs/development/RESOURCE_COUNTING_ENTRY_AUDIT.md"
    rows = {
        row.split("|")[1].strip().strip("`"): row.split("|")[2].strip().strip("`")
        for row in audit.read_text().splitlines()
        if row.startswith("| `")
    }
    assert set(names) == set(rows)
    for name, owner in rows.items():
        assert hasattr(source, name)
        module_name, _, qualified_name = owner.partition(":")
        native: object = import_module(module_name)
        for attribute in filter(None, qualified_name.split(".")):
            native = getattr(native, attribute)
        assert native is not None, name


def test_append_static_and_declared_fredkin_counts_use_distinct_owners() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    from qualtran.bloqs.basic_gates import TwoBitCSwap

    state = torch.tensor(
        [complex(i + 1, 2 * i - 3) for i in range(8)], dtype=torch.complex128
    )
    state /= state.norm()
    circuit = Circuit(3, inputs=state, dtype=torch.complex128)
    gates = controlled_swap_bit_ir(0, 1, 2)
    circuit.append(FieldOperation(3, gates))
    expected = state.clone()
    expected[5], expected[6] = state[6], state[5]
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)

    estimate = estimate_resources(circuit)
    assert estimate.n_operations == 17 and estimate.t_count == 7
    assert estimate.operation_counts == {"cx": 8, "h": 2, "t": 4, "tdg": 3}
    ref = source.get_cost_value(TwoBitCSwap(), source.QECGatesCost())
    assert (
        query_qec_gate_cost(FixedTCall("fredkin")).asdict()
        == ref.asdict()
        == {"cswap": 1}
    )
    report = resource_model_report(
        ComponentCost("Fredkin", 3, GateCost(cswap=1)),
        emitted_ir=ensure_circuit_ir(circuit),
    )
    assert report["t_equivalents"] == 4
    assert report["emitted_ir"]["t_count"] == estimate.t_count
    assert report["execution_measurements"] is None
