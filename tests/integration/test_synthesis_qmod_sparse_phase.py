"""Check sparse phase scale, distinct budgets and original planning costs."""

import hashlib
from pathlib import Path

import pytest

from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    ("name", "blob", "count", "operations", "depth"),
    [
        (
            "resiliency_planning",
            "0034d5aba2801be900a83a9171b36c1c0a5c77e8",
            40,
            2201,
            269,
        ),
        (
            "resiliency_planning_AMD",
            "0034d5aba2801be900a83a9171b36c1c0a5c77e8",
            40,
            2201,
            269,
        ),
        (
            "network_traffic_optimization",
            "9534cb1d36f6b7c77f207cdff43ca4d67030d689",
            10,
            851,
            110,
        ),
    ],
)
def test_original_sparse_phase_planning(name, blob, count, operations, depth):
    path = FIXTURES / f"{name}.qmod"
    raw = path.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    program = _synthesize(
        path,
        classical_inputs={"params": [0.01 * (index + 1) for index in range(count)]},
        memory_limit_bytes=512 * 1024**2,
    )
    resources = program.report["resources"]
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (
        24,
        operations,
        depth,
    )
    assert program.report["workspace_qubits"] == 0
    with pytest.raises(ValueError, match="state requires|memory budget"):
        _synthesize(
            path,
            classical_inputs={"params": [0.01 * (index + 1) for index in range(count)]},
            memory_limit_bytes=128 * 1024**2,
        )


@pytest.mark.parametrize(
    ("expression", "diagnostic"),
    [
        ("*".join(f"(1+q[{bit}])" for bit in range(9)), "256 monomials"),
        (
            "(" + "*".join(f"(1+q[{bit}])" for bit in range(8)) + ")**2",
            "20000 monomial products",
        ),
    ],
)
def test_sparse_phase_retains_expansion_budgets(expression, diagnostic):
    source = (
        f"qfunc layer(g:real,q:qbit[]) {{phase({expression},g);}}"
        "qfunc main(p:real[1],output q:qbit[9]) {allocate(q); layer(p[0],q);}"
    )
    with pytest.raises(ValueError, match=diagnostic):
        _synthesize(source, source_format="qmod", classical_inputs={"p": [0.1]})
