"""Original QAOA state vectors and reusable bounded Boolean phase factors."""

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum import synth
from flagquantum.compiler._construction.boolean_lowering import lower_boolean_value
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.ecosystem._qmod.syntax import Parser, tokenize
from tools.validate_qmod_phase_factors import FIXTURES, validate

pytestmark = pytest.mark.integration


def test_signed_cube_on_a_wide_owner_all_basis_inputs_and_nonzero_xor_output() -> None:
    source = """qfunc main(output x:qnum<12>) {
      allocate(x);flag:qnum<1>;
      within {flag=(x&2730)==2082;} apply {phase(flag,0.37);}
    }"""
    imported = parse_qmod(
        source, filename="mask.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    graph = imported.functions[0]
    argument = graph.body.blocks[0].arguments[0]
    candidate = lower_boolean_value(
        (graph,),
        {argument.id: tuple(range(12))},
        (12,),
        entry=graph.name,
        n_qubits=17,
        workspace=(13, 14, 15, 16),
        workspace_state="fresh_zero",
    )
    assert len(candidate.circuit.instructions) <= 17
    assert len(candidate.workspace) == 4
    for x in range(4096):
        for target in (0, 1):
            registers = (tuple(range(12)), (12,))
            initial = encode((x, target), registers)
            expected = encode((x, target ^ int(x & 2730 == 2082)), registers)
            assert classical_apply(candidate.circuit, initial) == expected


def test_native_qfunc_qmod_and_circuit_user_example() -> None:
    from examples.synthesis_boolean_factors import run_example

    run_example()


def test_mixed_cube_and_cubic_outputs_report_all_reused_workspace() -> None:
    source = """qfunc main(output x:qnum<12>) {
      allocate(x);flag:qnum<2>;
      within {flag=((x&15)==0)+2*((x&7)==0);} apply {phase(flag,0.37);}
    }"""
    imported = parse_qmod(
        source, filename="mixed.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    graph = imported.functions[0]
    argument = graph.body.blocks[0].arguments[0]
    candidate = lower_boolean_value(
        (graph,),
        {argument.id: tuple(range(12))},
        (12, 13),
        entry=graph.name,
        n_qubits=16,
        workspace=(14, 15),
        workspace_state="fresh_zero",
    )
    assert candidate.workspace == (14, 15)
    registers = (tuple(range(12)), (12, 13))
    for x in range(32):
        for target in range(4):
            answer = int(x & 15 == 0) + 2 * int(x & 7 == 0)
            assert classical_apply(
                candidate.circuit, encode((x, target), registers)
            ) == encode((x, target ^ answer), registers)


@pytest.mark.parametrize("case", ["facility_location", "traveling_salesman_problem"])
@pytest.mark.parametrize("scale", [0.003, -0.007])
def test_original_qaoa_against_independent_arithmetic_circuit(
    case: str, scale: float
) -> None:
    report = validate(case, scale)
    assert report.workspace == (5 if case == "facility_location" else 0)
    assert report.operations == (444 if case == "facility_location" else 929)


def factor_source(groups: int, size: int, *, initialized: bool = True) -> str:
    factors = [
        "*".join(
            (
                f"q[{group*size+bit}]"
                if groups == 2 and bit % 2
                else f"(1-q[{group*size+bit}])"
            )
            for bit in range(size)
        )
        for group in range(groups)
    ]
    cost = "+".join(f"({factor})" for factor in factors)
    # Reversed literal order must share the same Boolean computation.
    repeated = "*".join(reversed(factors[0].split("*")))
    preparation = "allocate(q);hadamard_transform(q);" if initialized else ""
    return f"""qfunc main(output q:qbit[{groups*size}]) {{
      {preparation}phase(-(({cost})-1)**2+({repeated})+0.25*q[0]+0.125,0.37);
    }}"""


@pytest.mark.parametrize(("groups", "size"), [(2, 5), (2, 6), (3, 4), (4, 3)])
def test_factor_shapes_signed_literals_shared_reads_absolute_phase_and_zero_work(
    groups: int, size: int
) -> None:
    source = factor_source(groups, size)
    imported = parse_qmod(
        source, filename="factor.qmod", source_version=SUPPORTED_SOURCE_VERSION
    )
    operations = imported.program.body.blocks[0].operations
    assert sum(op.name == "qreg.local" for op in operations) == groups
    assert sum(op.name == "qreg.uncompute" for op in operations) == groups
    program = synth.synthesize(source, source_format="qmod")
    state = fq.run(program).statevector().reshape(-1)
    width = groups * size
    basis = torch.arange(1 << width, dtype=torch.int64)
    factors = []
    for group in range(groups):
        factor = torch.ones(1 << width, dtype=torch.float64)
        for bit in range(size):
            values = (basis >> (group * size + bit)) & 1
            factor *= values if groups == 2 and bit % 2 else 1 - values
        factors.append(factor)
    total = torch.stack(factors).sum(0)
    cost = -(total - 1).square() + factors[0] + 0.25 * (basis & 1) + 0.125
    work = program.circuit.n_wires - width
    expected = torch.zeros_like(state)
    expected[:: 1 << work] = torch.exp(0.37j * cost) / (1 << width) ** 0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert float(state.reshape(-1, 1 << work)[:, 1:].abs().square().sum()) < 1e-24
    assert tuple(program.registers) == ("q",)


def test_factored_phase_still_rejects_uninitialized_physical_reads() -> None:
    with pytest.raises(ValueError, match="uninitialized"):
        synth.synthesize(factor_source(3, 4, initialized=False), source_format="qmod")


@pytest.mark.parametrize(
    ("constraint", "value"),
    [
        ("max_workspace", 4),
        ("max_qubits", 16),
        ("max_operations", 400),
        ("memory_limit_bytes", 1024),
    ],
)
def test_original_factor_candidate_obeys_actual_resource_budgets(
    constraint: str, value: int
) -> None:
    with pytest.raises(
        ValueError, match="workspace|qubit|operation|memory|limit|budget|candidate"
    ):
        synth.synthesize(
            FIXTURES / "facility_location.qmod",
            classical_inputs={"params": [0.003 * (i + 1) for i in range(10)]},
            **{constraint: value},
        )


def test_large_expression_preserves_order_and_has_a_separate_depth_bound() -> None:
    cost = "+".join(["q[0]"] * 200)
    source = f"qfunc main(output q:qbit[1]) {{allocate(q);hadamard_transform(q);phase({cost},0.003);}}"
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(1, dtype=torch.complex128).h(0).phase(0, theta=0.6)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    oversized = "+".join(["q[0]"] * 257)
    with pytest.raises(ValueError, match="depth 256"):
        Parser(tokenize(source.replace(cost, oversized), "deep.qmod")).parse()
    nested = "(" * 257 + "q[0]" + ")" * 257
    with pytest.raises(ValueError, match="nesting exceeds 256"):
        Parser(tokenize(source.replace(cost, nested), "deep.qmod")).parse()


def test_non_factorizable_phase_keeps_its_budget_refusal() -> None:
    expression = "*".join(f"(1+q[{i}])" for i in range(9))
    source = f"qfunc main(output q:qbit[9]) {{allocate(q);phase({expression},0.37);}}"
    with pytest.raises(ValueError, match="256 monomials|degree eight"):
        synth.synthesize(source, source_format="qmod")


def test_factorization_cannot_hide_more_than_twenty_four_physical_reads() -> None:
    with pytest.raises(ValueError, match="24 quantum input bits"):
        synth.synthesize(factor_source(6, 5), source_format="qmod")
