"""Check terminal Qmod partition transfers against independent Circuits."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration

CROSS_OWNER = """
qstruct State { first:qbit; second:qbit; }
qfunc main(output kept:qbit, output mixed:qnum<2>, output rest:qnum) {
    allocate(kept);
    state:State;
    aux:qbit;
    allocate(state);
    allocate(aux);
    X(kept);
    H(aux); S(aux);
    X(state.first); H(state.second);
    {aux,state} -> {mixed,rest};
}
"""


def test_cross_owner_publication_reorders_without_gates_or_workspace() -> None:
    program = synth.synthesize(
        CROSS_OWNER, source_format="qmod", max_qubits=4, max_workspace=0
    )
    assert dict(program.registers) == {"kept": (0,), "mixed": (1, 2), "rest": (3,)}
    reference = fq.Circuit(4, dtype=torch.complex128).x(0).h(2).s(2).x(1).h(3)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["resources"].n_operations == 4
    assert program.report["workspace_qubits"] == 0


def test_one_owner_split_keeps_unpublished_hidden_data() -> None:
    source = """
    qfunc main(output high:qnum<2>,output low:qbit) {
      data:qbit[3]; hidden:qbit[1];
      allocate(data); allocate(hidden);
      H(data[0]); S(data[0]); X(data[1]); H(data[2]); X(hidden[0]);
      drop(hidden);
      data -> {low, high};
    }
    """
    program = synth.synthesize(
        source, source_format="qmod", max_qubits=4, max_workspace=0
    )
    assert dict(program.registers) == {"high": (0, 1), "low": (2,)}
    reference = fq.Circuit(4, dtype=torch.complex128).h(2).s(2).x(1).h(0).x(3)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["resources"].n_qubits == 4
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    [
        ("{aux,state}", "{aux,aux}", "names must be distinct"),
        ("output rest:qnum", "output rest:qnum<2>", "widths must cover"),
        ("output mixed:qnum<2>", "output mixed:qnum", "infer at most one"),
        ("{mixed,rest};", "{mixed,rest}; X(kept);", "terminal publication"),
        ("{aux,state}", "{kept,state}", "requires complete owned hidden storage"),
    ],
)
def test_publication_refuses_aliases_or_nonterminal_transfer(
    before, after, reason
) -> None:
    with pytest.raises(InteropConversionError, match=reason):
        synth.synthesize(CROSS_OWNER.replace(before, after, 1), source_format="qmod")


def test_signed_output_can_span_two_complete_owners() -> None:
    source = """
    qfunc main(output phase:qnum<2, True, 2>) {
      first:qbit; second:qbit;
      allocate(first); allocate(second);
      X(first); H(second); S(second);
      {first,second} -> phase;
    }
    """
    program = synth.synthesize(
        source, source_format="qmod", max_qubits=2, max_workspace=0
    )
    reference = fq.Circuit(2, dtype=torch.complex128).x(1).h(0).s(0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["qmod_signed_registers"] == ("phase",)
    assert program.report["qmod_fractional_bits"] == {"phase": 2}
