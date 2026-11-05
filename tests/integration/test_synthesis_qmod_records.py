"""Check flat record storage, composition and named field readout contracts."""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def test_record_fields_share_storage_and_preserve_named_readout():
    source = """qstruct State {flags:qbit[2]; tag:qbit;}
      qfunc flip(s:State) {X(s.flags[0]); S(s.tag);}
      qfunc main(output s:State) {
        allocate(s.size,s); H(s.flags[1]); X(s.tag); flip(s);
      }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(3, dtype=DTYPE).h(1).x(0).x(2).s(0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {"s.flags": (1, 2), "s.tag": (0,)}
    assert dict(
        program.probabilities(result, registers=("s.flags", "s.tag"))
    ) == pytest.approx({(1, 1): 0.5, (3, 1): 0.5})
    assert program.report["qmod_records"] == (
        ("s", "State", (("flags", "qbit[]", 2), ("tag", "qbit", 1))),
    )
    assert program.report["workspace_qubits"] == 0


def test_record_output_helper_and_whole_field_hadamard():
    source = """qstruct State {flags:qbit[2];}
      qfunc prepare(output s:State) {allocate(s); hadamard_transform(s.flags);}
      qfunc main(output s:State) {prepare(s);}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        ("qstruct R {} qfunc main(output r:R) {allocate(r);}", "at least one field"),
        ("qstruct R {a:qbit;} qfunc helper(", "expected an Output"),
        ("qstruct R {a:qbit;a:qbit;} qfunc main(output r:R) {}", "unique"),
        ("qstruct R {size:qbit;} qfunc main(output r:R) {}", "shadow size"),
        ("qstruct R {a:qbit[256];b:qbit;} qfunc main(output r:R) {}", "exceeds 256"),
        ("qstruct R {a:qnum<2>;} qfunc main(output r:R) {}", "flat record fields"),
        ("qstruct R {a:qbit[2][2];} qfunc main(output r:R) {}", "one array dimension"),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {allocate(r);X(r.b);}",
            "unknown record field",
        ),
        (
            "qstruct R {a:qbit[2];} qfunc main(output r:R) {allocate(r);X(r.a[2]);}",
            "field index",
        ),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {X(r.a);allocate(r);}",
            "uninitialized",
        ),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {allocate(2,r);}",
            "declared width",
        ),
    ],
)
def test_record_admission_rejections(source, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_equal_width_record_types_are_not_interchangeable():
    source = """qstruct R {a:qbit;} qstruct S {b:qbit;}
      qfunc flip(r:R) {X(r.a);}
      qfunc main(output s:S) {allocate(s);flip(s);}"""
    with pytest.raises(InteropConversionError, match="matching record type"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "fields",
    [
        (("a", "qbit", 2),),
        (("a", "qbit[]", 1),),
        (("a", "qbit", 1), ("a", "qbit", 1)),
        (("a", "qbit", True), ("b", "qbit", 1)),
    ],
)
def test_compiler_rechecks_owned_record_layout(fields):
    imported = parse_qmod(
        "qstruct R {a:qbit;b:qbit;} qfunc main(output r:R) {allocate(r);}",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    block = imported.program.body.blocks[0]
    declaration = block.operations[0]
    broken = replace(
        declaration, attributes={**declaration.attributes, "fields": fields}
    )
    program = replace(
        imported.program,
        body=replace(
            imported.program.body,
            blocks=(replace(block, operations=(broken, *block.operations[1:])),),
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.record"):
        verify_register_program(program)


@pytest.mark.parametrize("name", ["ising_model", "kidney_exchange_problem"])
@pytest.mark.parametrize("scale", [0.003, -0.005])
def test_small_original_records_against_independent_cost(name, scale):
    import hashlib

    from tools.validate_qmod_records import BLOBS, FIXTURES, RecordCost
    from tools.validate_qmod_sparse_phase import independent_terms, reference_circuit

    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == BLOBS[name]
    oracle = RecordCost(raw.decode())
    params = [
        scale * (index + 1) * (-1 if index % 2 else 1) for index in range(oracle.count)
    ]
    program = synth.synthesize(source, classical_inputs={"params": params})
    terms = independent_terms(oracle, oracle.degree, oracle.width)
    midpoint = oracle.count // 2
    interleaved = [
        value
        for pair in zip(params[:midpoint], params[midpoint:], strict=True)
        for value in pair
    ]
    reference = reference_circuit(interleaved, terms, oracle.width)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    distribution = program.probabilities(result, registers=tuple(program.registers))
    assert sum(distribution.values()) == pytest.approx(1)
