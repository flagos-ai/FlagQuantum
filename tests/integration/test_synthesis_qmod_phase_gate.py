"""Original PHASE plus coherent probes that distinguish global conventions."""

import cmath
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_synthesis_register_workflow import check_state
from tests.register_program_cases import Registers

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("angle", [0, 1, 2, 2**53])
def test_coherent_reference_keeps_exact_phase_convention(
    objective: Literal["width", "depth"], angle: int
) -> None:
    source = f"qfunc main(output q:qbit,output ref:qbit) {{allocate(q);H(q);ref=q;PHASE({angle},q);}}"
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    check_state(
        p, fq.run(p), {(0, 0): 2**-0.5, (1, 1): cmath.exp(1j * angle) * 2**-0.5}
    )


@pytest.mark.parametrize("prefix,amplitude", [("", 1), ("X(q);", cmath.exp(1j))])
def test_basis_states_retain_absolute_global_phase(
    prefix: str, amplitude: complex
) -> None:
    p = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);{prefix}PHASE(1,q);}}",
        source_format="qmod",
    )
    check_state(p, fq.run(p), {(int(bool(prefix)),): amplitude})


@pytest.mark.parametrize("angle", [0, 1, 7])
def test_native_and_owned_record_replace_producer_without_changing_consumer(
    angle: int,
) -> None:
    p = synth.synthesize(
        f"qfunc main(output q:qbit) {{allocate(q);H(q);PHASE({angle},q);}}",
        source_format="qmod",
    )
    r = Registers()
    q = r.declare("q", 1)
    r.emit("qreg.allocate", q)
    r.emit("qreg.gate", q, gate="h", index=0)
    r.emit("qreg.gate", q, gate="phase", index=0, angle=float(angle))
    owned = synthesize_registers(r.finish(), memory_limit_bytes=1024**2)
    assert p.circuit.instructions == owned.circuit.instructions
    check_state(
        owned, fq.run(owned), {(0,): 2**-0.5, (1,): cmath.exp(1j * angle) * 2**-0.5}
    )
    check_state(p, fq.run(p), {(0,): 2**-0.5, (1,): cmath.exp(1j * angle) * 2**-0.5})


def test_output_helper_composition_and_lowercase_predicate_remain_distinct() -> None:
    source = """qfunc prepare(output q:qbit) {allocate(q);H(q);PHASE(1,q);}
    qfunc main(output q:qbit) {prepare(q);phase(q,pi);PHASE(2,q);}"""
    p = synth.synthesize(source, source_format="qmod")
    check_state(p, fq.run(p), {(0,): 2**-0.5, (1,): -cmath.exp(3j) * 2**-0.5})


@pytest.mark.parametrize(
    "statement",
    [
        "PHASE(q,q);",
        "PHASE(pi,q);",
        "PHASE(1+1,q);",
        "PHASE(9007199254740993,q);",
        "PHASE(1);",
        "PHASE(1,q,q);",
        "PHASE(-1,q);",
    ],
)
def test_unreviewed_angle_forms_are_refused_before_execution(statement: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output q:qbit) {{allocate(q);{statement}}}",
            source_format="qmod",
        )


@pytest.mark.parametrize(
    "source",
    [
        "qfunc PHASE(output q:qbit) {allocate(q);} qfunc main(output q:qbit) {PHASE(q);}",
        "qfunc main(output q:qbit) {PHASE(1,q);}",
        "qfunc main(output q:qnum<1>) {allocate(q);PHASE(1,q);}",
        "qfunc act(const q:qbit) {PHASE(1,q);} qfunc main(output q:qbit) {allocate(q);act(q);}",
        "qfunc main(output q:qbit) {allocate(q);t:qbit;within {t=q;} apply {PHASE(1,t);}}",
    ],
)
def test_storage_role_and_scope_refusals_remain(source: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("input_kind", ["file", "text"])
def test_unchanged_original_source_and_license_execute_at_original_size(
    input_kind: str,
) -> None:
    raw = (CORPUS / "PHASE.qmod").read_bytes()
    m = json.loads((CORPUS / "phase_gate_sources.json").read_text())
    entry = m["fixtures"][0]
    assert entry["git_blob_sha1"] == "8163d1ad91e916fcc5fa6c5c5e71f61d934efbed"
    assert (
        entry["sha256"]
        == "8cd253f41e1ae55f32592e1a8af20c09409d970e365d4e84b5e0eed06bcca035"
    )
    assert m["repository"] == "https://github.com/Classiq/classiq-library"
    assert (
        m["license_sha256"]
        == "241f68c9f35357f97e8a2a195e447d4f80098c140f0b3b3c770a11eed96f58e0"
    )
    assert m["commit"] == "2bfd835e73b6882fcc515a725098b04766c238e7"
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == entry["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((CORPUS / "LICENSE.txt").read_bytes()).hexdigest()
        == m["license_sha256"]
    )
    if input_kind == "file":
        p = synth.synthesize(CORPUS / "PHASE.qmod", memory_limit_bytes=1024**2)
    else:
        p = synth.synthesize(
            raw.decode(), source_format="qmod", memory_limit_bytes=1024**2
        )
    check_state(p, fq.run(p), {(0,): 1})
    assert p.report["resources"].n_qubits == 1
    assert p.report["workspace_qubits"] == 0
    assert p.execution.state_bytes == 32
    assert [(g.name, g.params) for g in p.circuit.instructions] == [
        ("phase", {"theta": 1.0})
    ]


def test_phase_keeps_checked_memory_limit() -> None:
    from flagquantum.errors import PlanningError

    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(
            "qfunc main(output q:qbit) {allocate(q);PHASE(1,q);}",
            source_format="qmod",
            memory_limit_bytes=16,
        )


def test_phase_requires_no_workspace_and_respects_output_width_limit() -> None:
    p = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);PHASE(1,q);}",
        source_format="qmod",
        max_qubits=1,
        max_workspace=0,
        max_operations=1,
        max_depth=1,
    )
    check_state(p, fq.run(p), {(0,): 1})
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            "qfunc main(output q:qbit,output ref:qbit) {allocate(q);allocate(ref);PHASE(1,q);}",
            source_format="qmod",
            max_qubits=1,
        )


def test_target_basis_preserves_basis_one_global_phase() -> None:
    from tests.integration.test_construction_register_execution import BASIS

    basis = (*BASIS, "phase")
    p = synth.synthesize(
        "qfunc main(output q:qbit) {allocate(q);X(q);PHASE(1,q);}",
        source_format="qmod",
        basis=basis,
    )
    check_state(p, fq.run(p), {(1,): cmath.exp(1j)})
    assert all(g.name in basis for g in p.circuit.instructions)


def test_routed_native_basis_preserves_entangled_phase() -> None:
    from tests.integration.test_construction_register_execution import BASIS

    p = synth.synthesize(
        "qfunc main(output q:qbit,output ref:qbit) {allocate(q);H(q);ref=q;PHASE(1,q);}",
        source_format="qmod",
        basis=(*BASIS, "phase"),
        topology="line",
    )
    check_state(p, fq.run(p), {(0, 0): 2**-0.5, (1, 1): cmath.exp(1j) * 2**-0.5})
    assert all(g.name in (*BASIS, "phase") for g in p.circuit.instructions)


def test_basis_without_verified_phase_decomposition_refuses() -> None:
    with pytest.raises(synth.SynthesisError, match="phase.*no verified decomposition"):
        synth.synthesize(
            "qfunc main(output q:qbit) {allocate(q);PHASE(1,q);}",
            source_format="qmod",
            basis=("x", "h"),
        )
