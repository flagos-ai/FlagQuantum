from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler.qir import ENTRY_POINT_NAME, emit_qir
from flagquantum.compiler.target_conformance import (
    TargetConformanceError,
    verify_target_emission,
)
from flagquantum.compiler.target_emission import (
    TargetEmissionError,
    emit_legalized_target,
)
from flagquantum.compiler.target_legalization import legalize_circuit_for_target
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 16, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("target:0",))

# The QIR base profile is the specification, not a vendor artifact. No QIR
# validator is available in this environment, so the static check below uses the
# LLVM IR parser of the platform C compiler for syntax acceptance only. It does
# not validate base-profile semantics; `verify_target_emission` is the semantic
# authority and the round trip below is the execution-equivalence evidence.
_CLANG = shutil.which("clang")

# Distinct gate angles, so a decomposition that swaps or drops a parameter
# produces a different state rather than the same one.
_PARAMETER_VALUES = {"theta": 0.7, "phi": 0.3, "lbd": 0.4}

_UNITARY_OPCODES = tuple(
    sorted(name for name, schema in OPERATOR_SCHEMAS.items() if schema.unitary)
)


def _snapshot() -> TargetCapabilitySnapshot:
    values = {
        "qubits.logical_capacity": 4,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("expectation", "samples"),
        "limits.maximum_shots": 4096,
        "gates.native": (
            {"name": "h", "parameters": ()},
            {"name": "cx", "parameters": ()},
            {"name": "rx", "parameters": ("theta",)},
        ),
    }
    source = FactSource(kind="qir2_emission_test", ref="qir2-evidence")
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="qir2-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="qir2-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=tuple(
            CapabilityFact(
                name=name,
                value=value,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=(
                    FactExposure.OBSERVED
                    if name == "precision.effective_dtype"
                    else FactExposure.DECLARED
                ),
                source=source,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="qir2-evidence",
                sha256="d" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def _legalized():
    circuit = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rx", (1,), params={"theta": 0.37}),
        ),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1), shots=100),),
    )
    return legalize_circuit_for_target(
        circuit,
        backend="qir",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )


def _assert_state_equivalent(left: CircuitIR, right: CircuitIR, atol: float) -> None:
    left_state = Circuit.from_ir(left).state()[0]
    right_state = Circuit.from_ir(right).state()[0]
    overlap = torch.vdot(left_state, right_state).abs()
    torch.testing.assert_close(
        overlap,
        torch.ones((), dtype=overlap.dtype),
        atol=atol,
        rtol=0,
    )


def test_qir_text_carries_the_base_profile_module_contract() -> None:
    legalized = _legalized()
    text = emit_legalized_target(legalized, profile="qir-2.0").text
    lines = text.splitlines()

    assert f"define i64 @{ENTRY_POINT_NAME}() #0 {{" in lines
    assert "  call void @__quantum__rt__initialize(ptr null)" in lines
    assert "  call void @__quantum__qis__h__body(ptr null)" in lines
    assert (
        "  call void @__quantum__qis__cnot__body(ptr null, ptr inttoptr (i64 1 to ptr))"
        in lines
    )
    assert (
        '  call void @__quantum__qis__rx__body(double 3.69999999999999996e-01, '
        "ptr inttoptr (i64 1 to ptr))" in lines
    )
    assert "  call void @__quantum__rt__tuple_record_output(i64 2, ptr @0)" in lines
    assert "  call void @__quantum__rt__result_record_output(ptr null, ptr @1)" in lines
    assert "  ret i64 0" in lines
    assert (
        'attributes #0 = { "entry_point" "qir_profiles"="base_profile" '
        '"output_labeling_schema"="labeled" "required_num_qubits"="2" '
        '"required_num_results"="2" }' in lines
    )
    assert 'attributes #1 = { "irreversible" }' in lines
    assert 'declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1' in lines
    assert '!0 = !{i32 1, !"qir_major_version", i32 2}' in lines
    assert '!1 = !{i32 7, !"qir_minor_version", i32 0}' in lines
    assert '!2 = !{i32 1, !"dynamic_qubit_management", i1 false}' in lines
    assert '!3 = !{i32 1, !"dynamic_result_management", i1 false}' in lines
    # Every basic block is terminated exactly once, by the required instruction.
    assert text.count("br label %") == 3
    assert text.count("ret i64 0") == 1


@pytest.mark.skipif(_CLANG is None, reason="no LLVM IR parser available locally")
def test_qir_text_is_accepted_by_an_llvm_ir_parser(tmp_path: Path) -> None:
    emitted = emit_legalized_target(_legalized(), profile="qir-2.0")
    source = tmp_path / "module.ll"
    source.write_text(emitted.text, encoding="utf-8")

    def parse(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                str(_CLANG),
                "-x",
                "ir",
                "-c",
                "-o",
                "/dev/null",
                "-Wno-override-module",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    accepted = parse(source)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stderr == ""

    # A parser that accepted anything would prove nothing, so the same call must
    # reject a module whose used declaration is missing.
    mutated = tmp_path / "missing_declaration.ll"
    mutated.write_text(
        emitted.text.replace("declare void @__quantum__qis__h__body(ptr)\n", ""),
        encoding="utf-8",
    )
    rejected = parse(mutated)
    assert rejected.returncode != 0
    assert "undefined value" in rejected.stderr


def _opcode_legalized(opcode: str) -> tuple[CircuitIR, tuple[Instruction, ...]]:
    """Legalize a prepared one-gate circuit whose snapshot declares it native.

    Each case prepares `h` then `s` on every wire before the gate under test, so
    each wire is in `S|+> = (|0> + i|1>)/sqrt(2)`. Applying a gate to the all-zero
    state hides real errors: every controlled gate is the identity on a zero
    control, and `phase` is the identity on |0>. `|+>` alone is not enough either,
    because it is an eigenstate of `x` and of `sx`, and because `s` and `sdg` act
    on |1> as a global phase. `S|+>` is an eigenstate of none of the tables'
    error modes and makes each gate change the compared state rather than only
    its phase.
    """

    schema = OPERATOR_SCHEMAS[opcode]
    wires = tuple(range(schema.arity))
    n_wires = max(schema.arity, 1)
    preparation = tuple(
        Instruction(name, (wire,))
        for name in ("h", "s")
        for wire in range(n_wires)
    )
    circuit = CircuitIR(
        n_wires,
        (
            *preparation,
            Instruction(
                opcode,
                wires,
                params={name: _PARAMETER_VALUES[name] for name in schema.parameters},
            ),
        ),
        dtype="complex128",
        measurements=(MeasurementNode("samples", tuple(range(n_wires)), shots=100),),
    )
    legalized = legalize_circuit_for_target(
        circuit,
        backend="qir",
        snapshot=_opcode_snapshot(
            {"h": (), "s": (), opcode: schema.parameters},
        ),
        evaluated_at=_NOW,
    )
    return legalized, preparation


def _opcode_snapshot(native: dict[str, tuple[str, ...]]):
    """The emission test snapshot narrowed to the declared native gate set."""

    values = {
        "qubits.logical_capacity": 4,
        "limits.maximum_program_operations": 128,
        "precision.effective_dtype": "complex128",
        "measurements.results": ("samples",),
        "limits.maximum_shots": 4096,
        "gates.native": tuple(
            {"name": name, "parameters": parameters}
            for name, parameters in sorted(native.items())
        ),
    }
    source = FactSource(kind="qir2_emission_test", ref="qir2-evidence")
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="qir2-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="qir2-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=tuple(
            CapabilityFact(
                name=name,
                value=value,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=(
                    FactExposure.OBSERVED
                    if name == "precision.effective_dtype"
                    else FactExposure.DECLARED
                ),
                source=source,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="qir2-evidence",
                sha256="d" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=_SCOPE,
            ),
        ),
    )


def test_qir_round_trip_is_strict_and_statevector_equivalent() -> None:
    legalized = _legalized()
    emitted = emit_legalized_target(legalized, profile="qir-2.0")

    first = verify_target_emission(emitted, legalized)
    second = verify_target_emission(emitted, legalized)

    measurement = first.reconstructed_program.measurements[0]
    assert measurement.kind == "samples"
    assert measurement.wires == (0, 1)
    assert measurement.shots is None
    assert first.parsed_operation_count == len(legalized.program.instructions)
    assert first.conformance_identity == second.conformance_identity
    assert len(first.conformance_identity) == 64
    # The QCIS precedent: a decomposed native gate set is compared up to an
    # unconditional global phase, so the overlap carries the tolerance.
    _assert_state_equivalent(legalized.program, first.reconstructed_program, 1e-6)


def test_qir_emission_is_deterministic_and_bound_to_its_legalization() -> None:
    legalized = _legalized()
    first = emit_legalized_target(legalized, profile="qir-2.0")
    second = emit_legalized_target(legalized, profile="qir-2.0")

    assert first == second

    for change in (
        {"text": "; FlagQuantum QIR base profile module\n"},
        {"payload_sha256": "0" * 64},
        {"target_snapshot_id": "wrong-snapshot"},
        {"emission_identity": "0" * 64},
    ):
        with pytest.raises(TargetConformanceError, match="does not match legalization"):
            verify_target_emission(replace(first, **change), legalized)


def test_qir_module_without_measurement_records_an_empty_tuple() -> None:
    """Zero results is a permitted module, not a degenerate one.

    The base profile allows an entry point that measures nothing, so the emitter
    must not invent a measurement, must not declare the irreversible group, and
    must report zero required results. Only `emit_qir` reaches this shape: the
    target-emission boundary requires a terminal full-register samples request,
    so this module has no legalized projection to round-trip against.
    """

    program = CircuitIR(2, (Instruction("h", (0,)),), dtype="complex128")
    text = emit_qir(program)

    assert "  call void @__quantum__rt__tuple_record_output(i64 0, ptr @0)" in text
    assert "__quantum__qis__mz__body" not in text
    assert "__quantum__rt__result_record_output" not in text
    assert "irreversible" not in text
    assert '"required_num_results"="0"' in text
    assert text.count("br label %") == 3


def test_qir_result_projection_is_explicit_and_validated() -> None:
    """A caller-supplied result projection replaces the declared measurement."""

    program = CircuitIR(3, (Instruction("h", (0,)),), dtype="complex128")
    text = emit_qir(program, result_wires=(1,))

    assert (
        "  call void @__quantum__qis__mz__body"
        "(ptr inttoptr (i64 1 to ptr), ptr writeonly null)" in text
    )
    assert "  call void @__quantum__rt__tuple_record_output(i64 1, ptr @0)" in text

    for rejected in ((0, 0), (), (0, 3), (0, -1), (0, "1")):
        with pytest.raises(ValueError, match="unique in-range integers"):
            emit_qir(program, result_wires=rejected)  # type: ignore[arg-type]

    twice = CircuitIR(
        2,
        (Instruction("h", (0,)),),
        dtype="complex128",
        measurements=(
            MeasurementNode("samples", (0,)),
            MeasurementNode("samples", (1,)),
        ),
    )
    with pytest.raises(ValueError, match="at most one terminal measurement"):
        emit_qir(twice)


def test_qir_emission_refuses_semantics_the_base_profile_cannot_express() -> None:
    # The base profile records results only, so an expectation request and a
    # channel matrix are refused at the emitter instead of being approximated.
    expectation = CircuitIR(
        2,
        (Instruction("h", (0,)), Instruction("cx", (0, 1))),
        dtype="complex128",
        measurements=(MeasurementNode("expectation", (0, 1)),),
    )
    with pytest.raises(ValueError, match="cannot record a 'expectation' measurement"):
        emit_qir(expectation)

    channel = CircuitIR(
        2,
        (
            Instruction(
                "depolarizing",
                (0,),
                matrix=torch.eye(4, dtype=torch.complex128).reshape(2, 2, 2, 2),
                metadata={"is_channel": True},
            ),
        ),
        dtype="complex128",
        measurements=(MeasurementNode("samples", (0, 1)),),
    )
    with pytest.raises(ValueError, match="channel"):
        emit_qir(channel)

    legalized = legalize_circuit_for_target(
        expectation,
        backend="qir",
        snapshot=_snapshot(),
        evaluated_at=_NOW,
    )
    with pytest.raises(TargetEmissionError, match="terminal full-register samples"):
        emit_legalized_target(legalized, profile="qir-2.0")


@pytest.mark.parametrize("opcode", _UNITARY_OPCODES)
def test_every_unitary_gate_survives_the_qir_round_trip(opcode: str) -> None:
    """The whole unitary gate set, not a hand-picked sample.

    A decomposition is the only thing that carries a gate the QIS set does not
    name, so each row is a separate claim. The comparison is the global-phase
    invariant overlap used for QCIS, and `atol=1e-6` is that precedent: a
    decomposition with the wrong relative phase, a swapped control, or a sign
    error in one rotation moves the overlap by more than this by orders of
    magnitude, while complex128 accumulation over one gate is at float epsilon.
    """

    legalized, preparation = _opcode_legalized(opcode)

    emitted = emit_legalized_target(legalized, profile="qir-2.0")
    verified = verify_target_emission(emitted, legalized)

    # The gate under test must reach the module: `i` is the only gate the base
    # profile legitimately drops, and it may not drop anything else.
    assert verified.parsed_operation_count >= len(preparation) + (opcode != "i")
    _assert_state_equivalent(
        legalized.program, verified.reconstructed_program, atol=1e-6
    )


# `verify_target_emission` re-emits and compares the whole result, so it rejects a
# tampered text before any parser runs. The base-profile rules below are therefore
# exercised on the parser directly: each row is one specification rule, and a row
# whose violation is accepted is a rule the emitter's own output cannot reveal.
_MZ_SECOND = (
    "  call void @__quantum__qis__mz__body"
    "(ptr inttoptr (i64 1 to ptr), ptr writeonly inttoptr (i64 1 to ptr))\n"
)
_H_BODY = "  call void @__quantum__qis__h__body(ptr null)\n"
_DEFINE_LINE = "define i64 @FlagQuantumEntryPoint() #0 {\n"

_QIR_PROFILE_RULES: tuple[tuple[str, str, str, str], ...] = (
    (
        "a foreign module header",
        "; FlagQuantum QIR base profile module\n",
        "; another module\n",
        "header is not canonical",
    ),
    (
        "a label constant that skips a number",
        '@2 = internal constant [3 x i8] c"r1\\00"\n',
        '@3 = internal constant [3 x i8] c"r1\\00"\n',
        "label numbering is not consecutive",
    ),
    (
        "a label constant without its null terminator",
        '@0 = internal constant [3 x i8] c"t0\\00"\n',
        '@0 = internal constant [4 x i8] c"t0\\00"\n',
        "label constant is inconsistent",
    ),
    (
        "two output labels sharing one string",
        '@2 = internal constant [3 x i8] c"r1\\00"\n',
        '@2 = internal constant [3 x i8] c"r0\\00"\n',
        "reuses an output label string",
    ),
    (
        "an output label no recording call references",
        "@0 = internal constant",
        '@3 = internal constant [3 x i8] c"x0\\00"\n@0 = internal constant',
        "leaves an output label unreferenced",
    ),
    (
        "a second entry point",
        _DEFINE_LINE,
        _DEFINE_LINE + "define i64 @Other() #0 {\n",
        "must define exactly one entry point",
    ),
    (
        "an entry point that takes a parameter",
        "define i64 @FlagQuantumEntryPoint() #0 {",
        "define i64 @FlagQuantumEntryPoint(i64) #0 {",
        "signature is not canonical",
    ),
    (
        "an entry point without the entry-point attribute",
        '"entry_point" ',
        "",
        "entry point attributes are incomplete",
    ),
    (
        "an entry point declaring the adaptive profile",
        '"qir_profiles"="base_profile"',
        '"qir_profiles"="adaptive_profile"',
        "entry point attributes are incomplete",
    ),
    (
        "an entry point without an output labeling schema",
        '"output_labeling_schema"="labeled" ',
        "",
        "entry point attributes are incomplete",
    ),
    (
        "a qubit count that is not the legalized width",
        '"required_num_qubits"="2"',
        '"required_num_qubits"="3"',
        "required qubit count does not match",
    ),
    (
        "a result count that is not the measured width",
        '"required_num_results"="2"',
        '"required_num_results"="1"',
        "required result count does not match",
    ),
    (
        "three basic blocks instead of four",
        "\nbody:\n",
        "\n",
        "needs four basic blocks",
    ),
    (
        "two basic blocks with the same name",
        "\nbody:\n",
        "\nentry:\n",
        "needs four basic blocks",
    ),
    (
        "a block that branches past the next one",
        "  br label %body\n",
        "  br label %measurements\n",
        "not an unconditional chain",
    ),
    (
        "a body call inside the initialization block",
        "  call void @__quantum__rt__initialize(ptr null)\n",
        _H_BODY,
        "initialization block is not canonical",
    ),
    (
        "a statement before the first basic block",
        _DEFINE_LINE,
        _DEFINE_LINE + _H_BODY,
        "statement outside a basic block",
    ),
    (
        "a non-zero exit code",
        "  ret i64 0\n",
        "  ret i64 1\n",
        "exit code is not a static zero",
    ),
    (
        "measurement results that skip a result index",
        _MZ_SECOND,
        _MZ_SECOND.replace("i64 1 to ptr))\n", "i64 0 to ptr))\n"),
        "measurement results are not consecutively indexed",
    ),
    (
        "a measurement of the wrong qubit",
        _MZ_SECOND,
        "  call void @__quantum__qis__mz__body"
        "(ptr null, ptr writeonly inttoptr (i64 1 to ptr))\n",
        "measurement target does not match the legalized projection",
    ),
    (
        "one measurement fewer than the projection",
        _MZ_SECOND,
        "",
        "measurement count does not match the legalized projection",
    ),
    (
        "a tuple record naming the wrong width",
        "tuple_record_output(i64 2, ptr @0)",
        "tuple_record_output(i64 1, ptr @0)",
        "tuple record is not canonical",
    ),
    (
        "a tuple record naming the wrong label",
        "tuple_record_output(i64 2, ptr @0)",
        "tuple_record_output(i64 2, ptr @1)",
        "tuple label is not canonical",
    ),
    (
        "a result record naming the wrong label",
        "result_record_output(ptr null, ptr @1)",
        "result_record_output(ptr null, ptr @2)",
        "result label is not canonical",
    ),
    (
        "an output block without its closing record",
        "  call void @__quantum__rt__result_record_output(ptr null, ptr @1)\n",
        "",
        "output block is not canonical",
    ),
    (
        "a declared function that is not in the base profile",
        "declare void @__quantum__qis__h__body(ptr)\n",
        "declare void @__quantum__qis__madeup__body(ptr)\n",
        "declares an unknown function",
    ),
    (
        "a declaration with the wrong signature",
        "declare void @__quantum__qis__h__body(ptr)\n",
        "declare void @__quantum__qis__h__body(ptr, ptr)\n",
        "has the wrong signature",
    ),
    (
        "a declaration of a function nothing calls",
        "declare void @__quantum__qis__h__body(ptr)\n",
        "declare void @__quantum__qis__h__body(ptr)\n"
        "declare void @__quantum__qis__x__body(ptr)\n",
        "declarations do not match the calls",
    ),
    (
        "a call with no declaration",
        "declare void @__quantum__qis__rx__body(double, ptr)\n",
        "",
        "declarations do not match the calls",
    ),
    (
        "a measurement declaration without its marker",
        "declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1\n",
        "declare void @__quantum__qis__mz__body(ptr, ptr writeonly)\n",
        "wrong irreversibility marker",
    ),
    (
        "an irreversible marker on a reversible function",
        "declare void @__quantum__qis__h__body(ptr)\n",
        "declare void @__quantum__qis__h__body(ptr) #1\n",
        "wrong irreversibility marker",
    ),
    (
        "a marker that names a group without the attribute",
        'attributes #1 = { "irreversible" }',
        'attributes #1 = { "reversible" }',
        "does not name a canonical attribute group",
    ),
    (
        "a module flag with a drifted value",
        '!1 = !{i32 7, !"qir_minor_version", i32 0}',
        '!1 = !{i32 7, !"qir_minor_version", i32 1}',
        "module flags are not canonical",
    ),
    (
        "a module flag list that repeats an entry",
        "!llvm.module.flags = !{!0, !1, !2, !3}",
        "!llvm.module.flags = !{!1, !1, !2, !3}",
        "module flag numbering is not consecutive",
    ),
    (
        "a module flag list shorter than the flags it declares",
        "!llvm.module.flags = !{!0, !1, !2, !3}",
        "!llvm.module.flags = !{!0, !1, !2}",
        "module flag list is inconsistent",
    ),
    (
        "a body call to a function outside the QIS set",
        _H_BODY,
        "  call void @__quantum__qis__madeup__body(ptr null)\n",
        "uses unsupported QIS function",
    ),
    (
        "a body call on an out-of-range qubit",
        _H_BODY,
        "  call void @__quantum__qis__h__body(ptr inttoptr (i64 2 to ptr))\n",
        "references an out-of-range qubit",
    ),
    (
        "a body call with too many operands",
        _H_BODY,
        "  call void @__quantum__qis__h__body(ptr null, ptr null)\n",
        "wrong operand count",
    ),
    (
        "an angle operand outside the canonical decimal form",
        "double 3.69999999999999996e-01",
        "double 1e-3",
        "malformed angle operand",
    ),
    (
        "a measurement result operand without its qualifier",
        "call void @__quantum__qis__mz__body(ptr null, ptr writeonly null)",
        "call void @__quantum__qis__mz__body(ptr null, ptr null)",
        "wrongly qualified result operand",
    ),
    (
        "a measurement call with an extra operand",
        "call void @__quantum__qis__mz__body(ptr null, ptr writeonly null)",
        "call void @__quantum__qis__mz__body(ptr null, ptr writeonly null, ptr null)",
        "not a canonical mz call",
    ),
    (
        "a rotation on an out-of-range qubit",
        "call void @__quantum__qis__rx__body(double 3.69999999999999996e-01, ptr inttoptr (i64 1 to ptr))",
        "call void @__quantum__qis__rx__body(double 3.69999999999999996e-01, ptr inttoptr (i64 2 to ptr))",
        "references an out-of-range qubit",
    ),
    (
        "a declaration repeated verbatim",
        "declare void @__quantum__qis__h__body(ptr)\n",
        "declare void @__quantum__qis__h__body(ptr)\ndeclare void @__quantum__qis__h__body(ptr)\n",
        "repeats a function declaration",
    ),
)


@pytest.mark.parametrize(
    ("label", "source", "replacement", "message"),
    _QIR_PROFILE_RULES,
    ids=[rule[0] for rule in _QIR_PROFILE_RULES],
)
def test_qir_parser_rejects_each_base_profile_violation(
    label: str, source: str, replacement: str, message: str
) -> None:
    legalized = _legalized()
    text = emit_legalized_target(legalized, profile="qir-2.0").text
    assert text.count(source) == 1, f"the emitter no longer writes {label}"

    with pytest.raises(TargetConformanceError, match=re.escape(message)):
        _parse_qir(text.replace(source, replacement), template=legalized.program)


def test_qir_parser_accepts_the_unmodified_module() -> None:
    """The rule table is only evidence if the parser still accepts valid text."""

    legalized = _legalized()
    text = emit_legalized_target(legalized, profile="qir-2.0").text
    reconstructed = _parse_qir(text, template=legalized.program)

    assert reconstructed.content_hash == verify_target_emission(
        emit_legalized_target(legalized, profile="qir-2.0"), legalized
    ).reconstructed_circuit_hash
