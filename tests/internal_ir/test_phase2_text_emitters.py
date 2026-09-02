from __future__ import annotations

import re

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.exporters.text import (
    TextEmissionStatus,
    emit_openqasm2,
    emit_openqasm3,
    emit_qcis_v1,
)
from flagquantum._compiler.ir.modules import Block, QuantumModule, Region
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    StaticCanonicalizationPass,
)
from flagquantum._compiler.passes.target_decomposition import (
    DecomposeToTargetGateSetPass,
)

pytestmark = pytest.mark.unit


def _target_module(source: fq.CircuitIR) -> QuantumModule:
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    result = PassManager(
        (StaticCanonicalizationPass(), DecomposeToTargetGateSetPass())
    ).run(sealed.artifact.imported.module)
    assert result.ok, result.diagnostics
    return result.module


def _fixture() -> tuple[fq.CircuitIR, QuantumModule]:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("rx", (2,), {"theta": -0.0}),
            fq.Instruction("ry", (0,), {"theta": 0.125}),
            fq.Instruction("rz", (1,), {"theta": -0.25}),
            fq.Instruction("cx", (2, 0)),
        ),
        dtype="complex128",
    )
    return source, _target_module(source)


def _parse_qasm(text: str) -> fq.CircuitIR:
    declaration = re.search(r"(?:qreg q|qubit)\[(\d+)\](?: q)?;", text)
    assert declaration is not None
    instructions = []
    gate_pattern = re.compile(r"^(rx|ry|rz)(?:\(([^)]+)\)) q\[(\d+)\];$")
    cx_pattern = re.compile(r"^cx q\[(\d+)\], ?q\[(\d+)\];$")
    for line in text.splitlines():
        gate = gate_pattern.match(line)
        if gate:
            instructions.append(
                fq.Instruction(
                    gate.group(1),
                    (int(gate.group(3)),),
                    {"theta": float(gate.group(2))},
                )
            )
        cx = cx_pattern.match(line)
        if cx:
            instructions.append(
                fq.Instruction("cx", (int(cx.group(1)), int(cx.group(2))))
            )
    return fq.CircuitIR(
        int(declaration.group(1)), tuple(instructions), dtype="complex128"
    )


def _parse_qcis(text: str, n_qubits: int) -> fq.CircuitIR:
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    instructions = []
    index = 0
    while index < len(lines):
        tokens = lines[index].split()
        if tokens[0] == "RZ":
            instructions.append(
                fq.Instruction("rz", (int(tokens[1][1:]),), {"theta": float(tokens[2])})
            )
            index += 1
            continue
        first, middle, last = (lines[index + offset].split() for offset in range(3))
        wire = int(first[1][1:])
        if first[0] == "Y2M" and middle[0] == "RZ" and last[0] == "Y2P":
            instructions.append(
                fq.Instruction("rx", (wire,), {"theta": float(middle[2])})
            )
        elif first[0] == "X2P" and middle[0] == "RZ" and last[0] == "X2M":
            instructions.append(
                fq.Instruction("ry", (wire,), {"theta": float(middle[2])})
            )
        else:
            assert first[0] == "Y2M" and middle[0] == "CZ" and last[0] == "Y2P"
            instructions.append(
                fq.Instruction("cx", (int(middle[1][1:]), int(middle[2][1:])))
            )
        index += 3
    return fq.CircuitIR(n_qubits, tuple(instructions), dtype="complex128")


def _assert_same_state(actual: fq.CircuitIR, expected: fq.CircuitIR) -> None:
    torch.testing.assert_close(
        fq.run(actual).state,
        fq.run(expected).state,
        atol=1e-12,
        rtol=0,
    )


def test_openqasm_golden_format_parse_semantics_and_qubit_order() -> None:
    source, module = _fixture()
    qasm2 = emit_openqasm2(module)
    qasm3 = emit_openqasm3(module)

    assert qasm2.ok and qasm2.text == (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
        "ry(0.125) q[0];\nrz(-0.25) q[1];\ncx q[2],q[0];\n"
    )
    assert qasm3.ok and qasm3.text == (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\n'
        "ry(0.125) q[0];\nrz(-0.25) q[1];\ncx q[2], q[0];\n"
    )
    _assert_same_state(_parse_qasm(qasm2.text), source)
    _assert_same_state(_parse_qasm(qasm3.text), source)


def test_qcis_golden_parse_semantics_and_native_cx_order() -> None:
    source, module = _fixture()
    result = emit_qcis_v1(module)

    assert result.ok and result.text == (
        "X2P Q0\nRZ Q0 0.125\nX2M Q0\n" "RZ Q1 -0.25\nY2M Q0\nCZ Q2 Q0\nY2P Q0\n"
    )
    _assert_same_state(_parse_qcis(result.text, 3), source)


@pytest.mark.parametrize("emitter", (emit_openqasm2, emit_openqasm3, emit_qcis_v1))
def test_emission_is_character_and_hash_deterministic(emitter) -> None:
    _, module = _fixture()
    first = emitter(module)
    second = emitter(module)

    assert first == second
    assert first.content_hash is not None and len(first.content_hash) == 64
    assert first.source_program_identity == module.program_identity


@pytest.mark.parametrize("emitter", (emit_openqasm2, emit_openqasm3, emit_qcis_v1))
def test_unbound_and_unsupported_operations_fail_closed(emitter) -> None:
    symbolic = fq.CircuitIR(
        1,
        (fq.Instruction("rx", (0,), {"theta": fq.Parameter("theta")}),),
    )
    unsupported = fq.CircuitIR(1, (fq.Instruction("h", (0,)),))
    trainable = fq.CircuitIR(
        1,
        (
            fq.Instruction(
                "ry",
                (0,),
                {"theta": torch.tensor(0.2, requires_grad=True)},
            ),
        ),
    )

    symbolic_result = emitter(
        seal_circuit_ir_round_trip(symbolic).artifact.imported.module
    )
    unsupported_result = emitter(
        seal_circuit_ir_round_trip(unsupported).artifact.imported.module
    )
    trainable_result = emitter(
        seal_circuit_ir_round_trip(trainable).artifact.imported.module
    )

    assert symbolic_result.status is TextEmissionStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert symbolic_result.text is None
    assert "unbound parameters" in symbolic_result.diagnostics[0].message
    assert unsupported_result.status is TextEmissionStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert unsupported_result.text is None
    assert "outside the target profile" in unsupported_result.diagnostics[0].message
    assert trainable_result.status is TextEmissionStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert "unbound parameters" in trainable_result.diagnostics[0].message


def test_invalid_module_and_non_module_fail_closed() -> None:
    _, module = _fixture()
    block = module.body.blocks[0]
    invalid = QuantumModule(
        Region((Block(block.arguments, (block.operations[0], block.operations[0])),))
    )

    assert emit_openqasm2(object()).status is TextEmissionStatus.INVALID_MODULE
    result = emit_openqasm2(invalid)
    assert result.status is TextEmissionStatus.INVALID_MODULE
    assert result.text is None
    assert result.diagnostics


def test_emitters_remain_private_and_do_not_change_default_path() -> None:
    assert not hasattr(fq, "emit_openqasm2")
    assert not hasattr(fq, "emit_openqasm3")
    assert not hasattr(fq, "emit_qcis_v1")
