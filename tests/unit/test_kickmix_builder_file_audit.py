"""Source file coverage for the original CircuitBuilder diagram and scalar tests."""

import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import (
    _broadcast_controlled_flip,
    _Control,
    _flip,
    _XBasisTarget,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream

pytestmark = pytest.mark.unit
_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/kickmix_builder_diagrams.json").read_text()
)
_PREFIX = tuple(ClassicalXor((), slot).instruction() for slot in range(1, 5))
_SHARED: tuple[_Control, ...] = (True, False, ClassicalFieldCoefficient(1), 1)


def _kmx_body(gates: tuple[Instruction, ...]) -> str:
    ir = CircuitIR(8, (*_PREFIX, *gates))
    stream = encode_kmx(lower_reversible_phase(ir))
    return "\n".join(stream.splitlines()[len(_PREFIX) + 1 :])


def _diagram_cases() -> dict[str, tuple[tuple[Instruction, ...], ...]]:
    b1 = ClassicalFieldCoefficient(1)
    b2 = ClassicalFieldCoefficient(2)
    cases: dict[str, tuple[tuple[Instruction, ...], ...]] = {
        "cccz_qqqqa": (_flip(None, 1, 2, 3, 4, scratch=5, measurement_bit=0),),
        "cccz_qbqqa": (_flip(None, 1, b2, 3, 4, scratch=5, measurement_bit=0),),
        "cccz_bqcqa": (_flip(None, b1, 2, True, 4, scratch=5, measurement_bit=0),),
        "cccz_bqfqa": (_flip(None, b1, 2, False, 4, scratch=5, measurement_bit=0),),
        "cccz_qqqq": (_flip(None, 1, 2, 3, 4),),
    }
    quantum = (2, 3, 4)
    classical = tuple(ClassicalFieldCoefficient(slot) for slot in quantum)
    targets = (5, 6, 7)
    for index, control in enumerate(_SHARED, 1):
        cases[f"broadcast_ccx_aqq_{index}"] = (
            _broadcast_controlled_flip(control, quantum, targets),
        )
        cases[f"broadcast_ccx_acq_{index}"] = (
            _broadcast_controlled_flip(control, classical, targets),
        )
    cases["broadcast_ccx_abq"] = tuple(
        _broadcast_controlled_flip(control, (False, True, True), targets)
        for control in _SHARED
    )
    cases["broadcast_ccx_amq"] = tuple(
        _broadcast_controlled_flip(control, (False, True, b1, 2), (4, *targets))
        for control in _SHARED
    )
    return cases


_CASES = _diagram_cases()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_all_original_diagram_operands_follow_native_layout(name: str) -> None:
    assert _FIXTURE["source_blob"] == "d01e4c1aa3ad397312e52229cc1883c4a15debd5"
    assert set(_FIXTURE["diagram_test_streams"]) == set(_CASES)
    for gates, source in zip(
        _CASES[name], _FIXTURE["diagram_test_streams"][name], strict=True
    ):
        body = _kmx_body(gates)
        resets = [line for line in source.splitlines() if line.startswith("R ")]
        assert resets == (["R q5"] if name == "cccz_qqqqa" else [])
        assert not any(line.startswith("R ") for line in body.splitlines())
        assert normalized_stream(body) == normalized_stream(source)


def _z_operands(index: int) -> tuple[_Control, ...]:
    return False, True, index, ClassicalFieldCoefficient(index)


def test_all_original_scalar_truth_table_operands_match_source() -> None:
    assert _kmx_body(_flip(3, 0, 1)) == "CCX q0 q1 q3"
    prefix = tuple(ClassicalXor((), bit).instruction() for bit in range(3))
    observed = {}
    for opcode in ("CX", "CZ", "CCX", "CCZ"):
        for tags in product(range(4), repeat=len(opcode)):
            wire = len(tags) - 1
            if opcode.endswith("X"):
                target = (
                    _XBasisTarget(False),
                    _XBasisTarget(True),
                    wire,
                    _XBasisTarget(ClassicalFieldCoefficient(wire)),
                )[tags[-1]]
                controls = tags[:-1]
            else:
                target = None
                controls = tags
            gates = _flip(
                target, *(_z_operands(index)[tag] for index, tag in enumerate(controls))
            )
            lowered = encode_kmx(
                lower_reversible_phase(CircuitIR(4, (*prefix, *gates)))
            )
            key = " ".join((opcode, *map(str, tags)))
            observed[key] = normalized_stream("\n".join(lowered.splitlines()[4:]))
    assert len(observed) == 160
    digest = sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == _FIXTURE["scalar_stream_digest"]
