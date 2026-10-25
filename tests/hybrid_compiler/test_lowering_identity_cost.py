import hashlib
import json

import pytest

from flagquantum.compiler._hybrid import (
    DEFAULT_HYBRID_PASSES,
    HybridProgram,
    analysis,
    capture_source,
    lower_dynamic_program,
    run_pass_pipeline,
    specialize_program,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_lowering_uses_verified_pipeline_identity_evidence(
    monkeypatch: pytest.MonkeyPatch, dynamic: bool, optimize: bool
) -> None:
    source = (
        "def sample():\n    bit = qp.measure(wires=0)\n    return bit\n"
        if dynamic
        else "def sample():\n    return qp.expval(qp.PauliZ(0))\n"
    )
    program = capture_source(source, ())
    expected = run_pass_pipeline(program, DEFAULT_HYBRID_PASSES if optimize else ())
    identity_calls = verification_calls = 0
    identity = HybridProgram.semantic_identity.fget
    assert identity is not None
    verify = analysis.verify_program

    def measured_identity(value: HybridProgram) -> str:
        nonlocal identity_calls
        identity_calls += 1
        return identity(value)

    def measured_verify(value: HybridProgram) -> HybridProgram:
        nonlocal verification_calls
        verification_calls += 1
        return verify(value)

    def canonical(value: HybridProgram) -> str:
        data = json.dumps(
            value.to_data(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    monkeypatch.setattr(HybridProgram, "semantic_identity", property(measured_identity))
    monkeypatch.setattr(analysis, "verify_program", measured_verify)
    result = (
        lower_dynamic_program(program, (), optimize=optimize)
        if dynamic
        else specialize_program(program, (), optimize=optimize)
    )
    assert result.program_identity == canonical(program)
    assert result.optimized_program_identity == canonical(expected.program)
    assert result.optimization_records == expected.records
    stages = len(expected.records) + 1
    assert verification_calls == stages
    assert identity_calls == stages
