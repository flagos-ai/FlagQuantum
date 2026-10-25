import hashlib
import json
from dataclasses import replace

import pytest

from flagquantum.compiler._hybrid import (
    HybridProgram,
    analysis,
    capture_source,
    run_pass_pipeline,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("steps", [0, 3])
def test_pipeline_serializes_each_immutable_stage_once(
    monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    program = capture_source("def sample():\n    return qp.expval(qp.PauliZ(0))\n", ())
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

    class Rename:
        name = "rename"

        def run(self, value: HybridProgram, facts: object) -> HybridProgram:
            return replace(value, name=value.name + "_stage")

    monkeypatch.setattr(HybridProgram, "semantic_identity", property(measured_identity))
    monkeypatch.setattr(analysis, "verify_program", measured_verify)
    result = run_pass_pipeline(program, (Rename(),) * steps)

    def canonical(value: HybridProgram) -> str:
        data = json.dumps(
            value.to_data(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    assert result.source_identity == canonical(program)
    assert result.optimized_identity == canonical(result.program)
    current = program
    for record in result.records:
        following = replace(current, name=current.name + "_stage")
        assert record.input_identity == canonical(current)
        assert record.output_identity == canonical(following)
        current = following
    assert verification_calls == steps + 1
    assert identity_calls == steps + 1

    # A later invocation computes fresh evidence rather than retaining a cache.
    run_pass_pipeline(program, ())
    assert identity_calls == steps + 2
    assert verification_calls == steps + 2
