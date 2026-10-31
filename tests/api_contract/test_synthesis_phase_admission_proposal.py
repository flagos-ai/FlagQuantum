"""Public phase input stays proposed while private Compiler evidence is real."""

import inspect
import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import select_arithmetic
from flagquantum.services.synthesis import _complete

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = json.loads(
    (ROOT / "contracts/synthesis-phase-admission-proposal.json").read_text()
)


def predicate(x: int) -> bool:
    return x == 1


def test_proposal_records_current_api_and_precise_pending_delta_without_fake_approval():
    assert PROPOSAL["status"] == "proposed"
    assert PROPOSAL["public_implementation_authorized"] is False
    assert PROPOSAL["public_api_implemented"] is False
    assert PROPOSAL["private_compiler_implemented"] is True
    assert PROPOSAL["api_owner_approval"] is None
    assert PROPOSAL["release_authorized"] is False
    assert PROPOSAL["human_trial_records"] == []
    signature = str(inspect.signature(synth.synthesize))
    assert signature == PROPOSAL["current_synthesize_signature"]
    assert "output_mode" not in inspect.signature(synth.synthesize).parameters
    assert PROPOSAL["proposed_synthesize_signature"] == signature.replace(
        ") -> 'Program'",
        ", output_mode: \"Literal['value', 'phase']\" = 'value') -> 'Program'",
    )
    assert list(synth.__all__) == PROPOSAL["namespace_exports"]
    assert str(inspect.signature(fq.run)) == PROPOSAL["unchanged_run_signature"]
    assert (
        str(inspect.signature(synth.Program.probabilities))
        == PROPOSAL["unchanged_probabilities_signature"]
    )
    assert (
        PROPOSAL["new_public_types"] == []
        and PROPOSAL["new_executor"] is False
        and PROPOSAL["serialization_change"] is False
    )
    assert (
        PROPOSAL["phase_scope"]["basis"] is None
        and PROPOSAL["phase_scope"]["topology"] is None
    )
    doc = (ROOT / PROPOSAL["proposal"]).read_text()
    assert "## Proposed user code: requires later public implementation" in doc
    assert 'output_mode="phase"' in doc and "result = fq.run(program)" in doc


@pytest.mark.parametrize("kind", ["python", "text", "file"])
def test_public_phase_keyword_is_not_silently_implemented_or_dispatched(
    kind, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        _api,
        "plan",
        lambda *a, **k: pytest.fail("public phase planned before admission"),
    )
    value = {
        "python": predicate,
        "text": "qfunc main(output q:qbit){allocate(q);}",
        "file": tmp_path / "absent.qmod",
    }[kind]
    with pytest.raises(TypeError, match="output_mode"):
        synth.synthesize(value, output_mode="phase")


def test_private_phase_product_uses_the_canonical_program_owner_and_existing_run():
    graph = capture(predicate, {"x": 1})
    selection = select_arithmetic(
        (graph,), entry=graph.name, input_names=("x",), output_mode="phase"
    )
    product = _complete(selection, {"output_mode": "phase"}, memory_limit_bytes=1024**2)
    assert type(product) is synth.Program
    result = fq.run(product)
    assert result.plan.identity == product.execution.identity
    assert dict(product.probabilities(result, registers=("x",))) == {(0,): 1.0}
    assert "result" not in product.registers
    with pytest.raises(ValueError, match="available names"):
        product.probabilities(result)
