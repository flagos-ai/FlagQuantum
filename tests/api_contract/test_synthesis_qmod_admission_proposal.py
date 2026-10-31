"""Retained source review gates and existing canonical consumer evidence."""

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.services._synthesis_registers import synthesize_registers

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = json.loads(
    (ROOT / "contracts/synthesis-qmod-admission-proposal.json").read_text()
)


def identity(x: int) -> int:
    return x


def test_proposal_does_not_activate_or_change_the_existing_public_api():
    active = json.loads(
        (ROOT / "contracts/synthesis-program-run-v1-candidate.json").read_text()
    )
    assert PROPOSAL["status"] == "proposed"
    assert PROPOSAL["implementation_authorized"] is False
    assert PROPOSAL["production_implemented"] is False
    assert PROPOSAL["api_owner_approval"] is None
    assert PROPOSAL["release_authorized"] is False
    assert not PROPOSAL["human_trial_records"]
    assert tuple(synth.__all__) == tuple(active["namespace_exports"])
    assert str(inspect.signature(synth.synthesize)) == active["synthesize_signature"]
    assert str(inspect.signature(fq.run)) == active["run_signature"]
    assert (
        str(inspect.signature(synth.Program.probabilities))
        == active["probabilities_signature"]
    )
    assert (
        PROPOSAL["proposed_synthesize_signature"]
        != PROPOSAL["current_synthesize_signature"]
    )


@pytest.mark.parametrize("producer", ["python", "qmod_text", "qmod_file"])
def test_current_producers_support_the_same_public_numerical_consumer(
    tmp_path, producer
):
    if producer == "python":
        program = synth.synthesize(
            identity,
            inputs={"x": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
        )
    else:
        source = "qfunc main(output x:qbit, output result:qbit) { allocate(x); H(x); result=x; }"
        name = "review-text.qmod"
        if producer == "qmod_file":
            path = tmp_path / "review-file.qmod"
            path.write_text(source)
            source, name = path.read_text(), str(path)
        # This is today's private producer, not a fake public source overload.
        imported = parse_qmod(
            source, filename=name, source_version=SUPPORTED_SOURCE_VERSION
        )
        program = synthesize_registers(
            imported.program, imported.functions, memory_limit_bytes=536_870_912
        )
    result = fq.run(program)
    assert type(program) is synth.Program
    assert dict(
        program.probabilities(result, registers=("x", "result"))
    ) == pytest.approx({(0, 0): 0.5, (1, 1): 0.5})
    torch.testing.assert_close(
        result.statevector().reshape(-1),
        torch.tensor(
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], dtype=torch.complex128
        ),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["source_resources"].n_operations == 2
    assert program.report["resources"].n_operations == 2
    if producer == "python":
        assert program.report["arithmetic_resources"].n_operations == 1
    else:
        # A segment has one CX, while the whole source also prepares H. Do not
        # freeze the private report's misleading arithmetic_resources label.
        segment = program.selection.selected.arithmetic.segments[0]
        assert estimate_resources(segment.circuit).n_operations == 1


def test_existing_named_reader_never_invents_a_result_output():
    imported = parse_qmod(
        "qfunc main(output q:qbit) { allocate(q); X(q); }",
        filename="review-output.qmod",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    program = synthesize_registers(imported.program, memory_limit_bytes=536_870_912)
    result = fq.run(program)
    with pytest.raises(ValueError, match="register"):
        program.probabilities(result)
    assert dict(program.probabilities(result, registers=("q",))) == {(1,): 1.0}
