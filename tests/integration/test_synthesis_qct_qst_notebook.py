"""Accept all four source sine/cosine workflows, signs and bounded execution."""

import json
from math import sqrt
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

import flagquantum as fq

pytest.importorskip("numpy")

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qct_qst.overview import (
    main as transforms,
)
from examples.synthesis.functions.qct_qst.overview.comparisons.circuit import (
    reference_circuits,
)
from examples.synthesis.functions.qct_qst.qct_qst_type1.main import (
    AMPLITUDES,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import (
    execute_code_cells,
)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/synthesis/functions/qct_qst/overview"


def test_all_four_original_stages_independent_circuit_phase_order_and_zero_work() -> (
    None
):
    circuits = reference_circuits()
    programs = transforms.run_workflow()
    assert tuple(programs) == ("type_one", "type_two", "cosine", "sine")
    for name, program in programs.items():
        registers = ("x", "q") if name == "type_two" else ("x",)
        compare_states(program, circuits[name], registers)
        assert program.report["workspace_qubits"] == 0
        assert sum(len(wires) for wires in program.registers.values()) == (
            4 if name in ("type_one", "type_two") else 3
        )
    cosine, sine, data = transforms.original_data()
    first = fq.run(programs["type_one"]).statevector().flatten()
    torch.testing.assert_close(
        first[:9],
        (transforms.dct(9, 1) @ cosine / sqrt(2)).to(torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        first[9:], 1j * (transforms.dst(7, 1) @ sine / sqrt(2)), atol=1e-12, rtol=0
    )
    combined = fq.run(programs["type_two"]).statevector().reshape(8, 2)
    torch.testing.assert_close(
        combined[:, 0],
        (transforms.dct(8, 2) @ data / sqrt(2)).to(torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        combined[:, 1],
        -(transforms.dst(8, 2) @ data / sqrt(2)).to(torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    assert float(first[9:].imag.abs().max()) > 0.1
    assert float(combined[:, 1].real.abs().max()) > 0.1


def test_seed123_sequential_inputs_match_original_frozen_vectors() -> None:
    cosine, sine, data = transforms.original_data()
    torch.testing.assert_close(
        torch.cat((cosine, sine)) / sqrt(2),
        torch.tensor(AMPLITUDES, dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )
    assert tuple(len(vector) for vector in (cosine, sine, data)) == (9, 7, 8)
    for vector in (cosine, sine, data):
        assert float(vector.norm()) == pytest.approx(1, abs=1e-15)
        assert bool((vector > 0).all())


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("size", [2, 3, 7, 8, 9])
def test_classical_references_are_orthonormal(kind: Literal[1, 2], size: int) -> None:
    for matrix in (transforms.dct(size, kind), transforms.dst(size, kind)):
        torch.testing.assert_close(
            matrix @ matrix.T, torch.eye(size, dtype=torch.float64), atol=4e-15, rtol=0
        )


@pytest.mark.parametrize(
    "size,kind", [(1, 1), (0, 2), (257, 2), (True, 1), (8, True), (8, 0)]
)
def test_unbounded_or_invalid_classical_reference_refuses(
    size: int, kind: Literal[1, 2]
) -> None:
    for transform in (transforms.dct, transforms.dst):
        with pytest.raises(ValueError, match="local reference"):
            transform(size, kind)


@pytest.mark.parametrize("limit", ["qubits", "memory", "operations", "depth"])
def test_explicit_budget_refusals(limit: str) -> None:
    data = transforms.original_data()[2].tolist()
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if limit == "qubits":
            synth.synthesize(
                transforms.cosine, classical_inputs={"data": data}, max_qubits=2
            )
        elif limit == "memory":
            synth.synthesize(
                transforms.cosine,
                classical_inputs={"data": data},
                memory_limit_bytes=16,
            )
        elif limit == "operations":
            synth.synthesize(
                transforms.cosine, classical_inputs={"data": data}, max_operations=1
            )
        else:
            synth.synthesize(
                transforms.cosine, classical_inputs={"data": data}, max_depth=1
            )


def test_notebook_runs_all_four_native_stages_and_records_complete_source_review() -> (
    None
):
    document = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    namespace, cells = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert cells == 6
    circuits = reference_circuits()
    for stage, variable in (
        ("type_one", "one"),
        ("type_two", "two"),
        ("cosine", "cosine"),
        ("sine", "sine"),
    ):
        program = cast(synth.Program, namespace[f"program_{variable}"])
        compare_states(
            program, circuits[stage], ("x", "q") if stage == "type_two" else ("x",)
        )
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
    ]
    assert proof["acceptance_test"] == str(Path(__file__).relative_to(ROOT))
