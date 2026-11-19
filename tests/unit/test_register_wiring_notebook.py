"""Execute every wiring tutorial cell and close its existing arithmetic dependencies."""

import builtins
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha1
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT.parent / "ftqc-reference-env/tutorials/quantum-programming/1-wiring-up.ipynb"
)
NATIVE = ROOT / "examples/notebooks/register_wiring.ipynb"


def execute(path: Path) -> dict[str, Any]:
    """Keep the arbitrary executed notebook namespace at this test boundary."""
    namespace: dict[str, Any] = {}
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"wiring-{index}", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "061fa74a7a2a4213bd406ba9fdd16cbad07276c2"
    )
    return execute(SOURCE)


def test_complete_source_cells_and_native_execution(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.loads(SOURCE.read_text())
    native = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == 19
    assert len(native["cells"]) == 20
    assert [c["cell_type"] for c in native["cells"][:-1]] == [
        c["cell_type"] for c in source["cells"]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in native["cells"][:-1]] == list(
        range(19)
    )
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native wiring tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        actual = execute(NATIVE)
    actual["do_nothing"].to_ir().validate()
    assert actual["do_nothing"].n_qubits == 8
    assert actual["do_nothing"].to_ir().instructions == ()
    assert actual["negate"].n_qubits == 23
    actual["negate"].to_ir().validate()
    assert actual["negated_value"] == original["negate"].call_classically(x=5)[0]
    assert [column.step.name for column in actual["negation_score"].columns] == [
        "BitwiseNot",
        "AddK(1)",
    ]
    source_steps = sorted(original["negate"].iter_bloqsoqs(), key=lambda row: row[0].i)
    assert [type(row[0].bloq).__name__ for row in source_steps] == [
        "BitwiseNot",
        "AddK",
    ]
    assert source_steps[1][1] == {"x": source_steps[0][2][0]}
    assert actual["negation_score"].output_order == ("x",)
    assert actual["identity_score"].output_order == ("x",)
    assert actual["negation_dot"].count("->") == 3


def test_every_eight_bit_signed_value_and_original_codec(
    original: dict[str, Any],
) -> None:
    from flagquantum.simulation.classical_negation import negate_classical_value
    from flagquantum.simulation.integer_encoding import integer_to_bits

    for value in range(-128, 128):
        assert negate_classical_value(value, 8, signed=True) == (
            original["negate"].call_classically(x=value)[0]
        )
        assert integer_to_bits(value, 8, signed=True) == original["QInt"](8).to_bits(
            value
        )
        assert original["do_nothing"].call_classically(x=value) == (value,)


def test_eight_bit_actual_primitive_stream_feedback_and_separate_model(
    original: dict[str, Any],
) -> None:
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    actual = execute(NATIVE)
    expected = reference_arithmetic_gates(
        original["negate"],
        {"x": actual["target"]},
        (*actual["constant"], *actual["carries"]),
        dict(zip(actual["carries"], actual["classical_bits"], strict=True)),
    )
    gates = actual["negate"].to_ir().instructions
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    assert counts == actual["primitive_counts"]
    assert counts["t"] + counts["tdg"] == 28
    assert counts["measure"] == counts["reset"] == 7
    assert counts["x"] == 10
    assert actual["model"].gates.total_t_count() == 28
    assert actual["model"].gates.measurement == 7
    assert actual["model"].peak_logical_qubits == 23


def test_every_small_measurement_branch_full_complex_amplitudes_and_clean_work() -> (
    None
):
    import flagquantum as fq
    from flagquantum.compiler.registers.arithmetic.addition import negate_ir
    from tests.unit.measurement_checks import forced_branch

    native = execute(NATIVE)
    width, work = 3, 5
    gates = negate_ir((0, 1, 2), (3, 4, 5), (6, 7), (0, 1))
    data = torch.arange(16, dtype=torch.int64)
    x, spectator = data >> 1, data & 1
    coefficients = data.to(torch.float64)
    amplitudes = torch.complex(coefficients + 1, 3 * coefficients - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (width + work + 1), dtype=torch.complex128)
    state[(x << (work + 1)) | spectator] = amplitudes
    expected = torch.zeros_like(state)
    expected[((-x % 8) << (work + 1)) | spectator] = amplitudes
    circuit = fq.Circuit(9, inputs=state, dtype=torch.complex128)
    circuit.append(native["RegisterOperation"](gates))
    assert circuit.to_ir().instructions == gates
    for outcomes in product((0, 1), repeat=2):
        branch = forced_branch(state, circuit.to_ir().instructions, outcomes)
        torch.testing.assert_close(branch, expected / 2, rtol=0, atol=2e-12)
        assert torch.count_nonzero(branch.reshape(8, 32, 2)[:, 1:]) == 0
