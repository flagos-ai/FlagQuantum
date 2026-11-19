"""Execute the complete native graph guide against its frozen conceptual source."""

import re
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
pytest.importorskip("cirq")
ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/development/REGISTER_COMPONENT_GRAPH_GUIDE.md"
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/_infra/Readme.md"


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "a6c706ec0d97a999b66da4e51b0c772eaca2514b"
    )
    namespace: dict[str, Any] = {}
    cells = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S)
    assert len(cells) == 2
    for index, code in enumerate(cells):
        exec(compile(code, f"graph-guide-{index}", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("repetitions", [1, 2])
def test_source_call_chain_and_physical_stream(
    native: dict[str, Any], repetitions: int
) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran.bloqs.basic_gates import Hadamard

    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    builder = BloqBuilder()
    q = builder.allocate(dtype=QBit())
    for _ in range(repetitions):
        q = builder.add(Hadamard(), q=q)
    builder.free(q)
    source = builder.finalize()
    record = native["hadamard_graph"](repetitions)
    calls = list(iter_component_calls(record))
    assert len(calls) == len(list(source.iter_bloqsoqs())) == repetitions + 2
    root = record.functions[0].blocks[0]
    assert root.arguments == root.return_values == ()
    for previous, current in zip(
        root.operations[:-1], root.operations[1:], strict=True
    ):
        assert current.operands == previous.results
    assert len(source.connections) == repetitions + 1
    bound = bind_register_program(record, (), clean=(0,))
    expected = reference_arithmetic_gates(source, {}, (0,), {})
    assert bound.circuit.instructions == expected
    assert [g.name for g in expected] == ["h"] * repetitions
    assert [event.kind for event in bound.events] == ["allocate", "free"]
    assert bound.outputs == ()


def test_echo_preserves_full_spectator_amplitudes_and_clean_work(
    native: dict[str, Any],
) -> None:
    import flagquantum as fq

    spectator = torch.tensor([1 + 2j, -3 + 5j], dtype=torch.complex128)
    spectator /= spectator.norm()
    state = torch.zeros(4, dtype=torch.complex128)
    state[:2] = spectator
    result = (
        fq.Circuit(2, inputs=state, dtype=torch.complex128)
        .append(native["HadamardEcho"]())
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, state, rtol=0, atol=3e-16)
    assert float(result[2:].abs().max()) < 3e-16
    record = native["hadamard_graph"](2)
    calls = record.functions[0].blocks[0].operations
    assert calls[1].callee == calls[2].callee
    assert calls[1].results != calls[2].results
