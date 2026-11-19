# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The complete protocol overview links domain owners to executable examples."""

import builtins
import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

pytest.importorskip("qualtran")
pytest.importorskip("numpy")

from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/Protocols.ipynb"
NATIVE = ROOT / "examples/notebooks/component_protocols.ipynb"


def test_complete_overview_executes_existing_native_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "8fe215cc166ccc223aeeb4edfa0dccfeec37ffb2"
    )
    source = json.loads(raw)
    native = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == len(native["cells"]) == 2
    assert all(cell["cell_type"] == "markdown" for cell in native["cells"])
    assert [cell["metadata"]["frozen_source_cell"] for cell in native["cells"]] == [
        0,
        1,
    ]
    source_text = "\n".join("".join(c["source"]) for c in source["cells"])
    text = "\n".join("".join(c["source"]) for c in native["cells"])
    for topic in ("Decomposition", "Call Graph", "Tensor", "Classical"):
        assert f"### {topic}" in source_text and f"### {topic}" in text
    assert (NATIVE.parent / "call_graph_protocol.ipynb").is_file()
    real_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Protocol overview examples must use native owners")
        return real_import(name, globals, locals, fromlist, level)

    examples = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(examples) == 2
    namespace: dict[str, Any] = {}
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        for index, example in enumerate(examples):
            exec(compile(example, f"component-protocol-{index}", "exec"), namespace)
    assert verify_program_record(namespace["program"]) == ()
    assert namespace["bound"].circuit.instructions == (
        Instruction("cx", (0, 1)),
        Instruction("cx", (0, 2)),
    )

    from qualtran import BloqBuilder, QUInt, Signature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    builder, values = BloqBuilder.from_signature(
        Signature.build_from_dtypes(control=QUInt(1), target=QUInt(2))
    )
    control = builder.split(values["control"])
    targets = builder.split(values["target"])
    for index in range(2):
        control[0], targets[index] = builder.add(
            CNOT(), ctrl=control[0], target=targets[index]
        )
    original = builder.finalize(
        control=builder.join(control, QUInt(1)),
        target=builder.join(targets, QUInt(2)),
    )
    operator = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    actual_operator = namespace["operator"].to(torch.complex128)
    torch.testing.assert_close(
        actual_operator, operator.reshape(8, 8), rtol=0, atol=1e-7
    )
    model = get_cost_value(original, QECGatesCost())
    assert model.clifford == 2 and model.total_t_count() == model.measurement == 0
    initial = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128)
    initial *= torch.exp(0.31j * torch.arange(16, dtype=torch.float64))
    initial /= initial.norm()
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(namespace["operation"])
    expected = (operator.reshape(8, 8) @ initial.reshape(8, 2)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-14)
    circuit.append(namespace["operation"])
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-14)
