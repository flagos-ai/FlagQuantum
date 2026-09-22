from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.ecosystem.cudaq import to_cudaq

cudaq = pytest.importorskip("cudaq")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.cudaq]
ROOT = Path(__file__).resolve().parents[1]
cudaq.set_target("qpp-cpu")

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONTRACT = tomllib.loads(
    (ROOT / "contracts" / "cudaq-export-contract.toml").read_text(encoding="utf-8")
)


def _state(kernel: object, n_wires: int) -> torch.Tensor:
    raw = np.asarray(cudaq.get_state(kernel))
    reordered = raw.reshape((2,) * n_wires).transpose(tuple(reversed(range(n_wires))))
    return torch.as_tensor(reordered.reshape(-1).copy(), dtype=torch.complex128)


def test_cudaq_export_preserves_operation_order_and_wire_semantics() -> None:
    source = (
        Circuit(3, dtype=torch.complex128)
        .x(2)
        .h(0)
        .ry(1, theta=0.231)
        .cx(0, 2)
        .cz(2, 1)
    )
    kernel = to_cudaq(source)
    expected = source.state()[0]
    torch.testing.assert_close(_state(kernel, 3), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mapping", CONTRACT["operations"])
def test_every_declared_gate_matches_statevector(mapping: dict[str, str]) -> None:
    opcode = mapping["flagquantum"]
    schema = OPERATOR_SCHEMAS[opcode]
    parameters = {name: 0.173 * (i + 1) for i, name in enumerate(schema.parameters)}
    preparations = tuple(
        Instruction("ry", (wire,), {"theta": 0.119 * (wire + 1)})
        for wire in range(schema.arity)
    )
    ir = CircuitIR(
        schema.arity,
        preparations + (Instruction(opcode, tuple(range(schema.arity)), parameters),),
        dtype="complex128",
    )
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(
        _state(to_cudaq(ir), schema.arity), expected, atol=1e-6, rtol=1e-6
    )
