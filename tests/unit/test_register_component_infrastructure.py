"""Compare complete tutorial dependencies to the untouched source call graph."""

from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.register_component_tutorial import (
    WordSwap,
    bit_swap_program,
    modular_exponentiation_program,
)
from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/_infra/Bloqs-Tutorial.ipynb"
)


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "89419aa389e3da01c6da1bd49783ce9d4e45e746"
    )
    return execute(SOURCE)


@pytest.mark.parametrize("width,shaped", [(1, False), (5, False), (4, True)])
def test_nested_swap_actual_gates_and_complete_complex_amplitudes(
    original: dict[str, Any], width: int, shaped: bool
) -> None:
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.simulation.unitary import get_unitary

    record = WordSwap(width).decompose(shaped=shaped, implemented=True)
    assert verify_program_record(record) == ()
    bound = bind_register_program(
        record,
        (
            tuple((i,) for i in range(2 * width))
            if shaped
            else (tuple(range(width)), tuple(range(width, 2 * width)))
        ),
        entry="SwapManyBits" if shaped else "Swap",
    )
    expected = tuple(
        Instruction("cx", pair)
        for i in range(width)
        for pair in ((i, width + i), (width + i, i), (i, width + i))
    )
    assert bound.circuit.instructions == expected
    assert bound.events == ()
    state = torch.arange(1 << (2 * width + 1), dtype=torch.float64)
    state = torch.complex(state + 1, state.flip(0) + 3)
    state /= state.norm()
    unitary = get_unitary(CircuitIR(2 * width, expected), dtype=torch.complex128)
    actual = (unitary @ state.reshape(1 << (2 * width), 2)).flatten()
    reference = state.reshape(1 << width, 1 << width, 2).transpose(0, 1).flatten()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    # The source tutorial's CNOT is intentionally only declared. Compare its
    # actual three-call operand chain, then give the native leaf an explicit CX.
    source = original["SwapTwoBits"]().decompose_bloq()
    calls = sorted(source.iter_bloqsoqs(), key=lambda row: row[0].i)
    assert len(calls) == 3
    assert calls[1][1]["control"] == calls[0][2][1]
    assert calls[2][1]["control"] == calls[1][2][1]
    native_calls = list(iter_component_calls(bit_swap_program(), "SwapTwoBits"))
    assert len(native_calls) == 3
    assert native_calls[1][1]["control"] == native_calls[0][2][1]
    assert native_calls[2][1]["control"] == native_calls[1][2][1]


@pytest.mark.parametrize(
    "base,modulus,e,n", [(8, 221, 3, 1024), (4, 15, 3, 4), (2, 7, 0, 3)]
)
def test_modexp_ordered_connections_match_source(
    original: dict[str, Any], base: int, modulus: int, e: int, n: int
) -> None:
    from qualtran.bloqs.cryptography.rsa import ModExp

    operation = ModularExponentiation(base, modulus, e, n)
    record = modular_exponentiation_program(operation)
    assert verify_program_record(record) == ()
    calls = sorted(iter_component_calls(record, "ModExp"), key=lambda row: row[0])
    source = sorted(
        ModExp(base=base, mod=modulus, exp_bitsize=e, x_bitsize=n)
        .decompose_bloq()
        .iter_bloqsoqs(),
        key=lambda row: row[0].i,
    )
    assert len(calls) == len(source) == e + 3
    native_mul = calls[2:-1]
    source_mul = source[2:-1]
    root = record.functions[0].blocks[0]
    assert [root.operations[pos].callee for pos, _, _ in native_mul] == [
        f"CModMulK({row[0].bloq.k})" for row in source_mul
    ]
    for index, ((_, inputs, outputs), (_, src_inputs, src_outputs)) in enumerate(
        zip(native_mul, source_mul, strict=True)
    ):
        assert isinstance(inputs["ctrl"], RegisterBinding)
        assert isinstance(outputs[1], RegisterBinding)
        assert inputs["ctrl"].index == src_inputs["ctrl"].idx == (e - index - 1,)
        predecessor = calls[0][2][0] if index == 0 else native_mul[index - 1][2][1]
        assert inputs["x"] == predecessor
        src_predecessor = source[0][2][0] if index == 0 else source_mul[index - 1][2][1]
        assert src_inputs["x"] == src_predecessor
        assert outputs[1].register.bitsize == n
        assert src_outputs[1].reg.bitsize == n
    if e:
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_register_program(record, (tuple(range(e)),), entry="ModExp")
    else:
        bound = bind_register_program(
            record,
            ((),),
            clean=tuple(range(n)),
            entry="ModExp",
            allow_workspace_outputs=True,
        )
        assert bound.circuit.instructions == (Instruction("x", (n - 1,)),)


def test_complete_notebook_and_source_state_effect_tensors(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import json
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    root = Path(__file__).resolve().parents[2]
    path = root / "examples/notebooks/register_component_infrastructure.ipynb"
    cells = json.loads(path.read_text())["cells"]
    source_cells = json.loads(SOURCE.read_text())["cells"]
    assert len(source_cells) == 70 and len(cells) == 72
    assert [cell["metadata"]["frozen_source_cell"] for cell in cells[:70]] == list(
        range(70)
    )
    assert [cell["cell_type"] for cell in cells[:70]] == [
        cell["cell_type"] for cell in source_cells
    ]
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(path)
    import numpy as np

    np.testing.assert_array_equal(
        native["zero_tensor"].numpy(), original["ZeroState"]().tensor_contract()
    )
    source_projection = original["cbloq"].tensor_contract()
    assert native["projection_amplitude"].item() == source_projection
    assert native["projection_probability"].item() == source_projection**2
    assert (
        native["array_register"].total_bits() == original["arr_reg"].total_bits() == 288
    )
    assert native["WordSwap"](10_000).declaration.blocks == ()
    assert native["WordSwap"](10_000).signature.n_qubits() == 20_000
    assert len(native["circuit"].to_ir().instructions) == 15


def test_append_binding_rejects_aliases_and_wrong_widths() -> None:
    for x, y in (((0,), (1, 2)), ((0, 1), (1, 2)), ((0, -1), (2, 3))):
        with pytest.raises(ValueError):
            WordSwap(2).make_on(x, y)
