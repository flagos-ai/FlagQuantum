"""Execute the whole adjoint guide and compare native graphs and operators."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip("NumPy reference dependency is absent", allow_module_level=True)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function_data,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/Adjoint.ipynb"
NATIVE = ROOT / "examples/notebooks/register_adjoints.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "d063fb12c9648da04de773dae977cec23973f004"
    )
    return execute(SOURCE)


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def test_all_source_cells_mapped_and_native_optional_sdk_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = json.loads(SOURCE.read_text())
    native = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == len(native["cells"]) == 17
    assert [c["cell_type"] for c in source["cells"]] == [
        c["cell_type"] for c in native["cells"]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in native["cells"]] == list(
        range(17)
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
            raise AssertionError("Native adjoint tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        actual = execute(NATIVE)
    assert actual["phase_counts"] == {"h": 1, "t": 1}
    assert actual["phase_adjoint_counts"] == {"tdg": 1, "h": 1}


def _supply(
    function: FunctionRecord,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[TensorNetworkNode, ...]:
    assert function.registers is not None
    signature = Signature(function.registers)
    basis = integer_basis_function_data(function)
    if basis is not None:
        value, state = basis
        return register_basis_vector_nodes(
            signature,
            outgoing if state else incoming,
            {"val": value},
            torch.empty((), dtype=torch.complex128),
            state=state,
        )
    inputs = tuple((index,) for index, _ in enumerate(signature.lefts()))
    bound = bind_register_program(
        ProgramRecord((function,)), inputs, entry=function.name
    )
    matrix = get_unitary(bound.circuit, dtype=torch.complex128)
    return (
        bind_register_tensor(matrix, signature, incoming, outgoing, name=function.name),
    )


def _dense(record: ProgramRecord) -> torch.Tensor:
    selected, _ = expand_components_for_tensor(record, lambda name: False)
    order = tuple(position for position, _, _ in iter_component_calls(selected))
    return contract_component_tensors(
        selected, order, _supply, torch.empty((), dtype=torch.complex128)
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_bell_full_tensor_and_lifetime_direction(
    original: dict[str, Any], native: dict[str, Any], inverse: bool
) -> None:
    record = native["bell_adjoint"] if inverse else native["bell"]
    source = original["bell_cbloq"].adjoint() if inverse else original["bell_cbloq"]
    actual = _dense(record)
    expected = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, atol=3e-16, rtol=0)
    assert torch.count_nonzero(actual[1:-1]) == 0
    bound = native["bell_inverse_bound"] if inverse else native["bell_bound"]
    assert [op.name for op in bound.circuit.instructions] == (
        ["cx", "cx", "cx", "h"] if inverse else ["h", "cx", "cx", "cx"]
    )
    assert [op.wires for op in bound.circuit.instructions] == (
        [(2, 3), (1, 2), (0, 1), (0,)] if inverse else [(0,), (0, 1), (1, 2), (2, 3)]
    )
    assert len(bound.events) == 4
    assert {event.kind for event in bound.events} == (
        {"free"} if inverse else {"allocate"}
    )
    assert bound.outputs == (() if inverse else ((0,), (1,), (2,), (3,)))
    assert torch.count_nonzero(native["recovered"][1:]) == 0


@pytest.mark.parametrize("inverse", [False, True])
def test_phase_same_gate_stream_complex_spectator_and_model(
    original: dict[str, Any], native: dict[str, Any], inverse: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = original["TStateMaker"]()
    if inverse:
        source = source.adjoint()
    bound = native["phase_inverse_bound"] if inverse else native["phase_bound"]
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source.decompose_bloq(),
            {"x": (0,)},
            (),
            {},
            cirq_style=True,
            keep_exchange_leaves=True,
        ),
    )
    matrix = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    record = native["phase_adjoint"] if inverse else native["phase"]
    torch.testing.assert_close(_dense(record), matrix, atol=3e-16, rtol=0)
    initial = torch.tensor([1 + 2j, 3 - 1j, 2 - 4j, -3 + 2j], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.kron(matrix.contiguous(), torch.eye(2, dtype=torch.complex128)) @ initial,
        atol=3e-16,
        rtol=0,
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.t == model.clifford == 1
    assert model.measurement == 0
    assert len(bound.circuit.instructions) == 2 and not bound.events


def test_serial_call_reversal_and_explicit_atom_model(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    for inverse in (False, True):
        record = native["serial_adjoint"] if inverse else native["serial"]
        calls = [
            record.functions[0].blocks[0].operations[i].callee
            for i, _, _ in iter_component_calls(record)
        ]
        assert calls == (
            ["atom2", "atom1", "atom0"] if inverse else ["atom0", "atom1", "atom2"]
        )
        bound = bind_register_program(record, ((0,),))
        assert [op.name for op in bound.circuit.instructions] == ["x", "x", "x"]
    # The source guide only draws these default atomic wrappers.
    source = original["bloq"]
    assert get_cost_value(source, QECGatesCost()).t == 300
    with pytest.raises(DecomposeTypeError, match="is atomic"):
        get_cost_value(source.adjoint(), QECGatesCost())


def test_and_exact_stream_models_and_both_measurement_branches(
    native: dict[str, Any],
) -> None:
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    forward, backward = native["and_forward"], native["and_backward"]
    for inverse, actual in ((False, forward), (True, backward)):
        source = And().adjoint() if inverse else And()
        assert_source(
            actual,
            reference_arithmetic_gates(
                source,
                {"ctrl": (0, 1), "target": (2,)},
                (),
                {2: 0},
                cirq_style=True,
                keep_exchange_leaves=True,
            ),
        )
        model = get_cost_value(source, QECGatesCost())
        actual_model = native["modeled_costs"]["erase" if inverse else "compute"]
        assert (model.t, model.and_bloq, model.clifford, model.measurement) == (
            actual_model.t,
            actual_model.and_bloq,
            actual_model.clifford,
            actual_model.measurement,
        )
    initial = torch.zeros(16, dtype=torch.complex128)
    indices = torch.tensor([i for i in range(16) if not i & 2])
    initial[indices] = torch.arange(1, 9, dtype=torch.float64) + 1j * torch.arange(
        8, dtype=torch.float64
    ).flip(0)
    initial /= initial.norm()
    for outcome in (0, 1):
        actual = forced_branch(initial, (*forward, *backward), (outcome,))
        torch.testing.assert_close(actual, initial / 2**0.5, atol=3e-16, rtol=0)
        assert torch.count_nonzero(actual[[i for i in range(16) if i & 2]]) == 0


def test_graph_outputs_render_without_changing_execution(
    native: dict[str, Any], tmp_path: Path
) -> None:
    for name in (
        "zero_graph",
        "zero_effect_graph",
        "bell_graph",
        "bell_adjoint_graph",
        "serial_graph",
        "serial_adjoint_graph",
        "phase_graph",
        "phase_adjoint_graph",
    ):
        svg = native[name].pipe(format="svg")
        assert b"<svg" in svg
        (tmp_path / f"{name}.svg").write_bytes(svg)
    assert native["phase_counts"] == {"h": 1, "t": 1}


@pytest.mark.parametrize("name", ["bell", "serial", "phase"])
@pytest.mark.parametrize("inverse", [False, True])
def test_complete_call_connections(
    original: dict[str, Any], native: dict[str, Any], name: str, inverse: bool
) -> None:
    from tests.unit.test_component_construction_notebook import normalized
    from tests.unit.test_component_graph import _native_trace, _source_trace
    from tests.unit.test_register_component_calls import SourceCalls

    source = {
        "bell": original["bell_cbloq"],
        "serial": original["bloq"].decompose_bloq(),
        "phase": original["TStateMaker"]().decompose_bloq(),
    }[name]
    if inverse:
        source = source.adjoint()
    record = native[f"{name}_adjoint" if inverse else name]

    def port_names(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(port_names(item) for item in value)
        return "q" if value == "val" else value

    # Basis functions use the owned port spelling val for the source q port.
    assert port_names(normalized(_native_trace(record))) == port_names(
        normalized(_source_trace(SourceCalls, source))
    )
