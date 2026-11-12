"""Original notebook Bell/MyCNOT definitions versus unchanged native consumers."""

import json
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.bell_tensor_workflow import (
    bell_program,
    bell_providers,
    contract_bell_program,
    main,
)
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
)
from flagquantum.compiler.registers.component_zero_initialization import (
    initialize_component_from_zero,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@lru_cache(maxsize=1)
def original_notebook() -> dict[str, Any]:
    """Execute unchanged definition cells from the pinned, hash-checked notebook."""
    import hashlib

    import qualtran

    path = Path(qualtran.__file__).parent / "simulation/tensor.ipynb"
    data = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == "d91373e7d8eb21c48a0b870d99d1bb2973971861"
    )
    notebook = json.loads(data)
    namespace: dict[str, Any] = {}
    for index in (1, 21, 24):
        exec(
            compile(
                "".join(notebook["cells"][index]["source"]),
                f"tensor.ipynb:{index}",
                "exec",
            ),
            namespace,
        )
    return namespace


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_original_state_matrix_initialization_and_physical_append(
    prepared: bool, factorized: bool, full: bool, dtype: torch.dtype
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CNOT, Hadamard
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.simulation.tensor import initialize_from_zero

    namespace = original_notebook()
    bb = BloqBuilder()
    q1 = bb.add_register("q1", 1)
    q2 = bb.add_register("q2", 1)
    q1 = bb.add(Hadamard(), q=q1)
    q1, q2 = bb.add(CNOT(), ctrl=q1, target=q2)
    source_circuit = bb.finalize(q1=q1, q2=q2)
    source = namespace["BellState"]() if prepared else source_circuit
    reference = source.tensor_contract()
    like = torch.empty((), dtype=dtype)
    record = bell_program(prepared=prepared)
    actual = contract_bell_program(
        record, like, factorized_cnot=factorized, full_flatten=full
    )
    tolerance = 2e-7 if dtype == torch.complex64 else 2e-14
    np.testing.assert_allclose(actual.numpy(), reference, atol=tolerance, rtol=0)
    initialized = initialize_component_from_zero(bell_program(prepared=False), "bell")
    state = contract_bell_program(
        initialized,
        like,
        entry="from_zero",
        factorized_cnot=factorized,
        full_flatten=full,
    )
    np.testing.assert_allclose(
        state.numpy(),
        initialize_from_zero(source_circuit).tensor_contract(),
        atol=tolerance,
        rtol=0,
    )
    if prepared:
        torch.testing.assert_close(actual, state, rtol=0, atol=0)
    bound = bind_register_program(
        record,
        () if prepared else ((0,), (1,)),
        clean=(0, 1) if prepared else (),
        entry="bell",
        allow_workspace_outputs=prepared,
    )
    assert tuple((g.name, g.wires) for g in bound.circuit.instructions) == (
        ("h", (0,)),
        ("cx", (0, 1)),
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 1, "cx": 1}
    # Prepared outputs transfer ownership; none is claimed to be clean scratch.
    assert bound.outputs == ((0,), (1,))
    assert get_cost_value(source, QubitCount()) == 2
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == model.total_t_count() == 0
    assert model.clifford == (1 if prepared else 2)
    if prepared:
        initial = torch.zeros(8, dtype=torch.complex128)
        initial[:2] = torch.tensor([1, 1j], dtype=torch.complex128) / np.sqrt(2)
        expected = torch.as_tensor(reference).reshape(4, 1) * initial[:2].reshape(1, 2)
    else:
        phase = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
        x = torch.arange(8, dtype=torch.float64)
        initial = (x + 1) * torch.exp(1j * phase * x)
        initial = initial / initial.norm()
        expected = torch.as_tensor(reference) @ initial.reshape(4, 2)
    observed = (
        Circuit(3, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(3, bound.circuit.instructions))
        .state()
        .reshape(4, 2)
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=2e-14)
    restored = forced_branch(
        observed.flatten(),
        (bound.circuit.instructions[1], bound.circuit.instructions[0]),
        (),
    )
    torch.testing.assert_close(restored, initial, rtol=0, atol=2e-14)
    if not prepared:
        torch.testing.assert_close(
            torch.autograd.grad(observed.real.sum(), phase, retain_graph=True)[0],
            torch.autograd.grad(expected.real.sum(), phase)[0],
            rtol=0,
            atol=2e-14,
        )


@pytest.mark.parametrize("factorized", [False, True])
def test_unflattened_refusal_and_explicit_cnot_provider_replacement(
    factorized: bool,
) -> None:
    from qualtran.simulation.tensor import cbloq_to_quimb

    source = original_notebook()["BellState"]()
    with pytest.raises(NotImplementedError):
        cbloq_to_quimb(source.as_composite_bloq())
    like = torch.empty((), dtype=torch.complex128)
    record = bell_program(prepared=True)
    providers = bell_providers(record, like, factorized_cnot=factorized)
    # Retain Bell itself, deliberately bypassing normal expansion.
    wrapped, _ = expand_component_root_for_tensor(
        record, lambda _: True, "bell", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(wrapped, "tensor_root"))
    with pytest.raises(ValueError, match="no tensor provider.*bell"):
        component_tensor_nodes_with_labels(
            wrapped, order, providers.supply, like, "tensor_root"
        )
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "bell", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    nodes, inc, out = component_tensor_nodes_with_labels(
        selected, order, providers.supply, like, "tensor_root"
    )
    assert len(nodes) == (4 if factorized else 3)
    result = contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), inc, out
    )
    np.testing.assert_allclose(
        result.numpy(), source.tensor_contract(), rtol=0, atol=2e-14
    )
    # Original MyCNOT's explicit dense data has the exact native unitary matrix.
    original = original_notebook()["MyCNOT"]().tensor_contract()
    cnot = next(f for f in record.functions if f.name == "cnot")
    supplied = tuple(
        providers.supply(
            cnot,
            {"ctrl": ((0,),), "target": ((1,),)},
            {"ctrl": ((2,),), "target": ((3,),)},
            lambda: 4,
        )
    )
    assert all(not isinstance(node, int) for node in supplied)
    nodes_only = tuple(node for node in supplied if not isinstance(node, int))
    native = contract_register_tensors(
        nodes_only,
        Signature(cnot.registers or ()),
        {"ctrl": ((0,),), "target": ((1,),)},
        {"ctrl": ((2,),), "target": ((3,),)},
    )
    np.testing.assert_array_equal(native.numpy(), original)
    with pytest.raises(ValueError, match="declared primitive"):
        tuple(providers.supply(replace(cnot, blocks=()), {}, {}, lambda: 4))


def test_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "circuit shape (4, 4)" in capsys.readouterr().out
