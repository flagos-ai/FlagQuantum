"""Pinned controlled tensors feed existing contractions and append execution."""

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor,
)
from flagquantum.simulation.tensor_network.models import ContractionGraph
from flagquantum.simulation.value_dimensions import SymbolicArrayShape
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

CASES = [
    (1, np.array(0)),
    (1, np.array(1)),
    (2, np.array(2)),
    (2, np.array(-1)),
    (1, np.array([[0, 1]])),
    (1, np.empty((0,), dtype=int)),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["h", "phase", "array"])
def test_original_dense_tensor_and_existing_contraction(case: Any, kind: str) -> None:
    from qualtran import Controlled, CtrlSpec, QInt, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import GlobalPhase, Hadamard

    width, pattern = case
    dtype = QInt(width) if np.any(pattern < 0) else QUInt(width)
    spec = CtrlSpec(qdtypes=dtype, cvs=pattern)
    rng = np.random.default_rng(51)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    matrix = np.linalg.qr(raw)[0]

    class ArrayChild:
        @property
        def signature(self) -> Any:
            return SourceSignature([SourceRegister("data", QUInt(1), shape=(1, 2))])

        def tensor_contract(
            self, *args: object, **kwargs: object
        ) -> NDArray[np.complex128]:
            return matrix

    child = {
        "h": Hadamard(),
        "phase": GlobalPhase(exponent=0.375),
        "array": ArrayChild(),
    }[kind]
    source = Controlled(child, spec)
    controls = native_signature(SourceSignature(source.ctrl_regs))
    child_signature = native_signature(child.signature)
    child_matrix = np.asarray(child.tensor_contract(), dtype=np.complex128)
    child_before = child_matrix.copy()
    calls: list[int] = []

    def callback() -> NDArray[np.complex128]:
        calls.append(1)
        return child_matrix

    if kind == "phase" and pattern.size == 0:
        with pytest.raises(TypeError):
            source._tensor_data()
        with pytest.raises(TypeError):
            controlled_register_tensor(controls, child_signature, (pattern,), callback)
        assert calls == [1]
        return
    expected = source._tensor_data()
    actual = controlled_register_tensor(controls, child_signature, (pattern,), callback)
    assert calls == [1]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(child_matrix, child_before)
    n = int(np.prod(actual.shape[: actual.ndim // 2]))
    state = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    state /= np.linalg.norm(state)
    # Two spectator amplitudes remain in the joint state, including global phase.
    graph = ContractionGraph()
    graph.add(torch.from_numpy(actual.reshape(n, n)), ("o", "i"))
    graph.add(torch.from_numpy(state), ("i", "s"))
    result = graph.to_einsum(("o", "s")).run().numpy()
    np.testing.assert_allclose(
        result,
        np.einsum("ij,js->is", expected.reshape(n, n), state),
        atol=1e-14,
        rtol=0,
    )
    restored = np.einsum("ji,js->is", actual.reshape(n, n).conj(), result)
    np.testing.assert_allclose(restored, state, atol=1e-14, rtol=0)


@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
def test_dense_tensor_matches_append_all_measurement_histories_and_resources(
    values: tuple[int, ...],
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component import (
        controlled_component_ir,
    )
    from flagquantum.core.ir import Instruction
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    pattern = np.array(values)
    source = Controlled(XGate(), CtrlSpec(cvs=pattern))
    data = controlled_register_tensor(
        native_signature(source.ctrl_regs),
        Signature.build(q=1),
        (pattern,),
        lambda: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    )
    gates = controlled_component_ir(
        ((0, 1),),
        (ControlRegister(1, values),),
        lambda c: (Instruction("cx", (c, 2)),),
        (3,),
        (0,),
    )
    decomposed = ControlledViaAnd(XGate(), source.ctrl_spec)
    original = reference_arithmetic_gates(
        decomposed, {decomposed.ctrl_reg_names[0]: (0, 1), "q": (2,)}, (3,), {3: 0}
    )
    assert_source(gates, original)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(decomposed, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4
    assert counts["measure"] == model.measurement == 1
    state = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128).reshape(8, 2)
    state += 1j * torch.flip(state.real, (0,))
    state /= state.norm()
    initial = torch.zeros((8, 2, 2), dtype=torch.complex128)
    initial[:, 0, :] = state  # wire 3 clean, wire 4 an entangled spectator
    expected = torch.from_numpy(data.reshape(8, 8)) @ state
    for history in ((0,), (1,)):
        branch = forced_branch(initial.flatten(), gates, history).reshape(8, 2, 2)
        torch.testing.assert_close(
            branch[:, 0, :], expected / np.sqrt(2), atol=2e-12, rtol=0
        )
        assert torch.count_nonzero(branch[:, 1, :]) == 0
    circuit = DynamicCircuit(5, inputs=initial.flatten(), dtype=torch.complex128)
    circuit.append(FieldOperation(5, gates))
    for seed in range(4):
        result = run_dynamic(circuit, shots=1, seed=seed, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(8, 2, 2)[:, 0, :], expected, atol=2e-12, rtol=0
        )


def test_multiple_control_registers_preserve_word_axes_and_callback_replacement() -> (
    None
):
    from qualtran import Controlled, CtrlSpec, QBit, QUInt
    from qualtran.bloqs.basic_gates import Hadamard

    patterns = (np.array([0, 1]), np.array(2))
    source = Controlled(Hadamard(), CtrlSpec(qdtypes=(QBit(), QUInt(2)), cvs=patterns))
    controls = native_signature(source.ctrl_regs)
    child = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    first = controlled_register_tensor(
        controls, Signature.build(q=1), patterns, lambda: child
    )
    second = controlled_register_tensor(
        controls, Signature.build(q=1), patterns, lambda: child.conj().T
    )
    np.testing.assert_allclose(first, source._tensor_data(), atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        np.einsum("ij,jk->ik", second.reshape(32, 32), first.reshape(32, 32)),
        np.eye(32),
        atol=1e-14,
        rtol=0,
    )
    assert first.shape == (2, 2, 4, 2) * 2


@pytest.mark.parametrize("failure", ["nonthru", "symbolic", "index", "childshape"])
def test_source_failure_order_and_child_invocation(failure: str) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Hadamard, ZeroState
    from qualtran.symbolics import Shaped

    child = ZeroState() if failure == "nonthru" else Hadamard()
    pattern = (
        Shaped(())
        if failure == "symbolic"
        else np.array(2 if failure == "index" else 1)
    )
    source = Controlled(child, CtrlSpec(cvs=pattern))
    calls: list[int] = []
    matrix = np.eye(3 if failure == "childshape" else 2, dtype=np.complex128)

    def callback() -> NDArray[np.complex128]:
        calls.append(1)
        return matrix

    if failure == "childshape":
        # Replace only the child supplier; the original control tensor method is untouched.
        from unittest.mock import patch

        with (
            patch.object(type(child), "tensor_contract", return_value=matrix),
            pytest.raises(ValueError),
        ):
            source._tensor_data()
    else:
        with pytest.raises(IndexError if failure == "index" else ValueError):
            source._tensor_data()
    native_pattern = SymbolicArrayShape(()) if failure == "symbolic" else pattern
    with pytest.raises(IndexError if failure == "index" else ValueError):
        controlled_register_tensor(
            native_signature(source.ctrl_regs),
            native_signature(child.signature),
            (native_pattern,),
            callback,
        )
    assert calls == ([] if failure == "nonthru" else [1])


@pytest.mark.parametrize("failure", ["arity", "shape", "control_side"])
def test_explicit_native_layout_validation(failure: str) -> None:
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )

    control = Register(
        "ctrl",
        RegisterType("bit", 1),
        side=RegisterSide.LEFT if failure == "control_side" else RegisterSide.THRU,
    )
    patterns = (
        ()
        if failure == "arity"
        else (np.array([1]) if failure == "shape" else np.array(1),)
    )
    with pytest.raises(ValueError):
        controlled_register_tensor(
            Signature([control]),
            Signature.build(q=1),
            patterns,
            lambda: np.eye(2, dtype=np.complex128),
        )
