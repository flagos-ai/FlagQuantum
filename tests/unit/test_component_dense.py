"""Native dense entries compose existing owners and match unchanged source calls."""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import BlockRecord, FunctionRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.matrices import H_MATRIX
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_superoperator_nodes import mixed_graph, supplier
from tests.unit.test_component_tensor_expansion import source_custom
from tests.unit.test_tensor_root_entry import Ports, positions, source_root

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complete_root_selection_dense_entry_and_custom_boundary(
    full: bool, mixed: bool, dtype: torch.dtype
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    record, _ = _hierarchy(1, 2)
    record = replace(
        record,
        functions=(replace(record.functions[0], name="custom"), record.functions[1]),
    )
    selected, _ = expand_component_root_for_tensor(
        record, lambda name: True, "custom", full_flatten=full
    )
    seen: list[str] = []
    like = torch.empty((), dtype=dtype)

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        seen.append(function.name)
        data = (
            torch.eye(2, dtype=dtype)
            if function.name == "custom"
            else torch.tensor([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
        )
        return (
            bind_register_tensor(data, Signature(function.registers or ()), inc, out),
        )

    def open_system(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        return pure(function, inc, out)

    if mixed:
        actual = contract_component_superoperator(
            selected, positions(selected), open_system, like, "tensor_root"
        )
    else:
        actual = contract_component_tensors(
            selected, positions(selected), pure, like, "tensor_root"
        )
    expected = bloq_to_dense(source_custom(), full_flatten=full, superoperator=mixed)
    tolerance = 2e-6 if dtype == torch.complex64 else 2e-14
    np.testing.assert_allclose(actual.numpy(), expected, atol=tolerance, rtol=0)
    assert seen == (["h", "h"] if full else ["custom"])
    assert actual.dtype == dtype and actual.device == like.device
    theta = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)
    state = torch.arange(1, 5) * torch.exp(1j * theta * torch.arange(4))
    state = state.reshape(2, 2) / state.norm()
    if mixed:
        density = torch.einsum("as,bt->abst", state, state.conj())
        observed = torch.einsum("abij,ijst->abst", actual.to(torch.complex128), density)
        oracle = torch.einsum(
            "abij,ijst->abst",
            torch.as_tensor(expected, dtype=torch.complex128),
            density,
        )
    else:
        observed = actual.to(torch.complex128) @ state
        oracle = torch.as_tensor(expected, dtype=torch.complex128) @ state
    torch.testing.assert_close(observed, oracle, atol=tolerance, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(observed.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(oracle.real.sum(), theta)[0],
        atol=tolerance,
        rtol=0,
    )
    bound = bind_register_program(selected, ((0,),), entry="tensor_root", clean=(2,))
    assert bound == bind_register_program(record, ((0,),), entry="custom", clean=(2,))
    gates = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert Counter(g.name for g in gates) == {"h": 2}
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = state.detach().flatten()
    result = forced_branch(initial, gates, ()).reshape(2, 2, 2)
    torch.testing.assert_close(result[..., 0], state.detach(), atol=2e-14, rtol=0)
    assert torch.count_nonzero(result[..., 1]) == 0


@pytest.mark.parametrize("side", [RegisterSide.LEFT, RegisterSide.RIGHT])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_directional_root_returns_vector_or_density_effect_matrix(
    side: RegisterSide, mixed: bool, full: bool
) -> None:
    from qualtran.bloqs.basic_gates import ZeroEffect, ZeroState
    from qualtran.simulation.tensor._dense import bloq_to_dense

    dtype = quantum_bit_type()
    leaf = (
        free_word_function if side == RegisterSide.LEFT else allocate_word_function
    )("zero", dtype, False)
    builder = ComponentProgramBuilder(
        Signature([Register("q", dtype, side=side)]), (leaf,)
    )
    if side == RegisterSide.LEFT:
        builder.call_dict("zero", {"reg": builder.inputs["q"]})
        record = builder.finalize({})
        source = ZeroEffect()
    else:
        record = builder.finalize({"q": builder.call_dict("zero", {})["reg"]})
        source = ZeroState()
    selected, _ = expand_component_root_for_tensor(
        record, lambda name: name != "main", full_flatten=full
    )
    like = torch.empty((), dtype=torch.complex128)

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        return (
            bind_register_tensor(
                torch.tensor([1, 0], dtype=like.dtype),
                Signature(function.registers or ()),
                inc,
                out,
            ),
        )

    def open_system(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        return pure(function, inc, out)

    actual = (
        contract_component_superoperator(
            selected, positions(selected), open_system, like, "tensor_root"
        )
        if mixed
        else contract_component_tensors(
            selected, positions(selected), pure, like, "tensor_root"
        )
    )
    np.testing.assert_array_equal(
        actual.numpy(), bloq_to_dense(source, full_flatten=full, superoperator=mixed)
    )
    assert actual.shape == ((2, 2) if mixed else (2,))


@pytest.mark.parametrize("mixed", [False, True])
def test_explicit_scalar_phase_retains_autograd_but_empty_network_is_refused(
    mixed: bool,
) -> None:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.simulation.tensor._dense import bloq_to_dense

    # This is a selected numerical leaf descriptor, not a physical global-phase lowering.
    leaf = FunctionRecord(
        "phase", (BlockRecord("entry", return_values=()),), registers=()
    )
    builder = ComponentProgramBuilder(Signature([]), (leaf,))
    builder.call_dict("phase", {})
    record = builder.finalize({})
    theta = torch.tensor(0.37 * np.pi, dtype=torch.float64, requires_grad=True)
    value = torch.exp(1j * theta)

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.name == "phase" and not inc and not out
        return (TensorNetworkNode(value, (), name="phase"),)

    def open_system(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        return pure(function, inc, out)

    like = torch.empty((), dtype=torch.complex128)
    if mixed:
        actual = contract_component_superoperator(record, (0,), open_system, like)
    else:
        actual = contract_component_tensors(record, (0,), pure, like)
    original = bloq_to_dense(
        GlobalPhase(exponent=0.37), full_flatten=False, superoperator=mixed
    )
    np.testing.assert_allclose(
        actual.detach().numpy(), np.asarray(original), rtol=0, atol=2e-15
    )
    assert actual.shape == () and actual.dtype == torch.complex128
    expected = value * value.conj() if mixed else value
    torch.testing.assert_close(
        torch.autograd.grad(actual.real, theta, retain_graph=True)[0],
        torch.autograd.grad(expected.real, theta)[0],
        rtol=0,
        atol=2e-15,
    )
    empty = ComponentProgramBuilder(Signature([])).finalize({})
    if mixed:
        with pytest.raises(ValueError, match="at least one tensor node"):
            contract_component_superoperator(empty, (), open_system, like)
    else:
        with pytest.raises(ValueError, match="at least one tensor node"):
            contract_component_tensors(empty, (), pure, like)
    bb, _ = BloqBuilder.from_signature(SourceSignature([]))
    with pytest.raises(ValueError, match="not enough values to unpack"):
        bloq_to_dense(bb.finalize(), superoperator=mixed)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complete_mixed_dense_channel_equals_all_physical_branch_densities(
    full: bool, discard: bool, dtype: torch.dtype
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX, MeasureZ
    from qualtran.simulation.tensor._dense import bloq_to_dense

    record, source = mixed_graph(discard)
    selected, _ = expand_component_root_for_tensor(
        record, lambda name: name != "main", full_flatten=full
    )
    like = torch.empty((), dtype=dtype)
    actual = contract_component_superoperator(
        selected, positions(selected), supplier(like), like, "tensor_root"
    )
    reference = bloq_to_dense(
        source_root(source), full_flatten=full, superoperator=True
    )
    tolerance = 2e-6 if dtype == torch.complex64 else 2e-14
    np.testing.assert_allclose(actual.numpy(), reference, atol=tolerance, rtol=0)
    replacement = contract_component_superoperator(
        selected,
        positions(selected),
        supplier(like, original=True),
        like,
        "tensor_root",
    )
    torch.testing.assert_close(actual, replacement)
    state = torch.arange(1, 9, dtype=torch.float64) * torch.exp(
        1j * 0.31 * torch.arange(8)
    )
    state /= state.norm()
    z = torch.as_tensor(MeasureZ().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    x = torch.as_tensor(MeasureX().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    source_instrument = torch.einsum("iae,jfb,ijs->abefs", z, x, state.reshape(2, 2, 2))
    for idle in (0, 1):
        bound = bind_register_program(
            selected,
            ((0,), (1,), (2,), ()),
            entry="tensor_root",
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        assert bound == bind_register_program(
            record,
            ((0,), (1,), (2,), ()),
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        gates = (
            Circuit(4)
            .append(FieldOperation(4, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        assert Counter(g.name for g in gates) == {
            "h": 1,
            "measure": 2,
            "classical_xor": 1,
        }
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[::2] = state
        input_state = torch.zeros((2, 2, 2, 2), dtype=torch.complex128)
        input_state[..., idle] = state.reshape(2, 2, 2)
        density = torch.outer(input_state.flatten(), input_state.flatten().conj())
        result = torch.einsum("abij,ij->ab", actual.to(torch.complex128), density)
        expected = torch.zeros_like(result)
        for a, b in product((0, 1), repeat=2):
            branch = forced_branch(initial, gates, (a, b)).reshape(2, 2, 2, 2)
            assert torch.count_nonzero(branch[..., 1]) == 0
            restored = torch.einsum("ij,ejk->eik", H_MATRIX, branch[..., 0])
            torch.testing.assert_close(
                restored, source_instrument[a, b], atol=3e-16, rtol=0
            )
            reduced = torch.einsum("efs,eft->st", restored, restored.conj())
            for s, t in product((0, 1), repeat=2):
                row = s * 2 + idle if discard else s * 8 + idle * 4 + b * 2 + a
                col = t * 2 + idle if discard else t * 8 + idle * 4 + b * 2 + a
                expected[row, col] += reduced[s, t]
        torch.testing.assert_close(result, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "problem", ["supplier", "generator", "empty", "order", "unknown", "invalid"]
)
def test_numerical_entry_preserves_validation_and_supplier_failure_identity(
    mixed: bool, problem: str
) -> None:
    record, _ = _hierarchy(1, 1)
    error = RuntimeError("selected numerical supplier failed")
    calls: list[str] = []

    def pure(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> Iterable[TensorNetworkNode]:
        calls.append(function.name)
        if problem == "supplier":
            raise error
        if problem == "generator":
            yield bind_register_tensor(
                torch.eye(2), Signature(function.registers or ()), inc, out
            )
            raise error
        if problem == "empty":
            return
        raise AssertionError("validation must happen before leaf supply")

    def open_system(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        return pure(function, inc, out)

    order = () if problem == "order" else (0,)
    entry = "missing" if problem == "unknown" else "main"
    if problem == "invalid":
        record = replace(record, functions=(*record.functions, record.functions[0]))
    with pytest.raises((RuntimeError, ValueError)) as failure:
        if mixed:
            contract_component_superoperator(
                record, order, open_system, torch.empty(()), entry
            )
        else:
            contract_component_tensors(record, order, pure, torch.empty(()), entry)
    if problem in {"supplier", "generator"}:
        assert failure.value is error and calls == ["h"]
    elif problem == "empty":
        assert "at least one tensor node" in str(failure.value) and calls == ["h"]
    else:
        assert not calls


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    "preparation,mixed", [(False, False), (False, True), (True, False)]
)
def test_phaseful_shaped_root_and_prepared_state_preserve_dense_complex_entries(
    full: bool, preparation: bool, mixed: bool
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._quimb import initialize_from_zero

    from flagquantum.compiler.registers.component_zero_initialization import (
        initialize_component_from_zero,
    )
    from tests.unit.test_component_zero_initialization import phase_fixture
    from tests.unit.test_register_zero_state import supplier as phase_supplier

    record, source = phase_fixture(1, ())
    if preparation:
        record, source = initialize_component_from_zero(record), initialize_from_zero(
            source
        )
    selected, _ = expand_component_root_for_tensor(
        record, lambda name: False, record.functions[0].name, full_flatten=full
    )
    like = torch.empty((), dtype=torch.complex128)

    def open_system(
        function: FunctionRecord, inc: Ports, out: Ports, fresh: Callable[[], int]
    ) -> Iterable[TensorNetworkNode | int]:
        return phase_supplier(function, inc, out)

    actual = (
        contract_component_superoperator(
            selected, positions(selected), open_system, like, "tensor_root"
        )
        if mixed
        else contract_component_tensors(
            selected, positions(selected), phase_supplier, like, "tensor_root"
        )
    )
    original = bloq_to_dense(
        source_root(source), full_flatten=full, superoperator=mixed
    )
    np.testing.assert_allclose(actual.numpy(), original, atol=2e-14, rtol=0)
    assert actual.shape == (
        (8,) if preparation else ((8, 8, 8, 8) if mixed else (8, 8))
    )
    assert bool(torch.count_nonzero(actual.imag))


@pytest.mark.parametrize("classical", [False, True])
def test_direct_input_trace_effect_extension_survives_dense_entry_without_hiding_source_failure(
    classical: bool,
) -> None:
    from qualtran.simulation.tensor._dense import bloq_to_dense

    from tests.unit.test_component_tensor_nodes import order
    from tests.unit.test_component_trace_only import graph, source_graph, trace_supplier

    record = graph(classical, (2,), True)
    actual = contract_component_superoperator(
        record, order(record), trace_supplier, torch.empty((), dtype=torch.complex128)
    )
    with pytest.raises(KeyError):
        bloq_to_dense(source_graph(classical, (2,), True, False), superoperator=True)
    # Executable equivalence uses unchanged Cast-wrapped source, not the broken direct graph.
    expected = bloq_to_dense(
        source_graph(classical, (2,), True, True), superoperator=True
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
