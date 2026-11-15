"""Compare original structural rewriting and append the resulting signed block."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.product_block_catalog import product_child
from flagquantum.algorithms.primitives.block_encoding_rewrites import collect_like_terms
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _original_native_tree() -> TensorProductBlockEncoding:
    x, h = product_child("x"), product_child("h")
    phase_h = PhaseBlockEncoding(h, phi=1, eps=1)
    lhs = LinearCombinationBlockEncoding((x, h), (2.0, 2.0), 1)
    rhs = LinearCombinationBlockEncoding((x, phase_h), (1.0, -1.0), 1)
    return TensorProductBlockEncoding(
        (
            LinearCombinationBlockEncoding((lhs, rhs), (1.0, -1.0), 1),
            x,
        )
    )


def test_atomic_phase_product_and_hash_identity() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary
    from qualtran.bloqs.block_encoding.rewrites import collect_like_terms as original

    x = product_child("x")
    root = ProductBlockEncoding((PhaseBlockEncoding(x, phi=1, eps=1), x))
    assert collect_like_terms(x) is x
    assert collect_like_terms(root) == root

    distinct = product_child("x")
    h = product_child("h")
    block = LinearCombinationBlockEncoding((x, distinct, h), (1.0, 2.0, 3.0), 1)
    result = collect_like_terms(block)
    assert isinstance(result, LinearCombinationBlockEncoding)
    assert result.block_encodings == (x, distinct, h)
    source = original(
        LinearCombination(
            (Unitary(XGate()), Unitary(XGate()), Unitary(Hadamard())),
            (1.0, 2.0, 3.0),
            1,
        )
    )
    assert source._lambd == (3.0, 3.0)


def test_original_nested_rewrite_order_coefficients_and_precision() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import (
        LinearCombination,
        Phase,
        TensorProduct,
        Unitary,
    )
    from qualtran.bloqs.block_encoding.rewrites import collect_like_terms as original

    x, h = Unitary(XGate()), Unitary(Hadamard())
    phase_h = Phase(h, phi=1, eps=1)
    lhs = LinearCombination((x, h), (2.0, 2.0), 1)
    rhs = LinearCombination((x, phase_h), (1.0, -1.0), 1)
    source = original(TensorProduct((LinearCombination((lhs, rhs), (1.0, -1.0), 1), x)))
    native = _original_native_tree()
    result = collect_like_terms(native)
    assert isinstance(result, TensorProductBlockEncoding)
    rewritten, tail = result.block_encodings
    assert isinstance(rewritten, LinearCombinationBlockEncoding)
    locked = source.block_encodings[0]
    assert locked._block_encodings == (x, h, phase_h)
    assert rewritten.coefficients == locked._lambd == (1.0, 2.0, 1.0)
    assert rewritten.coefficient_bits == locked.lambd_bits == 2
    assert tail is native.block_encodings[1]
    assert isinstance(rewritten.block_encodings[-1], PhaseBlockEncoding)
    assert (result.system_bitsize, result.ancilla_bitsize, result.resource_bitsize) == (
        source.system_bitsize,
        source.ancilla_bitsize,
        source.resource_bitsize,
    )
    assert result.alpha == source.alpha and result.epsilon == source.epsilon


def test_custom_oracle_blocks_are_opaque_and_cancellation_keeps_zero_terms() -> None:
    x, h = product_child("x"), product_child("h")
    explicit = LinearCombinationBlockEncoding((x, h), (1.0, 1.0), 1)
    custom = replace(explicit, prepare_override=explicit.prepare)
    assert collect_like_terms(custom) is custom
    enclosing = LinearCombinationBlockEncoding((custom, x), (2.0, -1.0), 1)
    rewritten = collect_like_terms(enclosing)
    assert isinstance(rewritten, LinearCombinationBlockEncoding)
    assert rewritten.block_encodings[0] is custom

    zero_term = collect_like_terms(
        LinearCombinationBlockEncoding((x, x, h), (1.0, -1.0, 2.0), 1)
    )
    assert isinstance(zero_term, LinearCombinationBlockEncoding)
    assert zero_term.block_encodings == (x, h)
    assert zero_term.coefficients == (0.0, 2.0)
    with pytest.raises(ValueError, match="at least two"):
        collect_like_terms(LinearCombinationBlockEncoding((x, x), (1.0, 2.0), 1))


def _controlled(
    name: str,
    *,
    negate: bool = False,
) -> Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    FieldOperation,
]:
    def bind(
        predicate: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        wires = (predicate, *ports["system"])
        phase = (Instruction("z", (predicate,)),) if negate else ()
        return FieldOperation(max(wires) + 1, (*phase, Instruction(name, wires)))

    return bind


def test_original_rewritten_terms_append_projection_and_modeled_cost(
    request: pytest.FixtureRequest,
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Phase, Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    from qualtran.bloqs.block_encoding.linear_combination_test import get_tensors

    result = collect_like_terms(_original_native_tree())
    assert isinstance(result, TensorProductBlockEncoding)
    block = result.block_encodings[0]
    assert isinstance(block, LinearCombinationBlockEncoding)
    h = Unitary(Hadamard())
    source = LinearCombination(
        (Unitary(XGate()), h, Phase(h, phi=1, eps=1)), (1.0, 2.0, 1.0), 2
    )
    expected = (
        torch.as_tensor(get_tensors(source), dtype=torch.complex128) / source.alpha
    )
    ancilla = tuple(range(1, 1 + block.ancilla_bitsize))
    resource = tuple(range(1 + len(ancilla), 1 + len(ancilla) + block.resource_bitsize))
    through = 1 + block.ancilla_bitsize + block.resource_bitsize
    prep_work, prep_slots = block.coefficient_preparation.workspace_requirements
    traversal = indexed_component_workspace((block.select.selection_bitsize,), 0)
    work = tuple(range(through, through + max(prep_work, traversal)))
    slots = tuple(range(max(prep_slots, traversal)))
    operation = block.make_on(
        (_controlled("cx"), _controlled("ch"), _controlled("ch", negate=True)),
        system=(0,),
        ancilla=ancilla,
        resource=resource,
        prepare_workspace=work[:prep_work],
        prepare_classical_bits=slots[:prep_slots],
        traversal_workspace=work[:traversal],
        traversal_classical_bits=slots[:traversal],
        workspaces=((),) * 3,
        classical_bits=((),) * 3,
    )
    width = through + len(work)
    gates = DynamicCircuit(width).append(operation).to_ir().instructions
    events = sum(gate.name == "measure" for gate in gates)
    assert events == 11
    data = torch.tensor([1, 1j], dtype=torch.complex128) / sqrt(2)
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    stride = 1 << (width - 1)
    initial[0], initial[stride] = data
    source_matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    logical_state = initial.reshape(-1, 1 << len(work))[:, 0]
    full_expected = torch.zeros_like(initial)
    full_expected.reshape(-1, 1 << len(work))[:, 0] = source_matrix @ logical_state
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            branch, full_expected / sqrt(2**events), atol=6e-12, rtol=0
        )
        torch.testing.assert_close(
            branch[[0, stride]], (expected @ data) / sqrt(2**events), atol=6e-12, rtol=0
        )
        assert branch.reshape(-1, 1 << len(work))[:, 1:].abs().max().item() < 6e-12

    prepare = block.coefficient_preparation.resource_model()
    select = block.select.resource_model(
        (
            ComponentCost("CX", 2, GateCost(clifford=1)),
            ComponentCost("CH", 2, GateCost(clifford=4, rotation=2)),
            ComponentCost("Controlled negative H", 2, GateCost(clifford=5, rotation=2)),
        )
    )
    cost = estimate_component_resources(block.resource_model(select, prepare, prepare))
    source_cost = get_cost_value(source, QECGatesCost())
    assert asdict(cost.gates) == {
        key: getattr(source_cost, key) for key in asdict(cost.gates)
    }
    assert cost.peak_logical_qubits == get_cost_value(source, QubitCount())
