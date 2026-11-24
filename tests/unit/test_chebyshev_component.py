"""Compare complete original Chebyshev examples through owned typed children."""

from dataclasses import replace
from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
    zero_state_reflection_program,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_unitary_block_factories import (
    unitary_block_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.preparation.chebyshev_component import (
    chebyshev_component_pair,
)
from flagquantum.compiler.registers.primitives.identity import register_identity_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._default_linear_combination_oracles import (
    default_linear_combination_prepare_pair,
    default_linear_combination_select_pair,
)
from flagquantum.ecosystem._linear_combination_component_library import (
    capture_linear_combination_context,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_chebyshev_external_file import (
    frozen_chebyshev_sources as frozen_chebyshev_sources,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_phased_block_factories import scalar_terminals
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_all_small_raw_histories(
    circuit: CircuitIR, logical: int, expected: torch.Tensor
) -> None:
    """Enumerate the small fixture's raw outcomes in addition to its certificate."""
    measurements = sum(g.name == "measure" for g in circuit.instructions)
    assert measurements <= 6
    stride = 1 << (circuit.n_wires - logical)
    amplitudes = torch.arange(1, (1 << logical) + 1, dtype=torch.float64).to(
        torch.complex128
    )
    amplitudes[1::2] *= 1j
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = amplitudes
    target = torch.zeros_like(initial)
    target[::stride] = expected @ amplitudes
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, circuit.instructions, history) * sqrt(
            2**measurements
        )
        torch.testing.assert_close(actual, target, atol=2e-10, rtol=0)


def child_pair(source: Any) -> tuple[ProgramRecord, ProgramRecord]:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination

    if isinstance(source, LinearCombination):
        context = capture_linear_combination_context(
            source,
            prepare_owner=default_linear_combination_prepare_pair,
            select_owner=default_linear_combination_select_pair,
        )
        return context.program(), context.program(adjoint=True)
    factories = unitary_block_factories()
    pair = tuple(
        resolve_component_program(_library(child, 0), 0, factories)
        for child in (source, Adjoint(source))
    )
    return pair[0], pair[1]


def chebyshev_terminals(record: ProgramRecord) -> set[str]:
    assert not verify_program_record(record)
    functions = {function.name: function for function in record.functions}
    terminals = atomic_leaves(record) | scalar_terminals(record)
    identity = register_identity_program(1)
    for function in record.functions[1:]:
        # A literal definition match includes the entire register declaration.
        # Reject impossible candidates before repeatedly verifying a large DAG.
        if function.registers != identity.functions[0].registers:
            continue
        # Whole-module validity was checked above. Literal matching ignores
        # unreachable definitions, so verify just this complete reachable
        # closure rather than the large unrelated graph for every candidate.
        reachable = {function.name: function}
        pending = [function.name]
        for name in pending:
            for block in functions[name].blocks:
                for operation in block.operations:
                    if (
                        operation.callee is not None
                        and operation.callee not in reachable
                    ):
                        reachable[operation.callee] = functions[operation.callee]
                        pending.append(operation.callee)
        candidate = replace(record, functions=tuple(reachable.values()))
        if matches_component_definition(candidate, function.name, identity):
            terminals.add(function.name)
    return terminals


@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("scaled", (False, True))
def test_external_capture_retains_small_exact_operator(
    controlled: bool, inverse: bool, scaled: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        ChebyshevPolynomial,
        ScaledChebyshevPolynomial,
    )
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum.ecosystem._chebyshev_component_library import (
        capture_chebyshev_pair,
    )

    constructor = ScaledChebyshevPolynomial if scaled else ChebyshevPolynomial
    source = constructor(Unitary(Hadamard(), alpha=3.14), order=1)
    record = capture_chebyshev_pair(source, controlled=controlled)[int(inverse)]
    if controlled:
        source = source.controlled()
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("odd", (False, True))
def test_original_scaled_property_captures_complete_native_dependencies(
    odd: bool,
) -> None:
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _scaled_chebyshev_poly_even,
        _scaled_chebyshev_poly_odd,
    )

    from flagquantum.ecosystem._chebyshev_component_library import (
        capture_chebyshev_pair,
    )

    source = (_scaled_chebyshev_poly_odd if odd else _scaled_chebyshev_poly_even).make()
    pair = capture_chebyshev_pair(source)
    for record in pair:
        assert not verify_program_record(record)
    # Scaled roots keep the single actual linear_combination property call;
    # descendants are translated immediately rather than retained as SDK bodies.
    compare_layers(pair[0], source, chebyshev_terminals(pair[0]))
    _assert_scaled_sparse_execution(source, pair, typed_source=not odd)


def _assert_scaled_sparse_execution(
    source: Any,
    pair: tuple[ProgramRecord, ProgramRecord],
    *,
    typed_source: bool = False,
) -> None:
    """Compare original large examples without a dense ancilla matrix.

    The even example uses the original typed flatten/export route. Direct
    Cirq recursion instead controls a non-THRU comparator and fails upstream;
    the typed visitor retains the actual LCU control specialization.
    """
    cirq = pytest.importorskip("cirq")
    from qualtran._infra.gate_with_registers import get_named_qubits

    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
    from tests.unit.test_sparse_chemistry_prepare import _branch

    forward, logical = bind(pair[0])
    backward, _ = bind(pair[1])
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    source_circuit = (
        source.decompose_bloq().flatten().to_cirq_circuit(cirq_quregs=named)
        if typed_source
        else source.on_registers(**named)
    )
    operations = list(
        cirq.decompose(
            source_circuit,
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    initial = {0: 0.6 + 0j, 1 << (logical - 1): 0.8j}
    expected = _source_sparse_branch(operations, visible, initial, 0)
    for pattern in (0, 1):
        work = forward.circuit.n_wires - logical
        events = certify_erasure_histories(
            forward.circuit.instructions,
            n_wires=forward.circuit.n_wires,
            zero_wires=tuple(range(logical, forward.circuit.n_wires)),
            check_clean_output=False,
        )
        actual = _branch(
            forward.circuit.instructions,
            forward.circuit.n_wires,
            (pattern,) * events,
            {k << work: v for k, v in initial.items()},
            normalize_measurements=True,
        )
        assert all(key % (1 << work) == 0 for key in actual)
        decoded = {k >> work: v for k, v in actual.items()}
        assert all(
            abs(decoded.get(k, 0) - expected.get(k, 0)) < 3e-10
            for k in decoded.keys() | expected.keys()
        )
        work = backward.circuit.n_wires - logical
        events = certify_erasure_histories(
            backward.circuit.instructions,
            n_wires=backward.circuit.n_wires,
            zero_wires=tuple(range(logical, backward.circuit.n_wires)),
            check_clean_output=False,
        )
        restored = _branch(
            backward.circuit.instructions,
            backward.circuit.n_wires,
            (pattern,) * events,
            {k << work: v for k, v in decoded.items()},
            normalize_measurements=True,
        )
        target = {k << work: v for k, v in initial.items()}
        assert all(
            abs(restored.get(k, 0) - target.get(k, 0)) < 4e-10
            for k in restored.keys() | target.keys()
        )


def test_scaled_positive_lcu_property_retains_public_bit_and_sparse_amplitudes() -> (
    None
):
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        ScaledChebyshevPolynomial,
    )
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.ecosystem._chebyshev_component_library import (
        capture_chebyshev_pair,
    )

    original = ScaledChebyshevPolynomial(
        Unitary(Hadamard(), alpha=1.25), 2, lambd_bits=1
    )
    source = original.controlled()
    pair = capture_chebyshev_pair(original, controlled=True)
    ports = pair[0].functions[0].registers
    assert ports is not None and ports[0].dtype == quantum_bit_type()
    for inverse, record in enumerate(pair):
        assert not verify_program_record(record)
        compare_layers(
            record, Adjoint(source) if inverse else source, chebyshev_terminals(record)
        )
    _assert_scaled_sparse_execution(source, pair, typed_source=True)


@pytest.mark.parametrize("odd", (False, True))
@pytest.mark.parametrize("order", (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_child_calls_complex_operator_and_work(
    odd: bool, order: int, inverse: bool
) -> None:
    from attrs import evolve
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _chebyshev_poly_even,
        _chebyshev_poly_odd,
    )

    source = evolve(
        (_chebyshev_poly_odd if odd else _chebyshev_poly_even).make(), order=order
    )
    pair = child_pair(source.block_encoding)
    width = int(source.ancilla_bitsize)
    reflection = (
        zero_state_reflection_program(
            PreparedStateReflection.around_zero((width,), global_phase=-1)
        )
        if width
        else None
    )
    record = chebyshev_component_pair(*pair, order, reflection=reflection)[int(inverse)]
    if inverse:
        source = Adjoint(source)
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-10,
        rtol=0,
    )


@pytest.mark.parametrize("order", (0, 2, 3))
@pytest.mark.parametrize("inverse", (False, True))
def test_zero_order_and_literal_nonhermitian_child_directions(
    order: int, inverse: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import ChebyshevPolynomial
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = ChebyshevPolynomial(
        Unitary(TGate(), ancilla_bitsize=1 if order == 0 else 0), order
    )
    record = chebyshev_component_pair(*child_pair(source.block_encoding), order)[
        int(inverse)
    ]
    if inverse:
        source = Adjoint(source)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("order", (0, 1, 2, 5))
@pytest.mark.parametrize("inverse", (False, True))
def test_positive_source_control_threads_child_and_reflection_calls(
    order: int, inverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import ChebyshevPolynomial

    from tests.unit.test_linear_combination_component import native_oracle_fixture

    child, prep, select = native_oracle_fixture(True)
    positive = capture_linear_combination_context(
        child, prepare_owner=lambda _: prep, select_owner=lambda _: select
    )
    # The source root being controlled has the same original un-controlled
    # block, while the exact supplied child already owns its specialization.
    from attrs import evolve

    original = evolve(child, is_controlled=False)
    source = ChebyshevPolynomial(original, order).controlled(CtrlSpec())
    width = int(child.ancilla_bitsize)
    reflection = (
        zero_state_reflection_program(
            PreparedStateReflection.around_zero((width,), 1, -1)
        )
        if order
        else None
    )
    pair = chebyshev_component_pair(
        positive.program(),
        positive.program(adjoint=True),
        order,
        reflection=reflection,
        controlled=True,
    )
    record = pair[int(inverse)]
    if inverse:
        source = Adjoint(source)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )
