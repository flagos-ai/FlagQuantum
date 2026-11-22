"""Retain the complete original alias factories and their executable children."""

from typing import Any

import pytest

from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_program,
)
from flagquantum.algorithms.primitives.alias_sampling import AliasSamplingTable
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_alias(source: Any) -> AliasSamplingTable | AliasPreparationShape:
    from qualtran.symbolics import is_symbolic

    sparse = hasattr(source, "index")
    if is_symbolic(source.mu):
        return AliasPreparationShape(
            source.n_coeff,
            source.l1_norm_of_coeffs,
            source.mu,
            source.n_nonzero_coeff if sparse else None,
        )
    return AliasSamplingTable(
        source.n_coeff,
        tuple(int(v) for v in source.alt),
        tuple(int(v) for v in source.keep),
        int(source.mu),
        float(source.l1_norm_of_coeffs),
        tuple(int(v) for v in source.index) if sparse else None,
    )


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("adjoint", [False, True])
def test_all_original_factories_keep_five_calls_and_recursive_dependencies(
    index: int, adjoint: bool
) -> None:
    from qualtran import BloqDocSpec
    from qualtran.bloqs.state_preparation import (
        state_preparation_alias_sampling as source_module,
    )

    sources = [
        example.make()
        for spec in vars(source_module).values()
        if isinstance(spec, BloqDocSpec)
        for example in spec.examples
    ]
    source = sources[index]
    record = alias_preparation_program(native_alias(source), adjoint=adjoint)
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 5
    compare_layers(
        record, source.adjoint() if adjoint else source, atomic_leaves(record)
    )


def ising_source() -> Any:
    from qualtran.bloqs.phase_estimation.qubitization_qpe import _qubitization_qpe_ising
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )

    pending = [_qubitization_qpe_ising.make()]
    seen = set()
    while pending:
        source = pending.pop(0)
        if source in seen:
            continue
        seen.add(source)
        if isinstance(source, StatePreparationAliasSampling):
            return source
        try:
            pending.extend(
                i.bloq for i, _, _ in source.decompose_bloq().iter_bloqsoqs()
            )
        except (NotImplementedError, ValueError, TypeError):
            pass
    raise AssertionError("Frozen Ising factory must retain its alias preparation")


@pytest.mark.parametrize("kind", ["dense", "sparse", "ising"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_factory_and_ising_exact_primitive_operands_feedback_and_separate_models(
    kind: str, adjoint: bool
) -> None:
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        _sparse_state_prep_alias,
        _state_prep_alias,
    )

    from flagquantum.algorithms.primitives.alias_preparation import (
        alias_preparation_requirements,
        source_dense_alias_preparation_cost,
        source_sparse_alias_preparation_cost,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_first_quantization_resource_model import assert_source_model
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = {
        "dense": _state_prep_alias.make,
        "sparse": _sparse_state_prep_alias.make,
        "ising": ising_source,
    }[kind]()
    table = native_alias(source)
    assert isinstance(table, AliasSamplingTable)
    graph = alias_preparation_program(table, adjoint=adjoint)
    expected = source.adjoint() if adjoint else source
    compare_layers(graph, expected, atomic_leaves(graph))
    cursor = 0
    ports = {}
    for reg in AliasPreparationShape.from_table(table).signature:
        width = int(reg.bitsize)
        ports[reg.name] = tuple(range(cursor, cursor + width))
        cursor += width
    work, _ = alias_preparation_requirements(table)
    clean = tuple(range(cursor, cursor + work))
    bound = bind_register_program(graph, tuple(ports.values()), clean=clean)
    # The Ising inverse exports the generic reversed graph. Bind its allocation
    # lifetimes explicitly, since the SDK pool recycles independent frees in a
    # different order. Other contexts use the original SDK pool directly.
    allocations = clean
    reuse = True
    if kind == "ising" and adjoint:
        allocations = tuple(
            wire
            for event in bound.events
            if event.kind == "allocate"
            for wire in event.wires
        )
        reuse = False
    gates = reference_arithmetic_gates(
        expected.decompose_bloq(),
        ports,
        allocations,
        MeasurementSlots(bound.circuit.instructions),
        keep_pauli_leaves=True,
        preserve_components=True,
        reuse_allocations=reuse,
    )
    assert_source(bound.circuit.instructions, gates)
    model = (
        source_sparse_alias_preparation_cost(table)
        if table.indices is not None
        else source_dense_alias_preparation_cost(table)
    )
    assert_source_model(model, source)


@pytest.mark.parametrize("rows,sparse", [(2, False), (3, False), (2, True)])
def test_complete_small_operators_every_raw_history_inverse_and_append(
    rows: int,
    sparse: bool,
) -> None:
    from itertools import product

    import numpy as np
    import torch
    from qualtran import BQUInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.algorithms.primitives.alias_preparation import (
        alias_preparation_requirements,
    )
    from flagquantum.compiler.registers.component_graph import copy_component_program
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_multiplexed_source_binding import logical_operator

    width = (rows - 1).bit_length()
    source = StatePreparationAliasSampling(
        SourceRegister("selection", BQUInt(width, rows)),
        np.asarray(tuple(range(rows))),
        np.asarray(tuple(i % 2 for i in range(rows))),
        1,
        float(rows),
    )
    if sparse:
        source = SparseStatePreparationAliasSampling(
            SourceRegister("selection", BQUInt(2, 4)),
            np.asarray([0, 3]),
            np.asarray([3, 0]),
            np.asarray([1, 0]),
            1,
            2.0,
        )
    table = native_alias(source)
    assert isinstance(table, AliasSamplingTable)
    cursor = 0
    inputs = []
    for reg in AliasPreparationShape.from_table(table).signature:
        inputs.append(tuple(range(cursor, cursor + int(reg.bitsize))))
        cursor += int(reg.bitsize)
    work, _ = alias_preparation_requirements(table)
    clean = tuple(range(cursor, cursor + work))
    graph = alias_preparation_program(table)
    forward = bind_register_program(graph, tuple(inputs), clean=clean).circuit
    backward = bind_register_program(
        alias_preparation_program(table, adjoint=True), tuple(inputs), clean=clean
    ).circuit
    assert (
        bind_register_program(
            copy_component_program(graph), tuple(inputs), clean=clean
        ).circuit
        == forward
    )
    assert (
        Circuit(forward.n_wires)
        .append(_BoundBlockOperation(forward.instructions, forward.n_wires))
        .to_ir()
        .instructions
        == forward.instructions
    )
    assert forward.from_json(forward.to_json()) == forward
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(forward, cursor), expected, atol=2e-11, rtol=0
    )
    torch.testing.assert_close(
        logical_operator(backward, cursor), expected.mH, atol=2e-11, rtol=0
    )
    stride = 1 << work
    state = torch.zeros(1 << forward.n_wires, dtype=torch.complex128)
    state[::stride] = torch.tensor([complex(i + 1, 3 - i) for i in range(1 << cursor)])
    state /= state.norm()
    desired = torch.zeros_like(state)
    desired[::stride] = expected @ state[::stride]
    for stream, initial, output in [
        (forward.instructions, state, desired),
        (backward.instructions, desired, state),
    ]:
        events = sum(g.name == "measure" for g in stream)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(initial, stream, history),
                output / 2 ** (events / 2),
                atol=2e-11,
                rtol=0,
            )
