"""Bind original alias declarations without erasing their logical junk outputs."""

from collections import Counter
from dataclasses import replace
from math import sqrt
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.alias_preparation_catalog import AliasPreparationExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.alias_preparation import alias_black_box_prepare
from flagquantum.algorithms.primitives.alias_sampling import AliasSamplingTable
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_alias_preparation_file import source_module as source_module

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolved_alias(source_module: Any, sparse: bool, name: str) -> Any:
    from attrs import evolve
    from qualtran import BQUInt, Register

    declaration = (
        source_module._sparse_state_prep_alias_symb.make()
        if sparse
        else source_module._state_prep_alias_symb.make()
    )
    fields = dict(
        selection_registers=(Register(name, BQUInt(2, 4)),),
        alt=np.array([3, 0] if sparse else [1, 0, 3, 2]),
        keep=np.array([1, 0] if sparse else [1, 0, 1, 0]),
        mu=1,
        sum_of_unnormalized_probabilities=2,
    )
    if sparse:
        fields["index"] = np.array([0, 3])
    return evolve(declaration, **fields)


@pytest.mark.parametrize(
    "sparse,kind",
    [(False, "bounded"), (True, "bounded"), (True, "unsigned"), (True, "opaque")],
)
def test_resolved_alias_complete_logical_junk_operators(
    source_module: Any, sparse: bool, kind: str
) -> None:
    source = resolved_alias(
        source_module, sparse, "selection" if sparse else "index_word"
    )
    table = AliasSamplingTable(
        4,
        tuple(map(int, source.alt)),
        tuple(map(int, source.keep)),
        1,
        2,
        tuple(map(int, source.index)) if sparse else None,
    )
    if kind != "bounded":
        from attrs import evolve
        from qualtran import QAny, QUInt, Register

        source = evolve(
            source,
            selection_registers=(
                Register("selection", QUInt(2) if kind == "unsigned" else QAny(2)),
            ),
        )
        assert source.n_coeff == 0
    else:
        from tests.unit.test_first_quantization_resource_model import (
            assert_source_model,
        )

        assert_source_model(AliasPreparationExample(table).resource_model(), source)
    native = alias_black_box_prepare(table)
    logical = native.selection_width + native.junk_width
    original = import_register_emitter(source)
    emitter = replace(
        original,
        snapshots=(
            CircuitIR.from_json(original.snapshots[0].to_json()),
            CircuitIR.from_json(original.snapshots[1].to_json()),
        ),
    )
    packed = BlackBoxPrepare(
        ((source.selection_registers[0].name, 2),),
        tuple((r.name, int(r.bitsize)) for r in source.junk_registers),
        emitter,
        2,
    )
    work, slots = emitter.requirements()
    source_irs = []
    native_irs = []
    for adjoint in (False, True):
        captured = (
            Circuit(logical + work)
            .append(
                packed.make_on(
                    selection=(0, 1),
                    junk=tuple(range(2, logical)),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        example = AliasPreparationExample(table, adjoint)
        actual = Circuit(example.qubit_count).append(example).to_ir()
        assert isinstance(captured, CircuitIR) and isinstance(actual, CircuitIR)
        assert Counter(g.name for g in captured.instructions) == Counter(
            g.name for g in actual.instructions
        )
        source_irs.append(captured)
        native_irs.append(actual)
    channels = (*source_irs, *native_irs)
    events = tuple(
        certify_erasure_histories(
            ir.instructions,
            n_wires=ir.n_wires,
            zero_wires=tuple(range(logical, ir.n_wires)),
            check_clean_output=False,
        )
        for ir in channels
    )
    operators = []
    for ir, count in zip(channels, events, strict=True):
        stride = 1 << (ir.n_wires - logical)
        columns = []
        for column in range(1 << logical):
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            result = forced_branch(initial, ir.instructions, (0,) * count) * sqrt(
                2**count
            )
            clean = torch.zeros_like(result)
            clean[::stride] = result[::stride]
            torch.testing.assert_close(result, clean, atol=8e-12, rtol=0)
            columns.append(result[::stride])
        operators.append(torch.stack(columns, dim=1))
    forward, backward, native_forward, native_backward = operators
    torch.testing.assert_close(forward, native_forward, atol=8e-12, rtol=0)
    torch.testing.assert_close(backward, native_backward, atol=8e-12, rtol=0)
    torch.testing.assert_close(
        backward @ forward,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    # Junk is part of the logical output, often entangled with selection. Only
    # private scratch must be zero at the PREPARE boundary.
    prepared = forward[:, 0].reshape(4, -1)
    probabilities = prepared.abs().square().sum(dim=1)
    expected_probabilities = (
        (0.75, 0, 0, 0.25) if sparse else (0.375, 0.125, 0.375, 0.125)
    )
    torch.testing.assert_close(
        probabilities,
        torch.tensor(expected_probabilities, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    reduced = prepared @ prepared.mH
    assert torch.trace(reduced @ reduced).real.item() < 1 - 1e-12
    for ir in source_irs:
        counts = Counter(g.name for g in ir.instructions)
        assert counts["measure"] == (1 if sparse else 3)
        assert counts["t"] + counts["tdg"] == (18 if sparse else 26)
        assert counts["cx"] == (36 if sparse else 51)
        assert len(ir.instructions) == (76 if sparse else 118)


@pytest.mark.parametrize("kind", ["unsigned", "opaque"])
def test_generic_register_source_admission_is_preserved(
    source_module: Any, kind: str
) -> None:
    from attrs import evolve
    from qualtran import QAny, QUInt, Register

    source = resolved_alias(source_module, False, "selection")
    source = evolve(
        source,
        selection_registers=(
            Register("selection", QUInt(2) if kind == "unsigned" else QAny(2)),
        ),
    )
    assert source.n_coeff == 0
    with pytest.raises(ValueError, match="math domain error"):
        import_register_emitter(source)


def test_sparse_custom_name_is_an_original_body_failure(source_module: Any) -> None:
    source = resolved_alias(source_module, True, "index_word")
    with pytest.raises(KeyError, match="selection"):
        source.decompose_bloq()
    with pytest.raises(KeyError, match="selection"):
        import_register_emitter(source)
