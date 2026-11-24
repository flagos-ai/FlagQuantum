"""Alias constructors preserve source metadata, five calls and exact inverses."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_program,
)
from flagquantum.compiler.formats.legacy_alias_preparation_factories import (
    alias_preparation_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_alias_stored_graph import native_alias
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def pinned_alias() -> None:
    import kickmix
    from qualtran.bloqs.state_preparation import state_preparation_alias_sampling

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        (
            "state_preparation_alias_sampling.py",
            "4850ef95d9ebf58b0e062505a3482db2f5307443",
        ),
        (
            "state_preparation_alias_sampling_test.py",
            "c5141c0e4beadb9a890dc3fc035601252a6c0454",
        ),
    ):
        raw = (
            Path(state_preparation_alias_sampling.__file__).parent / name
        ).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def _source(name: str) -> Any:
    from qualtran.bloqs.state_preparation import state_preparation_alias_sampling

    return getattr(state_preparation_alias_sampling, name).make()


@pytest.mark.parametrize(
    "example",
    (
        "_state_prep_alias",
        "_sparse_state_prep_alias",
        "_sparse_state_prep_alias_from_list",
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_all_concrete_original_examples_retain_calls_operands_and_resources(
    example: str, inverse: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    forward = _source(example)
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, alias_preparation_factories()
    )
    expected = alias_preparation_program(native_alias(forward), adjoint=inverse)
    assert verify_program_record(record) == ()
    assert matches_component_definition(record, "main", expected)
    deeper = resolve_component_program(
        _library(source, 1), 0, alias_preparation_factories()
    )
    assert verify_program_record(deeper) == ()
    assert matches_component_definition(deeper, "main", record)
    compare_layers(record, source, atomic_leaves(record))
    root = record.functions[0]
    assert root.registers is not None
    cursor = 0
    inputs = []
    for reg in root.registers:
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 15))
    )
    direct = bind_register_program(
        expected, tuple(inputs), clean=tuple(range(cursor, cursor + 15))
    )
    assert bound.circuit == direct.circuit
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] == model.t + 4 * model.and_bloq + 7 * model.cswap
    assert bound.outputs == tuple(inputs)


@pytest.mark.parametrize(
    "sparse,storage",
    (
        (False, "bounded"),
        (False, "named"),
        (True, "bounded"),
        (True, "uint"),
        (True, "opaque"),
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_small_constructor_full_complex_operator_all_histories_and_private_zero(
    sparse: bool, storage: str, inverse: bool
) -> None:
    from qualtran import BQUInt, QAny, QUInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )

    if sparse:
        forward = SparseStatePreparationAliasSampling(
            SourceRegister(
                "selection",
                {"bounded": BQUInt(2, 4), "uint": QUInt(2), "opaque": QAny(2)}[storage],
            ),
            np.array([0, 3]),
            np.array([3, 0]),
            np.array([1, 0]),
            1,
            2.0,
        )
    else:
        forward = StatePreparationAliasSampling(
            SourceRegister(
                "index_word" if storage == "named" else "selection", BQUInt(1, 2)
            ),
            np.array([1, 0]),
            np.array([1, 0]),
            1,
            2.0,
        )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, alias_preparation_factories()
    )
    cursor = 0
    inputs = []
    for reg in record.functions[0].registers or ():
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 8))
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor), expected, atol=2e-11, rtol=0
    )
    # Logical junk remains returned; only each child's private work is erased.
    assert bound.outputs == tuple(inputs)
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


@pytest.mark.parametrize(
    "example", ("_state_prep_alias_symb", "_sparse_state_prep_alias_symb")
)
@pytest.mark.parametrize("inverse", (False, True))
def test_original_symbolic_shape_codec_refusal_is_not_repaired(
    example: str, inverse: bool
) -> None:
    source = _source(example)
    if inverse:
        source = source.adjoint()
    with pytest.raises(ValueError, match="Cannot serialize Shaped"):
        _library(source, 0)


def test_original_alias_reflection_compiles_automatic_inverse_without_source_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.reflections.reflection_using_prepare import _refl_using_prep

    source = _refl_using_prep.make()
    libraries = (_library(source, 0), _library(source, 1))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        records = tuple(
            resolve_component_program(library, 0, alias_preparation_factories())
            for library in libraries
        )
    assert verify_program_record(records[0]) == verify_program_record(records[1]) == ()
    assert matches_component_definition(records[0], "main", records[1])
    compare_layers(records[0], source, atomic_leaves(records[0]))


@pytest.mark.parametrize("arguments", ({}, {"selection_registers": 3}, {"n": 4}))
def test_invalid_alias_metadata_refuses(arguments: dict[str, Any]) -> None:
    key = "qualtran.bloqs.state_preparation.state_preparation_alias_sampling.StatePreparationAliasSampling"
    with pytest.raises(SerializationError):
        alias_preparation_factories()[key]("main", None, arguments)


@pytest.mark.parametrize("sparse", (False, True))
def test_original_domain_and_hardcoded_name_failures_are_not_repaired(
    sparse: bool,
) -> None:
    import attrs
    from qualtran import QAny
    from qualtran import Register as SourceRegister

    original = _source("_sparse_state_prep_alias" if sparse else "_state_prep_alias")
    selection = original.selection_registers[0]
    source = attrs.evolve(
        original,
        selection_registers=(
            SourceRegister(
                "renamed" if sparse else "selection",
                selection.dtype if sparse else QAny(2),
            ),
        ),
    )
    if sparse:
        with pytest.raises(KeyError, match="selection"):
            source.decompose_bloq()
    else:
        with pytest.raises(ValueError, match="math domain"):
            source.decompose_bloq()
    library = _library(source, 0)
    with pytest.raises(
        SerializationError,
        match="source selection name" if sparse else "iteration bound",
    ):
        resolve_component_program(library, 0, alias_preparation_factories())
