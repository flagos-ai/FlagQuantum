"""Uniform constructors retain complete source calls and exact measured inverses."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_uniform_preparation_factories import (
    uniform_preparation_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_uniform() -> None:
    import kickmix
    from qualtran.bloqs.state_preparation import prepare_uniform_superposition

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(prepare_uniform_superposition.__file__).parent
    for name, digest in (
        (
            "prepare_uniform_superposition.py",
            "e567c913bff5777e899f38afae09e4c861c98d0a",
        ),
        (
            "prepare_uniform_superposition_test.py",
            "25f19b8539880ef9f0ce3ebe954cd885562780d4",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize(
    "count,values",
    (
        (2, ()),
        (3, ()),
        (4, (0,)),
        (5, ()),
        (5, (1,)),
        (3, (0, 1)),
        (8, (1, 0)),
        (12, (1,)),
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_constructors_retain_source_calls_complex_channel_and_resource_categories(
    count: int, values: tuple[int, ...], inverse: bool
) -> None:
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    forward = PrepareUniformSuperposition(count, values)
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, uniform_preparation_factories()
    )
    expected = _rename_program(
        uniform_preparation_program(count, values, adjoint=inverse), "main"
    )
    assert verify_program_record(record) == ()
    # Serialized inverse roots also retain an unused forward recipe. Match all
    # reachable operations literally, allowing only private function names.
    assert matches_component_definition(record, "main", expected)
    deeper = resolve_component_program(
        _library(source, 1), 0, uniform_preparation_factories()
    )
    assert verify_program_record(deeper) == ()
    assert matches_component_definition(deeper, "main", record)
    compare_layers(record, source, atomic_leaves(record))
    width = (count - 1).bit_length()
    logical = len(values) + width
    inputs = (
        *((tuple(range(len(values))),) if values else ()),
        tuple(range(len(values), logical)),
    )
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + 12))
    )
    direct = bind_register_program(
        expected, inputs, clean=tuple(range(logical, logical + 12))
    )
    assert bound == direct
    model = get_cost_value(source, QECGatesCost())
    actual = Counter(op.name for op in bound.circuit.instructions)
    assert actual["measure"] == model.measurement
    # total_t_count() additionally estimates synthesis of unsynthesized rotations.
    assert actual["t"] + actual["tdg"] == model.t + 4 * model.and_bloq
    assert_source_model(
        source_uniform_preparation_cost(count, control_values=values), forward
    )
    if count <= 5:
        tensor = torch.as_tensor(source.tensor_contract())
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical), tensor, atol=2e-11, rtol=0
        )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


@pytest.mark.parametrize("count", (1, sympy.Symbol("n", positive=True, integer=True)))
@pytest.mark.parametrize("inverse", (False, True))
def test_symbolic_and_original_singleton_keep_declaration(
    count: Any, inverse: bool
) -> None:
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    source = PrepareUniformSuperposition(count)
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, uniform_preparation_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(record, ())


def test_source_free_constructor_and_explicit_unknown_inverse_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    source = PrepareUniformSuperposition(5, (0, 1))
    library = _library(source.adjoint(), 0)
    expected = resolve_component_program(library, 0, uniform_preparation_factories())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        assert (
            resolve_component_program(library, 0, uniform_preparation_factories())
            == expected
        )
    key = "qualtran._infra.adjoint.Adjoint"
    with pytest.raises(SerializationError, match="context"):
        uniform_preparation_factories()[key](
            "main", None, {"subbloq": expected.functions[0]}
        )


@pytest.mark.parametrize(
    "arguments", ({}, {"n": True}, {"n": 3, "extra": 0}, {"n": 3, "cvs": ((0, 1),)})
)
def test_malformed_uniform_metadata_refuses(arguments: dict[str, Any]) -> None:
    factory = next(iter(uniform_preparation_factories().values()))
    with pytest.raises(SerializationError):
        factory("main", None, arguments)
