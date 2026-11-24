"""Frozen reflection constructors preserve typed dependencies and exact phases."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.identity_prepare import (
    IdentityPrepare,
    identity_prepare_program,
)
from flagquantum.algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
    zero_state_reflection_program,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_reflection_factories import (
    reflection_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("phase", (1, -1, 1j))
def test_registerless_preparation_keeps_original_moment_order_and_complex_scalar(
    control: int | None, phase: complex
) -> None:
    from cmath import phase as angle

    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    from flagquantum.compiler.registers.primitives.global_phase_component import (
        scalar_phase_program,
    )
    from tests.unit.test_default_linear_combination_oracles import bind

    source = ReflectionUsingPrepare.reflection_around_zero(
        (), control_val=control, global_phase=phase
    )
    record = resolve_component_program(_library(source, 0), 0, reflection_factories())
    assert not verify_program_record(record)
    terminals = atomic_leaves(record)
    if control is None and phase != 1:
        scalar = scalar_phase_program(angle(phase))
        terminals.update(
            function.name
            for function in record.functions[1:]
            if function.registers == ()
            and matches_component_definition(record, function.name, scalar)
        )
    compare_layers(record, source, terminals)
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            1 << logical, 1 << logical
        ),
        atol=3e-11,
        rtol=0,
    )


@pytest.fixture(scope="module", autouse=True)
def pinned_reflection() -> None:
    import kickmix
    from qualtran.bloqs.reflections import prepare_identity

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("prepare_identity.py", "cbd8eb35d88d5c02b2428e988e44de6615b7db24"),
        ("prepare_identity_test.py", "ee6cac40ed0eafc51f69594d94c93172ef70e8f8"),
        ("reflection_using_prepare.py", "4a8467fde126bd596be69286d88c198726a4557a"),
        (
            "reflection_using_prepare_test.py",
            "df403c18060daf172e68c4a80099c306743a7a3a",
        ),
    ):
        raw = (Path(prepare_identity.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("widths", ((), (1,), (2, 1), (10, 4, 1)))
def test_identity_constructor_reuses_shared_owner_and_source_calls(
    widths: tuple[int, ...],
) -> None:
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    source = PrepareIdentity.from_bitsizes(widths)
    record = resolve_component_program(_library(source, 0), 0, reflection_factories())
    expected = _rename_program(
        identity_prepare_program(IdentityPrepare.from_bitsizes(widths)), "main"
    )
    assert verify_program_record(record) == ()
    assert matches_component_definition(record, "main", expected)
    compare_layers(
        record,
        source,
        atomic_leaves(record)
        | {
            f.name
            for f in record.functions
            if f.name.endswith("main") and f.name != "main"
        },
    )


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("phase", (1, -1, 1j))
def test_zero_reflection_constructor_source_phases_and_small_complete_operators(
    control: int | None, phase: complex
) -> None:
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    source = ReflectionUsingPrepare.reflection_around_zero(
        (1, 2), control_val=control, global_phase=phase
    )
    record = resolve_component_program(_library(source, 0), 0, reflection_factories())
    expected = zero_state_reflection_program(
        PreparedStateReflection.around_zero((1, 2), control, phase)
    )
    # Constructor forward PREPARE references one external original definition;
    # the shared direct owner may namespace the same definition differently.
    assert verify_program_record(record) == ()
    inputs = (
        *(((0,),) if control is not None else ()),
        (int(control is not None),),
        tuple(range(1 + int(control is not None), 3 + int(control is not None))),
    )
    logical = 3 + int(control is not None)
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + 5))
    )
    direct = bind_register_program(
        expected, inputs, clean=tuple(range(logical, logical + 5))
    )
    assert bound.circuit == direct.circuit
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract()),
        atol=2e-11,
        rtol=0,
    )
    assert source.adjoint() is source


def test_explicit_scalar_dependency_retains_zero_port_and_control_ownership() -> None:
    from math import pi

    from flagquantum.compiler.registers.primitives.global_phase_component import (
        scalar_phase_program,
    )

    prepare = identity_prepare_program(IdentityPrepare.from_bitsizes((1,)))
    names = tuple(reg.name for reg in prepare.functions[0].registers or ())
    with pytest.raises(ValueError, match="explicit scalar owner"):
        prepared_reflection_program(prepare, prepare, names, global_phase=-1)
    scalar = scalar_phase_program(pi)
    for control, coefficient in ((1, -1), (None, 1)):
        with pytest.raises(ValueError, match="scalar owner"):
            prepared_reflection_program(
                prepare,
                prepare,
                names,
                control_value=control,
                global_phase=coefficient,
                scalar_phase=scalar,
            )
    with pytest.raises(ValueError, match="zero ports"):
        prepared_reflection_program(
            prepare, prepare, names, global_phase=-1, scalar_phase=prepare
        )


def test_general_array_junk_exact_inverse_preserves_full_complex_operator() -> None:
    bit = quantum_bit_type()
    signature = Signature((Register("selection", bit), Register("junk", bit, (2,))))
    h = bit_gate_function(
        "h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    cx = bit_gate_function(
        "cx",
        (Register("a", bit), Register("b", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    records = []
    for inverse in (False, True):
        builder = ComponentProgramBuilder(signature, (h, cx))
        selected = builder.inputs["selection"]
        junk = builder.inputs["junk"]
        assert isinstance(junk, np.ndarray)
        for stage in (("h", "cx") if inverse else ("cx", "h")):
            if stage == "h":
                selected = builder.call_dict("h", {"q": selected})["q"]
            else:
                result = builder.call_dict("cx", {"a": selected, "b": junk[0]})
                selected, junk[0] = result["a"], result["b"]
        records.append(builder.finalize({"selection": selected, "junk": junk}))
    record = prepared_reflection_program(records[0], records[1], ("selection",))
    bound = bind_register_program(record, ((0,),), clean=tuple(range(1, 8)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 1),
        -torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
    assert not live


def test_original_alias_example_uses_explicit_inverse_and_junk_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.reflections.reflection_using_prepare import _refl_using_prep

    from flagquantum.algorithms.primitives.alias_preparation import (
        alias_preparation_program,
    )
    from tests.unit.test_alias_stored_graph import native_alias

    source = _refl_using_prep.make()
    preparation = alias_preparation_program(native_alias(source.prepare_gate))
    inverse = alias_preparation_program(native_alias(source.prepare_gate), adjoint=True)
    source_key = "qualtran.bloqs.state_preparation.state_preparation_alias_sampling.StatePreparationAliasSampling"
    recipes = {}

    def prepare(name: str, registers: Any, arguments: Any) -> ProgramRecord:
        assert set(arguments) == {
            "alt",
            "keep",
            "mu",
            "selection_registers",
            "sum_of_unnormalized_probabilities",
        }
        assert int(arguments["mu"]) == source.prepare_gate.mu
        assert np.array_equal(arguments["alt"], source.prepare_gate.alt)
        assert np.array_equal(arguments["keep"], source.prepare_gate.keep)
        record = _rename_program(preparation, name)
        assert record.functions[0].registers == registers
        recipes[record.functions[0]] = inverse, ("selection",)
        return record

    catalog = {
        **reflection_factories(preparation_inverse=recipes.__getitem__),
        source_key: prepare,
    }
    library = _library(source, 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(library, 0, catalog)
    assert verify_program_record(record) == ()
    compare_layers(record, source, atomic_leaves(record))


def test_unknown_prepare_inverse_requires_owned_context() -> None:
    from qualtran.bloqs.reflections.reflection_using_prepare import _refl_using_prep

    source = _refl_using_prep.make()
    child = identity_prepare_program(IdentityPrepare.from_bitsizes((1,))).functions[0]
    key = "qualtran.bloqs.reflections.reflection_using_prepare.ReflectionUsingPrepare"
    with pytest.raises(SerializationError, match="context"):
        reflection_factories()[key]("main", None, {"prepare_gate": child})
    assert source.prepare_gate.junk_registers


def test_original_zero_example_depth_and_source_free_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.reflections.reflection_using_prepare import _refl_around_zero

    source = _refl_around_zero.make()
    libraries = (_library(source, 0), _library(source, 1))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        records = tuple(
            resolve_component_program(library, 0, reflection_factories())
            for library in libraries
        )
    assert verify_program_record(records[0]) == verify_program_record(records[1]) == ()
    assert matches_component_definition(records[0], "main", records[1])
    expected = zero_state_reflection_program(
        PreparedStateReflection.around_zero((1, 2, 3), 1, -1)
    )
    inputs = ((0,), (1,), (2, 3), (4, 5, 6))
    actual = bind_register_program(records[0], inputs, clean=tuple(range(7, 14)))
    direct = bind_register_program(expected, inputs, clean=tuple(range(7, 14)))
    assert actual.circuit == direct.circuit
    assert actual.circuit.from_json(actual.circuit.to_json()) == actual.circuit


@pytest.mark.parametrize(
    "arguments", ({}, {"selection_regs": 3}, {"selection_regs": (), "extra": 0})
)
def test_malformed_identity_metadata_refuses(arguments: dict[str, Any]) -> None:
    factory = reflection_factories()[
        "qualtran.bloqs.reflections.prepare_identity.PrepareIdentity"
    ]
    with pytest.raises(SerializationError):
        factory("main", None, arguments)
