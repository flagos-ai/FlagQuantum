"""Complete Hubbard constructors preserve source ports and unordered call metadata."""

from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_arguments import (
    serialize_component_argument,
)
from flagquantum.compiler.formats.legacy_component_export import (
    component_entry_from_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentDeclaration,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hubbard_factories import (
    HubbardKind,
    hubbard_component_library,
    hubbard_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_KINDS: tuple[HubbardKind, ...] = (
    "HoppingPlaquette",
    "HoppingTile",
    "HoppingTileHWP",
    "Interaction",
    "InteractionHWP",
)


@pytest.fixture(scope="module", autouse=True)
def frozen_sources_and_native_reference() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    hashes = {
        "hopping.py": "7e9a93856c658f93d728d74f9c3bc6cc357a3c0f",
        "hopping_test.py": "006c1c6c3b6a14ac11a8d6b3ea3fddaa417b1763",
        "interaction.py": "5a1d23cd0eeba8e7cf80b73c7b33b2ff7c07184b",
        "interaction_test.py": "deeccb2a5e59ff50afd0eeb101e5ac5556be7dd5",
    }
    source = import_module("qualtran.bloqs.chemistry.trotter.hubbard.hopping")
    assert source.__file__ is not None
    for filename, digest in hashes.items():
        raw = Path(source.__file__).with_name(filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def _class(kind: HubbardKind) -> Any:
    module = "hopping" if kind.startswith("Hopping") else "interaction"
    return getattr(
        import_module("qualtran.bloqs.chemistry.trotter.hubbard." + module), kind
    )


def _assert_declaration(source: Any, declaration: ComponentDeclaration) -> None:
    import attrs

    assert declaration.name == type(source).__module__ + "." + type(source).__name__
    assert declaration.registers == tuple(native_signature(source.signature))
    actual = {argument.name: argument.value for argument in declaration.arguments}
    expected = {
        field.name: getattr(source, field.name) for field in attrs.fields(type(source))
    }
    assert actual == expected
    for key, value in expected.items():
        # Numeric integer metadata must not be converted into a floating scalar.
        if type(value) is int:
            assert type(actual[key]) is int


def _assert_calls(source: Any, kind: HubbardKind, arguments: dict[str, Any]) -> None:
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.serialization.bloq import arg_from_proto, arg_to_proto

    library = hubbard_component_library(kind, arguments)
    root = library.entries[0]
    _assert_declaration(source, root.declaration)
    calls = source.build_call_graph(SympySymbolAllocator())
    entries = {entry.identifier: entry for entry in library.entries}
    assert not root.connections
    assert len(root.counts) == len(calls)
    for (identifier, count), (child, expected_count) in zip(
        root.counts, calls.items(), strict=True
    ):
        assert count == expected_count
        _assert_declaration(child, entries[identifier].declaration)
    raw = serialize_component_library(library)
    restored = deserialize_component_library(raw)
    for before, after in zip(library.entries, restored.entries, strict=True):
        expected_arguments = {}
        for argument in before.declaration.arguments:
            encoded = arg_to_proto(name=argument.name, val=argument.value)
            assert serialize_component_argument(argument) == encoded.SerializeToString()
            expected_arguments.update(arg_from_proto(encoded))
        assert {
            argument.name: argument.value for argument in after.declaration.arguments
        } == expected_arguments
    # Both codecs discard symbol assumptions and decode symbolic integer atoms
    # as Python ints. Check the same source normalization, not unsupported byte
    # identity across that first lossy read. Its normalized payload is stable.
    normalized = serialize_component_library(restored)
    assert (
        serialize_component_library(deserialize_component_library(normalized))
        == normalized
    )
    assert not restored.entries[0].connections
    program = resolve_component_program(library, 0, hubbard_factories())
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("profile", ("numeric", "symbolic", "default"))
def test_all_five_constructors_and_exact_direct_call_arguments(
    kind: HubbardKind, profile: str
) -> None:
    from qualtran import DecomposeNotImplementedError

    symbolic = profile == "symbolic"
    angle = sympy.Symbol("theta", real=True) if symbolic else -0.21
    args: dict[str, Any]
    if kind == "HoppingPlaquette":
        args = {"kappa": angle}
    else:
        args = {"length": sympy.Symbol("L") if symbolic else 8, "angle": angle}
        if not kind.startswith("Hopping"):
            args["hubb_u"] = sympy.Symbol("u") if symbolic else 4
        elif profile != "default":
            args.update(tau=2, pink=False)
    if profile != "default":
        args["eps"] = sympy.Symbol("eps", positive=True) if symbolic else 1e-6
    source = _class(kind)(**args)
    _assert_calls(source, kind, args)
    if symbolic and kind.endswith("HWP"):
        # Original protocol discovery recursively costs HWP and fails before
        # encoding its otherwise valid constructor metadata. Retain that failure.
        with pytest.raises(AttributeError, match="bit_length"):
            _library(source, 1)
        original_wire = hubbard_component_library(kind, args)
    else:
        original_wire = _library(source, 1)
    program = resolve_component_program(original_wire, 0, hubbard_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    assert not program.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize(
    "module,example,doc,kind",
    (
        ("hopping", "_plaquette", "_PLAQUETTE_DOC", "HoppingPlaquette"),
        ("hopping", "_hopping_tile", "_HOPPING_DOC", "HoppingTile"),
        ("hopping", "_hopping_tile_hwp", "_HOPPING_TILE_HWP_DOC", "HoppingTileHWP"),
        ("interaction", "_interaction", "_INTERACTION_DOC", "Interaction"),
        ("interaction", "_interaction_hwp", "_INTERACTION_HWP_DOC", "InteractionHWP"),
    ),
)
def test_complete_original_factories_and_docspecs(
    module: str, example: str, doc: str, kind: HubbardKind
) -> None:
    source_module = import_module("qualtran.bloqs.chemistry.trotter.hubbard." + module)
    factory = getattr(source_module, example)
    source = factory()
    specification = getattr(source_module, doc)
    assert specification.bloq_cls is type(source)
    assert specification.examples == (factory,)
    library = _library(source, 0)
    arguments = {
        argument.name: argument.value
        for argument in library.entries[0].declaration.arguments
    }
    _assert_calls(source, kind, arguments)
    if kind == "HoppingPlaquette":
        assert arguments == {"kappa": 8, "eps": 0.15}
    else:
        assert arguments["length"] == 8


@pytest.mark.parametrize("kind", ("HoppingTile", "HoppingTileHWP"))
def test_hopping_rejects_only_original_python_odd_constructor_lengths(
    kind: HubbardKind,
) -> None:
    for length in (-3, 1, 3):
        with pytest.raises(ValueError, match="Only even") as source:
            _class(kind)(length, 0.2)
        with pytest.raises(ValueError, match="Only even") as native:
            hubbard_component_library(kind, {"length": length, "angle": 0.2})
        assert str(native.value) == str(source.value)
    for length in (0, sympy.Integer(3)):
        # The original parity check deliberately uses isinstance(length, int).
        _assert_calls(_class(kind)(length, 0.2), kind, {"length": length, "angle": 0.2})
    library = hubbard_component_library(
        kind, {"length": sympy.Integer(3), "angle": 0.2}
    )
    normalized = deserialize_component_library(serialize_component_library(library))
    normalized_arguments = {
        argument.name: argument.value
        for argument in normalized.entries[0].declaration.arguments
    }
    assert type(normalized_arguments["length"]) is int
    with pytest.raises(ValueError, match="Only even"):
        _class(kind)(**normalized_arguments)
    with pytest.raises(ValueError, match="Only even"):
        resolve_component_program(normalized, 0, hubbard_factories())


def test_one_site_hwp_retains_zero_width_declared_child_and_unsupported_execution() -> (
    None
):
    args = {"length": 1, "angle": 0.21, "hubb_u": 4}
    _assert_calls(_class("InteractionHWP")(**args), "InteractionHWP", args)
    library = hubbard_component_library("InteractionHWP", args)
    child = library.entries[1].declaration
    assert child.registers == ()
    assert {argument.name: argument.value for argument in child.arguments}[
        "bitsize"
    ] == 0


@pytest.mark.parametrize("pink", (False, True))
def test_hwp_short_name_labels_library_without_changing_inherited_wire_symbol(
    pink: bool,
) -> None:
    source = _class("HoppingTileHWP")(8, 0.21, pink=pink)
    arguments = {"length": 8, "angle": 0.21, "pink": pink}
    library = hubbard_component_library("HoppingTileHWP", arguments)
    assert library.name == source.short_name() == f"H_h^{'p' if pink else 'g'}(HWP)"
    assert library.name != source.wire_symbol(None).text
    assert hubbard_component_library("HoppingTileHWP", arguments, name="").name == ""


@pytest.mark.parametrize("hamming_weight", (False, True))
@pytest.mark.parametrize("strip_layer", (False, True))
def test_original_suzuki_consumer_resolves_exact_stored_connections_without_changes(
    hamming_weight: bool, strip_layer: bool
) -> None:
    from qualtran.bloqs.chemistry.trotter.hubbard import trotter_step

    factory = (
        trotter_step.build_plaq_hwp_unitary_second_order_suzuki
        if hamming_weight
        else trotter_step.build_plaq_unitary_second_order_suzuki
    )
    source = factory(8, 4.0, -0.1234, 0.8, 1e-6, strip_layer=strip_layer)
    library = _library(source.decompose_bloq(), 1)
    program = resolve_component_program(library, 0, hubbard_factories())
    assert verify_program_record(program) == ()
    root = library.entries[0]
    identifiers = {
        (
            "main" if entry.identifier == 0 else f"legacy_component_{entry.identifier}"
        ): entry.identifier
        for entry in library.entries
    }
    exported = component_entry_from_program(
        program, "main", identifiers, root.declaration, counts=root.counts
    )
    assert exported == root
    assert len(program.functions[0].blocks[0].operations) == (4 if strip_layer else 5)
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


def test_reference_free_binding_and_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def no_reference(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reference)
    library = hubbard_component_library("HoppingTile", {"length": 8, "angle": 0.5})
    assert (
        not resolve_component_program(library, 0, hubbard_factories())
        .functions[0]
        .blocks
    )
    for bad in (
        {},
        {"length": True, "angle": 0.5},
        {"length": 8, "angle": 0.5, "pink": 2},
    ):
        with pytest.raises(SerializationError):
            hubbard_component_library("HoppingTile", bad)
    with pytest.raises(SerializationError, match="Unknown Hubbard"):
        hubbard_component_library(cast(HubbardKind, "Other"), {})
