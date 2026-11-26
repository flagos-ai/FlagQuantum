"""Source-ordered ECAdd components preserve coherent flags and private cleanup."""

import sys
from collections import Counter
from hashlib import sha1
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import sympy

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.elliptic_curve_addition_flags import (
    addition_flags_signature,
)
from flagquantum.compiler.formats.legacy_basis_vector_factories import (
    basis_vector_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_program,
)
from flagquantum.compiler.registers.component_definition_match import _named_definitions
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.compiler.registers.elliptic_curve.addition_flags_component import (
    addition_flags_program,
)
from flagquantum.compiler.registers.modular.modular_negation_component import (
    modular_negation_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import montgomery_register_type
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._elliptic_curve_addition_library import (
    elliptic_curve_addition_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_ecc_source() -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    assert module.__file__ is not None
    for name, digest in (
        ("ec_add.py", "2caf4f002152e51f1a09048d459c0b839280a4d1"),
        ("ec_add_test.py", "2fa735023da4c1b79d37079a7945aa2929e748b8"),
        ("ec_point.py", "eb885c54c88d663cbec19f44cdcbbb5e9cf424ef"),
    ):
        raw = (Path(module.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def flags_source_stream(
    source: Any, bound: BoundRegisterProgram, inputs: SourcePorts
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.arithmetic.comparison import Equals
    from qualtran.bloqs.basic_gates.z_basis import ZeroState
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg

    cursor = 0
    names = ("f1", "f2", "f3", "f4", "ctrl", "a", "b", "x", "y")
    outputs = dict(zip(names, bound.outputs, strict=True))
    composite = source.decompose_bloq()
    initial = {
        right[0]: bound.outputs[node.i]
        for node, _, right in composite.iter_bloqsoqs()
        if isinstance(node.bloq, ZeroState)
    }
    slots = MeasurementSlots(bound.circuit.instructions)
    live: dict[tuple[int, ...], int] = {}
    for index, event in enumerate(bound.events):
        if event.kind == "allocate":
            live[event.wires] = index
        elif event.kind == "free":
            live.pop(event.wires)
    flag_allocations = set(live.values())

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        if isinstance(bloq, ZeroState):
            return (), desired
        if not isinstance(bloq, (Equals, ModNeg, MultiControlX)):
            return None
        dummy = tuple(range(1000, 2000))
        dry = reference_arithmetic_gates(
            bloq,
            old,
            dummy,
            dict.fromkeys(dummy, 0),
            preserve_components=True,
            reuse_allocations=False,
        )
        plan = tuple(
            w
            for index, event in enumerate(bound.events)
            if index not in flag_allocations
            and event.kind == "allocate"
            and event.instruction_offset is not None
            and cursor <= event.instruction_offset < cursor + len(dry)
            for w in event.wires
        )
        gates = reference_arithmetic_gates(
            bloq, old, plan, slots, preserve_components=True, reuse_allocations=False
        )
        assert len(gates) == len(dry)
        cursor += len(gates)
        return gates, old

    gates, actual = walk_source_graph(
        composite, inputs, outputs, leaf, bindings=initial
    )
    assert actual == outputs and cursor == len(bound.circuit.instructions)
    return gates


def flags_program(n: int, modulus: int) -> ProgramRecord:
    dtype = montgomery_register_type(n)
    zero = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.ZeroState"](
        "zero", None, {}
    )
    assert not isinstance(zero, ProgramRecord)
    return addition_flags_program(
        addition_flags_signature(n),
        ProgramRecord((zero,)),
        equality_program(dtype),
        modular_negation_program(dtype, modulus),
        multi_control_x_program((0,) * (2 * n)),
        multi_control_x_program((0, 0, 0)),
    )


@pytest.mark.parametrize("n,modulus", [(2, 1), (3, 3), (5, 17)])
def test_literal_flags_calls_and_operand_producers(n: int, modulus: int) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne

    program = flags_program(n, modulus)
    assert verify_program_record(program) == ()
    expected = _named_inputs(
        _source_trace(SourceCalls, _ECAddStepOne(n, modulus).decompose_bloq())
    )
    assert _named_inputs(_native_trace(program)) == expected
    assert len(program.functions[0].blocks[0].operations) == 20
    inputs = tuple(tuple(range(i * n, (i + 1) * n)) for i in range(4))
    pool = tuple(range(4 * n, 4 * n + 32))
    bound = bind_register_program(
        program, inputs, clean=pool, allow_workspace_outputs=True
    )
    expected_gates = flags_source_stream(
        _ECAddStepOne(n, modulus),
        bound,
        dict(zip(("a", "b", "x", "y"), inputs, strict=True)),
    )
    assert_source(bound.circuit.instructions, expected_gates)


def test_flags_all_residues_complex_amplitudes_and_all_erasure_histories() -> None:
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne

    costs = import_module("qualtran.resource_counting")

    n, modulus = 3, 3
    program = flags_program(n, modulus)
    inputs = tuple(tuple(range(i * n, (i + 1) * n)) for i in range(4))
    bound = bind_register_program(
        program, inputs, clean=tuple(range(12, 44)), allow_workspace_outputs=True
    )
    used = {wire for event in bound.events for wire in event.wires}
    width = max(used) + 1
    bound = bind_register_program(
        program, inputs, clean=tuple(range(12, width)), allow_workspace_outputs=True
    )
    source = _ECAddStepOne(n, modulus)
    assert width == 21
    live: set[int] = set()
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        elif event.kind == "free":
            live.difference_update(event.wires)
    assert live == {wire for port in bound.outputs[:5] for wire in port}
    circuit = DynamicCircuit(width).append(
        FieldOperation(width, bound.circuit.instructions)
    )
    assert circuit.to_ir().instructions == bound.circuit.instructions
    names = ("f1", "f2", "f3", "f4", "ctrl", "a", "b", "x", "y")

    def basis(values: dict[str, int], ports: dict[str, tuple[int, ...]]) -> int:
        return sum(
            ((values[name] >> (len(wires) - i - 1)) & 1) << (width - wire - 1)
            for name, wires in ports.items()
            for i, wire in enumerate(wires)
        )

    before, after = {}, {}
    for index, coordinates in enumerate(product(range(modulus), repeat=4), 1):
        values = dict(zip(("a", "b", "x", "y"), coordinates, strict=True))
        amplitude = complex(index, 11 - index)
        before[basis(values, dict(zip(values, inputs, strict=True)))] = amplitude
        after[
            basis(
                {
                    name: int(value)
                    for name, value in source.on_classical_vals(**values).items()
                },
                dict(zip(names, bound.outputs, strict=True)),
            )
        ] = amplitude
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {key: value / norm for key, value in before.items()}
    after = {key: value / norm for key, value in after.items()}
    words = tuple(
        tuple((key >> (width - i - 1)) & 1 for i in range(width)) for key in before
    )
    mapped, measurements = certified_and_channel_map(bound.circuit.instructions, words)
    assert tuple(
        sum(bit << (width - i - 1) for i, bit in enumerate(row)) for row in mapped
    ) == tuple(after)
    for seed in (0, 1, 19):
        assert execute_sparse_register(
            circuit.to_ir(), before, seed=seed
        ) == pytest.approx(after, abs=1e-12)
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = costs.get_cost_value(source, costs.QECGatesCost())
    assert counts["measure"] == measurements == model.measurement == 28
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 112
    assert costs.get_cost_value(source, costs.QubitCount()) == 21


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_flags_constructor_sdk_free_roundtrip(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne

    library = _library(_ECAddStepOne(3, 3), depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        program = resolve_component_program(
            library, 0, elliptic_curve_addition_factories()
        )
    assert verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 20
    inputs = tuple(tuple(range(i * 3, (i + 1) * 3)) for i in range(4))
    actual = bind_register_program(
        program, inputs, clean=tuple(range(12, 30)), allow_workspace_outputs=True
    )
    expected = bind_register_program(
        flags_program(3, 3),
        inputs,
        clean=tuple(range(12, 30)),
        allow_workspace_outputs=True,
    )
    assert actual == expected


def test_flags_symbolic_declaration_and_invalid_constructor_contract() -> None:
    factory = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepOne"
    ]
    declaration = factory(
        "flags", None, {"n": sympy.Symbol("n", positive=True, integer=True), "mod": 3}
    )
    assert isinstance(declaration, FunctionRecord)
    assert not declaration.blocks
    with pytest.raises(SerializationError, match="original n and mod"):
        factory("flags", None, {"n": 3, "mod": 3, "window_size": 1})
    with pytest.raises(SerializationError, match="signature"):
        factory("flags", (), {"n": 3, "mod": 3})


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_slope_actual_constructor_literal_calls_and_sdk_free_replacement(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepTwo

    source = _ECAddStepTwo(3, 3, 1)
    library = _library(source, depth)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        graph = resolve_component_program(
            library, 0, elliptic_curve_addition_factories()
        )
    assert verify_program_record(graph) == ()
    assert len(graph.functions[0].blocks[0].operations) == 24
    assert _named_inputs(
        _native_trace(graph, graph.functions[0].name)
    ) == _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))


def test_typed_slope_replaces_flat_path_on_all_small_promised_inputs() -> None:
    from flagquantum.algorithms.primitives.elliptic_curve_addition_flags import (
        EllipticCurveAdditionFlags,
    )
    from flagquantum.algorithms.primitives.elliptic_curve_addition_slope import (
        EllipticCurveAdditionSlope,
    )
    from tests.unit.test_elliptic_curve_addition_slope import layout, pack

    factory = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepTwo"
    ]
    graph = factory("main", None, {"n": 3, "mod": 3, "window_size": 1})
    assert isinstance(graph, ProgramRecord)
    names = ("f1", "ctrl", "a", "b", "x", "y", "lam_r")
    sizes = (1, 1, 3, 3, 3, 3, 3)
    offset = 0
    ports = {}
    for name, size in zip(names, sizes, strict=True):
        ports[name] = tuple(range(offset, offset + size))
        offset += size
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, offset + 100)),
        allow_workspace_outputs=True,
        input_ownership=tuple((wire, False) for wire in range(offset)),
    )
    used = max(wire for event in bound.events for wire in event.wires) + 1
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, used)),
        allow_workspace_outputs=True,
        input_ownership=tuple((wire, False) for wire in range(offset)),
    )
    old, old_ports, _, _ = layout(EllipticCurveAdditionSlope(3, 3), reverse=True)
    old_words, new_words, amplitudes = [], [], []
    for index, (a, b, x, y, tangent) in enumerate(product(range(3), repeat=5), 1):
        f1, _, _, _, ctrl, *_ = EllipticCurveAdditionFlags(3, 3).classical_values(
            a, b, x, y
        )
        values = {
            "f1": f1,
            "ctrl": ctrl,
            "a": a,
            "b": b,
            "x": x,
            "y": y,
            "lam_r": tangent,
        }
        old_word, new_word = pack(values, old_ports, old.n_wires), pack(
            values, ports, used
        )
        old_words.append(
            tuple((old_word >> (old.n_wires - i - 1)) & 1 for i in range(old.n_wires))
        )
        new_words.append(tuple((new_word >> (used - i - 1)) & 1 for i in range(used)))
        amplitudes.append(complex(index, 11 - index))
    old_outputs, old_measurements = certified_and_channel_map(
        old.instructions, tuple(old_words)
    )
    mapped, measured = certified_and_channel_map(
        bound.circuit.instructions, tuple(new_words)
    )
    assert old_measurements == 470
    assert measured == 494
    output_ports = dict(
        zip(
            ("f1", "ctrl", "a", "b", "x", "y", "lam", "lam_r"),
            bound.outputs,
            strict=True,
        )
    )
    expected_words = []
    for row in old_outputs:
        values = {
            name: sum(row[wire] << (len(wires) - i - 1) for i, wire in enumerate(wires))
            for name, wires in old_ports.items()
        }
        expected_words.append(pack(values, output_ports, used))
    actual_words = [
        sum(bit << (used - i - 1) for i, bit in enumerate(row)) for row in mapped
    ]
    assert actual_words == expected_words
    norm = sum(abs(value) ** 2 for value in amplitudes) ** 0.5
    before = {
        sum(bit << (used - i - 1) for i, bit in enumerate(row)): value / norm
        for row, value in zip(new_words, amplitudes, strict=True)
    }
    expected = {
        word: value / norm
        for word, value in zip(expected_words, amplitudes, strict=True)
    }
    for seed in (0, 19):
        assert execute_sparse_register(
            bound.circuit, before, seed=seed
        ) == pytest.approx(expected, abs=2e-12)
    delta = Counter(g.name for g in bound.circuit.instructions)
    delta.subtract(g.name for g in old.instructions)
    assert delta == Counter({"h": 24, "measure": 24, "x": 24, "z": 12, "cx": -12})


@pytest.mark.parametrize("kind", ("Three", "Four"))
@pytest.mark.parametrize(
    "width,modulus,window,depth",
    [(3, 3, 1, 0), (3, 3, 1, 1), (3, 3, 1, 2), (5, 13, 5, 0)],
)
def test_coordinate_and_product_original_libraries_calls_and_nominal_views(
    kind: str,
    width: int,
    modulus: int,
    window: int,
    depth: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    source = getattr(module, "_ECAddStep" + kind)(width, modulus, window)
    library = _library(source, depth)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        graph = resolve_component_program(
            library, 0, elliptic_curve_addition_factories()
        )
    assert verify_program_record(graph) == ()
    assert len(graph.functions[0].blocks[0].operations) == (
        2 * width + 18 if kind == "Three" else 3 * width + 19
    )
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )


@pytest.mark.parametrize("kind", ("Three", "Four"))
def test_coordinate_and_product_complex_columns_all_histories_and_private_work(
    kind: str,
) -> None:
    from tests.unit.ec_stage_conformance import check_stage_channel
    from tests.unit.test_elliptic_curve_addition_coordinates import (
        layout as coordinates_layout,
    )
    from tests.unit.test_elliptic_curve_addition_products import (
        layout as products_layout,
    )

    factory = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStep" + kind
    ]
    graph = factory("main", None, {"n": 3, "mod": 3, "window_size": 1})
    assert isinstance(graph, ProgramRecord)
    if kind == "Three":
        original, ports, _, _ = coordinates_layout(3, 3)
        rows = tuple(
            dict(zip(("ctrl", "a", "b", "x", "y", "lam"), values, strict=True))
            for values in product(
                range(2), range(3), range(3), range(3), range(3), range(3)
            )
        )
        count = 1
    else:
        original, ports, _, _ = products_layout(3, 3)
        rows = tuple(
            dict(zip(("x", "y", "lam"), values, strict=True))
            for values in product(range(3), range(8), range(3))
        )
        count = 2
    check_stage_channel(
        graph,
        original,
        ports,
        rows,
        delta={
            "h": 24 * count,
            "measure": 24 * count,
            "x": 24 * count,
            "z": 12 * count,
            "cx": -12 * count,
        },
    )


@pytest.mark.parametrize(
    "kind,calls", [("_ECAddStepFive", 16), ("_ECAddStepSix", 61), ("ECAdd", 6)]
)
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_cleanup_finish_and_root_actual_libraries_literal_ports_and_feedback(
    kind: str, calls: int, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    source = getattr(module, kind)(3, 3)
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        graph = resolve_component_program(
            library, 0, elliptic_curve_addition_factories()
        )
    assert verify_program_record(graph) == ()
    assert len(graph.functions[0].blocks[0].operations) == calls
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )


@pytest.mark.parametrize("kind", ("clear", "zero_words"))
def test_structured_ecc_control_leaf_exact_gates_all_dirty_inputs_and_histories(
    kind: str,
) -> None:
    from math import prod

    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.ecosystem._elliptic_curve_addition_library import _clear_slope
    from tests.unit.ec_stage_conformance import pack
    from tests.unit.test_component_hierarchy_control import compare_layers

    source_module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    if kind == "clear":
        source = next(
            node.bloq
            for node, _, _ in source_module._ECAddStepFive(3, 3)
            .decompose_bloq()
            .iter_bloqsoqs()
            if node.i == 9
        )
        graph = _clear_slope(montgomery_register_type(3))
    else:
        source = next(
            node.bloq
            for node, _, _ in source_module._ECAddStepSix(3, 3)
            .decompose_bloq()
            .iter_bloqsoqs()
            if node.i == 1
        )
        record = elliptic_curve_addition_factories()[
            "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepSix"
        ]("finish", None, {"n": 3, "mod": 3})
        assert isinstance(record, ProgramRecord)
        entry = record.functions[0].blocks[0].operations[1].callee
        assert entry is not None
        graph = ProgramRecord(_named_definitions(record, entry))
    graph = _rename_program(graph, "main")
    signature = Signature(graph.functions[0].registers or ())
    compare_layers(graph, source, {f.name for f in graph.functions[1:]})
    inputs = []
    ports = {}
    offset = 0
    for register in signature.lefts():
        size = int(register.bitsize)
        start = offset
        for _ in range(prod(register.shape)):
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
        ports[register.name] = tuple(range(start, offset))
    pool = tuple(range(offset, offset + (3 if kind == "clear" else 8)))
    bound = bind_register_program(
        graph, tuple(inputs), clean=pool, entry=graph.functions[0].name
    )
    expected_gates = reference_arithmetic_gates(
        source,
        ports,
        pool,
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected_gates)
    width = bound.circuit.n_wires
    expected_words = []
    before = []
    for word in range(1 << offset):
        values = {
            name: sum(
                (word >> (offset - i - 1) & 1) << (len(wires) - j - 1)
                for j, i in enumerate(wires)
            )
            for name, wires in ports.items()
        }
        if kind == "clear":
            values["y"] ^= (
                values["x"] if values["ctrl1"] == 1 and values["ctrl2"] == 0 else 0
            )
        else:
            values["q"] ^= int(values["ctrl"] == 0)
        before.append(word << len(pool))
        expected_words.append(pack(values, ports, width))
    rows = tuple(
        tuple(word >> (width - i - 1) & 1 for i in range(width)) for word in before
    )
    mapped, events = certified_and_channel_map(bound.circuit.instructions, rows)
    assert events == len(pool)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in mapped
    ] == expected_words
    amplitudes = tuple(complex(i + 1, 7 - i) for i in range(len(before)))
    norm = sum(abs(value) ** 2 for value in amplitudes) ** 0.5
    initial = {
        word: value / norm for word, value in zip(before, amplitudes, strict=True)
    }
    final = {
        word: value / norm
        for word, value in zip(expected_words, amplitudes, strict=True)
    }
    for seed in (0, 19):
        assert execute_sparse_register(
            bound.circuit, initial, seed=seed
        ) == pytest.approx(final, abs=3e-12)


def test_cleanup_complete_promised_domain_complex_histories_and_consumed_slope() -> (
    None
):
    from tests.unit.ec_stage_conformance import check_stage_channel
    from tests.unit.test_elliptic_curve_addition_cleanup import layout

    graph = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFive"
    ]("main", None, {"n": 3, "mod": 3, "window_size": 1})
    assert isinstance(graph, ProgramRecord)
    original, ports, _, _ = layout(3, 3)
    rows = []
    word = import_module("qualtran").QMontgomeryUInt(3, 3)
    for ctrl, a, b, x, y, tangent in product(
        range(2), range(3), range(3), range(3), range(3), range(3)
    ):
        inverse = word.montgomery_inverse(x) if x else 0
        calculated = word.montgomery_product(inverse, y)
        assert type(inverse) is int and type(calculated) is int
        slope = (calculated if ctrl else 0) ^ (tangent if ctrl and inverse == 0 else 0)
        rows.append(dict(ctrl=ctrl, a=a, b=b, x=x, y=y, lam_r=tangent, lam=slope))
    check_stage_channel(
        graph,
        original,
        ports,
        rows,
        delta={"h": 24, "measure": 24, "x": 24, "z": 12, "cx": -12},
    )


def _small_curve_rows() -> list[dict[str, int]]:
    points = (
        (0, 0),
        *(
            (x, y)
            for x, y in product(range(3), repeat=2)
            if (y * y - x**3 - x - 1) % 3 == 0
        ),
    )
    rows = []
    for (a, b), (x, y) in product(points, repeat=2):
        tangent = (3 * a * a + 1) * pow(2 * b, -1, 3) % 3 if b else 0
        rows.append(
            {
                name: value * 8 % 3
                for name, value in zip(
                    ("a", "b", "x", "y", "lam_r"), (a, b, x, y, tangent), strict=True
                )
            }
        )
    return rows


@pytest.mark.parametrize("kind", ("finish", "root"))
def test_full_curve_pairs_absolute_complex_branches_all_work_and_flags_clear(
    kind: str,
) -> None:
    from flagquantum.algorithms.primitives.elliptic_curve_addition_finish import (
        EllipticCurveAdditionFinish,
    )
    from tests.unit.ec_stage_conformance import check_stage_channel
    from tests.unit.test_elliptic_curve_addition import layout

    factories = elliptic_curve_addition_factories()
    rows = _small_curve_rows()
    ports: dict[str, tuple[int, ...]]
    if kind == "finish":
        module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
        stages = [
            getattr(module, "_ECAddStep" + name)(3, 3).decompose_bloq()
            for name in ("One", "Two", "Three", "Four", "Five")
        ]
        for row in rows:
            for stage in stages:
                values = stage.on_classical_vals(
                    **{
                        register.name: row[register.name]
                        for register in stage.signature.lefts()
                    }
                )
                row.update({name: int(value) for name, value in values.items()})
        graph = factories["qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepSix"](
            "main", None, {"n": 3, "mod": 3}
        )
        operation = EllipticCurveAdditionFinish(3, 3)
        q, c = operation.workspace_requirements()
        ports = {
            name: (wire,) for wire, name in enumerate(("f1", "f2", "f3", "f4", "ctrl"))
        }
        ports.update(
            {
                name: tuple(range(5 + 3 * i, 8 + 3 * i))
                for i, name in enumerate(("a", "b", "x", "y"))
            }
        )
        original = (
            DynamicCircuit(17 + q)
            .append(
                operation.make_on(
                    a=ports["a"],
                    b=ports["b"],
                    x=ports["x"],
                    y=ports["y"],
                    flags=tuple(range(5)),
                    workspace=tuple(range(17, 17 + q)),
                    classical_bits=tuple(range(c)),
                )
            )
            .to_ir()
        )
        delta = {}
    else:
        graph = factories["qualtran.bloqs.cryptography.ecc.ec_add.ECAdd"](
            "main", None, {"n": 3, "mod": 3, "window_size": 1}
        )
        original, ports = layout(3, 3)
        delta = {"h": 120, "measure": 120, "x": 120, "z": 60, "cx": -60}
    assert isinstance(graph, ProgramRecord)
    check_stage_channel(graph, original, ports, rows, delta=delta)


def test_original_modulus_five_dirty_flag_release_is_not_repaired() -> None:
    from tests.unit.ec_stage_conformance import pack

    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    data = dict(a=0, b=0, x=2, y=1, lam_r=0)
    for name in ("One", "Two", "Three", "Four", "Five"):
        stage = getattr(module, "_ECAddStep" + name)(4, 5).decompose_bloq()
        values = stage.on_classical_vals(
            **{
                register.name: data[register.name]
                for register in stage.signature.lefts()
            }
        )
        data.update({name: int(value) for name, value in values.items()})
    finish = module._ECAddStepSix(4, 5).decompose_bloq()
    with pytest.raises(ValueError, match="free a non-zero register"):
        finish.on_classical_vals(
            **{
                register.name: data[register.name]
                for register in finish.signature.lefts()
            }
        )
    graph = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add.ECAdd"
    ]("main", None, {"n": 4, "mod": 5, "window_size": 1})
    assert isinstance(graph, ProgramRecord)
    names = ("a", "b", "x", "y", "lam_r")
    inputs = tuple(tuple(range(4 * i, 4 * i + 4)) for i in range(5))
    bound = bind_register_program(
        graph,
        inputs,
        clean=tuple(range(20, 160)),
        input_ownership=tuple((wire, False) for wire in range(20)),
        allow_workspace_outputs=True,
    )
    width = bound.circuit.n_wires
    flags = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepOne"
    ]("main", None, {"n": 4, "mod": 5})
    assert isinstance(flags, ProgramRecord)
    prefix = bind_register_program(
        flags,
        inputs[:4],
        clean=tuple(range(20, 160)),
        input_ownership=tuple((wire, False) for wire in range(16)),
        allow_workspace_outputs=True,
    )
    ghost = prefix.outputs[0][0]
    assert any(
        event.kind == "free" and not event.dirty and event.wires == (ghost,)
        for event in bound.events
    )
    raw = dict(a=0, b=0, x=2, y=1, lam_r=0)
    before = pack(raw, dict(zip(names, inputs, strict=True)), width)
    expected = pack(raw, dict(zip(names, bound.outputs, strict=True)), width) | (
        1 << (width - ghost - 1)
    )
    actual = execute_sparse_register(bound.circuit, {before: 1j}, seed=19)
    target = {expected: 1j}
    assert (
        max(
            abs(actual.get(key, 0) - target.get(key, 0))
            for key in actual.keys() | target.keys()
        )
        < 4e-12
    )


@pytest.mark.parametrize("kind", ("One", "Two", "Three", "Four", "Five", "Six", "root"))
def test_all_source_stage_symbolic_declarations_and_invalid_fields(kind: str) -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    source = (module.ECAdd if kind == "root" else getattr(module, "_ECAddStep" + kind))(
        sympy.Symbol("n", integer=True, positive=True), 3
    )
    library = _library(source, 0)
    factories = elliptic_curve_addition_factories()
    graph = resolve_component_program(library, 0, factories)
    assert verify_program_record(graph) == ()
    assert not graph.functions[0].blocks
    key = "qualtran.bloqs.cryptography.ecc.ec_add." + (
        "ECAdd" if kind == "root" else "_ECAddStep" + kind
    )
    with pytest.raises(SerializationError, match="fields"):
        factories[key]("bad", None, {"n": 3, "mod": 3, "unknown": 1})
    with pytest.raises(SerializationError, match="signature"):
        factories[key]("bad", (), {"n": 3, "mod": 3})


@pytest.mark.parametrize("name", ("_ec_add", "_ec_add_small"))
def test_unchanged_original_examples_declarations_or_complete_six_call_root(
    name: str,
) -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    source = getattr(module, name).make()
    graph = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_addition_factories()
    )
    assert verify_program_record(graph) == ()
    if name == "_ec_add":
        assert not graph.functions[0].blocks
    else:
        assert (source.n, source.mod, source.window_size) == (5, 7, 1)
        assert len(graph.functions[0].blocks[0].operations) == 6
        assert _named_inputs(_native_trace(graph)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )


def test_wide_window_source_reduction_counterexample_remains_in_typed_path() -> None:
    from tests.unit.ec_stage_conformance import pack

    module = import_module("qualtran.bloqs.cryptography.ecc.ec_add")
    source = module._ECAddStepFour(5, 13, 5)
    values = dict(x=12, y=0, lam=9)
    assert source.decompose_bloq().on_classical_vals(**values) == dict(x=5, y=1, lam=9)
    graph = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFour"
    ]("main", None, {"n": 5, "mod": 13, "window_size": 5})
    assert isinstance(graph, ProgramRecord)
    inputs = tuple(tuple(range(5 * i, 5 * i + 5)) for i in range(3))
    bound = bind_register_program(
        graph,
        inputs,
        clean=tuple(range(15, 265)),
        input_ownership=tuple((wire, False) for wire in range(15)),
        allow_workspace_outputs=True,
    )
    width = bound.circuit.n_wires
    before = pack(values, dict(zip(values, inputs, strict=True)), width)
    expected = pack(
        dict(x=5, y=27, lam=9), dict(zip(values, bound.outputs, strict=True)), width
    )
    for seed in (0, 19):
        actual = execute_sparse_register(bound.circuit, {before: 1j}, seed=seed)
        target = {expected: 1j}
        assert (
            max(
                abs(actual.get(key, 0) - target.get(key, 0))
                for key in actual.keys() | target.keys()
            )
            < 4e-12
        )
