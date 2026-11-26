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
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_program,
)
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
