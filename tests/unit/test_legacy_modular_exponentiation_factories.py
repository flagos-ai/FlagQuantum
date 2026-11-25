"""Complete ModExp constructor closure keeps original calls and exact channels."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_exponentiation_factories import (
    modular_exponentiation_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_modular_multiplication_component import source_stream
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    library = _library(source, depth)
    assert (
        deserialize_component_library(serialize_component_library(library)) == library
    )
    result = resolve_component_program(library, 0, modular_exponentiation_factories())
    assert verify_program_record(result) == ()
    return result


@pytest.mark.parametrize(
    "base,modulus,e,n",
    [(2, 3, 1, 2), (2, 3, 3, 2), (5, 6, 2, 3), (4, 15, 2, 4), (2, 3, 0, 2)],
)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_original_constructor_calls_full_primitives_feedback_and_resources(
    base: int, modulus: int, e: int, n: int, depth: int
) -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = ModExp(base, modulus, e, n)
    record = _resolve(source, depth)
    assert record == _resolve(source)
    pairs = compare_layers(
        record, source, {f.name for f in record.functions if f.name != "main"}
    )
    bound = bind_register_program(
        record,
        (tuple(range(e)),),
        clean=tuple(range(e, e + n + 30)),
        allow_workspace_outputs=True,
    )
    target = bound.outputs[1]
    # Compare each original root child with its own explicit allocation lifetime.
    # A single Cirq export hoists allocations across stages and aliases work.
    expected = [Instruction("x", (target[-1],))]
    cursor = 1
    root = record.functions[0].blocks[0]
    children = [
        op.callee
        for op in root.operations
        if op.callee in pairs and type(pairs[op.callee]).__name__ == "CModMulK"
    ]
    for index, callee in enumerate(children):
        assert callee is not None
        control = (e - 1 - index,)
        child_bound = bind_register_program(
            record,
            (control, target),
            entry=callee,
            clean=tuple(range(e + n, e + n + 30)),
        )
        size = len(child_bound.circuit.instructions)
        sliced = replace(
            bound,
            circuit=replace(
                bound.circuit,
                instructions=bound.circuit.instructions[cursor : cursor + size],
            ),
            events=tuple(
                replace(ev, instruction_offset=ev.instruction_offset - cursor)
                for ev in bound.events
                if ev.instruction_offset is not None
                and cursor <= ev.instruction_offset < cursor + size
            ),
        )
        expected.extend(
            source_stream(pairs[callee], sliced, {"ctrl": control, "x": target})
        )
        cursor += size
    assert cursor == len(bound.circuit.instructions)
    assert_source(bound.circuit.instructions, tuple(expected))
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == model.measurement
    active: set[int] = set()
    peak = 0
    for event in bound.events:
        if event.kind == "allocate":
            active.update(event.wires)
        else:
            active.difference_update(event.wires)
        peak = max(peak, len(active))
    assert active == set(target)
    assert e + peak == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("e", [0, 1, 2])
def test_actual_source_constructor_complete_complex_branches_and_logical_result_ownership(
    e: int,
) -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    source = ModExp(2, 3, e, 2)
    bound = bind_register_program(
        _resolve(source),
        (tuple(range(e)),),
        clean=tuple(range(e, e + 20)),
        allow_workspace_outputs=True,
    )
    width = bound.circuit.n_wires

    def pack(exponent: int, result: int) -> int:
        values = {
            wire: (exponent >> (e - 1 - i)) & 1
            for i, wire in enumerate(bound.outputs[0])
        }
        values.update(
            {wire: (result >> (1 - i)) & 1 for i, wire in enumerate(bound.outputs[1])}
        )
        return sum(bit << (width - 1 - wire) for wire, bit in values.items())

    raw = [complex(i + 1, 2 - i) for i in range(1 << e)]
    norm = sum(abs(v) ** 2 for v in raw) ** 0.5
    before = [pack(i, 0) for i in range(1 << e)]
    after = [pack(i, pow(2, i, 3)) for i in range(1 << e)]
    initial = {k: v / norm for k, v in zip(before, raw, strict=True)}
    target = {k: v / norm for k, v in zip(after, raw, strict=True)}

    def bits(values: list[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple((k >> (width - 1 - i)) & 1 for i in range(width)) for k in values
        )

    mapped, measurements = certified_and_channel_map(
        bound.circuit.instructions, bits(before)
    )
    assert mapped == bits(after)
    for history in (
        (0,) * measurements,
        (1,) * measurements,
        tuple(i % 2 for i in range(measurements)),
    ):
        normalized = measurements > 60
        actual = _branch(
            bound.circuit.instructions,
            width,
            history,
            initial.copy(),
            normalize_measurements=normalized,
        )
        scale = 1 if normalized else 2 ** (-measurements / 2)
        assert {k: v for k, v in actual.items() if abs(v) > 1e-30} == pytest.approx(
            {k: v * scale for k, v in target.items()}, abs=1e-12 * scale
        )


def test_original_symbolic_factory_and_explicit_constructor_failures() -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp, _modexp_symb

    record = _resolve(_modexp_symb())
    assert not record.functions[0].blocks
    factory = next(iter(modular_exponentiation_factories().values()))
    for args in (
        {},
        {"base": 6, "mod": 15, "exp_bitsize": 2, "x_bitsize": 4},
        {"base": True, "mod": 3, "exp_bitsize": 1, "x_bitsize": 2},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"base": 2, "mod": 3, "exp_bitsize": 1, "x_bitsize": 2})
    with pytest.raises(SerializationError):
        _resolve(ModExp(3, 8, 1, 3))
    # The original empty composite exists but its classical bit conversion fails.
    with pytest.raises(ValueError, match="invalid literal"):
        ModExp(2, 3, 0, 2).decompose_bloq().on_classical_vals(exponent=0)


def test_locked_source_and_native_reference_environment() -> None:
    import kickmix
    from qualtran.bloqs.cryptography.rsa import rsa_mod_exp

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    raw = Path(rsa_mod_exp.__file__).read_bytes()
    # The frozen digest is independently recorded in RSA_MODEXP_FILE_AUDIT.
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "280bc11da4d7a5adef3382989b6e9bfed95a83ff"
    )


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("e", [0, 1, 2])
def test_exact_generic_source_inverse_call_chain_and_image_roundtrip(
    depth: int, e: int
) -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    source = ModExp(2, 3, e, 2)
    forward = _resolve(source)
    inverse = _resolve(source.adjoint(), depth)
    compare_layers(
        inverse,
        source.adjoint(),
        {f.name for f in inverse.functions if f.name != "main"},
    )
    before = bind_register_program(
        forward,
        (tuple(range(e)),),
        clean=tuple(range(e, e + 24)),
        allow_workspace_outputs=True,
    )
    target = before.outputs[1]
    after = bind_register_program(
        inverse,
        (tuple(range(e)), target),
        clean=tuple(range(e + 2, e + 24)),
        input_ownership=tuple((w, False) for w in target),
    )
    assert after.outputs == (tuple(range(e)),)
    width = max(before.circuit.n_wires, after.circuit.n_wires)
    gates = before.circuit.instructions + after.circuit.instructions
    labels = [value << (width - e) for value in range(1 << e)]
    basis = tuple(
        tuple((v >> (width - 1 - i)) & 1 for i in range(width)) for v in labels
    )
    mapped, measurements = certified_and_channel_map(gates, basis)
    assert mapped == basis
    weights = [complex(i + 1, 1 - 2 * i) for i in range(1 << e)]
    norm = sum(abs(v) ** 2 for v in weights) ** 0.5
    initial = {k: v / norm for k, v in zip(labels, weights, strict=True)}
    for history in (
        (0,) * measurements,
        (1,) * measurements,
        tuple(i % 2 for i in range(measurements)),
    ):
        actual = _branch(
            gates, width, history, initial.copy(), normalize_measurements=True
        )
        assert {k: v for k, v in actual.items() if abs(v) > 1e-12} == pytest.approx(
            initial, abs=2e-12
        )


@pytest.mark.parametrize(
    "kind,cv",
    [
        ("controlled", 0),
        ("controlled", 1),
        ("constant", 1),
        ("scaled", 1),
        ("multiply", 1),
    ],
)
def test_exact_typed_dependency_adjoints_match_original_primitives(
    kind: str, cv: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic import (
        CModAdd,
        CModAddK,
        CModMulK,
        CtrlScaleModAdd,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.modular.controlled_modular_addition_component import (
        controlled_modular_addition_program,
    )
    from flagquantum.compiler.registers.modular.modular_constant_component import (
        controlled_modular_constant_program,
    )
    from flagquantum.compiler.registers.modular.modular_multiplication_component import (
        modular_multiplication_program,
    )
    from flagquantum.compiler.registers.modular.scaled_modular_addition_component import (
        scaled_modular_addition_program,
    )
    from flagquantum.core.ir.program.register_types import unsigned_register_type
    from tests.unit.test_modular_addition_component import bind_case

    word = unsigned_register_type(2)
    if kind == "controlled":
        source = CModAdd(QUInt(2), 3, cv).adjoint()
        record = controlled_modular_addition_program(
            word, 3, control_value=cv, adjoint=True
        )
    elif kind == "constant":
        source = CModAddK(1, 3, 2).adjoint()
        record = controlled_modular_constant_program(2, 3, 1, adjoint=True)
    elif kind == "scaled":
        source = CtrlScaleModAdd(1, 3, 2).adjoint()
        record = scaled_modular_addition_program(2, 3, 1, adjoint=True)
    else:
        source = CModMulK(QUInt(2), 2, 3).adjoint()
        record = modular_multiplication_program(word, 3, 2, adjoint=True)
    bound, ports, _ = bind_case(record)
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == model.measurement


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_literal_comparator_inverse_and_original_signed_port_failure(
    signed: bool, cv: int
) -> None:
    from qualtran import QInt, QMontgomeryUInt
    from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan

    from flagquantum.compiler.registers.comparison.controlled_component import (
        controlled_greater_program,
    )
    from flagquantum.core.ir.program.register_types import (
        montgomery_register_type,
        signed_register_type,
    )
    from tests.unit.test_modular_addition_component import bind_case

    dtype = signed_register_type(2) if signed else montgomery_register_type(2, 3)
    source = CLinearDepthGreaterThan(
        QInt(2) if signed else QMontgomeryUInt(2, 3), cv
    ).adjoint()
    if signed:
        from qualtran import BloqError

        with pytest.raises(BloqError, match="Soquet named"):
            source.decompose_bloq()
        with pytest.raises(ValueError, match="sign-extension port names"):
            controlled_greater_program(dtype, control_value=cv, adjoint=True)
        return
    bound, ports, _ = bind_case(
        controlled_greater_program(dtype, control_value=cv, adjoint=True)
    )
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
