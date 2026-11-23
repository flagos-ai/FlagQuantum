"""Comparison library constructors close complete source families and dependencies."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_comparison_factories import (
    comparison_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fanout_factories import fanout_factories
from flagquantum.compiler.formats.legacy_multi_control_factories import (
    multi_control_factories,
)
from flagquantum.compiler.formats.legacy_sign_extension_factories import (
    sign_extension_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_comparison_component import (
    allocation_order,
    assert_source_amplitudes,
    check_source,
)
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_comparison(source: Any, depth: int = 1) -> ProgramRecord:
    factories = {
        **_factories(),
        **comparison_factories(),
        **sign_extension_factories(),
        **multi_control_factories(),
        **fanout_factories(),
    }
    program = resolve_component_program(_library(source, depth), 0, factories)
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize(
    "kind",
    (
        "LessThanConstant",
        "EqualsAConstant",
        "LessThanEqual",
        "GreaterThan",
        "LinearDepthGreaterThan",
        "Equals",
        "CLinearDepthGreaterThan",
    ),
)
@pytest.mark.parametrize("width", (1, 2, 3, 5))
@pytest.mark.parametrize("signed", (False, True))
def test_complete_named_and_decomposed_source_streams(
    kind: str, width: int, signed: bool
) -> None:
    from qualtran import QInt, QUInt

    module = import_module("qualtran.bloqs.arithmetic.comparison")
    constructor = getattr(module, kind)
    dtype = QInt(width) if signed else QUInt(width)
    if kind in ("LessThanConstant", "EqualsAConstant"):
        source = constructor(width, (1 << width) - 1)
        ports = {"x": tuple(range(width)), "target": (width,)}
    elif kind == "LessThanEqual":
        source = constructor(width, width + 1)
        ports = {
            "x": tuple(range(width)),
            "y": tuple(range(width, 2 * width + 1)),
            "target": (2 * width + 1,),
        }
    elif kind == "GreaterThan":
        source = constructor(width, width + 1)
        ports = {
            "a": tuple(range(width)),
            "b": tuple(range(width, 2 * width + 1)),
            "target": (2 * width + 1,),
        }
    elif kind == "LinearDepthGreaterThan":
        source = constructor(width, signed=signed)
        ports = {
            "a": tuple(range(width)),
            "b": tuple(range(width, 2 * width)),
            "target": (2 * width,),
        }
    elif kind == "Equals":
        source = constructor(dtype)
        ports = {
            "x": tuple(range(width)),
            "y": tuple(range(width, 2 * width)),
            "target": (2 * width,),
        }
    else:
        source = constructor(dtype, cv=int(signed))
        ports = {
            "ctrl": (0,),
            "a": tuple(range(1, width + 1)),
            "b": tuple(range(width + 1, 2 * width + 1)),
            "target": (2 * width + 1,),
        }
    named = resolve_comparison(source)
    expected = check_source(named, source, ports, tree=kind == "LessThanConstant")
    for original in (source, source.decompose_bloq()):
        bound = bind_register_program(
            resolve_comparison(original, 2),
            tuple(ports.values()),
            clean=tuple(range(max(w for p in ports.values() for w in p) + 1, 100)),
        )
        assert bound.circuit.instructions == expected.circuit.instructions
        assert bound.outputs == expected.outputs


@pytest.mark.parametrize(
    "kind",
    (
        "LessThanConstant",
        "EqualsAConstant",
        "LessThanEqual",
        "GreaterThan",
        "LinearDepthGreaterThan",
        "Equals",
        "CLinearDepthGreaterThan",
    ),
)
def test_constructor_all_raw_branches_full_source_complex_amplitudes(kind: str) -> None:
    from qualtran import QUInt

    module = import_module("qualtran.bloqs.arithmetic.comparison")
    ports: tuple[tuple[int, ...], ...]
    if kind in ("LessThanConstant", "EqualsAConstant"):
        source, ports = getattr(module, kind)(2, 2), ((0, 1), (2,))
    elif kind in ("LessThanEqual", "GreaterThan"):
        source, ports = getattr(module, kind)(2, 2), ((0, 1), (2, 3), (4,))
    elif kind == "LinearDepthGreaterThan":
        source, ports = module.LinearDepthGreaterThan(2), ((0, 1), (2, 3), (4,))
    elif kind == "Equals":
        source, ports = module.Equals(QUInt(2)), ((0, 1), (2, 3), (4,))
    else:
        source, ports = module.CLinearDepthGreaterThan(QUInt(1), cv=0), (
            (0,),
            (1,),
            (2,),
            (3,),
        )
    assert_source_amplitudes(resolve_comparison(source), source, ports)


@pytest.mark.parametrize(
    "kind",
    (
        "_HalfLinearDepthGreaterThan",
        "LinearDepthHalfGreaterThan",
        "LinearDepthHalfGreaterThanEqual",
        "LinearDepthHalfLessThan",
        "LinearDepthHalfLessThanEqual",
    ),
)
@pytest.mark.parametrize("encoding", ("QUInt", "QInt", "QMontgomeryUInt"))
@pytest.mark.parametrize("width", (1, 2, 3))
def test_half_image_inverse_aliases_source_gates_and_separate_models(
    kind: str, encoding: str, width: int
) -> None:
    dtype = getattr(import_module("qualtran"), encoding)(width)
    source = getattr(import_module("qualtran.bloqs.arithmetic.comparison"), kind)(dtype)
    ports = {"a": tuple(range(width)), "b": tuple(range(width, 2 * width))}
    forward = bind_register_program(
        resolve_comparison(source),
        tuple(ports.values()),
        clean=tuple(range(2 * width, 6 * width + 5)),
        allow_workspace_outputs=True,
    )
    inverse = source.adjoint()
    inverse_clean = tuple(
        w
        for w in range(2 * width, 6 * width + 5)
        if all(w not in p for p in forward.outputs)
    )
    back = bind_register_program(
        resolve_comparison(inverse),
        forward.outputs,
        clean=inverse_clean,
        input_ownership=tuple((w, False) for p in forward.outputs[2:] for w in p),
    )
    swapped = kind in ("LinearDepthHalfGreaterThanEqual", "LinearDepthHalfLessThan")
    expected_ports = (
        tuple(reversed(tuple(ports.values()))) if swapped else tuple(ports.values())
    )
    assert back.outputs == expected_ports
    outputs = dict(zip(("a", "b", "c", "target"), forward.outputs, strict=True))
    for original, bound, before, after in (
        (source, forward, ports, outputs),
        (inverse, back, outputs, dict(zip(("a", "b"), back.outputs, strict=True))),
    ):
        reference = reference_arithmetic_gates(
            original,
            before,
            allocation_order(bound),
            MeasurementSlots(bound.circuit.instructions),
            expected_outputs=after,
            reuse_allocations=False,
        )
        assert_source(bound.circuit.instructions, reference)
        from qualtran.resource_counting import QECGatesCost, get_cost_value

        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(original, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == model.measurement
        direct = bind_register_program(
            resolve_comparison(original.decompose_bloq(), 1),
            tuple(before.values()),
            clean=(
                tuple(range(2 * width, 6 * width + 5))
                if original is source
                else inverse_clean
            ),
            allow_workspace_outputs=original is source,
            input_ownership=(
                ()
                if original is source
                else tuple((w, False) for p in forward.outputs[2:] for w in p)
            ),
        )
        assert direct.circuit.instructions == bound.circuit.instructions
        assert direct.outputs == bound.outputs
    if width > 1:
        return
    n = (
        max(
            w
            for bound in (forward, back)
            for g in bound.circuit.instructions
            for w in g.wires
        )
        + 1
    )
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    values = torch.tensor([1 + 2j, 3 - 4j, -2 + 5j, 7 + 1j], dtype=torch.complex128)
    values /= values.norm()
    initial[:: 1 << (n - 2)] = values
    prepared = forced_branch(initial, forward.circuit.instructions, ())
    # Compare the physical source stream above and every exact erasure history.
    measured = sum(g.name == "measure" for g in back.circuit.instructions)
    for history in product((0, 1), repeat=measured):
        actual = forced_branch(prepared, back.circuit.instructions, history)
        torch.testing.assert_close(
            actual, initial / 2 ** (measured / 2), rtol=0, atol=3e-12
        )
    circuit = DynamicCircuit(n, inputs=initial, dtype=torch.complex128)
    circuit.append(
        FieldOperation(n, (*forward.circuit.instructions, *back.circuit.instructions))
    )
    result = run_dynamic(circuit, shots=2, seed=13, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), initial.expand(2, -1), rtol=0, atol=3e-12
    )


def test_symbolic_model_only_and_invalid_metadata_boundaries() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import comparison as c

    n, k = sympy.symbols("n k", positive=True, integer=True)
    symbolic = (
        c.LessThanConstant(n, k),
        c.EqualsAConstant(n, k),
        c.LessThanEqual(n, n + 1),
        c.GreaterThan(n, n + 1),
        c.LinearDepthGreaterThan(n),
        c.Equals(QUInt(n)),
        c.CLinearDepthGreaterThan(QUInt(n)),
        c.LinearDepthHalfGreaterThan(QUInt(n)),
        c.LinearDepthHalfGreaterThanEqual(QUInt(n)),
        c.LinearDepthHalfLessThan(QUInt(n)),
        c.LinearDepthHalfLessThanEqual(QUInt(n)),
    )
    for original in (*symbolic, c.GreaterThanConstant(3, 2)):
        assert resolve_comparison(original, 0).functions[0].blocks == ()
    factories = comparison_factories()
    cases: tuple[tuple[str, dict[str, ResolvedArgument]], ...] = (
        ("LessThanEqual", {"x_bitsize": "bad", "y_bitsize": 2}),
        ("LessThanConstant", {}),
        ("LessThanEqual", {}),
        ("Equals", {}),
        ("LinearDepthGreaterThan", {"bitsize": 2, "bad": 1}),
        ("Equals", {"dtype": "bad"}),
        ("CLinearDepthGreaterThan", {"dtype": opaque_register_type(2)}),
        ("CLinearDepthGreaterThan", {"dtype": unsigned_register_type(2), "cv": 2}),
        ("BiQubitsMixer", {"bad": 1}),
    )
    for kind, args in cases:
        with pytest.raises(SerializationError):
            factories["qualtran.bloqs.arithmetic.comparison." + kind](
                "custom", None, args
            )
    result = factories["qualtran.bloqs.arithmetic.comparison.Equals"](
        "custom", None, {"dtype": unsigned_register_type(2)}
    )
    assert isinstance(result, ProgramRecord)
    with pytest.raises(SerializationError, match="signature"):
        factories["qualtran.bloqs.arithmetic.comparison.Equals"](
            "custom", (), {"dtype": unsigned_register_type(2)}
        )


@pytest.mark.parametrize("mixer", (False, True))
def test_directional_flag_constructor_source_isometry_and_erasure(mixer: bool) -> None:
    from qualtran.bloqs.arithmetic.comparison import BiQubitsMixer, SingleQubitCompare

    source = BiQubitsMixer() if mixer else SingleQubitCompare()
    logical, flags, measurements = (4, 3, 2) if mixer else (2, 2, 1)
    inputs = ((0, 1), (2, 3)) if mixer else ((0,), (1,))
    work = tuple(range(logical, logical + flags))
    forward = bind_register_program(
        resolve_comparison(source), inputs, clean=work, allow_workspace_outputs=True
    )
    back = bind_register_program(
        resolve_comparison(source.adjoint()),
        forward.outputs,
        input_ownership=tuple((w, False) for w in work),
    )
    before_names = ("x", "y") if mixer else ("a", "b")
    after_names = (
        ("x", "y", "ancilla") if mixer else ("a", "b", "less_than", "greater_than")
    )
    for original, bound, before, after in (
        (
            source,
            forward,
            dict(zip(before_names, inputs, strict=True)),
            dict(zip(after_names, forward.outputs, strict=True)),
        ),
        (
            source.adjoint(),
            back,
            dict(zip(after_names, forward.outputs, strict=True)),
            dict(zip(before_names, back.outputs, strict=True)),
        ),
    ):
        assert_source(
            bound.circuit.instructions,
            reference_arithmetic_gates(
                original,
                before,
                work if original is source else (),
                MeasurementSlots(bound.circuit.instructions),
                preserve_components=True,
                expected_outputs=after,
            ),
        )
    coefficients = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 3 - 7 * coefficients)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + flags + 1), dtype=torch.complex128)
    initial[:: 1 << (flags + 1)] = amplitudes[::2]
    initial[1 :: 1 << (flags + 1)] = amplitudes[1::2]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << (logical + flags), 1 << logical
    )
    expected = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    prepared = forced_branch(initial, forward.circuit.instructions, ())
    torch.testing.assert_close(prepared, expected, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(prepared, back.circuit.instructions, history),
            initial / 2 ** (measurements / 2),
            rtol=0,
            atol=3e-12,
        )


@pytest.mark.parametrize(
    "name",
    (
        "_lt_k",
        "_lt_k_symb",
        "_bi_qubits_mixer",
        "_sq_cmp",
        "_leq_symb",
        "_leq",
        "_greater_than",
        "_gt_k",
        "_equals",
        "_eq_k",
        "_clineardepthgreaterthan_example",
        "_lineardepthhalfgreaterthan_small",
        "_lineardepthhalflessthan_small",
        "_lineardepthhalfgreaterthanequal_small",
        "_lineardepthhalflessthanequal_small",
    ),
)
def test_all_original_examples_preserve_named_signatures(name: str) -> None:
    source = getattr(import_module("qualtran.bloqs.arithmetic.comparison"), name).make()
    program = resolve_comparison(source, 0)
    assert (
        program.functions[0].registers
        == _library(source, 0).entries[0].declaration.registers
    )


def test_unmodified_symbolic_dependency_codec_failure_is_retained() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import Equals

    n = sympy.Symbol("n", positive=True, integer=True)
    with pytest.raises(ValueError, match="Cannot serialize HasLength"):
        _library(Equals(QUInt(n)), 1)
    assert not resolve_comparison(Equals(QUInt(n)), 0).functions[0].blocks
