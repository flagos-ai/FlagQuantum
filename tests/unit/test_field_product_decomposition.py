"""The complete multiplication file retains recursive source graphs and exceptions."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import field_ui_export as native
from examples.algorithms.component_ui_dependencies import ComponentFunction
from examples.algorithms.field_product_decomposition import (
    field_product_child_decomposition,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_field_ui_export import originals as originals
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_gate_ui_export import _canonical_score

CONCRETE = tuple(native.INSTANCES)[:1] + tuple(native.INSTANCES)[2:9]
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_walk(
    identifier: str, source: Any, *, catalog: Any = native
) -> list[dict[str, Any]]:
    """Observe every named source child, including the frozen inverse defect."""
    from qualtran._infra.bloq import DecomposeNotImplementedError, DecomposeTypeError

    record = catalog.decomposition(identifier)
    assert record is not None and verify_program_record(record) == ()
    functions = {f.name: f for f in record.functions}
    entry = record.functions[0].name
    pending = [(entry, source)]
    visited: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    while pending:
        name, source = pending.pop(0)
        if name in visited:
            assert visited[name] == source
            continue
        visited[name] = source
        function = functions[name]
        row: dict[str, Any] = {"entry": name, "source_type": type(source).__name__}
        assert [
            (r.name, r.bitsize, r.shape, r.side.value) for r in function.registers or ()
        ] == [(r.name, r.bitsize, r.shape, r.side.value) for r in source.signature]
        try:
            composite = source.decompose_bloq()
        except (DecomposeNotImplementedError, DecomposeTypeError) as error:
            row.update(disposition="source_atomic_refusal", error=type(error).__name__)
            rows.append(row)
            continue
        except ValueError as error:
            assert identifier == "hamming_weight_phasing_via_phase_gradient"
            assert type(source).__name__ == "AddScaledValIntoPhaseReg"
            assert "Finalizing register dtypes are not consistent x:" in str(error)
            row.update(
                disposition="source_dtype_refusal",
                error=type(error).__name__,
                message=str(error),
            )
            rows.append(row)
            continue
        actual = _native_trace(record, name)
        expected = _source_trace(SourceCalls, composite)
        observed, original = [], []
        for left, right in zip(actual, expected, strict=True):
            assert isinstance(left, tuple) and isinstance(right, tuple)
            observed.append((left[0], dict(left[1]), left[2]))
            original.append((right[0], dict(right[1]), right[2]))
        exact = observed == original
        if not exact:
            assert identifier in {"gf2mulviakaratsuba", "gf16_inverse"}
            assert type(source).__name__ == "SynthesizeLRCircuit" and source.is_adjoint
            assert (
                _source_trace(SourceCalls, source.adjoint().decompose_bloq())
                == expected
            )
        row.update(
            disposition="exact_graph" if exact else "original_ignored_adjoint_defect",
            native_calls=len(actual),
            source_calls=len(expected),
        )
        if name == entry:
            drawing = catalog.score(identifier)
        else:
            node = ComponentFunction(
                identifier, name, record, identifier, catalog._child_presentation
            )
            assert node.presentation is not None, (identifier, name, source)
            label = _label(source)
            if not exact or (
                type(source).__name__ == "SynthesizeLRCircuit" and source.is_adjoint
            ):
                label += "†"
            assert node.presentation[0] == label
            row["qualified"] = label
            drawing = node.score()
        assert isinstance(drawing, dict)
        try:
            expected_score = _source_score(composite)
        except ValueError as error:
            assert type(source).__name__ in {
                "ControlledAddOrSubtract",
                "ZPowConstViaPhaseGradient",
            }
            assert "should return a `Text` object" in str(error)
            row["drawing"] = "original_controlled_title_refusal"
            row["drawing_error"] = str(error)
        else:
            if exact:
                assert _canonical_score(drawing) == _canonical_score(expected_score), (
                    identifier,
                    source,
                    name,
                )
                row["drawing"] = "source_exact"
            else:
                assert _canonical_score(drawing) != _canonical_score(expected_score)
                row["drawing"] = "native_exact_inverse_original_defect_retained"
        rows.append(row)
        children = {b.i: b.bloq for b, _, _ in composite.iter_bloqsoqs()}
        pending.extend(
            (op.callee or "", children[i])
            for i, op in enumerate(function.blocks[0].operations)
        )
    return rows


@pytest.mark.parametrize("identifier", CONCRETE)
def test_complete_source_file_recursive_calls_classes_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier])
    counts = Counter(row["disposition"] for row in rows)
    assert counts["exact_graph"] > 0
    assert counts["original_ignored_adjoint_defect"] == 0


def test_original_symbolic_product_refusal(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran._infra.bloq import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1]["gf2_multiplication_symbolic"].decompose_bloq()
    assert native.decomposition("gf2_multiplication_symbolic") is None


def test_source_child_labels_reject_changed_primitive_body() -> None:
    identifier = "gf2mulviakaratsuba"
    record = native.decomposition(identifier)
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            (
                                replace(op, opcode="cz")
                                if op.opcode == "cx"
                                else (
                                    replace(op, operands=op.operands[::-1])
                                    if op.opcode == "ccx"
                                    else op
                                )
                            )
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    assert changed != record
    rejected = 0
    for function in record.functions:
        arguments = (
            function.name,
            native.signature(identifier),
            native.attributes(identifier),
        )
        if field_product_child_decomposition(record, *arguments) is not None:
            assert field_product_child_decomposition(changed, *arguments) is None
            rejected += 1
    assert rejected > 0


def test_literal_inverse_labels_follow_complete_caller_with_opaque_function_names() -> (
    None
):
    identifier = "gf2mulviakaratsuba"
    record = native.decomposition(identifier)
    assert record is not None
    names = {f.name: f"opaque_{i}" for i, f in enumerate(record.functions)}
    renamed = replace(
        record,
        functions=tuple(
            replace(
                f,
                name=names[f.name],
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            (
                                replace(op, callee=names[op.callee])
                                if op.callee is not None
                                else op
                            )
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    functions = {f.name: f for f in renamed.functions}
    scales = [
        op.callee
        for b in renamed.functions[0].blocks
        for op in b.operations
        if op.callee is not None
        and tuple(r.name for r in functions[op.callee].registers or ()) == ("g",)
    ]
    assert len(scales) == 2 and scales[0] != scales[1]
    for scale, expected in zip(scales, ("GF2MulK†", "GF2MulK"), strict=True):
        assert scale is not None
        presentation = field_product_child_decomposition(
            renamed, scale, native.signature(identifier), native.attributes(identifier)
        )
        assert presentation is not None and presentation[0] == expected
        child = next(
            op.callee
            for b in functions[scale].blocks
            for op in b.operations
            if op.callee is not None
            and tuple(r.name for r in functions[op.callee].registers or ()) == ("q",)
        )
        assert child is not None
        linear = field_product_child_decomposition(
            renamed, child, native.signature(identifier), native.attributes(identifier)
        )
        assert linear is not None and linear[0] == "SynthesizeLRCircuit" + (
            "†" if expected.endswith("†") else ""
        )


@pytest.mark.parametrize("characteristic,degree", ((2, 1), (3, 1), (3, 2), (5, 1)))
def test_literal_quadratic_keeps_declared_left_right_identity_with_equal_bodies(
    characteristic: int, degree: int
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    from examples.algorithms.field_product_definitions import (
        field_product_catalog_program,
    )
    from flagquantum.compiler.registers.finite_field.multiplication_component import (
        field_multiplication_decomposition_program,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    encoding = FiniteFieldEncoding(characteristic, degree)
    parameters: dict[str, object] = {"plus_equal_prod": True}
    record = field_multiplication_decomposition_program(
        encoding.register_type(),
        modulus=int(encoding.gf_type.irreducible_poly),
        plus_equal_prod=True,
    )
    ports = Signature(record.functions[0].registers or ())
    assert (
        field_product_catalog_program("GF2Multiplication", ports, parameters) == record
    )
    original = GF2Multiplication(
        QGF(characteristic, degree), plus_equal_prod=True
    ).decompose_bloq()
    assert _native_trace(record) == _source_trace(SourceCalls, original)
    functions = {f.name: f for f in record.functions}
    shifts = [
        op.callee
        for b in record.functions[0].blocks
        for op in b.operations
        if op.callee is not None
        and tuple(r.name for r in functions[op.callee].registers or ()) == ("f",)
    ]
    assert len(shifts) == 2 and shifts[0] != shifts[1]
    for shift, label in zip(shifts, ("GF2ShiftLeft", "GF2ShiftRight"), strict=True):
        assert shift is not None
        presentation = field_product_child_decomposition(
            record, shift, ports, parameters
        )
        assert presentation is not None and presentation[0] == label


@pytest.mark.parametrize(
    "operation", ("MultiplyPolyByOnePlusXk", "GF2ShiftLeft", "GF2ShiftRight")
)
def test_source_integral_shift_aliases_reach_the_same_catalog_body(
    operation: str,
) -> None:
    from examples.algorithms.field_product_definitions import (
        field_product_catalog_program,
    )
    from flagquantum.compiler.registers.finite_field.monomial_component import (
        monomial_decomposition_program,
    )
    from flagquantum.compiler.registers.finite_field.polynomial_product_component import (
        one_plus_polynomial_program,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_field_frobenius_component import dtype

    canonical = (
        one_plus_polynomial_program(5, 1)
        if operation == "MultiplyPolyByOnePlusXk"
        else monomial_decomposition_program(
            dtype(3, 11), 1, modulus=11, adjoint=operation == "GF2ShiftLeft"
        )
    )
    ports = Signature(canonical.functions[0].registers or ())
    assert (
        field_product_catalog_program(operation, ports, {"n": 5, "k": True})
        == canonical
    )


def test_original_internal_degree_one_karatsuba_refusal_does_not_remove_math_base() -> (
    None
):
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import _GF2MulViaKaratsubaImpl

    from flagquantum.compiler.registers.finite_field.karatsuba_component import (
        karatsuba_decomposition_implementation,
        karatsuba_implementation_program,
    )
    from tests.unit.test_field_frobenius_component import dtype
    from tests.unit.test_field_linear import _source_type

    with pytest.raises(ValueError, match="negative dimensions"):
        _GF2MulViaKaratsubaImpl(_source_type(1, 3)).decompose_bloq()
    with pytest.raises(ValueError, match="degree at least two"):
        karatsuba_decomposition_implementation(dtype(1, 3), modulus=3)
    assert (
        verify_program_record(karatsuba_implementation_program(dtype(1, 3), modulus=3))
        == ()
    )
