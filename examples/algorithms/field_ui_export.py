# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 field and polynomial notebook catalogs.

"""Export complete finite-field and polynomial example interface catalogs.

Run ``python -m examples.algorithms.field_ui_export``. Existing numerical codecs
and native catalogs own field identity, coefficient order and signatures. This
UI workflow draws stored addition/product graphs and atomic interfaces without
allocating simulation state.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from sympy import Symbol, symbols

from examples.algorithms.component_ui_dependencies import (
    ChildPresentation,
    ComponentFunction,
    dependency_graph,
)
from examples.algorithms.field_addition_catalog import field_addition_examples
from examples.algorithms.field_addition_decomposition import (
    field_addition_catalog_program,
    field_addition_score,
)
from examples.algorithms.field_inverse_catalog import field_inverse_example
from examples.algorithms.field_multiplication_catalog import (
    field_multiplication_examples,
)
from examples.algorithms.field_polynomial_catalog import (
    polynomial_addition_examples,
    polynomial_view_examples,
)
from examples.algorithms.field_polynomial_decomposition import (
    field_polynomial_child,
    field_polynomial_program,
    field_polynomial_score,
)
from examples.algorithms.field_power_decomposition import (
    field_power_child,
    field_power_program,
    field_power_score,
)
from examples.algorithms.field_product_decomposition import (
    field_product_child_decomposition,
    field_product_score,
)
from examples.algorithms.field_product_definitions import field_product_catalog_program
from examples.algorithms.field_square_catalog import field_square_examples
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.component_ui_export import (
    component_filename,
    export_component_ui,
    register_interface_score,
)
from tools.development.notebook_autogen import NotebookSpec

# UI label, authoritative native signature, parameter metadata and native owner.
_Interface: TypeAlias = tuple[str, Signature, Mapping[str, object], str]


def _field_parameters(encoding: FiniteFieldEncoding) -> Mapping[str, object]:
    return {
        "characteristic": encoding.characteristic,
        "degree": encoding.degree,
        "irreducible_poly": encoding.irreducible_poly,
    }


def _field_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    m = Symbol("m")
    gf4 = _field_parameters(binary_field_encoding(4))
    symbolic = _field_parameters(binary_field_encoding(m))
    gf3 = _field_parameters(binary_field_encoding([3, 1, 0]))
    gf5 = _field_parameters(binary_field_encoding([5, 2, 0]))
    products: tuple[tuple[str, str, Mapping[str, object]], ...] = (
        (
            "gf16_multiplication",
            "GF2Multiplication",
            {"qgf": gf4, "plus_equal_prod": True},
        ),
        (
            "gf2_multiplication_symbolic",
            "GF2Multiplication",
            {"qgf": symbolic, "plus_equal_prod": False},
        ),
        ("gf2_multiply_by_constant", "GF2MulK", {"dtype": gf3, "const": 5}),
        ("gf2_poly_multiply_by_constant", "GF2MulK", {"dtype": gf3, "const": 5}),
        ("multiplypolybyoneplusxk", "MultiplyPolyByOnePlusXk", {"n": 5, "k": 3}),
        ("binarypolynomialmultiplication", "BinaryPolynomialMultiplication", {"n": 5}),
        ("gf2shiftright", "GF2ShiftRight", {"qgf": gf5, "k": 3}),
        ("gf2mulviakaratsuba", "GF2MulViaKaratsuba", {"dtype": gf5}),
        ("gf2shiftleft", "GF2ShiftLeft", {"qgf": gf5, "k": 3}),
    )
    for (_, ports, _), (identifier, operation, parameters) in zip(
        field_multiplication_examples(), products, strict=True
    ):
        result[identifier] = (
            operation,
            ports,
            parameters,
            field_multiplication_examples.__module__,
        )
    for (_, ports, _), identifier, parameters in zip(
        field_addition_examples(),
        ("gf16_addition", "gf2_addition_symbolic"),
        ({"qgf": gf4}, {"qgf": symbolic}),
        strict=True,
    ):
        result[identifier] = (
            "GF2Addition",
            ports,
            parameters,
            field_addition_examples.__module__,
        )
    positive_m, positive_k = symbols("m, k", positive=True, integers=True)
    for (_, ports, _), identifier, width, constant in zip(
        field_addition_examples(constant=True),
        ("gf16_add_k", "gf2_add_k_symbolic"),
        (4, positive_m),
        (1, positive_k),
        strict=True,
    ):
        result[identifier] = (
            "GF2AddK",
            ports,
            {"bitsize": width, "k": constant},
            field_addition_examples.__module__,
        )
    for (_, ports, _), identifier, field in zip(
        field_square_examples(),
        ("gf16_square", "gf2_square_symbolic"),
        (gf4, symbolic),
        strict=True,
    ):
        result[identifier] = (
            "GF2Square",
            ports,
            {"qgf": field, "k": 1, "uncompute": False},
            field_square_examples.__module__,
        )
    _, ports, _ = field_inverse_example(4)
    result["gf16_inverse"] = (
        "GF2Inverse",
        ports,
        {"qgf": gf4},
        field_inverse_example.__module__,
    )
    return result


def _polynomial_parameters(
    degree: SymbolicInt, field: FiniteFieldEncoding
) -> Mapping[str, object]:
    return {"degree": degree, "qgf": _field_parameters(field)}


def _polynomial_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    n, m = symbols("n, m", positive=True, integers=True)
    concrete = binary_field_encoding(3)
    metadata = _polynomial_parameters(4, concrete)
    symbolic = _polynomial_parameters(n, binary_field_encoding(m))
    for (_, ports), identifier, operation in zip(
        polynomial_view_examples(),
        ("gf_poly_split", "gf_poly_join"),
        ("GFPolySplit", "GFPolyJoin"),
        strict=True,
    ):
        result[identifier] = (
            operation,
            ports,
            {"dtype": metadata},
            polynomial_view_examples.__module__,
        )
    polynomial = FiniteFieldPolynomialEncoding(4, concrete).from_gf_coefficients(
        [1, 2, 3, 4, 5]
    )
    # The original symbolic constant factory supplies a zero polynomial over
    # GF(2), independently of its symbolic coefficient-field descriptor.
    zero = FiniteFieldPolynomialEncoding(
        0, binary_field_encoding(1)
    ).from_gf_coefficients([0])
    for (_, ports, _), identifier, dtype, constant in zip(
        polynomial_addition_examples(constant=True),
        ("gf2_poly_4_8_add_k", "gf2_poly_add_k_symbolic"),
        (metadata, symbolic),
        (polynomial, zero),
        strict=True,
    ):
        result[identifier] = (
            "GF2PolyAddK",
            ports,
            {"qgf_poly": dtype, "g_x": constant},
            polynomial_addition_examples.__module__,
        )
    for (_, ports, _), identifier, dtype in zip(
        polynomial_addition_examples(),
        ("gf2_poly_4_8_add", "gf2_poly_add_symbolic"),
        (metadata, symbolic),
        strict=True,
    ):
        result[identifier] = (
            "GF2PolyAdd",
            ports,
            {"qgf_poly": dtype},
            polynomial_addition_examples.__module__,
        )
    return result


INSTANCES = {**_field_instances(), **_polynomial_instances()}
DOCUMENTS = (
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "GF2Multiplication",
        ("gf16_multiplication", "gf2_multiplication_symbolic"),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "GF2MulK",
        ("gf2_multiply_by_constant", "gf2_poly_multiply_by_constant"),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "MultiplyPolyByOnePlusXk",
        ("multiplypolybyoneplusxk",),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "BinaryPolynomialMultiplication",
        ("binarypolynomialmultiplication",),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "GF2ShiftRight",
        ("gf2shiftright",),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "GF2ShiftLeft",
        ("gf2shiftleft",),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Multiplication",
        "binary_field_multiplication",
        "GF2MulViaKaratsuba",
        ("gf2mulviakaratsuba",),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Addition",
        "binary_field_addition",
        "GF2Addition",
        ("gf16_addition", "gf2_addition_symbolic"),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Add Constant",
        "binary_field_add_constant",
        "GF2AddK",
        ("gf16_add_k", "gf2_add_k_symbolic"),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Square",
        "binary_field_square",
        "GF2Square",
        ("gf16_square", "gf2_square_symbolic"),
    ),
    (
        "GF Arithmetic",
        "GF($2^m$) Inverse",
        "binary_field_inverse",
        "GF2Inverse",
        ("gf16_inverse",),
    ),
    (
        "Polynomials over Galois Fields",
        "Polynomials over GF($p^m$) - Split and Join",
        "field_polynomial_views",
        "GFPolySplit",
        ("gf_poly_split",),
    ),
    (
        "Polynomials over Galois Fields",
        "Polynomials over GF($p^m$) - Split and Join",
        "field_polynomial_views",
        "GFPolyJoin",
        ("gf_poly_join",),
    ),
    (
        "Polynomials over Galois Fields",
        "Polynomials over GF($2^m$) - Add Constant",
        "binary_field_polynomial_add_constant",
        "GF2PolyAddK",
        ("gf2_poly_4_8_add_k", "gf2_poly_add_k_symbolic"),
    ),
    (
        "Polynomials over Galois Fields",
        "Polynomials over GF($2^m$) - Addition",
        "binary_field_polynomial_addition",
        "GF2PolyAdd",
        ("gf2_poly_4_8_add", "gf2_poly_add_symbolic"),
    ),
)


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError("Field UI requires a registered native example identifier")
    return INSTANCES[component]


def name(component: object) -> str:
    if isinstance(component, ComponentFunction):
        return "ComponentFunction"
    return _instance(component)[0]


def signature(component: object) -> Signature:
    return _instance(component)[1]


def attributes(component: object) -> Mapping[str, object]:
    if isinstance(component, ComponentFunction):
        return component.attributes()
    return _instance(component)[2]


def filename(component: object) -> str:
    if isinstance(component, ComponentFunction):
        return component.filename()
    return component_filename(attributes(component))


def score(component: object) -> object:
    if isinstance(component, ComponentFunction):
        return component.score()
    decomposed = field_addition_score(
        name(component), signature(component), attributes(component)
    )
    if decomposed is None:
        decomposed = field_product_score(
            name(component), signature(component), attributes(component)
        )
    if decomposed is None:
        decomposed = field_power_score(
            name(component), signature(component), attributes(component)
        )
    if decomposed is None:
        decomposed = field_polynomial_score(
            name(component), signature(component), attributes(component)
        )
    if decomposed is not None:
        return decomposed
    return register_interface_score(signature(component), name(component))


def decomposition(component: object) -> ProgramRecord | None:
    """Traverse stored field arithmetic through its authoritative Compiler owners."""
    record = field_addition_catalog_program(
        name(component), signature(component), attributes(component)
    )
    if record is None:
        record = field_product_catalog_program(
            name(component), signature(component), attributes(component)
        )
    if record is None:
        record = field_power_program(
            name(component), signature(component), attributes(component)
        )
    return (
        record
        if record is not None
        else field_polynomial_program(
            name(component), signature(component), attributes(component)
        )
    )


def _child_presentation(
    record: ProgramRecord, entry: str, root: object
) -> ChildPresentation | None:
    if name(root) in {"GF2PolyAdd", "GF2PolyAddK"}:
        return field_polynomial_child(record, entry, signature(root), attributes(root))
    if name(root) in {"GF2Square", "GF2Inverse"}:
        return field_power_child(record, entry, signature(root), attributes(root))
    return field_product_child_decomposition(
        record, entry, signature(root), attributes(root)
    )


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = _instance(documentation.examples[0].make())[3]
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "Concrete field and polynomial arithmetic retain complete stored decompositions. "
        "Other diagrams show atomic typed interfaces. Execution and resource "
        "estimates retain separate evidence. "
        "Karatsuba and inverse diagrams retain original literal graphs and declared inverse identities. "
        "Inverse diagrams include original unproved clean releases and do not grant execution admission. "
        "Correct mathematical inverse operations and frozen-source defects remain separate in FIELD_MULTIPLICATION_FILE_AUDIT.md. "
        "Large and symbolic instances allocate no execution wires.\n"
    )


def _identity(identifier: str) -> object:
    return identifier


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, dict[str, NotebookSpec]] = {}
    for section, title, stem, _, identifiers in DOCUMENTS:
        notebooks = grouped.setdefault(section, {})
        if title not in notebooks:
            notebooks[title] = NotebookSpec(
                title=title,
                module=sys.modules[__name__],
                component_specs=[],
                directory=str(Path(__file__).resolve().parents[1] / "notebooks"),
                path_stem=stem,
            )
        notebooks[title].component_specs.append(
            ComponentDocumentation(
                component_type=object,
                examples=tuple(
                    ComponentExample(partial(_identity, identifier), identifier, object)
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.field_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete field and polynomial metadata and fresh interfaces."""
    export_component_ui(
        destination,
        sections(),
        dependency_graph(
            INSTANCES,
            owner=lambda component: name(component) + "/" + filename(component),
            program=decomposition,
            child_presentation=_child_presentation,
        ),
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )


if __name__ == "__main__":
    export_example(Path("field_ui_export"))
