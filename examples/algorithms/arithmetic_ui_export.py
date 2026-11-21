# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Arithmetic notebook catalog.

"""Export every Arithmetic example through native register metadata owners.

Run ``python -m examples.algorithms.arithmetic_ui_export``. The report associates
existing domain signatures with UI parameters. Its atomic interface drawings
have no execution or resource-count semantics. Physical execution remains with
the existing compiler components and ordinary ``circuit.append(op)``.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from sympy import Symbol

from examples.algorithms.addition_catalog import addition_examples
from examples.algorithms.comparison_catalog import comparison_examples
from examples.algorithms.integer_multiplication_catalog import (
    integer_multiplication_examples,
)
from examples.algorithms.negation_catalog import negation_examples
from examples.algorithms.permutation_catalog import permutation_examples
from examples.algorithms.sorting_catalog import sorting_examples
from examples.algorithms.subtraction_catalog import subtraction_examples
from examples.algorithms.trigonometric_catalog import arcsine_example
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.sorting.in_place_sort_declaration import (
    in_place_sort_signature,
)
from flagquantum.compiler.registers.sorting.sorted_duplicates import (
    sorted_duplicates_signature,
)
from flagquantum.compiler.registers.sorting.symmetric_difference_declaration import (
    symmetric_difference_signature,
)
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import (
    ones_complement_register_type,
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.component_ui_export import (
    component_filename,
    export_component_ui,
)
from tools.development.notebook_autogen import NotebookSpec


@dataclass(frozen=True)
class _Interface:
    # A UI-only association is needed because native functions, catalog tuples
    # and declarations do not share an operation superclass or parameter API.
    operation: str
    owner: str
    ports: Signature
    parameters: Mapping[str, object]


def _instances() -> dict[str, _Interface]:
    instances: dict[str, _Interface] = {}
    for identifier, call, ports in addition_examples():
        if call.operation == "add":
            operation = "Add"
            parameters: Mapping[str, object] = {
                "a_dtype": {"bitsize": ports[0].bitsize},
                "b_dtype": {"bitsize": ports[1].bitsize},
            }
        elif call.operation == "out_of_place_add":
            operation = "OutOfPlaceAdder"
            parameters = {
                "bitsize": call.widths[0],
                "is_adjoint": call.adjoint,
                "include_most_significant_bit": call.include_most_significant_bit,
            }
        else:
            operation = "AddK"
            parameters = {"dtype": {"bitsize": call.widths[0]}, "k": call.constant}
        instances[identifier] = _Interface(
            operation, addition_examples.__module__, ports, parameters
        )

    for (identifier, ports, _), alias in zip(
        negation_examples(), ("negate", "negate_symb"), strict=True
    ):
        instances[alias] = _Interface(
            "Negate",
            negation_examples.__module__,
            ports,
            {"dtype": {"bitsize": ports[0].bitsize}},
        )

    subtraction_aliases = (
        "sub_symb",
        "sub_small",
        "sub_large",
        "sub_diff_size_regs",
        "sub_symp_decomposition",
        "sub_from_symb",
        "sub_from_small",
        "sub_from_large",
    )
    for (_, ports, _), identifier in zip(
        subtraction_examples(), subtraction_aliases, strict=True
    ):
        operation = "SubtractFrom" if identifier.startswith("sub_from") else "Subtract"
        if identifier == "sub_symp_decomposition":
            # This factory returns a composite, whose source attrs are empty.
            operation, parameters = "CompositeBloq", {}
        elif operation == "SubtractFrom":
            parameters = {"dtype": {"bitsize": ports[0].bitsize}}
        else:
            parameters = {
                "a_dtype": {"bitsize": ports[0].bitsize},
                "b_dtype": {"bitsize": ports[1].bitsize},
            }
        instances[identifier] = _Interface(
            operation, subtraction_examples.__module__, ports, parameters
        )

    products = (
        (
            "plus_equal_product",
            "PlusEqualProduct",
            {"a_bitsize": 2, "b_bitsize": 2, "result_bitsize": 4, "is_adjoint": False},
        ),
        ("product", "Product", {"a_bitsize": 4, "b_bitsize": 6}),
        ("square", "Square", {"bitsize": 8, "uncompute": False}),
        ("sum_of_squares", "SumOfSquares", {"bitsize": 8, "k": 4}),
        ("scale_int_by_real", "ScaleIntByReal", {"r_bitsize": 12, "i_bitsize": 4}),
        ("multiply_two_reals", "MultiplyTwoReals", {"bitsize": 10}),
        ("square_real_number", "SquareRealNumber", {"bitsize": 10}),
        ("invert_real_number", "InvertRealNumber", {"bitsize": 10, "num_frac": 7}),
    )
    for (_, ports, _), (identifier, operation, parameters) in zip(
        integer_multiplication_examples(), products, strict=True
    ):
        instances[identifier] = _Interface(
            operation, integer_multiplication_examples.__module__, ports, parameters
        )

    n, k = Symbol("n"), Symbol("k")
    n1, n2 = Symbol("n1"), Symbol("n2")
    comparisons: dict[str, tuple[str, dict[str, object]]] = {
        "lt_k": ("LessThanConstant", {"bitsize": 8, "less_than_val": 5}),
        "lt_k_symb": ("LessThanConstant", {"bitsize": n, "less_than_val": k}),
        "greater_than": ("GreaterThan", {"a_bitsize": 4, "b_bitsize": 4}),
        "gt_k": ("GreaterThanConstant", {"bitsize": 4, "val": 13}),
        "equals": ("Equals", {"dtype": {"bitsize": 4}}),
        "eq_k": ("EqualsAConstant", {"bitsize": 4, "val": 13}),
        "bi_qubits_mixer": ("BiQubitsMixer", {"is_adjoint": False}),
        "sq_cmp": ("SingleQubitCompare", {"is_adjoint": False}),
        "leq": ("LessThanEqual", {"x_bitsize": 4, "y_bitsize": 8}),
        "leq_symb": ("LessThanEqual", {"x_bitsize": n1, "y_bitsize": n2}),
        "clineardepthgreaterthan_example": (
            "CLinearDepthGreaterThan",
            {"dtype": {"bitsize": 5}, "cv": 1},
        ),
    }
    for suffix, operation, symbol in (
        ("greaterthan", "LinearDepthHalfGreaterThan", ">"),
        ("greaterthanequal", "LinearDepthHalfGreaterThanEqual", ">="),
        ("lessthan", "LinearDepthHalfLessThan", "<"),
        ("lessthanequal", "LinearDepthHalfLessThanEqual", "<="),
    ):
        comparisons[f"lineardepthhalf{suffix}_small"] = (
            operation,
            {"dtype": {"bitsize": 3}, "uncompute": False, "_op_symbol": symbol},
        )
    for identifier, ports, _ in comparison_examples():
        operation, parameters = comparisons[identifier]
        instances[identifier] = _Interface(
            operation, comparison_examples.__module__, ports, parameters
        )

    sorts = (
        ("comparator", "Comparator", {"bitsize": 3}),
        ("comparator_symb", "Comparator", {"bitsize": n}),
        ("bitonic_sort", "BitonicSort", {"k": 8, "bitsize": 4}),
        ("bitonic_sort_symb", "BitonicSort", {"k": 4, "bitsize": n}),
        (
            "parallel_compare",
            "ParallelComparators",
            {"k": 7, "offset": 2, "bitsize": 3},
        ),
        ("bitonic_merge", "BitonicMerge", {"half_length": 4, "bitsize": 7}),
    )
    for (_, ports, _), (identifier, operation, parameters) in zip(
        sorting_examples(), sorts, strict=True
    ):
        instances[identifier] = _Interface(
            operation, sorting_examples.__module__, ports, parameters
        )

    size = Symbol("N", positive=True, integer=True)
    count = Symbol("k", positive=True, integer=True)
    length = Symbol("L", positive=True, integer=True)
    lower_n = Symbol("n", positive=True, integer=True)
    permutations = (
        ("permutation", 4),
        ("permutation_symb", size),
        ("permutation_symb_with_cycles", size),
        ("sparse_permutation", 16),
        ("sparse_permutation_with_symbolic_N", size),
        ("permutation_cycle_symb_N", lower_n),
        ("permutation_cycle_symb", size),
        ("permutation_cycle", 4),
    )
    for (_, ports, cycle_values, _), (identifier, bound) in zip(
        permutation_examples(), permutations, strict=True
    ):
        values: object = cycle_values
        cycle = identifier.startswith("permutation_cycle")
        operation = "PermutationCycle" if cycle else "Permutation"
        if identifier == "permutation_symb":
            values = {"shape": (count,)}
        elif identifier == "permutation_symb_with_cycles":
            assert isinstance(values, tuple)
            values = tuple({"shape": (value,)} for value in values)
        elif identifier == "permutation_cycle_symb":
            values = {"shape": (length,)}
        parameters = {
            "N": bound,
            "cycle" if cycle else "cycles": values,
            "bitsize": ports[0].bitsize,
        }
        instances[identifier] = _Interface(
            operation, permutation_examples.__module__, ports, parameters
        )

    _, ports, _ = arcsine_example()
    instances["arcsin"] = _Interface(
        "ArcSin",
        arcsine_example.__module__,
        ports,
        {"bitsize": 10, "num_frac": 7, "num_iters": 4, "degree": 4},
    )

    for identifier, width in (("cadd_small", 3), ("cadd_large", 1000)):
        ports = Signature(
            (
                Register("ctrl", quantum_bit_type()),
                Register("a", unsigned_register_type(width)),
                Register("b", unsigned_register_type(width)),
            )
        )
        instances[identifier] = _Interface(
            "CAdd",
            "flagquantum.compiler.registers.arithmetic.controlled_addition_component",
            ports,
            {"a_dtype": {"bitsize": width}, "b_dtype": {"bitsize": width}, "cv": 1},
        )
    for identifier, dtype in (
        ("ctrl_add_or_sub_signed_symb", signed_register_type(n)),
        ("ctrl_add_or_sub_unsigned", unsigned_register_type(8)),
        ("ctrl_add_or_sub_signed", signed_register_type(8)),
    ):
        ports = Signature(
            (
                Register("ctrl", quantum_bit_type()),
                Register("a", dtype),
                Register("b", dtype),
            )
        )
        # Retain nominal encoding. The source attrs erase it and collide for
        # signed and unsigned width-eight instances of this operation.
        parameters = {
            "a_dtype": {"bitsize": dtype.quantum_bits, "encoding": dtype.encoding},
            "b_dtype": {"bitsize": dtype.quantum_bits, "encoding": dtype.encoding},
            "add_when_ctrl_is_on": True,
        }
        instances[identifier] = _Interface(
            "ControlledAddOrSubtract",
            "flagquantum.compiler.registers.arithmetic.conditional_addition_component",
            ports,
            parameters,
        )

    for identifier, operation, inp, out in (
        (
            "signed_to_twos",
            "SignedIntegerToTwosComplement",
            ones_complement_register_type(10),
            signed_register_type(10),
        ),
        (
            "sign_extend",
            "SignExtend",
            signed_register_type(8),
            signed_register_type(16),
        ),
        (
            "sign_truncate",
            "SignTruncate",
            signed_register_type(16),
            signed_register_type(8),
        ),
    ):
        ports = Signature(
            (
                Register("x", inp, side=RegisterSide.LEFT),
                Register("y", out, side=RegisterSide.RIGHT),
            )
        )
        parameters = (
            {"bitsize": 10}
            if identifier == "signed_to_twos"
            else {
                "inp_dtype": {"bitsize": inp.quantum_bits},
                "out_dtype": {"bitsize": out.quantum_bits},
            }
        )
        instances[identifier] = _Interface(
            operation, "examples.notebooks.register_conversions", ports, parameters
        )

    for identifier, operation, width in (
        ("xor", "Xor", 4),
        ("xor_symb", "Xor", n),
        ("bitwise_not", "BitwiseNot", 4),
        ("bitwise_not_symb", "BitwiseNot", n),
    ):
        dtype = (
            opaque_register_type(width)
            if operation == "Xor"
            else unsigned_register_type(width)
        )
        ports = Signature(
            tuple(
                Register(port, dtype)
                for port in (("x", "y") if operation == "Xor" else ("x",))
            )
        )
        instances[identifier] = _Interface(
            operation,
            "examples.notebooks.register_bitwise",
            ports,
            {"dtype": {"bitsize": width}},
        )
    ports = Signature(
        tuple(
            Register(port, unsigned_register_type(width))
            for port, width in (("mu", 4), ("nu", 4), ("s", 8))
        )
    )
    instances["to_contg_index"] = _Interface(
        "ToContiguousIndex",
        "flagquantum.compiler.registers.sorting.contiguous_index",
        ports,
        {"bitsize": 4, "s_bitsize": 8},
    )
    instances["symm_diff"] = _Interface(
        "SymmetricDifference",
        symmetric_difference_signature.__module__,
        symmetric_difference_signature(4, 2, 4, unsigned_register_type(4)),
        {"n_lhs": 4, "n_rhs": 2, "n_diff": 4, "dtype": {"bitsize": 4}},
    )
    for identifier, width in (("has_duplicates_symb", n), ("has_duplicates", 3)):
        instances[identifier] = _Interface(
            "HasDuplicates",
            sorted_duplicates_signature.__module__,
            sorted_duplicates_signature(4, unsigned_register_type(width)),
            {"l": 4, "dtype": {"bitsize": width}, "is_controlled": False},
        )
    return instances


INSTANCES = _instances()

# Original documentation and example order, including SortInPlace's empty group.
DOCUMENTS = (
    (
        "Addition",
        "register_addition",
        "Add",
        ("add_symb", "add_small", "add_large", "add_diff_size_regs"),
    ),
    (
        "Addition",
        "register_addition",
        "OutOfPlaceAdder",
        ("add_oop_symb", "add_oop_small", "add_oop_large"),
    ),
    ("Addition", "register_addition", "AddK", ("add_k", "add_k_small", "add_k_large")),
    (
        "Controlled Addition",
        "register_controlled_addition",
        "CAdd",
        ("cadd_small", "cadd_large"),
    ),
    ("Negation", "register_negation", "Negate", ("negate", "negate_symb")),
    (
        "Subtraction",
        "register_subtraction",
        "Subtract",
        (
            "sub_symb",
            "sub_small",
            "sub_large",
            "sub_diff_size_regs",
            "sub_symp_decomposition",
        ),
    ),
    (
        "Subtraction",
        "register_subtraction",
        "SubtractFrom",
        ("sub_from_symb", "sub_from_small", "sub_from_large"),
    ),
    (
        "Controlled Add-or-Subtract",
        "register_conditional_arithmetic",
        "ControlledAddOrSubtract",
        (
            "ctrl_add_or_sub_signed_symb",
            "ctrl_add_or_sub_unsigned",
            "ctrl_add_or_sub_signed",
        ),
    ),
    (
        "Multiplication",
        "register_multiplication",
        "PlusEqualProduct",
        ("plus_equal_product",),
    ),
    ("Multiplication", "register_multiplication", "Product", ("product",)),
    ("Multiplication", "register_multiplication", "Square", ("square",)),
    ("Multiplication", "register_multiplication", "SumOfSquares", ("sum_of_squares",)),
    (
        "Multiplication",
        "register_multiplication",
        "ScaleIntByReal",
        ("scale_int_by_real",),
    ),
    (
        "Multiplication",
        "register_multiplication",
        "MultiplyTwoReals",
        ("multiply_two_reals",),
    ),
    (
        "Multiplication",
        "register_multiplication",
        "SquareRealNumber",
        ("square_real_number",),
    ),
    (
        "Multiplication",
        "register_multiplication",
        "InvertRealNumber",
        ("invert_real_number",),
    ),
    ("Comparison", "register_comparison", "LessThanConstant", ("lt_k", "lt_k_symb")),
    ("Comparison", "register_comparison", "GreaterThan", ("greater_than",)),
    ("Comparison", "register_comparison", "GreaterThanConstant", ("gt_k",)),
    ("Comparison", "register_comparison", "Equals", ("equals",)),
    ("Comparison", "register_comparison", "EqualsAConstant", ("eq_k",)),
    ("Comparison", "register_comparison", "BiQubitsMixer", ("bi_qubits_mixer",)),
    ("Comparison", "register_comparison", "SingleQubitCompare", ("sq_cmp",)),
    ("Comparison", "register_comparison", "LessThanEqual", ("leq", "leq_symb")),
    (
        "Comparison",
        "register_comparison",
        "CLinearDepthGreaterThan",
        ("clineardepthgreaterthan_example",),
    ),
    (
        "Comparison",
        "register_comparison",
        "LinearDepthHalfGreaterThan",
        ("lineardepthhalfgreaterthan_small",),
    ),
    (
        "Comparison",
        "register_comparison",
        "LinearDepthHalfGreaterThanEqual",
        ("lineardepthhalfgreaterthanequal_small",),
    ),
    (
        "Comparison",
        "register_comparison",
        "LinearDepthHalfLessThan",
        ("lineardepthhalflessthan_small",),
    ),
    (
        "Comparison",
        "register_comparison",
        "LinearDepthHalfLessThanEqual",
        ("lineardepthhalflessthanequal_small",),
    ),
    (
        "Integer Conversions",
        "register_conversions",
        "SignedIntegerToTwosComplement",
        ("signed_to_twos",),
    ),
    ("Integer Conversions", "register_conversions", "SignExtend", ("sign_extend",)),
    ("Integer Conversions", "register_conversions", "SignTruncate", ("sign_truncate",)),
    ("Sorting", "register_sorting", "Comparator", ("comparator", "comparator_symb")),
    ("Sorting", "register_sorting", "ParallelComparators", ("parallel_compare",)),
    ("Sorting", "register_sorting", "BitonicMerge", ("bitonic_merge",)),
    (
        "Sorting",
        "register_sorting",
        "BitonicSort",
        ("bitonic_sort", "bitonic_sort_symb"),
    ),
    ("Indexing", "triangular_register_index", "ToContiguousIndex", ("to_contg_index",)),
    (
        "Permutations",
        "register_permutation",
        "Permutation",
        (
            "permutation",
            "permutation_symb",
            "permutation_symb_with_cycles",
            "sparse_permutation",
            "sparse_permutation_with_symbolic_N",
        ),
    ),
    (
        "Permutations",
        "register_permutation",
        "PermutationCycle",
        ("permutation_cycle_symb_N", "permutation_cycle_symb", "permutation_cycle"),
    ),
    ("Bitwise Operations", "register_bitwise", "Xor", ("xor", "xor_symb")),
    (
        "Bitwise Operations",
        "register_bitwise",
        "BitwiseNot",
        ("bitwise_not", "bitwise_not_symb"),
    ),
    ("Trigonometric Functions", "register_trigonometric", "ArcSin", ("arcsin",)),
    ("List Functions", "register_lists", "SortInPlace", ()),
    ("List Functions", "register_lists", "SymmetricDifference", ("symm_diff",)),
    (
        "List Functions",
        "register_lists",
        "HasDuplicates",
        ("has_duplicates_symb", "has_duplicates"),
    ),
)


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError("Arithmetic UI requires a registered native example identifier")
    return INSTANCES[component]


def name(component: object) -> str:
    return _instance(component).operation


def attributes(component: object) -> Mapping[str, object]:
    return _instance(component).parameters


def signature(component: object) -> Signature:
    return _instance(component).ports


def filename(component: object) -> str:
    return component_filename(attributes(component))


def score(component: object) -> object:
    ports = signature(component)
    leaf = FunctionRecord(name(component), (), registers=tuple(ports))
    builder = ComponentProgramBuilder(ports, (leaf,), name="ArithmeticInterface")
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    return register_score_data(component_register_score(record, "ArithmeticInterface"))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    if not documentation.examples:
        return "SortInPlace"
    # The composite example belongs to the Subtract documentation group.
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = (
        _instance(documentation.examples[0].make()).owner
        if documentation.examples
        else in_place_sort_signature.__module__
    )
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "Diagrams show atomic typed interfaces. Complete decomposition glyphs "
        "and resource models retain separate acceptance evidence. "
        "Symbolic and large instances allocate no execution wires.\n"
    )


def _identity(identifier: str) -> object:
    return identifier


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, NotebookSpec] = {}
    for title, stem, _, identifiers in DOCUMENTS:
        if title not in grouped:
            grouped[title] = NotebookSpec(
                title=title,
                module=sys.modules[__name__],
                component_specs=[],
                directory=str(Path(__file__).resolve().parents[1] / "notebooks"),
                path_stem=stem,
            )
        grouped[title].component_specs.append(
            ComponentDocumentation(
                component_type=object,
                examples=tuple(
                    ComponentExample(partial(_identity, identifier), identifier, object)
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.arithmetic_ui_export import sections",
            )
        )
    return (("Arithmetic", tuple(grouped.values())),)


def export_example(destination: Path) -> None:
    """Publish the complete Arithmetic catalog with fresh typed port diagrams."""
    export_component_ui(
        destination,
        sections(),
        dict.fromkeys(INSTANCES, ()),
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )


if __name__ == "__main__":
    export_example(Path("arithmetic_ui_export"))
