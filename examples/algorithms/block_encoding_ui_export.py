# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Block Encoding notebook catalog.

"""Export native block-encoding interfaces and their original factory metadata.

Run ``python -m examples.algorithms.block_encoding_ui_export``. Symbolic examples
use existing declaration signatures, while concrete cases use native components.
"""

import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TypeAlias

import numpy as np
from sympy import Symbol, symbols

from examples.algorithms.hubbard_lcu_catalog import HubbardLCUExample
from examples.algorithms.linear_combination_catalog import linear_combination_example
from examples.algorithms.product_block_catalog import product_child
from flagquantum.algorithms.primitives._block_metadata import _BlockDeclaration
from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
    ChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
    UniformSqrtEntryOracle,
    source_hermitian_sparse_signature,
)
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
    source_product_block_signature,
)
from flagquantum.algorithms.primitives.scaled_chebyshev_block_encoding import (
    ScaledChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.select_oracle import SelectOracle
from flagquantum.algorithms.primitives.sparse_matrix_block_encoding import (
    SparseMatrixBlockEncoding,
    source_sparse_matrix_signature,
)
from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
    ExplicitEntryOracle,
    SymmetricBandedRowColumnOracle,
    TopLeftRowColumnOracle,
    UniformEntryOracle,
)
from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
    source_tensor_product_block_signature,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
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

_Interface: TypeAlias = tuple[str, Signature, Mapping[str, object], str]


def _unitary_parameters(
    block: UnitaryBlockEncoding, *, phase_gate: bool = False
) -> dict[str, object]:
    return {
        "U": {"is_adjoint": False} if phase_gate else {},
        "alpha": block.alpha,
        "ancilla_bitsize": block.ancilla_bitsize,
        "resource_bitsize": block.resource_bitsize,
        "epsilon": block.epsilon,
    }


def _controlled_not(
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return (Instruction("cx", (*ports["control"], *ports["target"])),)


def _unitary_instances() -> dict[str, _Interface]:
    t, h = (replace(product_child(gate), alpha=1, epsilon=0) for gate in ("t", "h"))
    t_properties = replace(
        t, alpha=0.5, ancilla_bitsize=2, resource_bitsize=1, epsilon=0.01
    )
    h_properties = replace(
        h, alpha=0.5, ancilla_bitsize=1, resource_bitsize=1, epsilon=0.1
    )
    cx_properties = UnitaryBlockEncoding(
        (("control", 1), ("target", 1)),
        _controlled_not,
        alpha=0.5,
        ancilla_bitsize=1,
        resource_bitsize=1,
        epsilon=0.1,
    )
    result: dict[str, _Interface] = {}
    for identifier, block in (
        ("unitary_block_encoding", t),
        ("unitary_block_encoding_properties", t_properties),
    ):
        result[identifier] = (
            "Unitary",
            block.signature,
            _unitary_parameters(block, phase_gate=True),
            type(block).__module__,
        )
    for identifier, left, right, tensor in (
        ("tensor_product_block_encoding", t, h, True),
        ("tensor_product_block_encoding_properties", t_properties, cx_properties, True),
        ("product_block_encoding", t, h, False),
        ("product_block_encoding_properties", t_properties, h_properties, False),
    ):
        encoding = (
            TensorProductBlockEncoding((left, right))
            if tensor
            else ProductBlockEncoding((left, right))
        )
        result[identifier] = (
            "TensorProduct" if tensor else "Product",
            encoding.signature,
            {
                "block_encodings": (
                    _unitary_parameters(left, phase_gate=True),
                    _unitary_parameters(right),
                )
            },
            type(encoding).__module__,
        )
    alpha1, alpha2, a1, a2, eps1, eps2 = symbols("alpha1 alpha2 a1 a2 eps1 eps2")
    symbolic_children = tuple(
        {
            "U": gate,
            "alpha": alpha,
            "ancilla_bitsize": ancilla,
            "resource_bitsize": 0,
            "epsilon": epsilon,
        }
        for gate, alpha, ancilla, epsilon in (
            ({"is_adjoint": False}, alpha1, a1, eps1),
            ({}, alpha2, a2, eps2),
        )
    )
    result["tensor_product_block_encoding_symb"] = (
        "TensorProduct",
        source_tensor_product_block_signature(((1, a1, 0), (1, a2, 0))),
        {"block_encodings": symbolic_children},
        source_tensor_product_block_signature.__module__,
    )
    result["product_block_encoding_symb"] = (
        "Product",
        source_product_block_signature(1, (a1, a2), (0, 0)),
        {"block_encodings": symbolic_children},
        source_product_block_signature.__module__,
    )
    phase = PhaseBlockEncoding(h, 0.25, 0)
    result["phase_block_encoding"] = (
        "Phase",
        phase.signature,
        {"block_encoding": _unitary_parameters(h), "phi": phase.phi, "eps": phase.eps},
        type(phase).__module__,
    )
    linear = linear_combination_example()
    child_parameters = []
    for index, child in enumerate(linear.block_encodings):
        assert isinstance(child, UnitaryBlockEncoding)
        child_parameters.append(_unitary_parameters(child, phase_gate=index == 0))
    result["linear_combination_block_encoding"] = (
        "LinearCombination",
        linear.signature,
        {
            "_block_encodings": tuple(child_parameters),
            "_lambd": linear.coefficients,
            "lambd_bits": linear.coefficient_bits,
            "_prepare": None,
            "_select": None,
            "is_controlled": False,
        },
        type(linear).__module__,
    )
    return result


def _polynomial_instances() -> dict[str, _Interface]:
    x, h = (replace(product_child(gate), alpha=1, epsilon=0) for gate in ("x", "h"))
    result: dict[str, _Interface] = {}
    for identifier, coefficients, scaled, order, alpha in (
        ("chebyshev_poly_even", (0.5, 0.5), False, 4, 1),
        ("chebyshev_poly_odd", (), False, 5, 1),
        ("scaled_chebyshev_poly_even", (1.0, 1.0), True, 4, 1),
        ("scaled_chebyshev_poly_odd", (), True, 5, 3.14),
    ):
        child: LinearCombinationBlockEncoding | UnitaryBlockEncoding
        if coefficients:
            child = LinearCombinationBlockEncoding((x, h), coefficients, 1)
            parameters: dict[str, object] = {
                "_block_encodings": (_unitary_parameters(x), _unitary_parameters(h)),
                "_lambd": child.coefficients,
                "lambd_bits": child.coefficient_bits,
                "_prepare": None,
                "_select": None,
                "is_controlled": False,
            }
        else:
            child = replace(h, alpha=alpha)
            parameters = _unitary_parameters(child)
        encoding = (
            ScaledChebyshevBlockEncoding(child, order)
            if scaled
            else ChebyshevBlockEncoding(child, order)
        )
        metadata: dict[str, object] = {
            "block_encoding": parameters,
            "order": encoding.order,
        }
        if isinstance(encoding, ScaledChebyshevBlockEncoding):
            metadata["lambd_bits"] = encoding.coefficient_bits
        result[identifier] = (
            "ScaledChebyshevPolynomial" if scaled else "ChebyshevPolynomial",
            encoding.signature,
            metadata,
            type(encoding).__module__,
        )
    return result


def _sparse_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, width, data, banded in (
        ("sparse_matrix_block_encoding", 2, False, False),
        ("explicit_matrix_block_encoding", 1, True, False),
        ("symmetric_banded_matrix_block_encoding", 3, False, True),
    ):
        oracle = (
            SymmetricBandedRowColumnOracle(width, 1)
            if banded
            else TopLeftRowColumnOracle(width)
        )
        entry = (
            ExplicitEntryOracle(width, ((0.0, 0.25), (1 / 3, 0.467)), 10)
            if data
            else UniformEntryOracle(width, 0.3)
        )
        block = SparseMatrixBlockEncoding(oracle, oracle, entry)
        row: dict[str, object] = {"system_bitsize": oracle.system_bitsize}
        row.update(
            {"bandsize": oracle.bandsize}
            if isinstance(oracle, SymmetricBandedRowColumnOracle)
            else {"_num_nonzero": oracle.num_nonzero}
        )
        values: dict[str, object] = {"system_bitsize": entry.system_bitsize}
        if isinstance(entry, ExplicitEntryOracle):
            values.update(
                data=np.asarray(entry.data), entry_bitsize=entry.entry_bitsize
            )
        else:
            values["entry"] = entry.entry
        result[identifier] = (
            "SparseMatrix",
            block.signature,
            {
                "row_oracle": row,
                "col_oracle": row,
                "entry_oracle": values,
                "eps": block.eps,
            },
            type(block).__module__,
        )
    n = Symbol("n", positive=True, integer=True)
    row = {"system_bitsize": n, "_num_nonzero": 2**n}
    result["sparse_matrix_symb_block_encoding"] = (
        "SparseMatrix",
        source_sparse_matrix_signature(n),
        {
            "row_oracle": row,
            "col_oracle": row,
            "entry_oracle": {"system_bitsize": n, "entry": 0.3},
            "eps": 0,
        },
        source_sparse_matrix_signature.__module__,
    )
    column = TopLeftRowColumnOracle(2)
    sqrt_entry = UniformSqrtEntryOracle(2, 0.3)
    hermitian = HermitianSparseMatrixBlockEncoding(column, sqrt_entry)
    result["sparse_matrix_hermitian_block_encoding"] = (
        "SparseMatrixHermitian",
        hermitian.signature,
        {
            "col_oracle": {
                "system_bitsize": column.system_bitsize,
                "_num_nonzero": column.num_nonzero,
            },
            "entry_oracle": {
                "system_bitsize": sqrt_entry.system_bitsize,
                "entry": sqrt_entry.entry,
                "eps": sqrt_entry.epsilon,
            },
            "eps": hermitian.eps,
        },
        type(hermitian).__module__,
    )
    result["sparse_matrix_symb_hermitian_block_encoding"] = (
        "SparseMatrixHermitian",
        source_hermitian_sparse_signature(n),
        {
            "col_oracle": row,
            "entry_oracle": {"system_bitsize": n, "entry": 0.3, "eps": 1e-11},
            "eps": 0,
        },
        source_hermitian_sparse_signature.__module__,
    )
    return result


def _oracle_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for packed in (False, True):
        example = HubbardLCUExample(packed=packed)
        select: dict[str, object] = {
            "x_dim": example.side,
            "y_dim": example.side,
            "control_val": None,
        }
        prepare: dict[str, object] = {
            "x_dim": example.side,
            "y_dim": example.side,
            "t": 1,
            "u": 4,
        }
        if packed:
            select, prepare = {"select": select}, {"prepare": prepare}
        metadata = {"select": select, "prepare": prepare}
        prefix = "black_box_" if packed else ""
        result[prefix + "select_block"] = (
            "SelectBlockEncoding",
            example.select_signature(),
            metadata,
            HubbardLCUExample.__module__,
        )
        result[prefix + "lcu_block"] = (
            "LCUBlockEncoding",
            example.lcu_signature(),
            {**metadata, "control_val": None},
            HubbardLCUExample.__module__,
        )
    two = HubbardLCUExample(side=2, packed=True)
    result["black_box_select"] = (
        "BlackBoxSelect",
        two.select_signature(),
        {"select": {"x_dim": two.side, "y_dim": two.side, "control_val": None}},
        HubbardLCUExample.__module__,
    )
    result["black_box_prepare"] = (
        "BlackBoxPrepare",
        PackedPrepareDeclaration(two.preparation).signature,
        {"prepare": {"x_dim": two.side, "y_dim": two.side, "t": 1, "u": 4}},
        PackedPrepareDeclaration.__module__,
    )
    return result


DOCUMENTS = (
    (
        "Block Encoding",
        "Block Encoding Interface",
        "block_encoding_interface",
        "BlockEncoding",
        (),
    ),
    (
        "Block Encoding",
        "Unitary",
        "register_unitary_block_encoding",
        "Unitary",
        ("unitary_block_encoding", "unitary_block_encoding_properties"),
    ),
    (
        "Block Encoding",
        "Tensor Product",
        "register_tensor_product_block_encoding",
        "TensorProduct",
        (
            "tensor_product_block_encoding",
            "tensor_product_block_encoding_properties",
            "tensor_product_block_encoding_symb",
        ),
    ),
    (
        "Block Encoding",
        "Product",
        "register_product_block_encoding",
        "Product",
        (
            "product_block_encoding",
            "product_block_encoding_properties",
            "product_block_encoding_symb",
        ),
    ),
    (
        "Block Encoding",
        "Phase",
        "register_phase_block_encoding",
        "Phase",
        ("phase_block_encoding",),
    ),
    (
        "Block Encoding",
        "Linear Combination",
        "register_linear_combination",
        "LinearCombination",
        ("linear_combination_block_encoding",),
    ),
    (
        "Block Encoding",
        "Sparse Matrix",
        "register_sparse_matrix_block_encoding",
        "SparseMatrix",
        (
            "sparse_matrix_block_encoding",
            "sparse_matrix_symb_block_encoding",
            "explicit_matrix_block_encoding",
            "symmetric_banded_matrix_block_encoding",
        ),
    ),
    (
        "Block Encoding",
        "Sparse Matrix Hermitian",
        "register_hermitian_sparse_matrix",
        "SparseMatrixHermitian",
        (
            "sparse_matrix_symb_hermitian_block_encoding",
            "sparse_matrix_hermitian_block_encoding",
        ),
    ),
    (
        "Block Encoding",
        "Chebyshev Polynomial",
        "register_chebyshev",
        "ChebyshevPolynomial",
        ("chebyshev_poly_even", "chebyshev_poly_odd"),
    ),
    (
        "Block Encoding",
        "Chebyshev Polynomial",
        "register_chebyshev",
        "ScaledChebyshevPolynomial",
        ("scaled_chebyshev_poly_even", "scaled_chebyshev_poly_odd"),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "SelectBlockEncoding",
        ("select_block", "black_box_select_block"),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "LCUBlockEncoding",
        ("lcu_block", "black_box_lcu_block"),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "SelectOracle",
        (),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "PrepareOracle",
        (),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "BlackBoxSelect",
        ("black_box_select",),
    ),
    (
        "Block Encoding",
        "LCU Select/Prepare Oracles",
        "register_lcu_block_encoding",
        "BlackBoxPrepare",
        ("black_box_prepare",),
    ),
)

_CANDIDATES = {
    **_unitary_instances(),
    **_sparse_instances(),
    **_polynomial_instances(),
    **_oracle_instances(),
}
INSTANCES = {
    identifier: _CANDIDATES[identifier]
    for _, _, _, _, identifiers in DOCUMENTS
    for identifier in identifiers
}
_EMPTY_DOCUMENTS = {
    "BlockEncoding": _BlockDeclaration,
    "SelectOracle": SelectOracle,
    "PrepareOracle": PrepareOracle,
}


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Block encoding UI requires a registered native example identifier"
        )
    return INSTANCES[component]


def name(component: object) -> str:
    return _instance(component)[0]


def signature(component: object) -> Signature:
    return _instance(component)[1]


def attributes(component: object) -> Mapping[str, object]:
    return _instance(component)[2]


def filename(component: object) -> str:
    return component_filename(attributes(component))


def score(component: object) -> object:
    return register_interface_score(signature(component), name(component))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    if documentation.examples:
        return name(documentation.examples[0].make())
    return next(
        label
        for label, owner in _EMPTY_DOCUMENTS.items()
        if documentation.component_type is owner
    )


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = (
        _instance(documentation.examples[0].make())[3]
        if documentation.examples
        else documentation.component_type.__module__
    )
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "This diagram shows the atomic typed interface. Complete decomposition "
        "graphs, glyphs, execution and resource estimates retain separate evidence. "
        "Large and symbolic instances allocate no execution wires.\n"
    )


def _identity(identifier: str) -> object:
    return identifier


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, dict[str, NotebookSpec]] = {}
    for section, title, stem, label, identifiers in DOCUMENTS:
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
                component_type=_EMPTY_DOCUMENTS.get(label, object),
                examples=tuple(
                    ComponentExample(partial(_identity, identifier), identifier, object)
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.block_encoding_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(
    destination: Path,
    *,
    documentation_name: Callable[
        [ComponentDocumentation[object]], str
    ] = documentation_name,
    relative_example_paths: bool = False,
) -> None:
    """Publish complete block-encoding metadata and fresh interfaces."""
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
        relative_example_paths=relative_example_paths,
    )


if __name__ == "__main__":
    export_example(Path("block_encoding_ui_export"))
