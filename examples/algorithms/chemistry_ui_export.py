# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Chemistry notebook catalog.

"""Export native chemistry metadata and atomic typed interfaces.

Run ``python -m examples.algorithms.chemistry_ui_export``. Execution and
analytical resource evidence remain with each owning component family.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from sympy import Symbol

from examples.algorithms.double_factorization_resources import notebook_signatures
from examples.algorithms.first_quantization_catalog import notebook_cases
from examples.algorithms.first_quantization_projectile_catalog import (
    notebook_cases as projectile_cases,
)
from examples.algorithms.hubbard_qubitization_resources import (
    notebook_consumer_signature,
    notebook_signature,
)
from examples.algorithms.sparse_chemistry_resources import sparse_integral_fixture
from examples.algorithms.thc_chemistry_resources import thc_tutorial_integrals
from flagquantum.algorithms.primitives.gradient_givens import gradient_givens_signature
from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
    grid_inverse_sqrt_qrom_data,
    source_grid_inverse_sqrt_newton_signature,
    source_grid_inverse_sqrt_polynomial_signature,
)
from flagquantum.algorithms.primitives.grid_kinetic import source_grid_kinetic_signature
from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_signature,
    source_grid_potential_signature,
)
from flagquantum.algorithms.primitives.grid_qvr import source_grid_qvr_signature
from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_plaquette_signature,
    source_hubbard_hopping_tile_signature,
    source_hubbard_interaction_signature,
)
from flagquantum.algorithms.primitives.ising_evolution import (
    IsingXUnitary,
    IsingZZUnitary,
)
from flagquantum.algorithms.primitives.multiplexers.select_sparse import SelectSparse
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_signature,
)
from flagquantum.algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_signature,
    source_single_factorization_one_body_signature,
)
from flagquantum.algorithms.primitives.sparse_chemistry_walk import SparseChemistryWalk
from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.algorithms.primitives.thc_uniform_preparation import (
    THCUniformPreparation,
)
from flagquantum.algorithms.primitives.trotter_product import TrotterProduct
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
_ROOT = "flagquantum.algorithms.primitives."


def _sparse_instances() -> dict[str, _Interface]:
    one, two = sparse_integral_fixture(3)
    walk = SparseChemistryWalk.from_integrals(
        one, two, probability_bits=4, rotation_bits=8
    )
    data = walk.prepare.data
    cols = dict(
        zip((name for name, _ in data.target_layout), data.datasets, strict=True)
    )
    params = {
        "num_spin_orb": walk.num_spin_orb,
        "num_non_zero": len(data.indices),
        "num_bits_state_prep": data.probability_bits,
        "alt_pqrs": tuple(cols["alt_" + p] for p in "pqrs"),
        "alt_theta": cols["alt_theta"],
        "alt_one_body": cols["alt_flag_1b"],
        "ind_pqrs": tuple(cols[p] for p in "pqrs"),
        "theta": cols["theta"],
        "one_body": cols["flag_1b"],
        "keep": cols["keep"],
        "num_bits_rot_aa": data.rotation_bits,
        "sum_of_l1_coeffs": data.normalization,
        "log_block_size": data.log_block_size,
    }
    return {
        "prep_sparse": (
            "PrepareSparse",
            walk.prepare.signature,
            params,
            _ROOT + "sparse_chemistry_prepare",
        ),
        "sel_sparse": (
            "SelectSparse",
            SelectSparse.source_signature(4),
            {"num_spin_orb": 4, "control_val": None},
            _ROOT + "multiplexers.select_sparse",
        ),
    }


def _factorization_instances() -> dict[str, _Interface]:
    sf = {"num_aux": 50, "num_spin_orb": 10, "num_bits_state_prep": 12}
    blocks = {"kp1": 1, "kp2": 1, "kp1_inv": 1, "kp2_inv": 1}
    df = {**sf, "num_eig": 250, "num_bits_rot": 7}
    df_one, df_block = notebook_signatures()
    return {
        "sf_one_body": (
            "SingleFactorizationOneBody",
            source_single_factorization_one_body_signature(50, 10),
            {**sf, "num_bits_rot_aa": 7, "is_adjoint": False, **blocks},
            _ROOT + "single_factorization_encoding_model",
        ),
        "sf_block_encoding": (
            "SingleFactorizationBlockEncoding",
            source_single_factorization_block_signature(50, 10),
            {**sf, "num_bits_rot_aa_outer": 7, "num_bits_rot_aa_inner": 7, **blocks},
            _ROOT + "single_factorization_encoding_model",
        ),
        "df_one_body": (
            "DoubleFactorizationOneBody",
            df_one,
            {**df, "num_bits_rot_aa": 8},
            _ROOT + "double_factorization_encoding_model",
        ),
        "df_block_encoding": (
            "DoubleFactorizationBlockEncoding",
            df_block,
            {**df, "num_bits_rot_aa_outer": 1, "num_bits_rot_aa_inner": 7},
            _ROOT + "double_factorization_encoding_model",
        ),
    }


def _thc_instances() -> dict[str, _Interface]:
    data = thc_chemistry_data(
        *thc_tutorial_integrals(), probability_bits=8, log_block_size=2
    )
    prep = PrepareTHCChemistry(data)
    cols = dict(
        zip((name for name, _ in data.target_layout), data.source_datasets, strict=True)
    )
    params = {
        "num_mu": 8,
        "num_spin_orb": 8,
        **{
            name: cols[name]
            for name in ("alt_mu", "alt_nu", "alt_theta", "theta", "keep")
        },
        "keep_bitsize": 8,
        "sum_of_l1_coeffs": data.normalization,
        "log_block_size": data.log_block_size,
    }
    return {
        "thc_uni": (
            "UniformSuperpositionTHC",
            THCUniformPreparation(10, 4).source_signature,
            {"num_mu": 10, "num_spin_orb": 4},
            _ROOT + "thc_uniform_preparation",
        ),
        "thc_prep": (
            "PrepareTHC",
            prep.source_signature,
            params,
            _ROOT + "thc_chemistry_prepare",
        ),
        "thc_sel": (
            "SelectTHC",
            select_thc_signature(10, 8, 10),
            {
                "num_mu": 10,
                "num_spin_orb": 8,
                "num_bits_theta": 12,
                "keep_bitsize": 10,
                "kr1": 1,
                "kr2": 1,
                "control_val": None,
            },
            _ROOT + "multiplexers.thc_resource_model",
        ),
    }


def _first_quantization_instances() -> dict[str, _Interface]:
    cases, projected = notebook_cases(), projectile_cases()
    shared = {
        "num_bits_p": 6,
        "eta": 10,
        "num_atoms": 10,
        "lambda_zeta": 10,
        "m_param": 256,
        "num_bits_nuc_pos": 16,
        "num_bits_t": 16,
        "num_bits_rot_aa": 8,
    }
    owner = "flagquantum.algorithms.first_quantization_resource_model"
    result: dict[str, _Interface] = {
        "prep_first_quant": (
            "PrepareFirstQuantization",
            cases["prepare"][0],
            {**shared, "sum_of_l1_coeffs": None},
            owner,
        ),
        "sel_first_quant": (
            "SelectFirstQuantization",
            cases["select"][0],
            shared,
            owner,
        ),
        "projectile.prep_first_quant": (
            "PrepareFirstQuantizationWithProj",
            projected["prepare"][0],
            {**shared, "num_bits_n": 8},
            "flagquantum.algorithms.first_quantization_projectile_resource_model",
        ),
        "projectile.sel_first_quant": (
            "SelectFirstQuantizationWithProj",
            projected["select"][0],
            {**shared, "num_bits_n": 8},
            "flagquantum.algorithms.first_quantization_projectile_resource_model",
        ),
    }
    for identifier, label, case, params in (
        (
            "prepare_t",
            "PrepareTFirstQuantization",
            "prepare_t",
            {"num_bits_p": 5, "eta": 10, "num_bits_rot_aa": 8},
        ),
        (
            "prepare_uv",
            "PrepareUVFirstQuantization",
            "prepare_uv",
            {
                "num_bits_p": 5,
                "eta": 10,
                "num_atoms": 10,
                "m_param": 256,
                "lambda_zeta": 10,
                "num_bits_nuc_pos": 16,
            },
        ),
        (
            "select_t",
            "SelectTFirstQuantization",
            "select_t",
            {"num_bits_p": 5, "eta": 10},
        ),
        (
            "select_uv",
            "SelectUVFirstQuantization",
            "select_uv",
            {"num_bits_p": 5, "eta": 10, "num_atoms": 10, "num_bits_nuc_pos": 16},
        ),
        (
            "prepare_zeta",
            "PrepareZetaState",
            "zeta",
            {
                "num_atoms": 10,
                "lambda_zeta": 10,
                "num_bits_nuc_pos": 8,
                "is_adjoint": False,
            },
        ),
    ):
        result[identifier] = label, cases[case][0], params, owner
    return result


def _grid_instances() -> dict[str, _Interface]:
    return {
        "poly_inv_sqrt": (
            "PolynmomialEvaluationInverseSquareRoot",
            source_grid_inverse_sqrt_polynomial_signature(7, 8, 12),
            {"x_sq_bitsize": 7, "poly_bitsize": 8, "out_bitsize": 12},
            _ROOT + "grid_inverse_sqrt",
        ),
        "nr_inv_sqrt": (
            "NewtonRaphsonApproxInverseSquareRoot",
            source_grid_inverse_sqrt_newton_signature(7, 8, 12),
            {"x_sq_bitsize": 7, "poly_bitsize": 8, "target_bitsize": 12},
            _ROOT + "grid_inverse_sqrt",
        ),
        "qvr": (
            "QuantumVariableRotation",
            source_grid_qvr_signature(12),
            {"phi_bitsize": 12},
            _ROOT + "grid_qvr",
        ),
        "kinetic_energy": (
            "KineticEnergy",
            source_grid_kinetic_signature(12, 17),
            {"num_elec": 12, "num_grid": 17},
            _ROOT + "grid_kinetic",
        ),
        "pair_potential": (
            "PairPotential",
            source_grid_pair_potential_signature(7),
            {
                "bitsize": 7,
                "qrom_data": grid_inverse_sqrt_qrom_data(16, 15),
                "poly_bitsize": 15,
                "inv_sqrt_bitsize": 24,
                "label": "V",
            },
            _ROOT + "grid_potential",
        ),
        "potential_energy": (
            "PotentialEnergy",
            source_grid_potential_signature(12, 17),
            {"num_elec": 12, "num_grid": 17, "poly_bitsize": 15, "label": "V"},
            _ROOT + "grid_potential",
        ),
    }


def _trotter_instances() -> dict[str, _Interface]:
    product = TrotterProduct(
        (IsingXUnitary(3), IsingZZUnitary(3)), (0, 1, 0), (0.05, 2, 0.05), 0.01
    )
    result: dict[str, _Interface] = {
        "trott_unitary": (
            "TrotterizedUnitary",
            product.signature,
            {
                "bloqs": (
                    {"nsites": 3, "angle": 0.001, "eps": 1e-10},
                    {"nsites": 3, "angle": 0.04, "eps": 1e-10},
                ),
                "indices": product.indices,
                "coeffs": product.coefficients,
                "timestep": product.timestep,
            },
            _ROOT + "trotter_product",
        ),
    }
    for identifier, label, operation in (
        ("ising_x", "IsingXUnitary", IsingXUnitary(3)),
        ("ising_zz", "IsingZZUnitary", IsingZZUnitary(3)),
    ):
        result[identifier] = (
            label,
            operation.signature,
            {"nsites": operation.nsites, "angle": 0.04, "eps": operation.eps},
            _ROOT + "ising_evolution",
        )
    for identifier, label, angle in (
        ("hopping_tile", "HoppingTile", 0.5),
        ("hopping_tile_hwp", "HoppingTileHWP", 0.15),
    ):
        result[identifier] = (
            label,
            source_hubbard_hopping_tile_signature(8),
            {"length": 8, "angle": angle, "tau": 1.0, "eps": 1e-9, "pink": True},
            _ROOT + "hubbard_lattice",
        )
    result["plaquette"] = (
        "HoppingPlaquette",
        source_hubbard_hopping_plaquette_signature(),
        {"kappa": 8, "eps": 0.15},
        _ROOT + "hubbard_lattice",
    )
    for identifier, label, angle in (
        ("interaction", "Interaction", 0.5),
        ("interaction_hwp", "InteractionHWP", 0.52728),
    ):
        result[identifier] = (
            label,
            source_hubbard_interaction_signature(8),
            {"length": 8, "angle": angle, "hubb_u": 4.0, "eps": 1e-9},
            _ROOT + "hubbard_lattice",
        )
    return result


def _hubbard_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {
        "sel_hubb": (
            "SelectHubbard",
            notebook_signature(4),
            {"x_dim": 4, "y_dim": 4, "control_val": None},
            _ROOT + "hubbard_select",
        ),
        "prep_hubb": (
            "PrepareHubbard",
            notebook_signature(4, prepare=True),
            {"x_dim": 4, "y_dim": 4, "t": 1.0, "u": 4.0},
            _ROOT + "hubbard_prepare",
        ),
    }
    n = Symbol("n")
    for identifier, label, majorana in (
        ("hubb_majoranna", "HubbardMajorannaOperator", True),
        ("hubb_spin_up_z", "HubbardSpinUpZ", False),
    ):
        for small in (False, True):
            side = 3 if small else n
            params: dict[str, object] = {
                "x_dim": side,
                "y_dim": side,
                "control_val": None if small else 1,
            }
            if majorana:
                params["gate"] = "Y" if small else "X"
            result[identifier + ("_small" if small else "")] = (
                label,
                notebook_consumer_signature(
                    side, majorana=majorana, controlled=not small
                ),
                params,
                _ROOT + "hubbard_select",
            )
    for identifier, label, complex_rotation in (
        ("real_givens", "RealGivensRotationByPhaseGradient", False),
        ("cplx_givens", "ComplexGivensRotationByPhaseGradient", True),
    ):
        result[identifier] = (
            label,
            gradient_givens_signature(4, complex_rotation=complex_rotation),
            {"phasegrad_bitsize": 4},
            _ROOT + "gradient_givens",
        )
    return result


DOCUMENTS = (
    (
        "Chemistry",
        "Sparse",
        "register_sparse_chemistry",
        "PrepareSparse",
        ("prep_sparse",),
    ),
    (
        "Chemistry",
        "Sparse",
        "register_sparse_chemistry",
        "SelectSparse",
        ("sel_sparse",),
    ),
    (
        "Chemistry",
        "Single Factorization",
        "register_single_factorization",
        "SingleFactorizationOneBody",
        ("sf_one_body",),
    ),
    (
        "Chemistry",
        "Single Factorization",
        "register_single_factorization",
        "SingleFactorizationBlockEncoding",
        ("sf_block_encoding",),
    ),
    (
        "Chemistry",
        "Double Factorization",
        "register_double_factorization",
        "DoubleFactorizationOneBody",
        ("df_one_body",),
    ),
    (
        "Chemistry",
        "Double Factorization",
        "register_double_factorization",
        "DoubleFactorizationBlockEncoding",
        ("df_block_encoding",),
    ),
    (
        "Chemistry",
        "Tensor Hypercontraction",
        "register_thc_chemistry",
        "UniformSuperpositionTHC",
        ("thc_uni",),
    ),
    (
        "Chemistry",
        "Tensor Hypercontraction",
        "register_thc_chemistry",
        "PrepareTHC",
        ("thc_prep",),
    ),
    (
        "Chemistry",
        "Tensor Hypercontraction",
        "register_thc_chemistry",
        "SelectTHC",
        ("thc_sel",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "PrepareFirstQuantization",
        ("prep_first_quant",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "SelectFirstQuantization",
        ("sel_first_quant",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "PrepareTFirstQuantization",
        ("prepare_t",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "PrepareUVFirstQuantization",
        ("prepare_uv",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "SelectTFirstQuantization",
        ("select_t",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "SelectUVFirstQuantization",
        ("select_uv",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian",
        "register_first_quantization",
        "PrepareZetaState",
        ("prepare_zeta",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian with Quantum Projectile",
        "register_first_quantization_projectile",
        "PrepareFirstQuantizationWithProj",
        ("projectile.prep_first_quant",),
    ),
    (
        "Chemistry",
        "First Quantized Hamiltonian with Quantum Projectile",
        "register_first_quantization_projectile",
        "SelectFirstQuantizationWithProj",
        ("projectile.sel_first_quant",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "PolynmomialEvaluationInverseSquareRoot",
        ("poly_inv_sqrt",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "NewtonRaphsonApproxInverseSquareRoot",
        ("nr_inv_sqrt",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "QuantumVariableRotation",
        ("qvr",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "KineticEnergy",
        ("kinetic_energy",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "PairPotential",
        ("pair_potential",),
    ),
    (
        "Chemistry",
        "Trotter Bloqs",
        "register_grid_trotter",
        "PotentialEnergy",
        ("potential_energy",),
    ),
    (
        "Chemistry",
        "Trotterization",
        "register_trotter_product",
        "TrotterizedUnitary",
        ("trott_unitary",),
    ),
    (
        "Chemistry",
        "Ising Trotter Bloqs",
        "register_ising_trotter",
        "IsingXUnitary",
        ("ising_x",),
    ),
    (
        "Chemistry",
        "Ising Trotter Bloqs",
        "register_ising_trotter",
        "IsingZZUnitary",
        ("ising_zz",),
    ),
    (
        "Chemistry",
        "Trotterized Hubbard",
        "register_hubbard_trotter",
        "HoppingTile",
        ("hopping_tile",),
    ),
    (
        "Chemistry",
        "Trotterized Hubbard",
        "register_hubbard_trotter",
        "HoppingPlaquette",
        ("plaquette",),
    ),
    (
        "Chemistry",
        "Trotterized Hubbard",
        "register_hubbard_trotter",
        "HoppingTileHWP",
        ("hopping_tile_hwp",),
    ),
    (
        "Chemistry",
        "Trotterized Hubbard",
        "register_hubbard_trotter",
        "Interaction",
        ("interaction",),
    ),
    (
        "Chemistry",
        "Trotterized Hubbard",
        "register_hubbard_trotter",
        "InteractionHWP",
        ("interaction_hwp",),
    ),
    (
        "Chemistry",
        "Qubitized Hubbard",
        "register_hubbard_qubitization",
        "SelectHubbard",
        ("sel_hubb",),
    ),
    (
        "Chemistry",
        "Qubitized Hubbard",
        "register_hubbard_qubitization",
        "PrepareHubbard",
        ("prep_hubb",),
    ),
    (
        "Chemistry",
        "Qubitized Hubbard Select",
        "register_hubbard_select",
        "SelectHubbard",
        ("sel_hubb",),
    ),
    (
        "Chemistry",
        "Qubitized Hubbard Select",
        "register_hubbard_select",
        "HubbardMajorannaOperator",
        ("hubb_majoranna", "hubb_majoranna_small"),
    ),
    (
        "Chemistry",
        "Qubitized Hubbard Select",
        "register_hubbard_select",
        "HubbardSpinUpZ",
        ("hubb_spin_up_z", "hubb_spin_up_z_small"),
    ),
    (
        "Chemistry",
        "Givens Rotations",
        "register_gradient_givens",
        "RealGivensRotationByPhaseGradient",
        ("real_givens",),
    ),
    (
        "Chemistry",
        "Givens Rotations",
        "register_gradient_givens",
        "ComplexGivensRotationByPhaseGradient",
        ("cplx_givens",),
    ),
)

_CANDIDATES = {
    **_sparse_instances(),
    **_factorization_instances(),
    **_thc_instances(),
    **_first_quantization_instances(),
    **_grid_instances(),
    **_trotter_instances(),
    **_hubbard_instances(),
}
INSTANCES = {
    identifier: _CANDIDATES[identifier]
    for _, _, _, _, identifiers in DOCUMENTS
    for identifier in identifiers
}


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Chemistry catalog UI requires a registered native example identifier"
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
    raise ValueError("Chemistry documentation requires a registered example")


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
                component_type=object,
                examples=tuple(
                    ComponentExample(
                        partial(_identity, identifier),
                        identifier.removeprefix("projectile."),
                        object,
                    )
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.chemistry_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete chemistry metadata and fresh interfaces."""
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
    export_example(Path("chemistry_ui_export"))
