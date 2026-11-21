# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Rotations notebook catalog.

"""Export native rotation, Fourier, phase-window and walk interfaces.

Run ``python -m examples.algorithms.rotations_ui_export``. Large chemistry
instances retain declaration-only interfaces and their seeded metadata.
"""

import sys
from collections.abc import Mapping
from functools import partial
from math import pi
from pathlib import Path
from typing import TypeAlias

import numpy as np
from sympy import Symbol

from examples.algorithms.approximate_qft_catalog import approximate_qft_examples
from examples.algorithms.basic_gate_child_decomposition import (
    child_decomposition as _child_presentation,
)
from examples.algorithms.component_ui_dependencies import (
    ComponentFunction,
    dependency_graph,
)
from examples.algorithms.hamming_phasing_catalog import HammingPhasingExample
from examples.algorithms.hubbard_qubitization_resources import notebook_signature
from examples.algorithms.phase_gradient_catalog import (
    GradientAdditionExample,
    GradientStateExample,
    GradientUnitaryExample,
)
from examples.algorithms.programmable_rotation_array_catalog import (
    ProgrammableRotationArrayExample,
)
from examples.algorithms.qvr_catalog import QvrExample
from examples.algorithms.rotation_gate_decomposition import (
    rotation_gate_program,
    rotation_gate_score,
)
from examples.algorithms.rz_phase_gradient_catalog import GradientRzExample
from examples.algorithms.sparse_chemistry_resources import sparse_integral_fixture
from examples.algorithms.square_cost_phasing_catalog import SquareCostPhasingExample
from examples.algorithms.textbook_qft_catalog import textbook_qft_examples
from examples.algorithms.zpow_phase_gradient_catalog import GradientZPowExample
from flagquantum.algorithms.primitives.approximate_qft_resource_model import (
    source_approximate_qft_signature,
)
from flagquantum.algorithms.primitives.fermionic_fourier import (
    source_two_bit_fermionic_fourier_signature,
)
from flagquantum.algorithms.primitives.ising_walk import ising_walk
from flagquantum.algorithms.primitives.kaiser_window import KaiserWindowState
from flagquantum.algorithms.primitives.luis_perina_state import (
    LuisPerinaInterim,
    LuisPerinaState,
)
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_signature,
)
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    PauliHamiltonianWalk,
)
from flagquantum.algorithms.primitives.phase_estimation_window import (
    PhaseEstimationWindow,
    RectangularWindowState,
)
from flagquantum.algorithms.primitives.qft_phase_gradient_resource_model import (
    source_phase_gradient_qft_signature,
)
from flagquantum.algorithms.primitives.qft_resource_model import (
    source_textbook_qft_signature,
)
from flagquantum.algorithms.primitives.qubitization_walk import source_walk_signature
from flagquantum.algorithms.primitives.sparse_chemistry_walk import SparseChemistryWalk
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.compiler.registers.lookup.source_qroam_clean import (
    optimal_qroam_clean_log_block_size,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
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


def _fixed_parameters(width: SymbolicInt) -> Mapping[str, object]:
    return {"bitsize": width, "num_frac": width, "signed": False}


def _cost_parameters(name: str, width: int) -> Mapping[str, object]:
    return {
        "name": name,
        "dtype": _fixed_parameters(width),
        "_shape": (),
        "side": RegisterSide.THRU,
    }


def _basic_rotation_instances() -> dict[str, _Interface]:
    q = Signature.build(q=1)
    owner = "flagquantum.compiler.registers.primitives.pauli_rotation"
    result: dict[str, _Interface] = {
        "z_pow": ("ZPowGate", q, {"exponent": 0.123, "eps": 1e-8}, owner),
        "cz_pow": (
            "CZPowGate",
            Signature((Register("q", quantum_bit_type(), (2,)),)),
            {"exponent": 0.123, "eps": 1e-11},
            "flagquantum.compiler.registers.primitives.controlled_rotation",
        ),
        "rz": ("Rz", q, {"angle": Symbol("a"), "eps": 1e-11}, owner),
        "crz": (
            "CRz",
            Signature.build(ctrl=1, q=1),
            {"angle": Symbol(r"\theta"), "eps": 1e-11},
            "flagquantum.compiler.registers.primitives.controlled_rotation",
        ),
    }
    for axis in ("x", "y"):
        result[axis + "_pow"] = (
            axis.upper() + "PowGate",
            q,
            {"exponent": 0.123, "global_shift": 0.0, "eps": 1e-8},
            owner,
        )
    for identifier, theta, phi, lambd, shift in (
        ("su2_rotation_gate", pi / 4, pi / 2, pi / 2, 0),
        ("hadamard", pi / 4, 0, 0, 0),
        ("t_gate", 0, 3 * pi / 4, 0, -3 * pi / 4),
    ):
        result[identifier] = (
            "SU2RotationGate",
            q,
            {
                "theta": theta,
                "phi": phi,
                "lambd": lambd,
                "global_shift": shift,
                "eps": 1e-11,
            },
            "flagquantum.compiler.registers.phase.su2_rotation",
        )
    return result


def _cost_phasing_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for gradient in (False, True):
        variable = QvrExample(via_gradient=gradient, epsilon=1e-4 if gradient else 1e-2)
        identifier = "qvr_phase_gradient" if gradient else "qvr_zpow"
        result[identifier] = (
            "QvrPhaseGradient" if gradient else "QvrZPow",
            variable.signature,
            {
                "cost_reg": _cost_parameters("x", variable.width),
                "gamma": variable.coefficient,
                "eps": variable.epsilon,
            },
            type(variable).__module__,
        )
        square = SquareCostPhasingExample(via_gradient=gradient)
        result[
            "square_via_phase_gradient" if gradient else "square_via_zpow_phasing"
        ] = (
            "PhasingViaCostFunction",
            square.signature,
            {
                "cost_eval_oracle": {"bitsize": square.width, "uncompute": False},
                "phase_oracle": {
                    "cost_reg": _cost_parameters("result", 2 * square.width),
                    "gamma": square.coefficient,
                    "eps": square.epsilon,
                },
            },
            type(square).__module__,
        )
        hamming = HammingPhasingExample(via_gradient=gradient)
        result[
            (
                "hamming_weight_phasing_via_phase_gradient"
                if gradient
                else "hamming_weight_phasing"
            )
        ] = (
            (
                "HammingWeightPhasingViaPhaseGradient"
                if gradient
                else "HammingWeightPhasing"
            ),
            hamming.signature,
            {
                "bitsize": hamming.width,
                "exponent": hamming.exponent,
                "eps": hamming.epsilon,
            },
            type(hamming).__module__,
        )
    return result


def _gradient_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, width in (
        ("phase_gradient_unitary", 4),
        ("phase_gradient_unitary_symbolic", Symbol("n")),
    ):
        gradient_unitary = GradientUnitaryExample(width)
        result[identifier] = (
            "PhaseGradientUnitary",
            gradient_unitary.signature,
            {
                "bitsize": width,
                "exponent": 1,
                "is_controlled": False,
                "eps": gradient_unitary.epsilon,
            },
            type(gradient_unitary).__module__,
        )
    state = GradientStateExample()
    result["phase_gradient_state"] = (
        "PhaseGradientState",
        state.signature,
        {"bitsize": state.width, "exponent": -1, "eps": 1e-10},
        type(state).__module__,
    )
    for scaled in (False, True):
        addition = GradientAdditionExample(scaled)
        params = (
            {
                "x_dtype": _fixed_parameters(addition.width),
                "phase_bitsize": addition.width,
                "gamma": 2,
                "gamma_dtype": _fixed_parameters(2),
            }
            if scaled
            else {
                "x_bitsize": addition.width,
                "phase_bitsize": addition.width,
                "right_shift": 0,
                "sign": 1,
                "controlled_by": None,
            }
        )
        result["add_scaled_val_into_phase_reg" if scaled else "add_into_phase_grad"] = (
            "AddScaledValIntoPhaseReg" if scaled else "AddIntoPhaseGrad",
            addition.signature,
            params,
            type(addition).__module__,
        )
    for identifier, exponent, precision in (
        ("zpow_const_via_phase_grad", 3 / 8, 1e-11),
        ("zpow_const_via_phase_grad_symb_prec", 3 / 8, Symbol("eps")),
        ("zpow_const_via_phase_grad_symb_angle", Symbol("t"), 1e-11),
    ):
        zpow = GradientZPowExample.from_precision(exponent, epsilon=precision)
        result[identifier] = (
            "ZPowConstViaPhaseGradient",
            zpow.signature,
            {"exponent": zpow.exponent, "phase_grad_bitsize": zpow.width},
            type(zpow).__module__,
        )
    rz = GradientRzExample()
    result["rz_via_phase_gradient"] = (
        "RzViaPhaseGradient",
        rz.signature,
        {
            "angle_dtype": _fixed_parameters(rz.angle_width),
            "phasegrad_dtype": _fixed_parameters(rz.gradient_width),
        },
        type(rz).__module__,
    )
    array = ProgrammableRotationArrayExample()
    result["programmable_rotation_gate_array"] = (
        "ProgrammableRotationGateArray",
        array.signature,
        {},
        type(array).__module__,
    )
    return result


def _fourier_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {
        "two_bit_ffft": (
            "TwoBitFFFT",
            source_two_bit_fermionic_fourier_signature(),
            {"k": 2, "n": 3, "eps": 1e-10, "is_adjoint": False},
            source_two_bit_fermionic_fourier_signature.__module__,
        )
    }
    for identifier, (width, precision) in zip(
        ("approximate_qft_small", "approximate_qft_from_epsilon"),
        approximate_qft_examples(),
        strict=True,
    ):
        result[identifier] = (
            "ApproximateQFT",
            source_approximate_qft_signature(width, precision),
            {"bitsize": width, "phase_bitsize": precision, "with_reverse": True},
            source_approximate_qft_signature.__module__,
        )
    for identifier, width in zip(
        ("qft_text_book", "symbolic_qft"), textbook_qft_examples(), strict=True
    ):
        result[identifier] = (
            "QFTTextBook",
            source_textbook_qft_signature(width),
            {"bitsize": width, "with_reverse": True},
            source_textbook_qft_signature.__module__,
        )
    result["qft_phase_gradient_small"] = (
        "QFTPhaseGradient",
        source_phase_gradient_qft_signature(3),
        {"bitsize": 3, "with_reverse": True},
        source_phase_gradient_qft_signature.__module__,
    )
    return result


def _window_instances() -> dict[str, _Interface]:
    interim = LuisPerinaInterim(5)
    result: dict[str, _Interface] = {
        "lprs_interim_prep": (
            "LPRSInterimPrep",
            interim.signature,
            {"bitsize": interim.bitsize, "eps": interim.eps},
            type(interim).__module__,
        )
    }
    for symbolic in (False, True):
        width = Symbol("n") if symbolic else 5
        for label, stem, cls in (
            ("LPResourceState", "lp_resource_state", LuisPerinaState),
            (
                "RectangularWindowState",
                "rectangular_window_state",
                RectangularWindowState,
            ),
        ):
            window = cls(width)
            identifier = stem + ("_symbolic" if symbolic else "_small")
            result[identifier] = (
                label,
                window.signature,
                {"bitsize": window.bitsize},
                type(window).__module__,
            )
        kaiser = KaiserWindowState(width, Symbol("alpha") if symbolic else 2)
        result[
            "kaiser_window_state_symbolic" if symbolic else "kaiser_window_state_small"
        ] = (
            "KaiserWindowState",
            kaiser.signature,
            {"bitsize": kaiser.bitsize, "alpha": kaiser.alpha},
            type(kaiser).__module__,
        )
    theta = Symbol("theta")
    windows = (
        ("textbook_qpe_small", 0.468, RectangularWindowState(3)),
        ("textbook_qpe_using_m_bits", 2 * theta, RectangularWindowState(Symbol("m"))),
        (
            "textbook_qpe_from_standard_deviation_eps",
            2 * theta,
            RectangularWindowState.from_standard_deviation_eps(Symbol("epsilon")),
        ),
        (
            "textbook_qpe_from_precision_and_delta",
            2 * theta,
            RectangularWindowState.from_precision_and_delta(
                Symbol("n"), Symbol("delta")
            ),
        ),
    )
    for identifier, exponent, window in windows:
        result[identifier] = (
            "TextbookQPE",
            Signature((*window.signature, *Signature.build(q=1))),
            {
                "unitary": {"exponent": exponent, "eps": 1e-11},
                **_qpe_parameters(window),
            },
            "flagquantum.algorithms.primitives.textbook_phase_estimation",
        )
    return result


def _qpe_parameters(window: PhaseEstimationWindow) -> Mapping[str, object]:
    return {
        "ctrl_state_prep": {"bitsize": window.m_bits},
        "qft_inv": {"subbloq": {"bitsize": window.m_bits, "with_reverse": True}},
    }


def _ising_instance(epsilon: float) -> tuple[Signature, Mapping[str, object]]:
    walk, _ = ising_walk(4, epsilon)
    assert isinstance(walk, PauliHamiltonianWalk)
    shape = AliasPreparationShape.from_table(walk.table)
    ports = Signature(
        (shape.selection_registers[0], Register("target", opaque_register_type(4)))
    )
    rows = tuple(
        {
            "pauli": word.lstrip("-"),
            "coefficient": -1 + 0j if word.startswith("-") else 1 + 0j,
        }
        for word in walk.selector.select_unitaries
    )
    selection = shape.selection_registers[0]
    params = {
        "block_encoding": {
            "select": {
                "selection_bitsize": walk.selector.selection_bitsize,
                "target_bitsize": walk.selector.target_bitsize,
                "select_unitaries": rows,
                "control_val": None,
            },
            "prepare": {
                "selection_registers": (
                    {
                        "name": selection.name,
                        "dtype": {
                            "bitsize": selection.bitsize,
                            "iteration_length": walk.table.selection_size,
                        },
                        "_shape": (),
                        "side": selection.side,
                    },
                ),
                "alt": np.asarray(walk.table.alternates),
                "keep": np.asarray(walk.table.keep),
                "mu": walk.table.mu,
                "sum_of_unnormalized_probabilities": float(walk.table.normalization),
            },
        }
    }
    return ports, params


def _sparse_instance(
    spatial: int, probability_bits: int
) -> tuple[Signature, Mapping[str, object], float]:
    one, two = sparse_integral_fixture(spatial)
    walk = SparseChemistryWalk.from_integrals(
        one, two, probability_bits=probability_bits, rotation_bits=8
    )
    data = walk.prepare.data
    cols = dict(
        zip((name for name, _ in data.target_layout), data.datasets, strict=True)
    )
    prepare = {
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
    params = {
        "block_encoding": {
            "select": {"num_spin_orb": walk.num_spin_orb, "control_val": None},
            "prepare": prepare,
        }
    }
    norm = float(np.sum(np.abs(one)) + 0.5 * np.sum(np.abs(two)))
    return walk.source_signature, params, norm


def _thc_instance() -> tuple[Signature, Mapping[str, object], float]:
    rng = np.random.RandomState(7)
    one = rng.normal(size=(76, 76))
    t_l = np.linalg.eigvalsh(0.5 * (one + one.T))
    central = rng.normal(size=(450, 450))
    central = 0.5 * (central + central.T)
    eta = rng.normal(size=(450, 76))
    data = thc_chemistry_data(t_l, eta, central, probability_bits=10)
    preparation = PrepareTHCChemistry(data)
    ports = tuple(select_thc_signature(450, 152, 10))
    signature = source_walk_signature(
        ports[:-2], preparation.selection_registers, ports[-2:], ()
    )
    cols = dict(
        zip((name for name, _ in data.target_layout), data.source_datasets, strict=True)
    )
    block = np.power(2, optimal_qroam_clean_log_block_size(526, 1))
    params = {
        "block_encoding": {
            "select": {
                "num_mu": 450,
                "num_spin_orb": 152,
                "num_bits_theta": 20,
                "keep_bitsize": 10,
                "kr1": block,
                "kr2": block,
                "control_val": None,
            },
            "prepare": {
                "num_mu": 450,
                "num_spin_orb": 152,
                "alt_mu": cols["alt_mu"],
                "alt_nu": cols["alt_nu"],
                "alt_theta": cols["alt_theta"],
                "theta": cols["theta"],
                "keep": cols["keep"],
                "keep_bitsize": 10,
                "sum_of_l1_coeffs": data.normalization,
                "log_block_size": data.log_block_size,
            },
        }
    }
    return signature, params, data.normalization


def _walk_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    ising_ports, ising_params = _ising_instance(0.2)
    result["walk_op"] = (
        "QubitizationWalkOperator",
        ising_ports,
        ising_params,
        ising_walk.__module__,
    )
    sparse_ports, sparse_params, _ = _sparse_instance(4, 12)
    result["walk_op_chem_sparse"] = (
        "QubitizationWalkOperator",
        sparse_ports,
        sparse_params,
        SparseChemistryWalk.__module__,
    )
    thc_ports, thc_params, thc_norm = _thc_instance()
    result["thc_walk_op"] = (
        "QubitizationWalkOperator",
        thc_ports,
        thc_params,
        PrepareTHCChemistry.__module__,
    )
    hubbard_params = {
        "block_encoding": {
            "select": {"x_dim": 2, "y_dim": 2, "control_val": None},
            "prepare": {"x_dim": 2, "y_dim": 2, "t": 2, "u": 8},
        }
    }
    sparse_ports, sparse_params, sparse_norm = _sparse_instance(3, 16)
    ising_ports, ising_params = _ising_instance(0.1)
    for identifier, ports, params, window in (
        (
            "qubitization_qpe_hubbard_model_small",
            notebook_signature(2),
            hubbard_params,
            LuisPerinaState.from_standard_deviation_eps(0.02 / (64 * np.sqrt(2))),
        ),
        (
            "qubitization_qpe_sparse_chem",
            sparse_ports,
            sparse_params,
            LuisPerinaState.from_standard_deviation_eps(
                0.0016 / (sparse_norm * np.sqrt(2))
            ),
        ),
        (
            "qubitization_qpe_chem_thc",
            thc_ports,
            thc_params,
            LuisPerinaState.from_standard_deviation_eps(0.0016 / (thc_norm * 2**0.5)),
        ),
        (
            "qubitization_qpe_ising",
            ising_ports,
            ising_params,
            RectangularWindowState(4),
        ),
    ):
        result[identifier] = (
            "QubitizationQPE",
            Signature((*window.signature, *ports)),
            {"walk": params, **_qpe_parameters(window)},
            "flagquantum.algorithms.primitives.qubitization_phase_estimation",
        )
    return result


DOCUMENTS = (
    ("Rotations", "Basic Rotation Gates", "register_rotation", "ZPowGate", ("z_pow",)),
    (
        "Rotations",
        "Basic Rotation Gates",
        "register_rotation",
        "CZPowGate",
        ("cz_pow",),
    ),
    ("Rotations", "Basic Rotation Gates", "register_rotation", "Rz", ("rz",)),
    ("Rotations", "Basic Rotation Gates", "register_rotation", "CRz", ("crz",)),
    ("Rotations", "Basic Rotation Gates", "register_rotation", "XPowGate", ("x_pow",)),
    ("Rotations", "Basic Rotation Gates", "register_rotation", "YPowGate", ("y_pow",)),
    (
        "Rotations",
        "SU2 Rotation",
        "register_su2_rotation",
        "SU2RotationGate",
        ("su2_rotation_gate", "hadamard", "t_gate"),
    ),
    (
        "Rotations",
        "Quantum Variable Rotation",
        "quantum_variable_rotation",
        "QvrZPow",
        ("qvr_zpow",),
    ),
    (
        "Rotations",
        "Quantum Variable Rotation",
        "quantum_variable_rotation",
        "QvrPhaseGradient",
        ("qvr_phase_gradient",),
    ),
    (
        "Rotations",
        "Phasing via Cost function",
        "phasing_via_cost_function",
        "PhasingViaCostFunction",
        ("square_via_phase_gradient", "square_via_zpow_phasing"),
    ),
    (
        "Rotations",
        "Rotations via Phase Gradients",
        "phase_gradient",
        "PhaseGradientUnitary",
        ("phase_gradient_unitary", "phase_gradient_unitary_symbolic"),
    ),
    (
        "Rotations",
        "Rotations via Phase Gradients",
        "phase_gradient",
        "PhaseGradientState",
        ("phase_gradient_state",),
    ),
    (
        "Rotations",
        "Rotations via Phase Gradients",
        "phase_gradient",
        "AddIntoPhaseGrad",
        ("add_into_phase_grad",),
    ),
    (
        "Rotations",
        "Rotations via Phase Gradients",
        "phase_gradient",
        "AddScaledValIntoPhaseReg",
        ("add_scaled_val_into_phase_reg",),
    ),
    (
        "Rotations",
        "Z Rotations via Hamming Weight Phasing",
        "hamming_weight_phasing",
        "HammingWeightPhasing",
        ("hamming_weight_phasing",),
    ),
    (
        "Rotations",
        "Z Rotations via Hamming Weight Phasing",
        "hamming_weight_phasing",
        "HammingWeightPhasingViaPhaseGradient",
        ("hamming_weight_phasing_via_phase_gradient",),
    ),
    (
        "Rotations",
        "ZPow Rotation via Phase Gradient",
        "zpow_via_phase_gradient",
        "ZPowConstViaPhaseGradient",
        (
            "zpow_const_via_phase_grad",
            "zpow_const_via_phase_grad_symb_prec",
            "zpow_const_via_phase_grad_symb_angle",
        ),
    ),
    (
        "Rotations",
        "Rz Rotation via Phase Gradient",
        "rz_via_phase_gradient",
        "RzViaPhaseGradient",
        ("rz_via_phase_gradient",),
    ),
    (
        "Rotations",
        "Programmable Rotation Gate Array",
        "programmable_rotation_gate_array",
        "ProgrammableRotationGateArray",
        ("programmable_rotation_gate_array",),
    ),
    (
        "Rotations",
        "Two Bit FFFT Gate",
        "two_bit_fermionic_fourier",
        "TwoBitFFFT",
        ("two_bit_ffft",),
    ),
    (
        "Rotations",
        "Approximate QFT",
        "approximate_qft",
        "ApproximateQFT",
        ("approximate_qft_small", "approximate_qft_from_epsilon"),
    ),
    (
        "Rotations",
        "Textbook QFT",
        "textbook_qft",
        "QFTTextBook",
        ("qft_text_book", "symbolic_qft"),
    ),
    (
        "Rotations",
        "Phase Gradient QFT",
        "phase_gradient_qft",
        "QFTPhaseGradient",
        ("qft_phase_gradient_small",),
    ),
    (
        "Rotations",
        "Optimal resource states for Phase Estimation by A. Luis and J. Peřina",
        "luis_perina_phase_window",
        "LPRSInterimPrep",
        ("lprs_interim_prep",),
    ),
    (
        "Rotations",
        "Optimal resource states for Phase Estimation by A. Luis and J. Peřina",
        "luis_perina_phase_window",
        "LPResourceState",
        ("lp_resource_state_small", "lp_resource_state_symbolic"),
    ),
    (
        "Rotations",
        "Textbook Quantum Phase Estimation",
        "textbook_phase_estimation",
        "RectangularWindowState",
        ("rectangular_window_state_small", "rectangular_window_state_symbolic"),
    ),
    (
        "Rotations",
        "Textbook Quantum Phase Estimation",
        "textbook_phase_estimation",
        "TextbookQPE",
        (
            "textbook_qpe_small",
            "textbook_qpe_using_m_bits",
            "textbook_qpe_from_standard_deviation_eps",
            "textbook_qpe_from_precision_and_delta",
        ),
    ),
    (
        "Rotations",
        "Kaiser Window State for Quantum Phase Estimation",
        "kaiser_phase_window",
        "KaiserWindowState",
        ("kaiser_window_state_small", "kaiser_window_state_symbolic"),
    ),
    (
        "Rotations",
        "Qubitization Walk Operator",
        "qubitization_walk",
        "QubitizationWalkOperator",
        ("walk_op", "thc_walk_op", "walk_op_chem_sparse"),
    ),
    (
        "Rotations",
        "Qubitization Phase Estimation",
        "qubitization_phase_estimation",
        "QubitizationQPE",
        (
            "qubitization_qpe_hubbard_model_small",
            "qubitization_qpe_sparse_chem",
            "qubitization_qpe_chem_thc",
            "qubitization_qpe_ising",
        ),
    ),
)

_CANDIDATES = {
    **_basic_rotation_instances(),
    **_cost_phasing_instances(),
    **_gradient_instances(),
    **_fourier_instances(),
    **_window_instances(),
    **_walk_instances(),
}
INSTANCES = {
    identifier: _CANDIDATES[identifier]
    for _, _, _, _, identifiers in DOCUMENTS
    for identifier in identifiers
}


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Rotations catalog UI requires a registered native example identifier"
        )
    return INSTANCES[component]


def name(component: object) -> str:
    return (
        "ComponentFunction"
        if isinstance(component, ComponentFunction)
        else _instance(component)[0]
    )


def signature(component: object) -> Signature:
    return _instance(component)[1]


def attributes(component: object) -> Mapping[str, object]:
    return (
        component.attributes()
        if isinstance(component, ComponentFunction)
        else _instance(component)[2]
    )


def filename(component: object) -> str:
    return (
        component.filename()
        if isinstance(component, ComponentFunction)
        else component_filename(attributes(component))
    )


def score(component: object) -> object:
    if isinstance(component, ComponentFunction):
        return component.score()
    result = rotation_gate_score(name(component), attributes(component))
    return (
        result
        if result is not None
        else register_interface_score(signature(component), name(component))
    )


def decomposition(component: object) -> ProgramRecord | None:
    """Reuse controlled-rotation calls while retaining other source boundaries."""
    return rotation_gate_program(name(component), attributes(component))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    if documentation.examples:
        return name(documentation.examples[0].make())
    raise ValueError("Rotation documentation requires a registered example")


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = (
        _instance(documentation.examples[0].make())[3]
        if documentation.examples
        else documentation.component_type.__module__
    )
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "Controlled rotations retain complete stored dependency graphs. "
        "Other operations retain their atomic typed interfaces and separate evidence. "
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
                    ComponentExample(partial(_identity, identifier), identifier, object)
                    for identifier in identifiers
                ),
                import_line="from examples.algorithms.rotations_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete rotation, Fourier and phase-estimation metadata and fresh interfaces."""
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
    export_example(Path("rotations_ui_export"))
