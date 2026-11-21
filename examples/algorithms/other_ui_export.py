# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Other notebook catalog.

"""Export native lookup, preparation and signal-processing interfaces.

Run ``python -m examples.algorithms.other_ui_export``. Diagrams expose atomic
nominal interfaces. Existing component owners retain execution and resources.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

import numpy as np
from sympy import Symbol, symbols

from examples.algorithms.alias_preparation_catalog import (
    dense_alias_example,
    sparse_alias_example,
    symbolic_alias_example,
)
from examples.algorithms.hubbard_gqsp import hubbard_gqsp_signature
from examples.algorithms.lth_component_catalog import IndexedComponentExample
from examples.algorithms.multi_pauli_catalog import (
    ControlledPauliExample,
    FanoutCNOTExample,
)
from examples.algorithms.pauli_select_catalog import PauliSelectExample
from examples.algorithms.rotation_preparation_catalog import (
    ORIGINAL_COEFFICIENTS,
)
from examples.algorithms.target_position_catalog import apply_z_to_odd_example
from examples.algorithms.uniform_preparation_catalog import UniformPreparationExample
from flagquantum.algorithms.kikuchi_block_encoding import KikuchiHamiltonian
from flagquantum.algorithms.kikuchi_guiding_state import GuidingState
from flagquantum.algorithms.kxor_instance import KXorInstance
from flagquantum.algorithms.primitives.black_box_prepare import PackedPrepareDeclaration
from flagquantum.algorithms.primitives.generalized_qsp import source_gqsp_signature
from flagquantum.algorithms.primitives.guided_hamiltonian_resource_model import (
    GuidedHamiltonianModel,
    GuidedPhaseEstimationModel,
)
from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
from flagquantum.algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
)
from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_signature as rotation_signature,
)
from flagquantum.compiler.registers.lookup.qroam_clean_metadata import (
    qroam_clean_signature,
)
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import ceil, log2
from flagquantum.simulation.numerics.qsp_complement import qsp_complementary_polynomial
from flagquantum.simulation.value_dimensions import SymbolicSequenceLength
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


def _register_parameters(port: Register) -> Mapping[str, object]:
    dtype: dict[str, object] = {}
    if port.dtype.encoding != "bit":
        dtype["bitsize"] = port.bitsize
    if port.dtype.encoding == "uint" and "bound" in dict(port.dtype.parameters):
        dtype["iteration_length"] = dict(port.dtype.parameters)["bound"]
    return {"name": port.name, "dtype": dtype, "_shape": port.shape, "side": port.side}


def _lookup_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    small = np.arange(5)
    square = np.arange(9).reshape(3, 3)
    large = np.arange(25).reshape(5, 5)
    tables = {
        "qrom_small": QromTable.from_data(small, target_bitsizes=(3,)),
        "qrom_multi_data": QromTable.from_data(
            small, small + 1, target_bitsizes=(3, 4)
        ),
        "qrom_multi_dim": QromTable.from_data(
            square, square + 1, target_bitsizes=(8, 8)
        ),
    }
    n, m, b1, b2, c = symbols("N M b1 b2 c")
    tables["qrom_symb"] = QromTable.from_shape((n, m), (b1, b2), num_controls=c)
    for identifier, table in tables.items():
        result[identifier] = (
            "QROM",
            table.signature,
            _lookup_parameters(table),
            type(table).__module__,
        )
    for clean in (False, True):
        label = "QROAMClean" if clean else "SelectSwapQROM"
        for suffix, data in (("multi_data", small), ("multi_dim", large)):
            width = int(np.max(data)).bit_length()
            table = QromTable.from_data(data, data + 1, target_bitsizes=(width, width))
            logs = (int(clean),) * data.ndim
            params = {
                **_lookup_parameters(table),
                "log_block_sizes": logs,
                "use_dirty_ancilla": not clean,
            }
            ports = (
                qroam_clean_signature(table.shape, table.target_bitsizes, logs)
                if clean
                else table.signature
            )
            identifier = ("qroam_clean_" if clean else "qroam_") + suffix
            result[identifier] = (
                label,
                ports,
                params,
                qroam_clean_signature.__module__ if clean else type(table).__module__,
            )
    # The original factories use the literal assumption key ``integers``.
    n, m, b, b1, b2, k, k1, k2, c = symbols(
        "N M b b1 b2 k k1 k2 c", positive=True, integers=True
    )
    for dirty in (True, False):
        for rank in (1, 2):
            shape, widths, logs = (
                ((n,), (b,), (k,)) if rank == 1 else ((n, m), (b1, b2), (k1, k2))
            )
            table = QromTable.from_shape(shape, widths, num_controls=c)
            params = {
                **_lookup_parameters(table),
                "log_block_sizes": logs,
                "use_dirty_ancilla": dirty,
            }
            identifier = f"qroam_symb_{'dirty' if dirty else 'clean'}_{rank}d"
            result[identifier] = (
                "SelectSwapQROM",
                table.signature,
                params,
                type(table).__module__,
            )
    return result


def _lookup_parameters(table: QromTable) -> dict[str, object]:
    if table.datasets is None:
        data: tuple[object, ...] = tuple(
            {"shape": table.shape + target} for target in table.target_shapes
        )
    else:
        data = tuple(
            np.asarray(values).reshape(tuple(int(n) for n in table.shape + target))
            for values, target in zip(table.datasets, table.target_shapes, strict=True)
        )
    return {
        "data_or_shape": data,
        "selection_bitsizes": table.selection_bitsizes,
        "target_bitsizes": table.target_bitsizes,
        "target_shapes": table.target_shapes,
        "num_controls": table.num_controls,
    }


def _alias_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, example in (
        ("state_prep_alias", dense_alias_example()),
        ("sparse_state_prep_alias", sparse_alias_example()),
        ("sparse_state_prep_alias_from_list", sparse_alias_example(from_dense=True)),
    ):
        table = example.table
        params: dict[str, object] = {
            "selection_registers": tuple(
                _register_parameters(p)
                for p in AliasPreparationShape.from_table(table).selection_registers
            ),
            "alt": np.asarray(table.alternates),
            "keep": np.asarray(table.keep),
            "mu": table.mu,
            "sum_of_unnormalized_probabilities": float(table.normalization),
        }
        if table.indices is not None:
            params["index"] = np.asarray(table.indices)
        label = (
            "StatePreparationAliasSampling"
            if table.indices is None
            else "SparseStatePreparationAliasSampling"
        )
        result[identifier] = (
            label,
            example.signature,
            params,
            type(example).__module__,
        )
    for sparse in (False, True):
        declaration = symbolic_alias_example(sparse=sparse)
        shape = {"shape": (declaration.support_size if sparse else declaration.rows,)}
        params = {
            "selection_registers": tuple(
                _register_parameters(p) for p in declaration.selection_registers
            ),
            "alt": shape,
            "keep": shape,
            "mu": declaration.probability_bits,
            "sum_of_unnormalized_probabilities": declaration.normalization,
        }
        if sparse:
            params["index"] = shape
        identifier = (
            "sparse_state_prep_alias_symb" if sparse else "state_prep_alias_symb"
        )
        result[identifier] = (
            (
                "SparseStatePreparationAliasSampling"
                if sparse
                else "StatePreparationAliasSampling"
            ),
            declaration.signature,
            params,
            type(declaration).__module__,
        )
    return result


def _preparation_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, controls in (("prep_uniform", ()), ("c_prep_uniform", (1,))):
        uniform = UniformPreparationExample(5, controls)
        result[identifier] = (
            "PrepareUniformSuperposition",
            uniform.signature,
            {"n": uniform.count, "cvs": uniform.controls},
            type(uniform).__module__,
        )
    for identifier, widths in (
        ("prepare_identity", (10, 4, 1)),
        ("refl_around_zero", (1, 2, 3)),
    ):
        identity = IdentityPrepare.from_bitsizes(widths)
        params: dict[str, object] = {
            "selection_regs": tuple(
                _register_parameters(p) for p in identity.selection_regs
            )
        }
        if identifier == "prepare_identity":
            result[identifier] = (
                "PrepareIdentity",
                identity.signature,
                params,
                type(identity).__module__,
            )
        else:
            reflection = PreparedStateReflection(identity, 1, -1)
            result[identifier] = (
                "ReflectionUsingPrepare",
                reflection.signature,
                {
                    "prepare_gate": params,
                    "control_val": reflection.control_value,
                    "global_phase": reflection.global_phase,
                    "eps": reflection.precision,
                },
                type(reflection).__module__,
            )
    preparation = AliasPreparationShape(5, 5, 7)
    reflection = PreparedStateReflection(preparation)
    result["refl_using_prep"] = (
        "ReflectionUsingPrepare",
        reflection.signature,
        {
            "prepare_gate": {
                "selection_registers": tuple(
                    _register_parameters(p) for p in preparation.selection_registers
                ),
                "alt": np.arange(5),
                "keep": np.zeros(5, dtype=int),
                "mu": preparation.probability_bits,
                "sum_of_unnormalized_probabilities": preparation.normalization,
            },
            "control_val": reflection.control_value,
            "global_phase": reflection.global_phase,
            "eps": reflection.precision,
        },
        type(reflection).__module__,
    )
    for identifier, coefficients, length, phase in (
        ("state_prep_via_rotation", ORIGINAL_COEFFICIENTS, 8, 2),
        (
            "state_prep_via_rotation_symb",
            {"n": Symbol("L")},
            Symbol("L"),
            Symbol(r"\text{phase_bitsize}"),
        ),
        (
            "state_prep_via_rotation_symb_phasegrad",
            ORIGINAL_COEFFICIENTS,
            8,
            Symbol(r"b_\text{grad}"),
        ),
    ):
        result[identifier] = (
            "StatePreparationViaRotations",
            rotation_signature(length, phase, 0),
            {
                "state_coefficients": coefficients,
                "phase_bitsize": phase,
                "control_bitsize": 0,
                "uncompute": False,
            },
            rotation_signature.__module__,
        )
    return result


def _odd_target_gate(index: int) -> str:
    return apply_z_to_odd_example().paulis[index]


def _pauli_instances() -> dict[str, _Interface]:
    indexed = IndexedComponentExample()
    target = apply_z_to_odd_example()
    select = PauliSelectExample()
    result: dict[str, _Interface] = {
        "apply_z_to_odd": (
            "ApplyGateToLthQubit",
            target.signature,
            {
                "selection_regs": (
                    _register_parameters(target.signature.get_left("selection")),
                ),
                "nth_gate": _odd_target_gate,
                "control_regs": (
                    _register_parameters(target.signature.get_left("control")),
                ),
            },
            type(target).__module__,
        ),
        "apply_lth_bloq": (
            "ApplyLthBloq",
            indexed.signature,
            {
                "ops": np.asarray(("t", "h", "z", "x"), dtype=object),
                "selection_regs": None,
                "control_val": 1,
            },
            type(indexed).__module__,
        ),
        "select_pauli_lcu": (
            "SelectPauliLCU",
            select.signature,
            {
                "selection_bitsize": select.select.selection_bitsize,
                "target_bitsize": select.select.target_bitsize,
                "select_unitaries": tuple(
                    {"pauli": word, "coefficient": 1 + 0j}
                    for word in select.select.select_unitaries
                ),
                "control_val": select.control_value,
            },
            type(select).__module__,
        ),
    }
    for identifier, width in (("c_multi_not_symb", Symbol("n")), ("c_multi_not", 5)):
        fanout = FanoutCNOTExample(width)
        result[identifier] = (
            "MultiTargetCNOT",
            fanout.signature,
            {"bitsize": fanout.targets},
            type(fanout).__module__,
        )
    controlled = ControlledPauliExample("x", (1, 0, 1, 0, 1))
    result["ccpauli"] = (
        "MultiControlX",
        controlled.signature,
        {"cvs": controlled.activation},
        type(controlled).__module__,
    )
    return result


def _signal_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, p, negative_power in (
        ("gqsp", (0.5, 0.5), 0),
        ("gqsp_with_negative_power", (0.5, 0, 0.5), 1),
        ("gqsp_with_large_negative_power", (0.5, 0, 0.5), 5),
    ):
        result[identifier] = (
            "GeneralizedQSP",
            source_gqsp_signature(Signature.build(q=1)),
            {
                "U": {"exponent": 1.0, "global_shift": 0.0, "eps": 1e-11},
                "P": p,
                "Q": tuple(qsp_complementary_polynomial(p)),
                "negative_power": negative_power,
                "precision": 0,
            },
            source_gqsp_signature.__module__,
        )
    tau, time, inverse_precision = symbols(r"\tau t \epsilon^{-1}", positive=True)
    for identifier, hopping, interaction, t, precision in (
        ("hubbard_time_evolution_by_gqsp", 1, 1, 5, 1e-7),
        ("symbolic_hamsim_by_gqsp", tau, 4 * tau, time, 1 / inverse_precision),
    ):
        params = {
            "walk_operator": {
                "block_encoding": {
                    "select": {"x_dim": 2, "y_dim": 2, "control_val": None},
                    "prepare": {"x_dim": 2, "y_dim": 2, "t": hopping, "u": interaction},
                }
            },
            "t": t,
            "precision": precision,
        }
        result[identifier] = (
            "HamiltonianSimulationByGQSP",
            hubbard_gqsp_signature(),
            params,
            hubbard_gqsp_signature.__module__,
        )
    return result


def _kx_parameters(instance: KXorInstance) -> Mapping[str, object]:
    assert isinstance(instance.constraints, SymbolicSequenceLength)
    return {
        "n": instance.n,
        "k": instance.k,
        "constraints": {"n": instance.constraints.length},
        "max_rhs": instance.max_rhs,
    }


def _guided_instances() -> dict[str, _Interface]:
    n, k, c, m, m1, m2 = symbols("n k c m m_1 m_2", positive=True, integer=True)
    zeta = Symbol(r"\zeta", positive=True)
    lambd, alpha, gamma = symbols(r"\lambda \alpha \gamma", positive=True, real=True)
    eps, delta = symbols(r"\epsilon_{PE} \delta_{PE}", positive=True, real=True)
    result: dict[str, _Interface] = {}
    for guided in (True, False):
        guide_count, solve_count = ((1 - zeta) * m, zeta * m) if guided else (m1, m2)
        guide = GuidingState(KXorInstance.symbolic(n, guide_count, k, max_rhs=2), c * k)
        hamiltonian = KikuchiHamiltonian(
            KXorInstance.symbolic(n, solve_count, k, max_rhs=2),
            c * k,
            c * k * ceil(log2(n)),
        )
        packed = PackedPrepareDeclaration(guide)
        model = (
            GuidedHamiltonianModel(
                hamiltonian.system_bitsize,
                hamiltonian.ancilla_bitsize,
                packed.selection_width,
                packed.junk_width,
                hamiltonian.alpha,
                lambd,
                alpha,
                gamma,
            )
            if guided
            else GuidedPhaseEstimationModel(
                hamiltonian.system_bitsize,
                hamiltonian.ancilla_bitsize,
                packed.selection_width,
                packed.junk_width,
                eps,
                delta,
            )
        )
        params = {
            "hamiltonian": {
                "inst": _kx_parameters(hamiltonian.inst),
                "ell": hamiltonian.ell,
                "entry_bitsize": hamiltonian.entry_bitsize,
                "s": hamiltonian.s,
            },
            "guiding_state": {
                "prepare": {
                    "inst": _kx_parameters(guide.inst),
                    "ell": guide.ell,
                    "amplitude_good_part": guide.amplitude_good_part,
                    "eps": guide.eps,
                }
            },
            **(
                {"lambd": lambd, "alpha": alpha, "gamma": gamma}
                if guided
                else {"precision": eps, "fail_prob": delta}
            ),
        }
        identifier = (
            "guided_hamiltonian_symb" if guided else "guided_phase_estimate_symb"
        )
        result[identifier] = (
            "GuidedHamiltonian" if guided else "GuidedHamiltonianPhaseEstimation",
            model.signature,
            params,
            type(model).__module__,
        )
    return result


DOCUMENTS = (
    (
        "Other",
        "Prepare Uniform Superposition",
        "prepare_uniform_superposition",
        "PrepareUniformSuperposition",
        ("prep_uniform", "c_prep_uniform"),
    ),
    (
        "Other",
        "Apply to Lth Target",
        "apply_gate_to_lth_target",
        "ApplyGateToLthQubit",
        ("apply_z_to_odd",),
    ),
    ("Other", "Apply Lth Bloq", "apply_lth_bloq", "ApplyLthBloq", ("apply_lth_bloq",)),
    ("Other", "QROM", "register_qrom", "QROMBase", ()),
    (
        "Other",
        "QROM",
        "register_qrom",
        "QROM",
        ("qrom_small", "qrom_multi_data", "qrom_multi_dim", "qrom_symb"),
    ),
    ("Other", "SelectSwapQROM", "register_select_swap_qrom", "QROMBase", ()),
    (
        "Other",
        "SelectSwapQROM",
        "register_select_swap_qrom",
        "SelectSwapQROM",
        (
            "qroam_multi_data",
            "qroam_multi_dim",
            "qroam_symb_dirty_1d",
            "qroam_symb_dirty_2d",
            "qroam_symb_clean_1d",
            "qroam_symb_clean_2d",
        ),
    ),
    (
        "Other",
        "Advanced QROM (aka QROAM) using clean ancilla",
        "register_qroam_clean",
        "QROMBase",
        (),
    ),
    (
        "Other",
        "Advanced QROM (aka QROAM) using clean ancilla",
        "register_qroam_clean",
        "QROAMClean",
        ("qroam_clean_multi_data", "qroam_clean_multi_dim"),
    ),
    (
        "Other",
        "Reflections",
        "prepared_state_reflections",
        "PrepareIdentity",
        ("prepare_identity",),
    ),
    (
        "Other",
        "Reflections",
        "prepared_state_reflections",
        "ReflectionUsingPrepare",
        ("refl_using_prep", "refl_around_zero"),
    ),
    (
        "Other",
        "Multi-Paulis",
        "multi_paulis",
        "MultiTargetCNOT",
        ("c_multi_not_symb", "c_multi_not"),
    ),
    ("Other", "Multi-Paulis", "multi_paulis", "MultiControlX", ("ccpauli",)),
    (
        "Other",
        "Generic Select",
        "select_pauli_lcu",
        "SelectPauliLCU",
        ("select_pauli_lcu",),
    ),
    (
        "Other",
        "State Preparation via Alias Sampling",
        "state_preparation_alias_sampling",
        "StatePreparationAliasSampling",
        ("state_prep_alias", "state_prep_alias_symb"),
    ),
    (
        "Other",
        "State Preparation via Alias Sampling",
        "state_preparation_alias_sampling",
        "SparseStatePreparationAliasSampling",
        (
            "sparse_state_prep_alias",
            "sparse_state_prep_alias_from_list",
            "sparse_state_prep_alias_symb",
        ),
    ),
    (
        "Other",
        "State Preparation Using Rotations",
        "state_preparation_via_rotation",
        "StatePreparationViaRotations",
        (
            "state_prep_via_rotation",
            "state_prep_via_rotation_symb",
            "state_prep_via_rotation_symb_phasegrad",
        ),
    ),
    (
        "Other",
        "Generalized Quantum Signal Processing",
        "generalized_signal_processing",
        "GeneralizedQSP",
        ("gqsp", "gqsp_with_negative_power", "gqsp_with_large_negative_power"),
    ),
    (
        "Other",
        "Hamiltonian Simulation by Generalized Quantum Signal Processing",
        "hamiltonian_simulation_by_gqsp",
        "HamiltonianSimulationByGQSP",
        ("hubbard_time_evolution_by_gqsp", "symbolic_hamsim_by_gqsp"),
    ),
    (
        "Other",
        "Guided Hamiltonian Problem",
        "guided_hamiltonian",
        "GuidedHamiltonian",
        ("guided_hamiltonian_symb",),
    ),
    (
        "Other",
        "Guided Hamiltonian Problem",
        "guided_hamiltonian",
        "GuidedHamiltonianPhaseEstimation",
        ("guided_phase_estimate_symb",),
    ),
)

_CANDIDATES = {
    **_lookup_instances(),
    **_alias_instances(),
    **_preparation_instances(),
    **_pauli_instances(),
    **_signal_instances(),
    **_guided_instances(),
}
INSTANCES = {
    identifier: _CANDIDATES[identifier]
    for _, _, _, _, identifiers in DOCUMENTS
    for identifier in identifiers
}
_EMPTY_DOCUMENTS = {"QROMBase": QromTable}


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError(
            "Other catalog UI requires a registered native example identifier"
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
                import_line="from examples.algorithms.other_ui_export import sections",
            )
        )
    return tuple(
        (section, tuple(notebooks.values())) for section, notebooks in grouped.items()
    )


def export_example(destination: Path) -> None:
    """Publish complete lookup, preparation and signal-processing metadata and fresh interfaces."""
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
    export_example(Path("other_ui_export"))
