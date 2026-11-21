# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Modular Arithmetic notebook catalog.

"""Export the complete modular, RSA and ECC example interface catalog.

Run ``python -m examples.algorithms.modular_ui_export``. The existing domain
owners provide signatures and instance parameters. UI diagrams show atomic
interfaces. Physical execution and resources retain their independent owners.
"""

import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from sympy import Symbol

from examples.algorithms.modular_addition_catalog import modular_addition_examples
from examples.algorithms.modular_inverse_catalog import modular_inverse_examples
from examples.algorithms.modular_multiplication_catalog import (
    modular_multiplication_examples,
)
from examples.algorithms.modular_subtraction_catalog import modular_subtraction_examples
from flagquantum.algorithms.primitives.elliptic_curve_addition import (
    elliptic_curve_addition_signature,
)
from flagquantum.algorithms.primitives.elliptic_curve_double_phase import (
    elliptic_curve_double_phase_source_graph,
)
from flagquantum.algorithms.primitives.elliptic_curve_fixed_point import (
    EllipticCurveFixedPointAddition,
)
from flagquantum.algorithms.primitives.elliptic_curve_phase_estimation import (
    elliptic_curve_phase_signature,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.algorithms.primitives.elliptic_curve_window_addition import (
    EllipticCurveWindowAddition,
)
from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.algorithms.primitives.rsa_phase_estimation import RSAPhaseEstimation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import FunctionRecord
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

# Operation label, native signature, parameter metadata and authoritative owner.
# This UI association is not an executable operation or another program IR.
_Interface: TypeAlias = tuple[str, Signature, Mapping[str, object], str]


def _modular_instances() -> dict[str, _Interface]:
    result: dict[str, _Interface] = {}
    for identifier, addition in modular_addition_examples().items():
        n, p, k = addition.width, addition.modulus, addition.constant
        parameters: Mapping[str, object]
        if addition.kind == "ModAdd":
            parameters = {"bitsize": n, "mod": p}
        elif addition.kind == "ModAddK":
            parameters = {"bitsize": n, "mod": p, "add_val": k, "cvs": ()}
        elif addition.kind == "CModAdd":
            parameters = {"dtype": {"bitsize": n}, "mod": p, "cv": 1}
        else:
            parameters = {"k": k, "mod": p, "bitsize": n}
        result[identifier] = (
            addition.kind,
            addition.signature,
            parameters,
            type(addition).__module__,
        )
    for identifier, subtraction in modular_subtraction_examples().items():
        parameters = {
            "dtype": {"bitsize": subtraction.width},
            "mod": subtraction.modulus,
        }
        if subtraction.kind in ("CModNeg", "CModSub"):
            parameters = {**parameters, "cv": subtraction.control_value}
        result[identifier] = (
            subtraction.kind,
            subtraction.signature,
            parameters,
            type(subtraction).__module__,
        )
    for identifier, multiplication in modular_multiplication_examples().items():
        n, p = multiplication.width, multiplication.modulus
        if multiplication.kind == "ModDbl":
            parameters = {"dtype": {"bitsize": n}, "mod": p}
        elif multiplication.kind == "CModMulK":
            parameters = {
                "dtype": {"bitsize": n},
                "k": multiplication.constant,
                "mod": p,
            }
        else:
            parameters = {
                "bitsize": n,
                "window_size": multiplication.window_size,
                "mod": p,
                "uncompute": False,
            }
        result[identifier] = (
            multiplication.kind,
            multiplication.signature(),
            parameters,
            type(multiplication).__module__,
        )
    for identifier, inverse in modular_inverse_examples().items():
        result[identifier] = (
            "KaliskiModInverse",
            inverse.signature(),
            {"bitsize": inverse.width, "mod": inverse.modulus, "uncompute": False},
            type(inverse).__module__,
        )
    return result


def _rsa_instances() -> dict[str, _Interface]:
    n, p, g, size, exponent, word = (
        Symbol(s) for s in ("n", "p", "g", "N", "n_e", "n_x")
    )
    phases = (
        ("rsa_pe_small", RSAPhaseEstimation(8, 221, 9)),
        ("rsa_pe", RSAPhaseEstimation(n, p, g)),
    )
    result: dict[str, _Interface] = {
        identifier: (
            "RSAPhaseEstimate",
            operation.source_signature,
            {"n": operation.width, "mod": operation.modulus, "base": operation.base},
            type(operation).__module__,
        )
        for identifier, operation in phases
    }
    powers = (
        ("modexp_small", ModularExponentiation(4, 15, 3, 2048)),
        ("modexp", ModularExponentiation.for_shor(221, 9)),
        ("modexp_symb", ModularExponentiation(g, size, exponent, word)),
    )
    for identifier, operation in powers:
        result[identifier] = (
            "ModExp",
            operation.source_signature,
            {
                "base": operation.base,
                "mod": operation.modulus,
                "exp_bitsize": operation.exponent_width,
                "x_bitsize": operation.output_width,
            },
            type(operation).__module__,
        )
    return result


def _point_parameters(point: ECPoint) -> Mapping[str, object]:
    return {"x": point.x, "y": point.y, "mod": point.mod, "curve_a": point.curve_a}


def _curve_instances() -> dict[str, _Interface]:
    n, p, px, py, qx, qy, rx, ry = (
        Symbol(s) for s in ("n", "p", "P_x", "P_y", "Q_x", "Q_y", "R_x", "R_y")
    )
    base, public, point = ECPoint(px, py, p), ECPoint(qx, qy, p), ECPoint(rx, ry, p)
    result: dict[str, _Interface] = {
        "ec_add": (
            "ECAdd",
            elliptic_curve_addition_signature(n),
            {"n": n, "mod": p, "window_size": 1},
            elliptic_curve_addition_signature.__module__,
        ),
        "ec_add_small": (
            "ECAdd",
            elliptic_curve_addition_signature(5),
            {"n": 5, "mod": 7, "window_size": 1},
            elliptic_curve_addition_signature.__module__,
        ),
        # The double-phase root consumes its local coordinates and has no ports.
        # Symbolic declaration uses the signature directly, without passing a
        # symbolic width to an execution class whose width contract is int.
        "ecc": (
            "FindECCPrivateKey",
            Signature(()),
            {
                "n": n,
                "base_point": _point_parameters(base),
                "public_key": _point_parameters(public),
                "add_window_size": 1,
                "mul_window_size": 1,
            },
            elliptic_curve_double_phase_source_graph.__module__,
        ),
        "ec_pe": (
            "ECPhaseEstimateR",
            elliptic_curve_phase_signature(n),
            {
                "n": n,
                "point": _point_parameters(point),
                "add_window_size": 1,
                "mul_window_size": 1,
            },
            elliptic_curve_phase_signature.__module__,
        ),
        "ec_pe_small": (
            "ECPhaseEstimateR",
            elliptic_curve_phase_signature(3),
            {
                "n": 3,
                "point": _point_parameters(point),
                "add_window_size": 1,
                "mul_window_size": 1,
            },
            elliptic_curve_phase_signature.__module__,
        ),
    }
    for identifier, fixed in (
        ("ec_add_r", EllipticCurveFixedPointAddition(n, point)),
        ("ec_add_r_small", EllipticCurveFixedPointAddition(5, ECPoint(15, 13, 17))),
    ):
        result[identifier] = (
            "ECAddR",
            fixed.source_signature,
            {"n": fixed.width, "R": _point_parameters(fixed.point)},
            type(fixed).__module__,
        )
    window = EllipticCurveWindowAddition(16, ECPoint(2, 2, 7, 3), 4)
    result["ec_window_add_r_small"] = (
        "ECWindowAddR",
        window.source_signature,
        {
            "n": window.width,
            "R": _point_parameters(window.point),
            "add_window_size": window.window,
            "mul_window_size": window.multiplication_window,
        },
        type(window).__module__,
    )
    return result


INSTANCES = {**_modular_instances(), **_rsa_instances(), **_curve_instances()}
DOCUMENTS = (
    ("Modular Addition", "modular_addition", "ModAdd", ("mod_add",)),
    (
        "Modular Addition",
        "modular_addition",
        "ModAddK",
        ("mod_add_k", "mod_add_k_small", "mod_add_k_large"),
    ),
    (
        "Modular Addition",
        "modular_addition",
        "CModAdd",
        ("cmodadd_example", "cmodadd_symbolic"),
    ),
    (
        "Modular Addition",
        "modular_addition",
        "CModAddK",
        ("cmod_add_k", "cmod_add_k_small"),
    ),
    (
        "Modular Addition",
        "modular_addition",
        "CtrlScaleModAdd",
        ("ctrl_scale_mod_add", "ctrl_scale_mod_add_small"),
    ),
    ("Modular Subtraction", "modular_subtraction", "ModNeg", ("mod_neg",)),
    ("Modular Subtraction", "modular_subtraction", "CModNeg", ("cmod_neg",)),
    ("Modular Subtraction", "modular_subtraction", "ModSub", ("modsub_symb",)),
    ("Modular Subtraction", "modular_subtraction", "CModSub", ("cmodsub_symb",)),
    (
        "Modular Multiplication",
        "modular_multiplication",
        "ModDbl",
        ("moddbl_small", "moddbl_large"),
    ),
    (
        "Modular Multiplication",
        "modular_multiplication",
        "CModMulK",
        ("modmul_symb", "modmul"),
    ),
    (
        "Modular Multiplication",
        "modular_multiplication",
        "DirtyOutOfPlaceMontgomeryModMul",
        (
            "dirtyoutofplacemontgomerymodmul_small",
            "dirtyoutofplacemontgomerymodmul_medium",
        ),
    ),
    (
        "Modular Divison",
        "modular_inverse",
        "KaliskiModInverse",
        ("kaliskimodinverse_example", "kaliskimodinverse_symbolic"),
    ),
    (
        "Factoring RSA",
        "register_rsa_phase_estimation",
        "RSAPhaseEstimate",
        ("rsa_pe_small", "rsa_pe"),
    ),
    (
        "Factoring RSA",
        "register_rsa_phase_estimation",
        "ModExp",
        ("modexp_small", "modexp", "modexp_symb"),
    ),
    (
        "Elliptic Curve Addition",
        "register_elliptic_curve_addition",
        "ECAdd",
        ("ec_add", "ec_add_small"),
    ),
    (
        "Elliptic Curve Cryptography",
        "register_elliptic_curve_cryptography",
        "FindECCPrivateKey",
        ("ecc",),
    ),
    (
        "Elliptic Curve Cryptography",
        "register_elliptic_curve_cryptography",
        "ECPhaseEstimateR",
        ("ec_pe", "ec_pe_small"),
    ),
    (
        "Elliptic Curve Cryptography",
        "register_elliptic_curve_cryptography",
        "ECAddR",
        ("ec_add_r", "ec_add_r_small"),
    ),
    (
        "Elliptic Curve Cryptography",
        "register_elliptic_curve_cryptography",
        "ECWindowAddR",
        ("ec_window_add_r_small",),
    ),
)


def _instance(component: object) -> _Interface:
    if not isinstance(component, str) or component not in INSTANCES:
        raise TypeError("Modular UI requires a registered native example identifier")
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
    ports = signature(component)
    leaf = FunctionRecord(name(component), (), registers=tuple(ports))
    builder = ComponentProgramBuilder(ports, (leaf,), name="ModularInterface")
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    return register_score_data(component_register_score(record, "ModularInterface"))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    title = documentation_name(documentation)
    owner = _instance(documentation.examples[0].make())[3]
    return (
        f"# {title}\n\nNative metadata owner `{owner}`.\n\n"
        "This diagram shows the atomic typed interface. Complete decomposition "
        "graphs, glyphs, execution and resource estimates retain separate evidence. "
        "Large and symbolic instances allocate no execution wires.\n"
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
                import_line="from examples.algorithms.modular_ui_export import sections",
            )
        )
    return (("Modular Arithmetic", tuple(grouped.values())),)


def export_example(destination: Path) -> None:
    """Publish all modular, RSA and ECC instance metadata and fresh interfaces."""
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
    export_example(Path("modular_ui_export"))
