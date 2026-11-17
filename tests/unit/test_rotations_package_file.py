"""Certify the locked rotations package namespace and native consumers."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_SOURCE_PACKAGE = "qualtran.bloqs.rotations"
_NATIVE_PACKAGE = "flagquantum.compiler.registers.phase"
_SOURCE_BLOB = "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"
# Source module, native module, native function and consumer module.
_ROUTES = {
    "HammingWeightPhasing": (
        "hamming_weight_phasing",
        "hamming_weight_phasing",
        "hamming_weight_phasing_ir",
        "examples.algorithms.hamming_phasing_catalog",
    ),
    "HammingWeightPhasingViaPhaseGradient": (
        "hamming_weight_phasing",
        "hamming_weight_phasing",
        "hamming_weight_gradient_phasing_ir",
        "examples.algorithms.hamming_phasing_catalog",
    ),
    "AddIntoPhaseGrad": (
        "phase_gradient",
        "phase_addition",
        "phase_add_ir",
        "examples.algorithms.phase_gradient_catalog",
    ),
    "AddScaledValIntoPhaseReg": (
        "phase_gradient",
        "scaled_phase_addition",
        "scaled_phase_add_ir",
        "examples.algorithms.phase_gradient_catalog",
    ),
    "PhaseGradientState": (
        "phase_gradient",
        "phase_gradient",
        "phase_gradient_state_ir",
        "examples.algorithms.phase_gradient_catalog",
    ),
    "PhasingViaCostFunction": (
        "phasing_via_cost_function",
        "cost_function_phasing",
        "cost_function_phasing_ir",
        f"{_NATIVE_PACKAGE}.square_cost_phasing",
    ),
    "ProgrammableRotationGateArray": (
        "programmable_rotation_gate_array",
        "programmable_rotation_array",
        "programmable_rotation_array_ir",
        "examples.algorithms.programmable_rotation_array_catalog",
    ),
    "QvrPhaseGradient": (
        "quantum_variable_rotation",
        "qvr_phase_gradient",
        "qvr_phase_gradient_ir",
        "examples.algorithms.qvr_catalog",
    ),
    "QvrZPow": (
        "quantum_variable_rotation",
        "qvr_zpow",
        "qvr_zpow_ir",
        "examples.algorithms.qvr_catalog",
    ),
    "RzViaPhaseGradient": (
        "rz_via_phase_gradient",
        "rz_phase_gradient",
        "rz_via_phase_gradient_ir",
        "examples.algorithms.rz_phase_gradient_catalog",
    ),
    "ZPowConstViaPhaseGradient": (
        "zpow_via_phase_gradient",
        "zpow_phase_gradient",
        "zpow_via_phase_gradient_ir",
        "examples.algorithms.zpow_phase_gradient_catalog",
    ),
}


@pytest.mark.ftqc_reference
def test_locked_rotations_initializer_covers_every_export() -> None:
    package = pytest.importorskip(_SOURCE_PACKAGE)
    assert package.__file__ is not None
    content = Path(package.__file__).read_bytes()
    assert sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == _SOURCE_BLOB
    seen = set()
    for node in ast.parse(content).body:
        assert isinstance(node, ast.ImportFrom)
        assert node.level == 0
        for alias in node.names:
            name = alias.name
            assert alias.asname is None and name not in seen
            seen.add(name)
            source = f"{_SOURCE_PACKAGE}.{_ROUTES[name][0]}"
            assert node.module == source
            assert getattr(package, name) is getattr(import_module(source), name)
    assert seen == _ROUTES.keys()


@pytest.mark.parametrize("route", _ROUTES.values(), ids=_ROUTES.keys())
def test_rotation_consumer_uses_the_internal_compiler_owner(
    route: tuple[str, str, str, str],
) -> None:
    _, native, entry, consumer = route
    owner = import_module(f"{_NATIVE_PACKAGE}.{native}")
    implementation = getattr(owner, entry)
    assert callable(implementation)
    assert implementation.__module__ == owner.__name__
    assert getattr(import_module(consumer), entry) is implementation
