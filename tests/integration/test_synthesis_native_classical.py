"""Finite static bindings share the native owned-register program path."""

import importlib.util
import math
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.qml.main import (
    main,
)
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.synth import (
    CX,
    PHASE,
    RX,
    RZ,
    Output,
    QArray,
    QBit,
    allocate,
    qfunc,
)

pytestmark = pytest.mark.integration


class _ResourceLimits(TypedDict, total=False):
    max_qubits: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


@pytest.mark.parametrize(("input_0", "weight_0"), [(0.37, -0.81), (1.2, 6.1)])
def test_original_qml_two_nonzero_bindings_and_absolute_phase(
    input_0: float, weight_0: float
) -> None:
    bindings = {"input_0": input_0, "weight_0": weight_0}
    source = Path(
        "tests/fixtures/classiq_qmod/corpus/original/tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide.qmod"
    )
    reference = fq.run(
        fq.Circuit(1, dtype=torch.complex128).rx(0, theta=input_0 + weight_0)
    ).statevector()
    analytic = torch.tensor(
        [math.cos((input_0 + weight_0) / 2), -1j * math.sin((input_0 + weight_0) / 2)],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(reference.flatten(), analytic, atol=1e-12, rtol=0)
    for definition in (main, source):
        program = synth.synthesize(definition, classical_inputs=bindings)
        torch.testing.assert_close(
            fq.run(program).statevector(), reference, atol=1e-12, rtol=0
        )
        assert program.report["workspace_qubits"] == 0
        assert tuple(program.registers) == ("res",)


@qfunc
def rotate(angles: list[float], bias: float, q: QArray[QBit]) -> None:
    for index in range(len(angles)):
        RX(angles[index] + bias / 2, q[index])
        PHASE(-angles[index] * 0.5, q[index])


@qfunc
def nested(angles: list[float], bias: float, q: QArray[QBit]) -> None:
    rotate(angles, bias, q)
    RZ(angles[0] - bias, q[0])


@qfunc
def array_program(angles: list[float], bias: float, q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    nested(angles, bias, q)
    RX(bias, q[0])
    CX(q[0], q[1])


@pytest.mark.parametrize(
    ("angles", "bias"), [([0.17, -0.83], 0.29), ((-1.3, 0.41), -0.7)]
)
def test_array_helpers_scope_order_and_coherent_amplitudes(
    angles: list[float] | tuple[float, ...], bias: float
) -> None:
    reference = fq.Circuit(2, dtype=torch.complex128)
    for logical, angle in enumerate(angles):
        reference.rx(1 - logical, theta=angle + bias / 2).phase(
            1 - logical, theta=-angle * 0.5
        )
    reference.rz(1, theta=angles[0] - bias).rx(1, theta=bias).cx(1, 0)
    program = synth.synthesize(
        array_program, classical_inputs={"angles": angles, "bias": bias}
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def _definition(
    tmp_path: Path, signature: str, body: str, prelude: str = ""
) -> _QuantumFunction:
    path = tmp_path / "native_classical.py"
    path.write_text(
        "from flagquantum.synth import *\n"
        + prelude
        + f"\n@qfunc\ndef main({signature}) -> None:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("native_classical", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main
    assert isinstance(result, _QuantumFunction)
    return result


@pytest.mark.parametrize(
    "bindings",
    [
        None,
        {},
        {"input_0": 0.3},
        {"input_0": 0.3, "weight_0": 0.2, "typo": 0.1},
        {"input_0": True, "weight_0": 0.2},
        {"input_0": float("inf"), "weight_0": 0.2},
        {"input_0": float("nan"), "weight_0": 0.2},
        {"input_0": 2**54, "weight_0": 0.2},
        {"input_0": "0.3", "weight_0": 0.2},
    ],
)
def test_invalid_root_bindings_fail_before_lowering(bindings: object) -> None:
    with pytest.raises(ValueError, match="program.binding"):
        synth.synthesize(main, classical_inputs=bindings)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    "angles", [[], [True], [float("nan")], [[0.2]], "0.2", [0.1] * 4097]
)
def test_invalid_array_bindings(angles: object) -> None:
    with pytest.raises(ValueError, match="program.binding|budget"):
        synth.synthesize(array_program, classical_inputs={"angles": angles, "bias": 0.2})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(2,q)\nRX(angles[-1],q[0])", "program.binding"),
        ("allocate(2,q)\nRX(angles[2],q[0])", "program.binding"),
        ("allocate(2,q)\nRX(angles[True],q[0])", "program.loop"),
        ("allocate(2,q)\nRX(angles[0][0],q[0])", "program.angle"),
        ("allocate(2,q)\nRX(angles[0]/0,q[0])", "program.angle"),
        ("allocate(2,q)\nRX(angles[0](),q[0])", "program.angle"),
        ("allocate(2,q)\nRX(angles[0],angles)", "program.port"),
        ("allocate(2,q)\nfor i in range(angles[0]):\n    H(q[i])", "program.loop"),
        ("allocate(2,q)\nfor angles in range(2):\n    H(q[angles])", "program.loop"),
    ],
)
def test_closed_parameter_syntax(tmp_path: Path, body: str, diagnostic: str) -> None:
    definition = _definition(
        tmp_path, "angles: list[float], q: Output[QArray[QBit]]", body
    )
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(definition, classical_inputs={"angles": [0.3, -0.4]})


@pytest.mark.parametrize("name", ["RX", "len", "range"])
def test_classical_names_cannot_shadow_callables(tmp_path: Path, name: str) -> None:
    body = "allocate(2,q)\n" + (
        "RX(0.2,q[0])"
        if name == "RX"
        else (
            f"for i in range({name}(q)):\n    H(q[i])"
            if name == "len"
            else "for i in range(2):\n    H(q[i])"
        )
    )
    definition = _definition(tmp_path, f"{name}: float, q: Output[QArray[QBit]]", body)
    with pytest.raises(ValueError, match="program.call|program.loop"):
        synth.synthesize(definition, classical_inputs={name: 0.3})


@pytest.mark.parametrize("actuals", ["angles,q,q", "angles,q[:1],q[:1]"])
def test_parameterized_helpers_keep_alias_rejection(
    tmp_path: Path, actuals: str
) -> None:
    prelude = "@qfunc\ndef helper(angles: list[float], a: QArray[QBit], b: QArray[QBit]) -> None:\n    RX(angles[0],a[0])\n    CX(a[0],b[0])\n"
    definition = _definition(
        tmp_path,
        "angles: list[float], q: Output[QArray[QBit]]",
        f"allocate(2,q)\nhelper({actuals})",
        prelude,
    )
    with pytest.raises(ValueError, match="program.borrow"):
        synth.synthesize(definition, classical_inputs={"angles": [0.3]})


def test_same_definition_rebinds_without_stale_specialization() -> None:
    for value in (0.37, -0.82, 0.37):
        program = synth.synthesize(
            main, classical_inputs={"input_0": value, "weight_0": 0.31}
        )
        expected = fq.run(
            fq.Circuit(1, dtype=torch.complex128).rx(0, theta=value + 0.31)
        ).statevector()
        torch.testing.assert_close(
            fq.run(program).statevector(), expected, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 1},
        {"max_operations": 0},
        {"max_depth": 0},
        {"memory_limit_bytes": 1},
    ],
)
def test_parameter_binding_preserves_resource_rejection(
    limits: "_ResourceLimits",
) -> None:
    with pytest.raises((ValueError, MemoryError)):
        synth.synthesize(
            array_program,
            classical_inputs={"angles": [0.3, -0.4], "bias": 0.2},
            **limits,
        )


@pytest.mark.parametrize(
    ("prelude", "body", "diagnostic"),
    [
        (
            "@qfunc\ndef helper(theta: float, q: QArray[QBit]) -> None:\n    RX(theta,q[0])\n",
            "helper(angles,q)",
            "program.binding",
        ),
        (
            "@qfunc\ndef helper(theta: float, q: QArray[QBit]) -> None:\n    RX(theta,q[0])\n",
            "helper(angles[0])",
            "program.signature",
        ),
        (
            "@qfunc\ndef helper(theta: float, q: QArray[QBit]) -> None:\n    RX(theta,q[0])\n",
            "helper(theta=angles[0],q=q)",
            "program.signature",
        ),
        (
            "@qfunc\ndef helper(theta: float, q: Output[QBit]) -> None:\n    allocate(q)\n",
            "helper(angles[0],q)",
            "program.binding",
        ),
        (
            "@qfunc\ndef helper(theta: float, q: QArray[QBit]) -> None:\n    allocate(2,q)\n",
            "helper(angles[0],q)",
            "construction.register.initialized",
        ),
        (
            "@qfunc\ndef helper(theta: float, q: QArray[QBit]) -> None:\n    helper(theta,q)\n",
            "helper(angles[0],q)",
            "program.recursion",
        ),
    ],
)
def test_helper_signature_escape_and_recursion_refusals(
    tmp_path: Path, prelude: str, body: str, diagnostic: str
) -> None:
    definition = _definition(
        tmp_path,
        "angles: list[float], q: Output[QArray[QBit]]",
        "allocate(2,q)\n" + body,
        prelude,
    )
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(definition, classical_inputs={"angles": [0.3, -0.4]})


def test_literal_array_helper_and_interleaved_scalar_bit_ports(tmp_path: Path) -> None:
    prelude = "@qfunc\ndef helper(a: QBit, angles: list[float], b: QBit, theta: float) -> None:\n    RX(angles[0],a)\n    RZ(theta,b)\n    CX(a,b)\n"
    definition = _definition(
        tmp_path,
        "theta: float, q: Output[QArray[QBit]]",
        "allocate(2,q)\nhelper(q[1],[theta, -theta/2],q[0],theta*2)",
        prelude,
    )
    program = synth.synthesize(definition, classical_inputs={"theta": 0.37})
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .rx(0, theta=0.37)
        .rz(1, theta=0.74)
        .cx(0, 1)
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_values_are_snapshotted_and_host_calls_are_not_executed(tmp_path: Path) -> None:
    bindings: dict[str, float | list[float]] = {"angles": [0.17, -0.83], "bias": 0.29}
    program = synth.synthesize(array_program, classical_inputs=bindings)
    expected = fq.run(program).statevector().clone()
    bindings["angles"] = [0.9, 1.1]
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )
    prelude = "def unsafe():\n    raise AssertionError('host function executed')\n"
    definition = _definition(
        tmp_path, "q: Output[QBit]", "allocate(q)\nRX(unsafe(),q)", prelude
    )
    with pytest.raises(ValueError, match="program.angle"):
        synth.synthesize(definition)


@pytest.mark.parametrize(
    "annotation", ["list[list[float]]", "list[bool]", "Output[float]"]
)
def test_classical_annotations_do_not_expand_quantum_or_array_types(
    tmp_path: Path, annotation: str
) -> None:
    with pytest.raises(ValueError, match="annotation"):
        _definition(tmp_path, f"values: {annotation}, q: Output[QBit]", "allocate(q)")


def test_legacy_integer_entry_retains_its_separate_input_semantics() -> None:
    def energy(x: int) -> int:
        return x * x

    with pytest.raises(ValueError, match="typed qfunc"):
        synth.synthesize(
            energy, inputs={"x": synth.UInt(2)}, classical_inputs={"x": 0.3}
        )


@pytest.mark.parametrize(
    ("signature", "bindings", "body", "diagnostic"),
    [
        (
            "adjoint: float, q: Output[QBit]",
            {"adjoint": 0.3},
            "allocate(q)\nwith adjoint():\n    H(q)",
            "program.call",
        ),
        (
            "H: float, q: Output[QBit]",
            {"H": 0.3},
            "allocate(q)\ncontrol(q,lambda: H(q))",
            "program.call",
        ),
        (
            "theta: float, q: Output[QBit]",
            {"theta": 0.3},
            "allocate(q)\nwith within():\n    theta: QNum[2] = q + 1\nwith apply():\n    phase(theta)",
            "program.local",
        ),
    ],
)
def test_parameter_scope_is_checked_in_regions_and_callbacks(
    tmp_path: Path,
    signature: str,
    bindings: dict[str, float],
    body: str,
    diagnostic: str,
) -> None:
    definition = _definition(tmp_path, signature, body)
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(definition, classical_inputs=bindings)


def test_quoted_annotations_and_standard_type_aliases(tmp_path: Path) -> None:
    definition = _definition(
        tmp_path,
        "theta: 'Real', angles: 'Reals[Real]', q: 'Output[QBit]'",
        "allocate(q)\nRX(theta+angles[0],q)",
        "from builtins import float as Real, list as Reals\n",
    )
    program = synth.synthesize(
        definition, classical_inputs={"theta": 0.3, "angles": [-0.7]}
    )
    expected = fq.run(
        fq.Circuit(1, dtype=torch.complex128).rx(0, theta=-0.4)
    ).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )
    with pytest.raises(ValueError, match="annotation"):
        _definition(
            tmp_path, "theta: 'float', q: Output[QBit]", "allocate(q)", "float = None\n"
        )


def test_classical_helper_in_adjoint_preserves_the_inverse(tmp_path: Path) -> None:
    entry = _definition(
        tmp_path,
        "angles: list[float], q: Output[QArray[QBit]]",
        "allocate(2,q)\nwith adjoint():\n    helper(angles[0],q[0])",
        "@qfunc\ndef helper(theta: float, q: QBit) -> None:\n    RX(theta,q)\n",
    )
    program = synth.synthesize(entry, classical_inputs={"angles": [0.3, -0.4]})
    expected = fq.Circuit(2, dtype=torch.complex128).rx(1, theta=-0.3).state()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected.flatten(), atol=1e-12, rtol=0
    )
