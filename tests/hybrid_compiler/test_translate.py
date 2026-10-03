from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.compiler import TRANSLATION_FORMATS, translate
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.operator_lowering import UnsupportedLoweringError
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.compiler.qir import emit_qir
from flagquantum.compiler.target_emission import EMISSION_PROFILES
from flagquantum.core.ir import CircuitIR, Instruction, ObservableNode

pytestmark = pytest.mark.integration


def _circuit() -> fq.Circuit:
    """A two-wire program whose every gate is directly expressible."""

    return fq.Circuit(2).h(0).cx(0, 1).t(1)


def _channel_ir() -> CircuitIR:
    return CircuitIR(
        n_wires=2,
        instructions=(
            Instruction(
                name="bit_flip",
                wires=(0,),
                params={"probability": 0.1},
                metadata={"is_channel": True},
            ),
        ),
        dtype="complex64",
    )


def test_translation_formats_are_the_emission_profile_names() -> None:
    """The accepted vocabulary is the Compiler's profile set, not a second list."""

    assert tuple(sorted(EMISSION_PROFILES)) == TRANSLATION_FORMATS
    assert set(EMISSION_PROFILES) == {
        "openqasm-2.0",
        "openqasm-3.0",
        "qcis-1.0",
        "qir-2.0",
    }


def test_the_default_format_is_openqasm_3() -> None:
    assert translate(_circuit()) == emit_openqasm(_circuit(), version=3.0)


@pytest.mark.parametrize(
    ("translation_format", "expected"),
    (
        ("openqasm-2.0", lambda: emit_openqasm(_circuit(), version=2.0)),
        ("openqasm-3.0", lambda: emit_openqasm(_circuit(), version=3.0)),
        ("qcis-1.0", lambda: emit_qcis(_circuit())),
        ("qir-2.0", lambda: emit_qir(_circuit())),
    ),
)
def test_each_format_delegates_to_its_emitter_byte_for_byte(
    translation_format: str, expected: object
) -> None:
    """Translate is a dispatch, so a divergence here is a second implementation."""

    assert translate(_circuit(), format=translation_format) == expected()  # type: ignore[operator]


@pytest.mark.parametrize("translation_format", TRANSLATION_FORMATS)
def test_translation_is_deterministic(translation_format: str) -> None:
    first = translate(_circuit(), format=translation_format)

    assert first == translate(_circuit(), format=translation_format)
    assert first.endswith("\n") is False


@pytest.mark.parametrize("spelling", ("OpenQASM-3.0", " openqasm-3.0 ", "OPENQASM-3.0"))
def test_the_format_name_is_case_and_whitespace_insensitive(spelling: str) -> None:
    assert translate(_circuit(), format=spelling) == translate(
        _circuit(), format="openqasm-3.0"
    )


@pytest.mark.parametrize(
    "requested", ("qir:0.1", "qir", "openqasm", "OPENQASM2", "", "None")
)
def test_an_unknown_format_is_refused_with_the_accepted_set(requested: str) -> None:
    """A CUDA-Q spelling that names no FlagQuantum profile must not be guessed at."""

    with pytest.raises(ValueError) as excinfo:
        translate(_circuit(), format=requested)

    message = str(excinfo.value)
    assert repr(requested) in message
    for name in TRANSLATION_FORMATS:
        assert name in message


def test_a_non_string_format_is_refused_by_the_same_path() -> None:
    with pytest.raises(ValueError):
        translate(_circuit(), format=None)  # type: ignore[arg-type]


def test_a_program_with_an_unbound_parameter_is_refused_before_emission() -> None:
    """Every format names the unbound parameter instead of printing a placeholder."""

    circuit = fq.Circuit(2).rx(0, fq.Parameter("theta"))

    for translation_format in TRANSLATION_FORMATS:
        with pytest.raises(ValueError) as excinfo:
            translate(circuit, format=translation_format)
        assert "theta" in str(excinfo.value)
        assert "bind_parameters" in str(excinfo.value)


def test_binding_the_parameter_makes_the_program_translatable() -> None:
    circuit = fq.Circuit(2).rx(0, fq.Parameter("theta"))
    bound = circuit.bind_parameters({"theta": 0.7})

    assert "0.7" in translate(bound, format="openqasm-3.0")


@pytest.mark.parametrize("translation_format", TRANSLATION_FORMATS)
def test_a_noise_channel_is_refused_by_name(translation_format: str) -> None:
    """A channel has no text form in any profile; naming it beats emitting it."""

    with pytest.raises(UnsupportedLoweringError) as excinfo:
        translate(_channel_ir(), format=translation_format)

    assert "bit_flip" in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.parametrize("translation_format", TRANSLATION_FORMATS)
def test_an_arbitrary_matrix_gate_is_refused(translation_format: str) -> None:
    circuit = fq.Circuit(1).any(0, unitary=[[0, 1], [1, 0]])

    with pytest.raises(UnsupportedLoweringError) as excinfo:
        translate(circuit, format=translation_format)

    assert "any" in str(excinfo.value)


@pytest.mark.parametrize("translation_format", TRANSLATION_FORMATS)
def test_a_classical_condition_is_refused(translation_format: str) -> None:
    """A conditioned gate is not a static operation in any of the four languages."""

    ir = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction(
                name="x",
                wires=(1,),
                metadata={"condition_clauses": ({"wire": 0, "value": 1},)},
            ),
        ),
        dtype="complex64",
    )

    with pytest.raises(UnsupportedLoweringError):
        translate(ir, format=translation_format)


def test_a_qir_translation_carries_the_entry_point_and_labeling_schema() -> None:
    from flagquantum.compiler.qir import ENTRY_POINT_NAME, OUTPUT_LABELING_SCHEMA

    text = translate(_circuit(), format="qir-2.0")

    assert f"define i64 @{ENTRY_POINT_NAME}()" in text
    assert f'"output_labeling_schema"="{OUTPUT_LABELING_SCHEMA}"' in text


def test_qir_translation_decomposes_what_the_language_lacks() -> None:
    """``sxdg`` is not a base-profile QIS call, so the text must say something else."""

    text = translate(fq.Circuit(1).sxdg(0), format="qir-2.0")

    assert "__quantum__qis__sxdg__body" not in text
    assert text.count("call void @__quantum__qis__h__body") == 2
    assert "call void @__quantum__qis__s__adj(ptr null)" in text


def test_oq2_and_oq3_disagree_about_how_a_measurement_is_written() -> None:
    """The two OpenQASM versions are different languages, not a version keyword."""

    two = translate(_circuit(), format="openqasm-2.0")
    three = translate(_circuit(), format="openqasm-3.0")

    assert "OPENQASM 2.0;" in two and "measure q[0] -> c[0];" in two
    assert "OPENQASM 3.0;" in three and "c = measure q;" in three


def test_a_routed_program_keeps_its_allocated_result_projection() -> None:
    """The projection lives in the IR, so a bare translate must still honour it."""

    routed = CircuitIR(
        n_wires=3,
        instructions=(Instruction(name="h", wires=(2,)),),
        dtype="complex64",
        metadata={
            "routing": {
                "schema": "flagquantum_directed_routing_plan_v2",
                "logical_result_physical_slots": (2,),
            }
        },
    )

    text = translate(routed, format="qir-2.0")

    assert 'c"t0\\00"' in text
    assert 'c"t1\\00"' not in text


@pytest.mark.parametrize("translation_format", TRANSLATION_FORMATS)
def test_an_observable_request_is_refused(translation_format: str) -> None:
    """Translation emits gates; an observable request is not a gate and is not dropped."""

    ir = CircuitIR(
        n_wires=2,
        instructions=(Instruction(name="h", wires=(0,)),),
        dtype="complex64",
        observables=(ObservableNode(name="z", wires=(0,)),),
    )

    with pytest.raises(UnsupportedLoweringError) as excinfo:
        translate(ir, format=translation_format)

    assert "observable" in str(excinfo.value)
