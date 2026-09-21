"""Contract for what a gate parameter may be.

A gate parameter in the operator registry is a rotation angle, so it has to denote
a real number. The builder used to store whatever it was handed and leave the value
unread until the simulator resolved it, which made three different mistakes look
alike: a string or ``None`` surfaced as ``ExecutionError: planned execution failed``
wrapping an interpreter message that named neither the gate nor the parameter, and
``True`` was not an error at all -- it was the angle one radian, so the run succeeded
with a rotation nobody asked for.

Wrong Python type is ``TypeError``, which is what the errors-module boundary reserves
for it. Length and shape are not checked here: only the simulator knows the batch
size, so that part of the contract stays with it.

The file imports only ``torch``, ``pytest`` and the standard library on purpose.
A ``unit`` test is collected by every CPU lane, including ``triton-optional``,
which installs ``torch`` and not much else -- so a NumPy value cannot be probed
here even though the rule has an answer for one. NumPy scalars reach the rule
through ``numbers.Real`` and NumPy arrays through the element-dtype branch; the
first is covered below by ``Fraction``, the second by a torch tensor, because the
branch reads whatever dtype the value reports. The NumPy cases themselves are
measured in the pull request body and in
``gate_parameter_types_measurement.txt``.
"""

from __future__ import annotations

import decimal
import fractions
import math
import numbers

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit

# Every parameterized opcode in the registry, so the rule is about parameters and
# not about one gate. `u2` and `u3` name their parameters differently on purpose.
PARAMETERIZED_GATES = (
    ("rx", (0,), "theta"),
    ("ry", (0,), "theta"),
    ("rz", (0,), "theta"),
    ("phase", (0,), "theta"),
    ("u1", (0,), "theta"),
    ("crx", (0, 1), "theta"),
    ("cry", (0, 1), "theta"),
    ("crz", (0, 1), "theta"),
    ("cphase", (0, 1), "theta"),
    ("rxx", (0, 1), "theta"),
    ("ryy", (0, 1), "theta"),
    ("rzz", (0, 1), "theta"),
)

REFUSED_VALUES = [
    ("a string", "0.3"),
    ("an empty string", ""),
    ("a string that spells a complex", "3j"),
    ("bytes", b"0"),
    ("a bytearray", bytearray(b"0")),
    ("None", None),
    ("a complex number with no imaginary part", 0.3 + 0j),
    ("a complex number", 3j),
    ("a Decimal", decimal.Decimal("0.3")),
    ("a mapping", {"a": 1}),
    ("a set", {0.3}),
    ("a range", range(1)),
    ("a list holding a string", ["0.3"]),
    ("a list holding None", [None]),
    ("a list holding a bool", [True]),
    ("a list mixing an angle and None", [0.3, None]),
    ("a bool tensor", torch.tensor(True)),
    ("a bool tensor holding a vector", torch.tensor([True])),
    ("a complex tensor", torch.tensor([0.3 + 0j])),
]


def _refused_values() -> list[tuple[str, object]]:
    """Return the refused values as they will be handed to the gate.

    The table above is punctuation-free on purpose: it is evaluated while the
    module is imported, so anything it built would be built at collection time in
    every lane, including the ones that install no array library beyond torch.
    """

    return list(REFUSED_VALUES)


@pytest.mark.parametrize(("label", "value"), _refused_values(), ids=lambda item: None)
def test_a_gate_parameter_refuses_a_value_that_is_not_a_real_number(
    label: str, value: object
) -> None:
    """A wrong type is reported at the call, naming the gate and the parameter."""

    for gate, wires, parameter in PARAMETERIZED_GATES:
        with pytest.raises(
            TypeError, match=f"gate {gate!r} parameter {parameter!r} must be a real"
        ):
            getattr(fq.Circuit(2), gate)(*wires, value)


def test_a_gate_parameter_refuses_a_bool_instead_of_reading_it_as_one_radian() -> None:
    """``True`` is a flag, and reading it as one radian ran a different circuit."""

    with pytest.raises(TypeError, match="must be a real number, not a bool"):
        fq.Circuit(1).ry(0, True)
    with pytest.raises(TypeError, match="must be a real number, not a bool"):
        fq.Circuit(1).ry(0, False)
    # The angle it used to stand for, so the assertion above is not decorative.
    assert fq.Circuit(1).ry(0, 1.0).to_ir().instructions[0].params["theta"] == 1.0


def test_a_gate_parameter_refuses_a_bytes_angle_instead_of_reading_it_as_a_text() -> (
    None
):
    """``float(b"0")`` is ``0.0``, so this used to compile the gate away."""

    with pytest.raises(TypeError, match="gate 'ry' parameter 'theta' must be a real"):
        fq.Circuit(1).ry(0, b"0")
    assert float(b"0") == 0.0


def test_instruction_refuses_the_same_value_however_it_was_reached() -> None:
    """The builder, the IR constructor and the decoder share one contract."""

    with pytest.raises(TypeError, match="gate 'ry' parameter 'theta' must be a real"):
        Instruction("ry", (0,), {"theta": "0.3"})
    with pytest.raises(TypeError, match="gate 'u2' parameter 'phi' must be a real"):
        Instruction("u2", (0,), {"phi": None, "lbd": 0.5})
    with pytest.raises(TypeError, match="gate 'rx' parameter 'theta' must be a real"):
        Instruction("rx", (0,), {"theta": [0.5, "1.0"]})


def test_a_decoded_instruction_is_held_to_the_same_contract() -> None:
    """``from_dict`` is the other way in, and it does not get a laxer rule."""

    program = fq.Circuit(1).ry(0, 0.3).to_ir()
    payload = program.to_dict()
    payload["instructions"][0]["params"]["theta"] = "0.3"
    with pytest.raises(TypeError, match="gate 'ry' parameter 'theta' must be a real"):
        type(program).from_dict(payload)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("a float", 0.3),
        ("a negative float", -1.5),
        ("an int", 1),
        ("a whole float", 2.0),
        ("a Fraction", fractions.Fraction(1, 3)),
        ("an infinity", math.inf),
        ("a NaN", math.nan),
    ],
)
def test_a_gate_parameter_accepts_a_real_scalar(label: str, value: object) -> None:
    """A real number is an angle, and the value is kept exactly as it arrived."""

    circuit = fq.Circuit(1).ry(0, value)
    stored = circuit.to_ir().instructions[0].params["theta"]
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(stored)
    else:
        assert stored == value


def test_a_gate_parameter_accepts_a_real_number_from_another_library() -> None:
    """A scalar type from another array library is a real number, not a wrong type.

    ``numpy`` is neither a core dependency nor declared by any extra, so it is
    deliberately not probed here; the rule reaches a NumPy scalar through
    ``numbers.Real``, which is the same line ``Fraction`` covers below without
    the undeclared dependency.
    """

    assert isinstance(fractions.Fraction(1, 3), numbers.Real)
    assert isinstance(3, numbers.Real)
    assert fq.Circuit(1).ry(0, fractions.Fraction(1, 3)).to_ir().instructions[0].params[
        "theta"
    ] == fractions.Fraction(1, 3)


def test_a_gate_parameter_refuses_a_tensor_whose_elements_are_not_real() -> None:
    """The dtype rule reads the element type, on a tensor as on anything else.

    The same branch answers for any array-like that reports a dtype, so ``torch``
    covers it here and a NumPy array would take the same line. ``numpy`` is
    deliberately not probed: no extra declares it, so it is missing from lanes
    that still collect this file, and the repository's lane dependency policy
    fails on a probe no lane can satisfy (``tests/unit/test_algorithms_pca.py``
    records the same finding). The NumPy values are measured in the pull request
    body instead.
    """

    for value in (torch.tensor(True), torch.tensor([True]), torch.tensor([0.3 + 0j])):
        with pytest.raises(
            TypeError, match="gate 'ry' parameter 'theta' must be a real"
        ):
            fq.Circuit(1).ry(0, value)
    # A real tensor is the same branch, one dtype over.
    assert fq.Circuit(1).ry(0, torch.tensor(0.3)).to_ir().instructions[0].params[
        "theta"
    ].item() == pytest.approx(0.3)


def test_a_gate_parameter_accepts_a_real_sequence() -> None:
    """A list or tuple of reals is how a per-batch angle is written."""

    for value in ([0.3], (0.3,), [0.3, 0.4], []):
        circuit = fq.Circuit(1).ry(0, value)
        assert circuit.to_ir().instructions[0].params["theta"] == value


def test_a_gate_parameter_accepts_a_real_tensor_and_keeps_its_gradient() -> None:
    """A bound tensor is the autograd path, so it must survive as the same tensor."""

    angle = torch.tensor(0.3, requires_grad=True)
    circuit = fq.Circuit(1).ry(0, angle)
    stored = circuit.to_ir().instructions[0].params["theta"]
    assert stored is angle
    assert stored.requires_grad


def test_a_gate_parameter_accepts_the_double_single_carrier() -> None:
    """The P4 executor's own angle carrier is a real value, not a wrong type.

    ``split_real_imag_device_double_single_conformance`` builds an ``Instruction``
    whose ``theta`` is a ``DoubleSingleTensor``, so the carrier has to say what it
    holds. It reports float32, which is what it stores.
    """

    from flagquantum.simulation.numerics.double_single import DoubleSingleTensor

    pair = DoubleSingleTensor.from_float64(torch.tensor(0.3, dtype=torch.float64))
    assert pair.dtype == torch.float32
    circuit = fq.Circuit(1).ry(0, pair)
    assert circuit.to_ir().instructions[0].params["theta"] is pair


def test_a_gate_parameter_accepts_a_named_parameter_and_an_expression() -> None:
    """An unbound angle has no type to check yet, so it is not a wrong type."""

    named = fq.Parameter("t")
    assert fq.Circuit(1).ry(0, named).to_ir().instructions[0].params["theta"] is named
    expression = 1.0 + named
    assert (
        fq.Circuit(1).ry(0, expression).to_ir().instructions[0].params["theta"]
        is expression
    )


def test_a_refused_gate_parameter_leaves_the_circuit_unchanged() -> None:
    """A rejected call must not append a half-built instruction."""

    circuit = fq.Circuit(1).h(0)
    with pytest.raises(TypeError, match="must be a real number"):
        circuit.ry(0, "0.3")
    assert [instruction.name for instruction in circuit.to_ir().instructions] == ["h"]
