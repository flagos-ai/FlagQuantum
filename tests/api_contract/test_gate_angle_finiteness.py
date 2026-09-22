"""Contract for what magnitude a gate angle may have.

A gate parameter in the operator registry is a rotation angle, so it has to denote a
real number -- and a *finite* one. The type rule is the neighbouring contract, in
``test_gate_parameter_semantics.py``; this file is about the value, not the type.

The builder used to store whatever magnitude it was handed and leave the value
unread until the simulator resolved it, so ``ry(0, nan)`` returned
``tensor([[nan, nan]])`` **as its result**, with no error at all, and
``ry(0, nan).counts(64)`` failed a layer below the mistake in ``torch.multinomial``
with a message that named neither the gate nor the parameter. ``inf`` behaved the
same way.

A wrong magnitude is a ``ValidationError``, which is what the errors-module boundary
reserves for a value, and which is what the neighbouring type rule deliberately does
*not* use: a wrong type stays a ``TypeError``. ``nan`` is a real number, so it is not
a type error, and the assertions below check the class rather than only that
something was raised.

The file imports only ``torch``, ``pytest`` and the standard library on purpose, for
the same reason the neighbouring file gives: a ``unit`` test is collected by every
CPU lane, including ``triton-optional``, which installs ``torch`` and not much else.
A NumPy scalar reaches the rule through ``numbers.Real``, which ``float`` and
``Fraction`` already cover here; a one-dimensional NumPy array is walked as a
sequence, which the list and tuple cases cover; and a zero-dimensional one is read as
the scalar it is, which ``_ScalarOnlyCarrier`` covers. ``numpy`` itself is not
imported, because no extra declares it and a probe no lane can satisfy fails the
repository's lane dependency policy. The NumPy values are measured in the pull request
body instead.

The JAX path is covered by ``tests/test_hybrid_jax.py``, which the ``jax-optional``
lane runs: a builder compiled under ``jax.jit`` receives a tracer rather than a
number, and that lane fails without the deferral this file's rule depends on.
"""

from __future__ import annotations

import fractions
import math
from collections.abc import Iterator

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import ValidationError
from flagquantum.observables import probabilities
from flagquantum.simulation.numerics.double_single import DoubleSingleTensor

pytestmark = pytest.mark.unit

# Every parameterized opcode in the registry, with a filler angle for each of its
# parameters, so the rule is about angles and not about one gate. `u2` and `u3` name
# their parameters differently on purpose and carry more than one.
PARAMETERIZED_GATES = (
    ("rx", (0,), {"theta": 0.5}),
    ("ry", (0,), {"theta": 0.5}),
    ("rz", (0,), {"theta": 0.5}),
    ("phase", (0,), {"theta": 0.5}),
    ("u1", (0,), {"theta": 0.5}),
    ("crx", (0, 1), {"theta": 0.5}),
    ("cry", (0, 1), {"theta": 0.5}),
    ("crz", (0, 1), {"theta": 0.5}),
    ("cphase", (0, 1), {"theta": 0.5}),
    ("rxx", (0, 1), {"theta": 0.5}),
    ("ryy", (0, 1), {"theta": 0.5}),
    ("rzz", (0, 1), {"theta": 0.5}),
    ("u2", (0,), {"phi": 0.5, "lbd": 0.5}),
    ("u3", (0,), {"theta": 0.5, "phi": 0.5, "lbd": 0.5}),
)

# One case per (gate, parameter) pair, so `u3` is asked about three times and never
# about only the parameter that happens to be written first.
GATE_CASES = tuple(
    (gate, wires, parameter, dict(params))
    for gate, wires, params in PARAMETERIZED_GATES
    for parameter in params
)

NON_FINITE_SCALARS = (math.nan, math.inf, -math.inf)
NON_FINITE_LABELS = ("nan", "the positive infinity", "the negative infinity")


def _with_angle(
    gate: str,
    wires: tuple[int, ...],
    parameter: str,
    params: dict[str, float],
    value: object,
) -> object:
    """Call one parameterized gate, putting ``value`` in ``parameter``."""

    others = {name: angle for name, angle in params.items() if name != parameter}
    return getattr(fq.Circuit(2), gate)(*wires, **{parameter: value}, **others)


@pytest.mark.parametrize("value", NON_FINITE_SCALARS, ids=NON_FINITE_LABELS)
@pytest.mark.parametrize(
    ("gate", "wires", "parameter", "params"),
    GATE_CASES,
    ids=[f"{gate}-{parameter}" for gate, _, parameter, _ in GATE_CASES],
)
def test_a_gate_angle_refuses_a_non_finite_value(
    gate: str,
    wires: tuple[int, ...],
    parameter: str,
    params: dict[str, float],
    value: float,
) -> None:
    """Every parameterized gate refuses a non-finite angle where it is stored."""

    with pytest.raises(
        ValidationError,
        match=f"gate {gate!r} parameter {parameter!r} must be a finite real number",
    ):
        _with_angle(gate, wires, parameter, params, value)


def test_a_non_finite_angle_is_a_value_error_and_not_a_type_error() -> None:
    """``nan`` is a real number of the wrong magnitude, so the class says so.

    The neighbouring type rule is the contrast, and it keeps its own class; if this
    ever collapses to one exception type, the boundary has been lost.
    """

    with pytest.raises(ValidationError) as raised:
        fq.Circuit(1).ry(0, math.nan)
    assert not isinstance(raised.value, TypeError)
    assert isinstance(raised.value, ValueError)
    with pytest.raises(TypeError) as mistyped:
        fq.Circuit(1).ry(0, "0.3")
    assert not isinstance(mistyped.value, ValidationError)


def test_a_non_finite_angle_never_becomes_a_nan_distribution() -> None:
    """The reported symptom, and the finite control that makes it mean something.

    ``run`` used to hand back ``tensor([[nan, nan]])`` as the answer. The refusal now
    happens while the circuit is built, so no execution is reached; the control run
    shows that a finite angle still produces a finite distribution.
    """

    with pytest.raises(ValidationError, match="must be a finite real number"):
        fq.Circuit(1).ry(0, math.nan).run(outputs=probabilities())
    with pytest.raises(ValidationError, match="must be a finite real number"):
        fq.Circuit(1).ry(0, math.inf)
    finite = fq.Circuit(1).ry(0, 0.3).run(outputs=probabilities()).probabilities
    assert bool(torch.isfinite(finite).all())


@pytest.mark.parametrize(
    "value",
    [torch.tensor(math.nan), torch.tensor([0.3, math.nan]), torch.tensor([[math.inf]])],
    ids=["scalar", "vector", "matrix"],
)
def test_a_gate_angle_refuses_a_tensor_holding_a_non_finite_number(
    value: torch.Tensor,
) -> None:
    """A tensor answers for every number it carries, not only for the first."""

    with pytest.raises(
        ValidationError, match="gate 'ry' parameter 'theta' must be a finite real"
    ):
        fq.Circuit(1).ry(0, value)


@pytest.mark.parametrize(
    "value",
    [[0.3, math.nan], (math.inf,), [[math.nan]]],
    ids=["list", "tuple", "nested list"],
)
def test_a_gate_angle_refuses_a_sequence_holding_a_non_finite_number(
    value: object,
) -> None:
    """A per-batch angle is written as a sequence, so it is read element by element."""

    with pytest.raises(
        ValidationError, match="gate 'ry' parameter 'theta' must be a finite real"
    ):
        fq.Circuit(1).ry(0, value)


class _ScalarOnlyCarrier:
    """The shape of a zero-dimensional array from a library this file cannot import.

    ``numpy`` is neither a core dependency nor declared by an extra, and a probe for a
    package no lane installs fails the repository's own lane dependency policy, so a
    bare ``import numpy`` here would be worse than useless -- it would be missing from
    the lanes that collect this file. The rule reads a zero-dimensional value from its
    *shape* -- ``ndim`` and ``item`` -- rather than from its type, so a carrier with
    those two members reaches exactly the line a NumPy 0-d array reaches. ``__iter__``
    raises, as a real zero-dimensional array does, so a reader that walks the value
    instead of asking for its scalar fails here rather than passing by luck. The real
    NumPy values are measured in the pull request body.
    """

    def __init__(self, value: object) -> None:
        self._value = value
        self.dtype = "float64"

    @property
    def ndim(self) -> int:
        return 0

    def item(self) -> object:
        return self._value

    def __iter__(self) -> Iterator[object]:
        raise TypeError("iteration over a 0-d array")


def test_a_zero_dimensional_angle_is_read_as_the_scalar_it_is() -> None:
    """A zero-dimensional value is asked for its number, not walked.

    ``iter`` over a zero-dimensional array raises instead of yielding it, so walking
    one reported ``TypeError: iteration over a 0-d array`` -- a message about the array
    library, from a rule about gate angles, and not the ``ValidationError`` this
    contract promises. The magnitude it carries is still what decides the answer.
    """

    fq.Circuit(1).ry(0, _ScalarOnlyCarrier(0.3))
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(
            ValidationError, match="gate 'ry' parameter 'theta' must be a finite real"
        ):
            fq.Circuit(1).ry(0, _ScalarOnlyCarrier(value))


def test_a_double_single_angle_answers_whether_its_words_are_finite() -> None:
    """The pair carrier is read through the same question the rule asks a tensor.

    ``_require_pair`` fixes both words to float32 without refusing a non-finite one,
    so without an answer here this carrier would be the single value the rule waves
    through, and it does reach a gate parameter on the P4 executor path.
    """

    finite = DoubleSingleTensor.from_float32(torch.tensor([0.3]))
    assert bool(finite.isfinite().all())
    fq.Circuit(1).ry(0, finite)
    # The high word is refused...
    nan_high = DoubleSingleTensor(torch.tensor([math.nan]), torch.zeros(1))
    with pytest.raises(ValidationError, match="must be a finite real number"):
        fq.Circuit(1).ry(0, nan_high)
    # ...and so is the low one, which carries the residual of the same angle.
    inf_low = DoubleSingleTensor(torch.tensor([0.3]), torch.tensor([math.inf]))
    with pytest.raises(ValidationError, match="must be a finite real number"):
        fq.Circuit(1).ry(0, inf_low)


def test_a_finite_angle_outside_float_range_is_still_an_angle() -> None:
    """A magnitude the simulator may refuse for range is not this rule's business.

    ``math.isfinite`` raises ``OverflowError`` for these rather than answering, and
    that is its own answer: the value is finite and merely too large for a float.
    Refusing it here would be a range check wearing a finiteness message.
    """

    huge = 10**400
    with pytest.raises(OverflowError):
        math.isfinite(huge)
    fq.Circuit(1).ry(0, huge)
    fq.Circuit(1).ry(0, fractions.Fraction(10**400))
    fq.Circuit(1).ry(0, fractions.Fraction(1, 3))


def test_a_symbolic_angle_is_still_accepted_and_checked_when_it_is_bound() -> None:
    """A ``Parameter`` is an angle whose value arrives later, so the check waits."""

    circuit = fq.Circuit(1).ry(0, fq.Parameter("t"))
    assert circuit.to_ir().instructions[0].params == {"theta": fq.Parameter("t")}
    # Binding rebuilds every Instruction, so the same rule reaches the bound value.
    assert circuit.bind_parameters({"t": 0.3}).to_ir().instructions[0].params == {
        "theta": 0.3
    }
    with pytest.raises(
        ValidationError, match="gate 'ry' parameter 'theta' must be a finite real"
    ):
        circuit.bind_parameters({"t": math.nan})
    with pytest.raises(
        ValidationError, match="gate 'ry' parameter 'theta' must be a finite real"
    ):
        circuit.bind_parameters({"t": torch.tensor(math.inf)})


def test_instruction_refuses_a_non_finite_angle_however_it_was_reached() -> None:
    """The builder, the IR constructor and the decoder share one contract."""

    with pytest.raises(ValidationError, match="gate 'ry' parameter 'theta' must be"):
        Instruction("ry", (0,), {"theta": math.nan})
    with pytest.raises(ValidationError, match="gate 'u2' parameter 'phi' must be"):
        Instruction("u2", (0,), {"phi": math.inf, "lbd": 0.5})
    with pytest.raises(ValidationError, match="gate 'rx' parameter 'theta' must be"):
        Instruction("rx", (0,), {"theta": [0.5, math.nan]})


def test_a_decoded_program_is_held_to_the_same_contract() -> None:
    """``from_dict`` is the other way in, and it does not get a laxer rule.

    The payload is built by hand rather than by serializing an invalid circuit,
    because the builder no longer produces one -- which is the point of the change.
    """

    payload = fq.Circuit(1).ry(0, 0.3).to_ir().to_dict()
    payload["instructions"][0]["params"]["theta"] = math.nan
    with pytest.raises(ValidationError, match="gate 'ry' parameter 'theta' must be"):
        CircuitIR.from_dict(payload)
    # The same payload with a finite angle decodes, so the refusal above is about
    # the magnitude and not about the payload's shape.
    payload["instructions"][0]["params"]["theta"] = 0.3
    assert CircuitIR.from_dict(payload).instructions[0].params == {"theta": 0.3}
