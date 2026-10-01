"""Conformance of the detector error model text against an independent reader.

``tests/qec/test_dem_stim_text.py`` checks that the stimulus text format
round-trips through this module's own writer and reader. That is a
self-consistency property, and on its own it cannot distinguish a text format
that Stim accepts from one that only FlagQuantum accepts. This file supplies the
independent reader: the real ``stim`` distribution parses what
``DetectorErrorModel.to_stim_text`` emits, and its reading of every error
mechanism is compared against the model that produced the text.

What this file proves
---------------------
The outbound direction interoperates. Stim parses the emitted text, agrees on
the detector count, the observable count and the error count, and reports the
same set of (probability, detectors, observables) mechanisms the model states.

What this file does not prove, and the exact shape of the gap
------------------------------------------------------------
The inbound direction does not interoperate, and this file pins each refusal
rather than describing it in prose. A detector error model produced by Stim for
an ordinary repeated circuit is refused for three independent reasons:

1. ``shift_detectors``. Stim addresses detectors relatively between rounds, so
   every multi-round circuit produces this instruction. This reader requires
   absolute indices.
2. ``^``. Stim marks a decomposed error mechanism's separator with ``^``. This
   reader accepts only ``D`` and ``L`` targets.
3. A missing ``logical_observable`` declaration. Stim introduces an observable
   through the error targets that flip it and need not declare it, so
   ``num_observables`` must be inferred from the targets as well as from the
   declarations.

The three tests below are inverted guards, not satisfied contracts: they assert
the refusal that exists today. When the reader learns a construct its test must
be rewritten to assert the accepted model, and deleting one instead would drop
the only evidence that the gap was ever measured.

Stim is an optional dependency, so this file skips when it is absent.
"""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration

stim = pytest.importorskip("stim")


def _model() -> DetectorErrorModel:
    return DetectorErrorModel(
        num_detectors=3,
        num_observables=1,
        errors=(
            DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
            DemError(probability=0.125, detectors=(2,), observables=()),
        ),
    )


def _repeated_circuit() -> "stim.Circuit":
    """A circuit whose error model uses every construct the reader refuses.

    A single-round circuit would not exercise the relative addressing, so the
    gap tests below would pass vacuously.
    """

    return stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )


def _mechanisms(
    model: "stim.DetectorErrorModel",
) -> set[tuple[float, frozenset[int], frozenset[int]]]:
    """Read every error mechanism the way an independent consumer would.

    ``DemTarget.val`` is the index of a ``D`` or ``L`` target. It raises for a
    decomposition separator, which only a decomposed model carries, so this
    helper states the undecomposed model it is used on by refusing the case.
    """

    mechanisms = set()
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        (probability,) = instruction.args_copy()
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            assert (
                not target.is_separator()
            ), "this comparison reads an undecomposed error model"
            if target.is_relative_detector_id():
                detectors.add(target.val)
            elif target.is_logical_observable_id():
                observables.add(target.val)
        mechanisms.add((probability, frozenset(detectors), frozenset(observables)))
    return mechanisms


def test_stim_parses_the_text_this_module_emits() -> None:
    model = _model()

    parsed = stim.DetectorErrorModel(model.to_stim_text())

    assert parsed.num_detectors == model.num_detectors
    assert parsed.num_observables == model.num_observables
    assert parsed.num_errors == model.num_errors


def test_stim_reads_the_same_mechanisms_the_model_states() -> None:
    """Agreement on counts could still hide a moved detector index."""

    model = _model()

    parsed = stim.DetectorErrorModel(model.to_stim_text())

    assert _mechanisms(parsed) == {
        (error.probability, frozenset(error.detectors), frozenset(error.observables))
        for error in model.errors
    }


def test_an_error_free_model_survives_the_round_trip_through_stim() -> None:
    """The declaration lines are what carry a shape no error touches."""

    model = DetectorErrorModel(num_detectors=5, num_observables=2, errors=())

    parsed = stim.DetectorErrorModel(model.to_stim_text())

    assert parsed.num_detectors == 5
    assert parsed.num_observables == 2
    assert parsed.num_errors == 0


def test_the_repeated_reference_circuit_exercises_both_refused_constructs() -> None:
    """Without this the two gap tests below could pass on an empty premise."""

    model = _repeated_circuit().detector_error_model(decompose_errors=False)

    text = str(model)
    assert "shift_detectors" in text
    assert "logical_observable" not in text
    assert model.num_observables == 1


def test_stim_declares_an_observable_through_its_error_targets_alone() -> None:
    """The third refusal is a shape rule, not an unknown instruction."""

    text = str(
        _repeated_circuit().detector_error_model(decompose_errors=False).flattened()
    )

    assert "logical_observable" not in text
    assert "L0" in text
    assert stim.DetectorErrorModel(text).num_observables == 1


def test_a_relative_detector_index_is_refused_today() -> None:
    text = str(_repeated_circuit().detector_error_model(decompose_errors=False))

    with pytest.raises(ValueError, match="shift_detectors is not supported"):
        DetectorErrorModel.from_stim_text(text)


def test_a_decomposition_separator_is_refused_today() -> None:
    text = str(_repeated_circuit().detector_error_model(decompose_errors=True))

    assert "^" in text
    with pytest.raises(ValueError, match=r"not '\^'"):
        DetectorErrorModel.from_stim_text(text)


def test_an_undeclared_observable_is_refused_today() -> None:
    text = str(
        _repeated_circuit().detector_error_model(decompose_errors=False).flattened()
    )

    with pytest.raises(ValueError, match="outside the model shape"):
        DetectorErrorModel.from_stim_text(text)
