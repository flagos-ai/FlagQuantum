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
Both directions interoperate. Stim parses the emitted text, agrees on the
detector count, the observable count and the error count, and reports the same
set of (probability, detectors, observables) mechanisms the model states. And
this reader parses what Stim writes for an ordinary repeated circuit, agreeing
with Stim on the shape, on the error count, and on every mechanism.

The four constructs that used to be refused
-------------------------------------------
An earlier revision of this file pinned refusals instead, because a detector error
model produced by Stim for an ordinary repeated circuit was refused for four
independent reasons:

1. ``shift_detectors``. Stim addresses detectors relatively between rounds, so
   every multi-round circuit produces this instruction. The reader required
   absolute indices.
2. ``^``. Stim marks a decomposed error mechanism's separator with ``^``. The
   reader accepted only ``D`` and ``L`` targets.
3. A missing ``logical_observable`` declaration. Stim introduces an observable
   through the error targets that flip it and need not declare it, so
   ``num_observables`` had to be inferred from the targets as well as from the
   declarations.
4. ``repeat``. Stim keeps a round's instructions in a block once the circuit is
   long enough, even in the ``flattened()`` form, so a five-round circuit was
   refused outright and ``flatten_loops=True`` was not a route around it.

All four are now accepted, and the tests below assert the accepted model. Tests
are kept from the refusal era in inverted form, because they are the evidence that
each construct was measured rather than assumed: the tests that establish the
premise still exist, and they check that the repeated reference circuit still
exercises the constructs that were refused. A future change therefore cannot
quietly stop exercising any one of them.

Blocks and flattening are compared rather than described. Reading a block is an
expansion, so the test for it does not stop at "stim accepts the same text": it
requires this reader's model of a block-carrying text to equal this reader's model
of the text after Stim's own ``flattened()``, and it requires both to agree with
Stim's reading of the block. The two readers agree on the shape, the error count
and every mechanism across a sweep that includes both code families, so a reading
that hoisted a block's ``shift_detectors`` out of the loop -- which would place
every copy of a mechanism at the same detector -- fails there rather than passing.

What this reader still refuses, and why
---------------------------------------
Two texts stim's own printer can write are refused, for one reason each, and both
are narrowings this package chose rather than constructs it has not reached.

The first is a text that states no ``detector`` declaration. This reader takes the
detector count from the declarations, and the model refuses fewer than one
detector, so ``error(0.1) D0`` -- which is what stim writes for a flat
one-detector circuit such as ``X_ERROR(0.1) 0; M 0; DETECTOR rec[-1]`` -- and
``error(0.1) L0``, which is what it writes for an observable-only circuit, are
both refused. Stim reads either text and infers the count from the error targets,
and this reader infers the *observable* count that way because stim declares an
observable exactly when no mechanism references it. Reading the detector count off
the largest index an error mentions would accept a text that had lost its trailing
detectors, which is why the declaration is the authority here; and a model with no
detector is not one this record can hold, which is the model's own rule and not
the reader's. Every shape stim writes for a memory circuit declares the detectors
whose index the errors do not imply, so the 240-model sweep below is unaffected by
either narrowing.

The probability digits are stim's printer to choose. ``str(model)`` writes at
``std::setprecision(std::numeric_limits<long double>::digits10 + 1)``, so
nineteen significant digits on the x86-64 Linux runners, where ``long double``
is the 80-bit extended type, and sixteen on arm64 macOS, where it is a double.
Seventeen digits name a double uniquely, so the former returns the identical
double for every mechanism while the latter can differ from the in-memory
probability in the last place; across the 240-model developer-time sweep
described in ``flagquantum/qec/IMPLEMENTATION.md`` the largest relative
difference measured on the sixteen-digit platform was 4.9e-16. The text is the
interchange format, so the text's value is the one that survives, and the two
readers agree exactly on it.

Stim is an optional dependency, so this file skips when it is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RotatedSurfaceCode
from flagquantum.qec.dem import DemError, DetectorErrorModel

if TYPE_CHECKING:
    import stim

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
    """A circuit whose error model used to exercise every refused construct.

    A single-round circuit would not exercise the relative addressing, so the
    tests below would pass vacuously.
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

    A shot of a mechanism flips the symmetric difference of its targets: a
    repeated detector cancels, and a ``^`` separator partitions the targets
    without changing which ones are flipped. This helper implements that rule
    from ``stim``'s own objects rather than from this module's parser, so the
    two readings are independent.
    """

    mechanisms = set()
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        (probability,) = instruction.args_copy()
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
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


def test_the_repeated_reference_circuit_exercises_both_previously_refused_constructs() -> (
    None
):
    """Without this the two acceptance tests below could pass on an empty premise."""

    model = _repeated_circuit().detector_error_model(decompose_errors=False)

    text = str(model)
    assert "shift_detectors" in text
    assert "logical_observable" not in text
    assert model.num_observables == 1


def test_stim_declares_an_observable_through_its_error_targets_alone() -> None:
    """The third refusal was a shape rule, not an unknown instruction."""

    text = str(
        _repeated_circuit().detector_error_model(decompose_errors=False).flattened()
    )

    assert "logical_observable" not in text
    assert "L0" in text
    assert stim.DetectorErrorModel(text).num_observables == 1


def test_a_relative_detector_index_is_now_read_as_absolute() -> None:
    """The premise is a real circuit, so this cannot pass on hand-written text."""

    dem = _repeated_circuit().detector_error_model(decompose_errors=False)
    text = str(dem)

    assert "shift_detectors" in text
    parsed = DetectorErrorModel.from_stim_text(text)
    reread = stim.DetectorErrorModel(text)

    assert (parsed.num_detectors, parsed.num_observables, parsed.num_errors) == (
        reread.num_detectors,
        reread.num_observables,
        reread.num_errors,
    )
    assert {
        (e.probability, frozenset(e.detectors), frozenset(e.observables))
        for e in parsed.errors
    } == _mechanisms(reread)


def test_a_decomposition_separator_is_now_read_as_a_symmetric_difference() -> None:
    """Two readers of the same decomposed text agree on every mechanism."""

    text = str(_repeated_circuit().detector_error_model(decompose_errors=True))

    assert "^" in text
    parsed = DetectorErrorModel.from_stim_text(text)
    reread = stim.DetectorErrorModel(text)

    assert parsed.num_errors == reread.num_errors
    assert {
        (e.probability, frozenset(e.detectors), frozenset(e.observables))
        for e in parsed.errors
    } == _mechanisms(reread)


def _decomposing_circuit() -> "stim.Circuit":
    """A circuit whose decomposition separates the two readings of ``^``.

    The gap is what a test needs, and the repetition code does not supply one:
    at distance three, three rounds and one percent noise, reading only the
    first group moves every detector rate by at most 0.0024, which any
    reasonable tolerance absorbs. A rotated surface code at distance five, two
    rounds and two percent noise moves the worst rate by 0.094, because its
    decompositions are the ones that actually merge groups.
    """

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=2,
        distance=5,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.02,
    )


def test_the_decomposed_text_agrees_with_the_flat_text_on_every_marginal_rate() -> None:
    """The claim the separator route has to earn: both texts describe one model.

    Decomposition splits a composite mechanism into its graphlike parts, so the
    mechanism sets differ and the mechanism count differs -- this test asserts
    that they do, so it cannot pass on a model that never carried a separator.
    What may not differ is the physics, and the marginal detector and observable
    rates are where that shows.
    """

    circuit = _decomposing_circuit()
    decomposed_text = str(circuit.detector_error_model(decompose_errors=True))
    assert "^" in decomposed_text

    decomposed = DetectorErrorModel.from_stim_text(decomposed_text)
    flat = DetectorErrorModel.from_stim_text(
        str(circuit.detector_error_model(decompose_errors=False))
    )

    assert decomposed.errors != flat.errors
    assert decomposed.detector_error_matrix().shape != (
        flat.detector_error_matrix().shape
    )

    for mine, theirs in (
        (decomposed.detector_rates(), flat.detector_rates()),
        (decomposed.observable_rates(), flat.observable_rates()),
    ):
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs, strict=True):
            assert abs(a - b) < 1e-9


def test_the_separator_reading_reproduces_stim_sampler_and_only_the_symmetric_difference_would() -> (
    None
):
    """Ground truth for the separator rule, from stim's own sampler on the text.

    A tolerance is only evidence if a wrong rule misses it, so the budget and
    the discrimination are both stated. At 200000 shots a rate near 0.15 has a
    standard error of about 0.0008, which puts the 0.004 tolerance at five
    standard errors; this reader's worst measured deviation is 0.0018. Reading
    only the first group instead moves the worst detector rate by 0.094, more
    than twenty times the tolerance, so the assertion separates the two readings
    rather than merely passing.
    """

    text = str(_decomposing_circuit().detector_error_model(decompose_errors=True))
    model = DetectorErrorModel.from_stim_text(text)

    det, obs = (
        torch.as_tensor(part)
        for part in stim.DetectorErrorModel(text)
        .compile_sampler()
        .sample(shots=200000)[:2]
    )

    for mine, theirs in (
        (model.detector_rates(), det.to(torch.float64).mean(dim=0)),
        (model.observable_rates(), obs.to(torch.float64).mean(dim=0)),
    ):
        assert mine.shape == theirs.shape
        assert bool((torch.abs(mine - theirs) < 0.004).all())


def test_an_undeclared_observable_is_now_inferred_from_the_error_targets() -> None:
    """The shape comes from the declarations when there are any and from the targets otherwise."""

    text = str(
        _repeated_circuit().detector_error_model(decompose_errors=False).flattened()
    )

    parsed = DetectorErrorModel.from_stim_text(text)

    assert parsed.num_observables == stim.DetectorErrorModel(text).num_observables


def test_a_repeat_block_is_read_as_the_model_its_flattened_form_states() -> None:
    """The last asymmetry with stim, closed and measured against stim itself.

    A long enough circuit keeps its block even when the loops are flattened, and
    unflattening is an expansion rather than a second statement of the format, so
    the two readings have to be compared rather than assumed: stim reads the
    block, and this reader must reach the model stim's own ``flattened()`` form
    reaches, mechanism for mechanism. The premise is asserted first, so this
    cannot pass on a text that never carried a block.
    """

    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=5,
        distance=3,
        after_clifford_depolarization=0.01,
    )
    blocked = str(circuit.detector_error_model(decompose_errors=False))
    flattened = str(circuit.detector_error_model(decompose_errors=False).flattened())

    assert "repeat" in blocked
    assert "repeat" not in flattened

    parsed = DetectorErrorModel.from_stim_text(blocked)
    reread = stim.DetectorErrorModel(blocked)

    assert (parsed.num_detectors, parsed.num_observables, parsed.num_errors) == (
        reread.num_detectors,
        reread.num_observables,
        reread.num_errors,
    )
    assert {
        (e.probability, frozenset(e.detectors), frozenset(e.observables))
        for e in parsed.errors
    } == _mechanisms(reread)
    assert parsed == DetectorErrorModel.from_stim_text(flattened)


@pytest.mark.parametrize(
    ("task", "distance", "rounds"),
    [
        ("repetition_code:memory", 3, 5),
        ("repetition_code:memory", 5, 9),
        ("surface_code:rotated_memory_z", 3, 9),
        ("surface_code:rotated_memory_z", 5, 9),
    ],
)
def test_every_block_stim_writes_is_read_as_stim_reads_it(
    task: str, distance: int, rounds: int
) -> None:
    """Blocks appear where the round count is long enough, for two code families.

    The shapes are the ones stim 1.16.0 was measured to write a block in: a
    repetition code from five rounds and a rotated surface code from nine, while
    the shorter circuits of the same families write the same instructions out.
    The block is the format's only construct whose meaning is arithmetic rather
    than a field, so it is checked on texts stim wrote for several shapes rather
    than on one hand-picked example. Each case asserts that stim really emitted a
    block, so a stim release that stopped writing one fails here rather than
    turning the comparison vacuous, and then requires agreement with stim on the
    declared shape, on the error count and on every mechanism.
    """

    text = str(
        stim.Circuit.generated(
            task,
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01,
            after_reset_flip_probability=0.01,
        ).detector_error_model(decompose_errors=False)
    )

    assert "repeat" in text
    parsed = DetectorErrorModel.from_stim_text(text)
    reread = stim.DetectorErrorModel(text)

    assert (parsed.num_detectors, parsed.num_observables, parsed.num_errors) == (
        reread.num_detectors,
        reread.num_observables,
        reread.num_errors,
    )
    assert {
        (e.probability, frozenset(e.detectors), frozenset(e.observables))
        for e in parsed.errors
    } == _mechanisms(reread)


def test_the_two_texts_stim_writes_without_a_declaration_are_refused() -> None:
    """The narrowing that remains, measured on stim's own output.

    Stim infers a detector count from the error targets where this reader takes it
    from the declarations, so a flat one-detector circuit and an observable-only
    circuit produce texts stim reads and this reader refuses. Both premises are
    asserted before the refusals are, so a stim release that started declaring
    these detectors fails here rather than leaving the claim standing on a text it
    no longer writes.
    """

    flat = str(
        stim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]").detector_error_model()
    )
    observable_only = str(
        stim.Circuit(
            "X_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
        ).detector_error_model()
    )

    assert "detector" not in flat
    assert "detector" not in observable_only
    assert stim.DetectorErrorModel(flat).num_detectors == 1

    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text(flat)
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text(observable_only)


def test_the_parsed_model_predicts_the_circuit_it_came_from() -> None:
    """The end-to-end check: sample the model and the circuit, compare rates.

    This is the strongest available ground truth, because the text was derived
    from the circuit rather than from another reading of itself, and it is the
    only check here that exercises :meth:`dem_sampling` against a real device
    model. A rule that kept only the first group of a decomposed mechanism would
    move the worst detector rate by 0.094 on this circuit, twelve times the
    tolerance, so the assertion separates the readings.

    The budget is stated rather than implied: 100000 shots put the standard
    error on a rate near 0.2 at about 0.0013, and the two samples are
    independent, so the 0.008 tolerance is about four and a half standard errors
    of their difference. This reader's measured worst difference is 0.0041.
    """

    circuit = _decomposing_circuit()
    text = str(circuit.detector_error_model(decompose_errors=True))
    assert "^" in text

    mine = DetectorErrorModel.from_stim_text(text).dem_sampling(shots=100000, seed=7)
    det, obs = (
        torch.as_tensor(part)
        for part in circuit.compile_detector_sampler(seed=7).sample(
            shots=100000, separate_observables=True
        )
    )

    assert mine.detectors.shape == det.shape
    assert mine.observables.shape == obs.shape
    for mine_part, theirs in (
        (mine.detectors, det),
        (mine.observables, obs),
    ):
        difference = mine_part.to(torch.float64).mean(dim=0) - theirs.to(
            torch.float64
        ).mean(dim=0)
        assert float(difference.abs().max()) < 0.008


@pytest.mark.parametrize("rounds", (1, 2, 3, 4))
@pytest.mark.parametrize("distance", (2, 3, 4, 5, 6, 7, 8))
def test_the_surface_code_declares_the_detector_shape_stim_declares(
    distance: int, rounds: int
) -> None:
    """The generated rotated surface code pins the declared shape for the same code.

    Stim's ``surface_code:rotated_memory_z`` is a second, independent statement
    of how many detectors and observables a distance ``d`` patch of ``r`` rounds
    must declare. This module builds the same experiment from the code's checks,
    so comparing the two at 28 (distance, rounds) points checks the layout rule
    the checks imply rather than a number transcribed from Stim.

    The count has to follow the two check classes. Both the initial state and the
    terminal readout are in the Z basis, so each Z-type check is deterministic in
    round zero and at the terminal readout and gets ``rounds + 1`` detectors,
    while each X-type check is deterministic only against the round before it and
    gets ``rounds - 1``. Reading every check as Z-type instead — the rule this
    record used before X-type checks were expressible — would report
    ``(nx + nz) * (rounds + 1)``, which at ``d=3, rounds=2`` is 24 against Stim's
    16, so the assertion separates the two rules rather than merely agreeing with
    one of them.

    The comparison is on the declared shape. Which detector a particular error
    mechanism flips is a separate question, and answering it needs a
    mechanism-level correspondence between the two circuits rather than a count.
    """

    code = RotatedSurfaceCode(distance=distance)
    x_checks = sum(1 for check in code.checks if check.stabilizer.x_qubits)
    z_checks = len(code.checks) - x_checks
    memory = build_memory_circuit(code, rounds=rounds)

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=rounds
    )
    theirs = circuit.detector_error_model(decompose_errors=False)

    assert len(memory.detectors.detectors) == theirs.num_detectors
    assert len(memory.observables.observables) == theirs.num_observables
    assert len(memory.detectors.detectors) == (
        z_checks * (rounds + 1) + x_checks * (rounds - 1)
    )
    assert len(memory.detectors.detectors) < (x_checks + z_checks) * (rounds + 1)


def _sampled_rates(text: str, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stim's own marginals for a detector error model text.

    The sampler is the ground truth for what the text means, because the text is
    a statement about the distribution and not about a parse tree.
    """

    det, obs = (
        torch.as_tensor(part)
        for part in stim.DetectorErrorModel(text)
        .compile_sampler()
        .sample(shots=shots)[:2]
    )
    return det.to(torch.float64).mean(dim=0), obs.to(torch.float64).mean(dim=0)


def test_the_suggestion_reading_departs_from_stim_where_the_default_reading_does_not() -> (
    None
):
    """The cost of the decomposition suggestion, measured rather than asserted.

    Upstream offers the same flag, so the alignment is that this reader offers
    the same two readings and that the default one is stim's. Which reading stim
    itself means is not a matter of taste: stim's sampler on the same text is the
    reference, and at 200000 shots a rate near 0.15 carries a standard error of
    about 0.0008, so the 0.004 tolerance sits at five standard errors.

    The default reading is inside it -- worst measured deviation 0.0022 on the
    detectors and 0.0016 on the observables. Expanding the components at the
    parent probability is not: it holds the detector marginals (0.0022) but
    misses the observable ones by 0.048, twelve times the tolerance and sixty
    times the standard error. That separation is the point of the test. A
    tolerance that both readings pass would not be evidence about either.
    """

    text = str(_decomposing_circuit().detector_error_model(decompose_errors=True))
    assert "^" in text
    det, obs = _sampled_rates(text, shots=200000)

    stated = DetectorErrorModel.from_stim_text(text)
    expanded = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)

    # The two readings really are different models of the same text.
    assert expanded.num_errors > stated.num_errors

    stated_observable = float((stated.observable_rates() - obs).abs().max())
    expanded_observable = float((expanded.observable_rates() - obs).abs().max())

    assert float((stated.detector_rates() - det).abs().max()) < 0.004
    assert stated_observable < 0.004
    assert expanded_observable > 0.01
    assert expanded_observable > stated_observable


def test_stim_reads_a_detector_named_twice_as_a_symmetric_difference() -> None:
    """The rule both the default reading and stim use, on the text that shows it.

    ``D0`` is named in both groups, so the line's signature is ``D1`` alone and
    ``D0`` never flips. The suggestion reading instead returns one mechanism per
    group, which puts ``D0`` at probability 0.18 -- far above the 0.002 a
    standard error at this shot count would allow -- so the assertion tells the
    two readings apart instead of rounding them together.
    """

    text = "error(0.1) D0 D1 ^ D0\ndetector D0\ndetector D1\n"
    det, _ = _sampled_rates(text, shots=200000)

    stated = DetectorErrorModel.from_stim_text(text)
    expanded = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)

    assert float(det[0]) < 0.004
    assert float((stated.detector_rates() - det).abs().max()) < 0.008
    assert float(expanded.detector_rates()[0]) == pytest.approx(0.18)
