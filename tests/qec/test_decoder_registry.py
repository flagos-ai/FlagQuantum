"""The name-keyed registry of detector error model decoders.

`flagquantum/qec/registry.py` is the factory half of the decoder row: a name
reaches a class, and a call site that holds a detector error model does not have
to know which class that is. Three claims are asserted here, and the file is
arranged around them.

A name reaches a decoder that decodes. Every registered name is resolved, built
from a real model, and run, so a registration that resolved to a class which
cannot answer a syndrome would fail here rather than at the first caller. The
optional name is asked the same way, and the answer differs by host rather than
by claim: where the distribution is installed the decoder decodes, and where it
is not the name reaches this package's own refusal naming the extra, which is a
stated answer rather than a broken registration. Both hosts are asserted in the
one test, because a lane that installs no extras is a lane this file has to hold
on.

Registration is checked at registration. A class that carries neither `decode`
nor `from_detector_error_model` is refused while the registering module is being
imported, and a name is refused when it is empty, carries whitespace, is not a
string, or is already taken -- so the refusal is about the registry rather than
about one implementation.

The optional implementation is registered whether or not it is installed. Its
name is part of this package's surface and not part of the extra's, so the
registry lists it unconditionally and asking for it without PyMatching raises the
error that names the extra. The test that asserts this runs on both hosts:
installed and not installed are the same assertion about the name, and only the
success path differs.

The other half of the row is the boundary this module does not cross, and it is
pinned here too: the repetition-code decoders are *not* registered, because a
registry that held two input protocols would make a name mean one of two things.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    AUTHORITY_NAME,
    CROSS_CHECK_NAME,
    DecodingGraph,
    DetectorErrorModel,
    DetectorErrorModelDecoder,
    MatchingDependencyError,
    MinimumWeightMatchingDecoder,
    PyMatchingDecoder,
    RepetitionLookupDecoder,
    build_memory_circuit,
    decoder_names,
    get_decoder,
    register_decoder,
)
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.noise import PhenomenologicalNoise

pytestmark = pytest.mark.unit


def _model(rounds: int = 2, strength: float = 0.02) -> DetectorErrorModel:
    """A small merged graphlike model, which both decoders accept."""

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(RepetitionCode(distance=3), rounds=rounds),
        noise=PhenomenologicalNoise(data_flip=strength, measurement_flip=strength),
    )
    return model.merge_duplicate_mechanisms()


def _syndrome(model: DetectorErrorModel) -> list[int]:
    """One syndrome of the model, taken from its own sampler."""

    sample = model.dem_sampling(shots=1, seed=20261003)
    return [int(index) for index in sample.detectors[0].nonzero()[0]]


def test_the_registry_names_the_authority_and_the_cross_check() -> None:
    names = decoder_names()
    assert AUTHORITY_NAME in names
    assert CROSS_CHECK_NAME in names
    assert names == tuple(sorted(names))


@pytest.mark.parametrize("name", [AUTHORITY_NAME, CROSS_CHECK_NAME])
def test_every_registered_name_builds_a_decoder_that_decodes(name: str) -> None:
    """Every name answers on every host, and the optional one says which host it is.

    The authority carries no extra and so is built and run wherever this file
    runs. The cross-check carries one, so its answer depends on the host rather
    than on the registry: built and run where the distribution is installed, and
    otherwise the refusal that names the extra. The refusal is only accepted for
    the optional name, so a lane that lost the authority's own implementation
    cannot pass here by raising something that looks like a missing extra.
    """

    model = _model()
    try:
        decoder = get_decoder(name, model)
    except MatchingDependencyError as exc:
        assert name == CROSS_CHECK_NAME, (
            f"{name} needs no optional distribution, so its absence is a defect "
            "in the registry rather than a missing extra"
        )
        assert CROSS_CHECK_NAME in str(exc) or "pymatching" in str(exc)
        return

    result = decoder.decode(_syndrome(model))

    assert isinstance(result.observables, tuple)
    assert result.weight > 0.0


def test_the_authority_is_the_class_the_name_is_registered_against() -> None:
    decoder = get_decoder(AUTHORITY_NAME, _model())
    assert isinstance(decoder, MinimumWeightMatchingDecoder)


def test_the_authority_route_from_a_model_matches_a_direct_construction() -> None:
    model = _model()
    through_name = get_decoder(AUTHORITY_NAME, model)
    direct = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    syndrome = _syndrome(model)

    assert through_name.decode(syndrome) == direct.decode(syndrome)


def test_a_name_accepts_options_the_constructor_accepts() -> None:
    model = _model()
    decoder = get_decoder(AUTHORITY_NAME, model, max_defects=1)
    assert isinstance(decoder, MinimumWeightMatchingDecoder)
    assert decoder.max_defects == 1


def test_a_graph_is_taken_as_it_is_rather_than_lifted_through_a_model() -> None:
    model = _model()
    graph = DecodingGraph.from_detector_error_model(model)
    through_graph = get_decoder(AUTHORITY_NAME, graph)
    through_model = get_decoder(AUTHORITY_NAME, model)

    assert through_graph.graph is graph
    assert through_graph.decode(_syndrome(model)) == through_model.decode(
        _syndrome(model)
    )


def test_an_unregistered_name_lists_the_names_that_are_registered() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_decoder("chromobius", _model())
    message = str(excinfo.value)
    assert "chromobius" in message
    for name in decoder_names():
        assert name in message


def test_a_source_that_is_neither_a_model_nor_a_graph_is_refused() -> None:
    with pytest.raises(TypeError) as excinfo:
        get_decoder(AUTHORITY_NAME, [[1, 0], [0, 1]])  # type: ignore[arg-type]
    assert "DecodingGraph" in str(excinfo.value)


def test_stim_text_is_a_carrier_of_its_own() -> None:
    text = "error(0.1) D0 D1\ndetector D0\ndetector D1\n"
    through_text = get_decoder(AUTHORITY_NAME, text)
    through_model = get_decoder(AUTHORITY_NAME, DetectorErrorModel.from_stim_text(text))

    assert through_text.decode([0, 1]) == through_model.decode([0, 1])


def test_the_text_carrier_takes_the_text_readers_options() -> None:
    text = "error(0.1) D0 ^ D1\ndetector D0\ndetector D1\n"
    combined = get_decoder(AUTHORITY_NAME, text)
    split = get_decoder(AUTHORITY_NAME, text, use_decomp_suggestions=True)

    assert combined.graph.num_edges == 1
    assert split.graph.num_edges == 2


def test_the_text_carrier_refuses_options_that_belong_to_the_matcher() -> None:
    with pytest.raises(TypeError):
        get_decoder(AUTHORITY_NAME, "detector D0\n", max_defects=1)


def test_the_registry_refuses_a_class_without_the_two_members() -> None:
    class NoDecode:
        @classmethod
        def from_detector_error_model(cls, model: object) -> NoDecode:
            return cls()

    with pytest.raises(TypeError) as excinfo:
        register_decoder("no_decode")(NoDecode)  # type: ignore[arg-type]
    assert "decode" in str(excinfo.value)
    assert "no_decode" not in decoder_names()


def test_the_registry_refuses_a_class_without_a_model_constructor() -> None:
    class NoConstructor:
        def decode(self, detection_events: object) -> object:
            return detection_events

    with pytest.raises(TypeError) as excinfo:
        register_decoder("no_constructor")(NoConstructor)  # type: ignore[arg-type]
    assert "from_detector_error_model" in str(excinfo.value)
    assert "no_constructor" not in decoder_names()


@pytest.mark.parametrize("name", ["", "  ", "two words", "trailing "])
def test_a_name_that_is_not_one_token_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        register_decoder(name)


def test_a_name_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError):
        register_decoder(7)  # type: ignore[arg-type]


def test_a_second_registration_of_one_name_is_refused_by_default() -> None:
    with pytest.raises(ValueError) as excinfo:
        register_decoder(AUTHORITY_NAME)(MinimumWeightMatchingDecoder)
    assert "already registered" in str(excinfo.value)
    assert _model() is not None


def test_a_registration_can_be_replaced_when_the_caller_says_so() -> None:
    class Replacement:
        def decode(self, detection_events: object) -> object:
            return detection_events

        @classmethod
        def from_detector_error_model(cls, model: object) -> Replacement:
            return cls()

    try:
        register_decoder(AUTHORITY_NAME, replace=True)(Replacement)  # type: ignore[arg-type]
        assert get_decoder(AUTHORITY_NAME, _model()).__class__ is Replacement
    finally:
        register_decoder(AUTHORITY_NAME, replace=True)(MinimumWeightMatchingDecoder)
    assert (
        get_decoder(AUTHORITY_NAME, _model()).__class__ is MinimumWeightMatchingDecoder
    )


def test_the_registry_keeps_the_input_family_and_not_the_repetition_decoders() -> None:
    registered = set(decoder_names())
    assert {"minimum_weight_matching", "pymatching"} == registered
    assert RepetitionLookupDecoder is not None
    assert not hasattr(RepetitionLookupDecoder, "from_detector_error_model")
    assert isinstance(get_decoder(AUTHORITY_NAME, _model()), DetectorErrorModelDecoder)


def test_the_optional_name_is_registered_whether_or_not_it_is_installed() -> None:
    assert CROSS_CHECK_NAME in decoder_names()
    try:
        decoder = get_decoder(CROSS_CHECK_NAME, _model())
    except MatchingDependencyError as exc:
        assert CROSS_CHECK_NAME in str(exc) or "pymatching" in str(exc)
    else:
        assert isinstance(decoder, PyMatchingDecoder)


def test_the_registry_does_not_get_between_the_name_and_the_implementation() -> None:
    """The authority is returned for its own name even where the extra is present."""

    model = _model()
    authority = get_decoder(AUTHORITY_NAME, model)
    assert isinstance(authority, MinimumWeightMatchingDecoder)
    assert not isinstance(authority, PyMatchingDecoder)


def test_a_model_the_implementation_cannot_represent_keeps_its_reason() -> None:
    model = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1 D2\ndetector D0\ndetector D1\ndetector D2\n"
    )
    with pytest.raises(CapabilityError) as excinfo:
        get_decoder(AUTHORITY_NAME, model)
    assert "graphlike" in str(excinfo.value)


def test_importing_the_package_does_not_import_the_optional_distribution() -> None:
    """The registry holds the name without holding the extra's import.

    This is measured in a fresh interpreter rather than in this one, because a
    module another test already imported stays imported and would make the
    assertion about test order instead of about the package.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, flagquantum.qec; print('pymatching' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
