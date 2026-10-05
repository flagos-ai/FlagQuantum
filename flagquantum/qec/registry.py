"""A name-keyed registry of detector error model decoders.

Upstream `cudaq-qec` reaches a decoder through
``get_decoder(name, H_or_dem_text_or_sparse_matrix, **options)`` and registers one
with a decorator, so that a call site can accept a model and choose an
implementation by name without knowing which class implements it. This module is
that factory, and it is deliberately smaller than upstream's: it registers the
family of decoders whose input is a detector error model, and nothing else.

**The family is the point.** A detector error model decoder is built from a model
or from the graph that model defines, and it translates a syndrome -- a set of
detectors that fired -- into the mechanisms that explain it and the logical
observables those mechanisms flip. The repetition-code decoders in
:mod:`flagquantum.qec.decoders` are not in this registry and are not in it by
omission: their input is an ordered syndrome history and their output is a
correction on a known data wire, which is a different protocol over a different
record. Registering both under one name space would make a name mean one of two
things and make the factory's argument mean one of two things, so the registry
holds one family and the other stays directly constructed.

**Registration is checked at import rather than at the call site.** A registered
class must carry the two members the family shares -- a ``decode`` method and a
``from_detector_error_model`` constructor -- and a class that lacks either is
refused when it is registered, which is while the module that registers it is
being imported. Upstream checks the same thing for the same reason: a registry
that accepted any class would move the failure to the first caller, where the
name is all the caller has to go on.

**The optional name is always there; the optional distribution is not.**
``PyMatchingDecoder`` is imported and registered whether or not PyMatching is
installed, because the name is part of this package's surface and not part of the
extra's, and the adapter reaches the distribution through a function rather than
at import time, so importing this module does not import PyMatching. Asking for
the name without the extra raises
:class:`~flagquantum.qec.MatchingDependencyError` naming the extra, at the call
site and with the name the caller used, rather than an ``ImportError`` from an
import that happened while this module was being loaded.

**A name is not a promise about the implementation.** ``get_decoder`` returns the
class the name is registered against and nothing else; it does not compare two
implementations, and it does not prefer one because it is available. The
in-tree matcher is the authority for a decoded syndrome and the PyMatching
adapter is the cross-check that keeps that claim independent of the matcher's own
implementation, so the two are registered side by side and the caller chooses.

**The windowed member is the authority, not a third claim.** ``sliding_window_matching``
is registered against the decoder that decides a syndrome a band of detectors at a
time and calls the authority's matcher on each band, so it is the same authority
read through a window rather than a second opinion about the same syndrome. It is
registered beside the other two because a call site that already chooses a decoder
by name -- the reason this factory exists -- is exactly the call site that wants
the matcher that finishes a history the exact matcher refuses, and choosing it by
name keeps the caller from having to reach the class directly. The name carries
one obligation the other two do not: a window narrower than the whole graph
decides a band rather than the history, so a caller that names it states
``commit`` and ``window`` when the default band is not the one it wants.

**The source argument carries a model; it does not configure a decoder.** Upstream
takes ``H_or_dem_text_or_sparse_matrix``, and this factory takes the three forms
a caller can hold a model in: the model, stim's text for one, and the decoding
graph the model defines. A parity-check matrix is not among them on purpose --
lifting a matrix needs a noise model and a round count, because
:meth:`~flagquantum.qec.DetectorErrorModel.from_code_matrices` reads them rather
than defaulting them, so a factory that lifted a matrix would also be choosing
the noise the caller decodes against.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

from .adapters import PyMatchingDecoder
from .decoding_graph import DecodingGraph
from .dem import DetectorErrorModel
from .matching import MatchingDecodeResult, MinimumWeightMatchingDecoder
from .sliding_window import SlidingWindowMatchingDecoder

#: The name of the implementation this package is authoritative for.
AUTHORITY_NAME = "minimum_weight_matching"
#: The name of the optional implementation this package is cross-checked against.
CROSS_CHECK_NAME = "pymatching"
#: The name of the authority used one band of detectors at a time.
SLIDING_WINDOW_NAME = "sliding_window_matching"

_DecoderT = TypeVar("_DecoderT", bound="DetectorErrorModelDecoder")


@runtime_checkable
class DetectorErrorModelDecoder(Protocol):
    """What a decoder this registry hands back carries.

    The protocol states the instance contract and not the constructor one: a
    caller of :func:`get_decoder` holds a decoder and asks it for ``decode``, and
    reads ``graph`` to see what was built. Registration additionally requires the
    class to carry ``from_detector_error_model``, which is a classmethod and so is
    not part of the instance contract a return type can state; the two members
    are checked together in :func:`register_decoder`, where a class that lacks
    either is refused.

    ``graph`` is stated as a read-only property rather than as an attribute,
    because the decoders in this family are frozen records: a protocol that
    promised a writable ``graph`` would not be satisfied by a record whose
    ``graph`` cannot be reassigned, and a registry that returned one decoder type
    and refused another for being immutable would be checking a property no
    caller uses.
    """

    @property
    def graph(self) -> DecodingGraph: ...

    def decode(self, detection_events: Iterable[int]) -> MatchingDecodeResult: ...


#: The instance contract is what a caller gets back; the value here is the class
#: that builds one, so the graph route can call its constructor.
_DECODERS: dict[str, type[DetectorErrorModelDecoder]] = {}


def register_decoder(
    name: str, *, replace: bool = False
) -> Callable[[type[_DecoderT]], type[_DecoderT]]:
    """Name a detector error model decoder in the registry.

    Args:
        name: The name ``get_decoder`` will build this class from. It is a name
            and not a path: it must be a non-empty string with no whitespace in
            it, so that a name is one token and can be printed in a refusal.
        replace: Whether to overwrite an existing registration. Replacing is
            refused by default, because two classes answering to one name is a
            choice the registry cannot make on the caller's behalf and the second
            registration is usually a typo rather than an intent.

    Returns:
        The decorator that registers a class and returns it unchanged, so the
        decorated class keeps its name in the module that defines it.

    Raises:
        TypeError: The name is not a string, or the registered object does not
            carry the two members the family shares.
        ValueError: The name is empty or carries whitespace, or it is already
            registered and ``replace`` is not set.
    """

    if not isinstance(name, str):
        raise TypeError("a decoder name must be a string")
    if not name or name != name.strip() or any(char.isspace() for char in name):
        raise ValueError(
            f"a decoder name must be non-empty and carry no whitespace, not {name!r}"
        )
    if name in _DECODERS and not replace:
        raise ValueError(
            f"the name {name!r} is already registered against "
            f"{_DECODERS[name].__name__}; pass replace=True to overwrite it"
        )

    def decorate(candidate: type[_DecoderT]) -> type[_DecoderT]:
        missing = [
            member
            for member in ("decode", "from_detector_error_model")
            if not callable(getattr(candidate, member, None))
        ]
        if missing:
            raise TypeError(
                f"{candidate.__name__} cannot be registered as {name!r}: a detector "
                f"error model decoder needs a decode method and a "
                f"from_detector_error_model constructor, and "
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing "
                "or not callable"
            )
        _DECODERS[name] = candidate
        return candidate

    return decorate


def decoder_names() -> tuple[str, ...]:
    """Return the registered names, sorted, so a refusal can list them."""

    return tuple(sorted(_DECODERS))


def get_decoder(
    name: str,
    source: str | DecodingGraph | DetectorErrorModel,
    **options: Any,
) -> DetectorErrorModelDecoder:
    """Build the decoder registered under ``name`` for one model or graph.

    The source is a carrier and not a decoder setting, which is why the three
    accepted forms are the three a caller can hold a model in rather than three
    ways to configure one.

    Args:
        name: A name from :func:`decoder_names`.
        source: The detector error model to decode, stim's text for one, or the
            decoding graph that model defines. A model is lifted through the
            class's own ``from_detector_error_model``; a graph is passed to the
            constructor, because the graph is already the thing a matcher
            searches and rebuilding a model from it would lose the observable
            labels the caller has in hand; text is read through
            :meth:`DetectorErrorModel.from_stim_text`, so ``options`` must be the
            options that reader takes, and the model it returns is then lifted.
        **options: Forwarded to whichever route the source selects, so a name
            reaches the same configuration a direct construction would. This is
            also why the routes do not share their options: ``use_decomp_suggestions``
            belongs to the text reader and ``max_defects`` belongs to the matcher,
            and a caller who passes one to the wrong route is told so by that
            route rather than having it dropped here.

    Returns:
        The decoder, ready for ``decode``.

    Raises:
        ValueError: The name is not registered, or the text is not a detector
            error model stim could have written.
        TypeError: The source is not one of the three carriers.
        CapabilityError: The registered implementation cannot represent this
            model or graph, with the reason the implementation states.
        MatchingDependencyError: The registered implementation needs an optional
            distribution that is not installed, with the extra named.

    Note:
        A parity-check matrix is deliberately not a carrier, although upstream's
        ``H_or_dem_text_or_sparse_matrix`` is. Lifting a matrix here would need a
        noise model and a round count as well, because
        :meth:`DetectorErrorModel.from_code_matrices` reads them rather than
        assuming them, and a factory that silently supplied defaults for both
        would be choosing the noise the caller decodes against. The caller holds
        the matrix and the noise together, so the caller lifts it.
    """

    try:
        factory = _DECODERS[name]
    except KeyError:
        known = ", ".join(decoder_names())
        raise ValueError(
            f"no decoder is registered as {name!r}; the registry holds {known}"
        ) from None
    # A registered value is a class, and it is reached through two different
    # constructors: its own ``from_detector_error_model`` for a model or text, and
    # its ``__init__(graph=...)`` for a graph. The protocol above states the
    # instance a caller receives, because that is what this function returns and
    # what a caller then holds; no one protocol can state both class-level
    # signatures without widening one of them, so the routes call the class
    # through a constructor-shaped local and the contract is enforced where it can
    # be enforced -- at registration, on the members the class actually carries.
    build: Any = factory
    if isinstance(source, DetectorErrorModel):
        decoder: DetectorErrorModelDecoder = build.from_detector_error_model(
            source, **options
        )
    elif isinstance(source, DecodingGraph):
        decoder = build(graph=source, **options)
    elif isinstance(source, str):
        decoder = build.from_detector_error_model(
            DetectorErrorModel.from_stim_text(source, **options)
        )
    else:
        raise TypeError(
            "a decoder is built from a detector error model, from stim's text for one, "
            f"or from a DecodingGraph, not from {type(source).__name__}"
        )
    return decoder


register_decoder(AUTHORITY_NAME)(MinimumWeightMatchingDecoder)
register_decoder(CROSS_CHECK_NAME)(PyMatchingDecoder)
register_decoder(SLIDING_WINDOW_NAME)(SlidingWindowMatchingDecoder)
