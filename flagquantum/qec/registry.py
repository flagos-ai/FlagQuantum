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
correction on a known data qubit, which is a different protocol over a different
record. Registering both under one name space would make a name mean one of two
things and make the factory's argument mean one of two things, so the registry
holds one family and the other stays directly constructed.

**Registration is checked at import rather than at the call site.** A registered
class must carry the member the family's instances share -- a ``decode`` method --
and the two constructors ``get_decoder``'s carriers select: a
``from_detector_error_model`` for a model or for stim's text, and a
``from_decoding_graph`` for the graph a model defines. A class that lacks any of
the three is refused when it is registered, which is while the module that
registers it is being imported. Upstream checks the same thing for the same
reason: a registry that accepted any class would move the failure to the first
caller, where the name is all the caller has to go on. The two constructors are
required together because the three carriers are one documented surface: a name
that could be built from a model and not from a graph would be a name whose
accepted sources depend on which name was asked for.

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
from .belief_propagation import BeliefPropagationDecoder
from .decoding_graph import DecodingGraph
from .dem import DetectorErrorModel
from .matching import MinimumWeightMatchingDecoder

#: The name of the implementation this package is authoritative for.
AUTHORITY_NAME = "minimum_weight_matching"
#: The name of the optional implementation this package is cross-checked against.
CROSS_CHECK_NAME = "pymatching"
#: The name of the in-tree decoder for the models a matcher cannot represent.
BELIEF_PROPAGATION_NAME = "belief_propagation"

_DecoderT = TypeVar("_DecoderT", bound="DetectorErrorModelDecoder")


@runtime_checkable
class DetectorErrorModelDecoder(Protocol):
    """What a decoder this registry hands back carries.

    The protocol states the instance contract, and the instance contract of this
    family is one member: a caller of :func:`get_decoder` holds a decoder and asks
    it for ``decode``. The constructors are deliberately not stated here --
    ``from_detector_error_model`` and ``from_decoding_graph`` are classmethods, so
    no return type can state them -- and are checked in :func:`register_decoder`
    instead, where a class that lacks either is refused.

    What ``decode`` *returns* is likewise not stated, because the family does not
    share one result record and should not pretend to. A matcher's correction is a
    set of graph edges with a weight, and a belief-propagation decoder's is a set
    of mechanism indices with the beliefs' convergence; both answer "which
    mechanisms explain this syndrome, and what do they flip", and a protocol that
    named either record would refuse the other family for being a different
    answer rather than a different shape.

    ``graph`` is not part of this contract either. Two of the three carriers build
    a graph and the text route builds one indirectly, but a decoder whose input is
    the model as written -- a mechanism touching three detectors is a variable
    touching three checks -- has no graph to hand back for exactly the models it
    exists to read, so requiring one would make the protocol untrue of the family
    it names. A decoder that does build a graph exposes it as a read-only
    ``graph`` property, which is a fact about that decoder rather than about the
    family.
    """

    def decode(self, detection_events: Iterable[int]) -> Any: ...


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
            carry the three members the family shares -- a ``decode`` method and
            the two constructors the three carriers select.
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
            for member in ("decode", "from_detector_error_model", "from_decoding_graph")
            if not callable(getattr(candidate, member, None))
        ]
        if missing:
            raise TypeError(
                f"{candidate.__name__} cannot be registered as {name!r}: a detector "
                f"error model decoder needs a decode method, a "
                f"from_detector_error_model constructor and a from_decoding_graph "
                f"constructor, and {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing or not callable"
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
            class's own ``from_decoding_graph``, which for a matcher is the graph
            it already searches and for a decoder that reads the model as written is
            the lift of the graph's own edges back into a model; text is read
            through
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
    # its own ``from_decoding_graph`` for a graph. The protocol above states the
    # instance a caller receives, because that is what this function returns and
    # what a caller then holds; no one protocol can state class-level signatures
    # without widening it, so the routes call the class through a
    # constructor-shaped local and the contract is enforced where it can be
    # enforced -- at registration, on the members the class actually carries.
    build: Any = factory
    if isinstance(source, DetectorErrorModel):
        decoder: DetectorErrorModelDecoder = build.from_detector_error_model(
            source, **options
        )
    elif isinstance(source, DecodingGraph):
        decoder = build.from_decoding_graph(source, **options)
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
register_decoder(BELIEF_PROPAGATION_NAME)(BeliefPropagationDecoder)
