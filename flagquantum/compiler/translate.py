"""Translate a FlagQuantum program into another framework's text language.

The counterpart of ``cudaq.translate``. Translation is Compiler-owned: it turns
a validated program into text and performs no compilation, no target
legalization, and no numerical work. A caller who needs the emitted text bound
to a legalized target, with its evidence identities, uses
:func:`flagquantum.compiler.target_emission.emit_legalized_target` instead.
"""

from __future__ import annotations

from typing import Any

from .target_emission import EMISSION_PROFILES

TRANSLATION_FORMATS: tuple[str, ...] = tuple(sorted(EMISSION_PROFILES))
# Closed set of accepted ``format`` values, the emission profile names.


def translate(program: Any, *, format: str = "openqasm-3.0") -> str:
    """Translate a program into deterministic text in the requested format.

    The output is byte-for-byte the text the named emission profile produces, so
    a machine that consumes the language can be pointed at either entry point.
    This performs no target legalization and checks no device: it answers what
    the program says, not whether a named QPU accepts it. What it does refuse is
    any program whose semantics the language cannot carry, since emitting such a
    program means dropping the part that was not expressible.

    Args:
        program: A ``Circuit`` or canonical ``CircuitIR``. Unbound named
            parameters are refused; call ``Circuit.bind_parameters`` first.
        format: The target language, one of :data:`TRANSLATION_FORMATS`.
            ``"openqasm-3.0"`` is the default because OpenQASM 3.0 is the only
            language here that expresses every gate in the FlagQuantum operator
            set directly, so it is the one translation that decomposes nothing.
            ``"openqasm-2.0"``, ``"qcis-1.0"``, and ``"qir-2.0"`` decompose or
            drop a global phase of their own; ``"qir-2.0"`` is the QIR base
            profile, which expresses ``cphase``, ``u1``, ``u2``, and ``u3`` only
            up to an unconditional global phase that no measurement can observe.

    Returns:
        The translated text. No trailing newline is added beyond the one the
        language requires, and two calls on equal programs return equal strings.

    Raises:
        ValueError: If ``format`` is not one of :data:`TRANSLATION_FORMATS`, or
            a gate parameter is unbound or not a finite real number.
        UnsupportedLoweringError: If the selected language has no form for an
            instruction. A noise channel, a classical condition, a dynamic
            instruction, an arbitrary matrix gate, a reset, or an observable
            request is named by index and opcode rather than emitted with the
            part the language cannot carry left out. It is a ``ValueError``.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum.compiler import TRANSLATION_FORMATS, translate
        >>> print(TRANSLATION_FORMATS)
        ('openqasm-2.0', 'openqasm-3.0', 'qcis-1.0', 'qir-2.0')
        >>> print(translate(fq.Circuit(2).h(0).cx(0, 1)).splitlines()[2])
        qubit[2] q;
    """

    normalized = str(format).strip().lower()
    try:
        profile = EMISSION_PROFILES[normalized]
    except KeyError as error:
        supported = ", ".join(TRANSLATION_FORMATS)
        raise ValueError(
            f"unsupported translation format {format!r}; expected one of {supported}"
        ) from error
    return profile.emitter(program)


__all__ = ("TRANSLATION_FORMATS", "translate")
