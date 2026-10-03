"""Shared stateless parsing helpers for remote QPU providers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast


def _normalize_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): int(round(float(value))) for key, value in counts.items()}


def _flip_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    return {str(key)[::-1]: int(round(float(value))) for key, value in counts.items()}


def _unwrap_result_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result", payload)
    if (
        isinstance(result, Mapping)
        and "ok" in result
        and isinstance(result.get("result"), Mapping)
    ):
        return cast(Mapping[str, Any], result["result"])
    if isinstance(result, Mapping):
        return result
    return payload


def _extract_counts(
    payload: Mapping[str, Any], *, provider: str, flip: bool = False
) -> dict[str, int]:
    current: Any = _unwrap_result_envelope(payload)
    if isinstance(current, Mapping) and "counts" in current:
        current = current["counts"]
    elif isinstance(current, Mapping) and "count" in current:
        current = current["count"]
    elif isinstance(current, Mapping) and isinstance(current.get("data"), Mapping):
        current = current["data"]
        if "counts" in current:
            current = current["counts"]
        elif "count" in current:
            current = current["count"]
    if not isinstance(current, Mapping):
        raise RuntimeError(f"{provider} result response does not contain counts.")
    return _flip_counts(current) if flip else _normalize_counts(current)


def _strip_barrier(qasm: str) -> str:
    return "\n".join(
        line for line in qasm.splitlines() if not line.strip().startswith("barrier")
    )


def _format_circuit_source(source: str) -> str:
    return "\n".join(line.strip() for line in str(source).splitlines() if line.strip())


def _first_named(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return the first named outcome collection, mapping or per-shot list."""

    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, str | bytes):
            continue
        if isinstance(candidate, Mapping | Sequence):
            return candidate
    return None


def _outcome_collection(payload: Mapping[str, Any], *, keys: Sequence[str]) -> Any:
    """Locate the outcome collection of a provider result payload.

    A vendor nests it under one of a small set of names, sometimes inside a
    ``data`` envelope, and sometimes answers with the collection itself. It is
    either a mapping of outcome to weight or a list holding one entry per
    returned shot; both are located here, and which shape was found is left to
    the caller that knows how to read it.
    """

    current: Any = _unwrap_result_envelope(payload)
    for _ in range(2):
        if not isinstance(current, Mapping):
            break
        found = _first_named(current, keys)
        if found is not None:
            return found
        nested = current.get("data")
        if not isinstance(nested, Mapping):
            break
        current = nested
    return current


def _outcome_table(
    payload: Mapping[str, Any], *, provider: str, keys: Sequence[str]
) -> dict[Any, Any]:
    """Return the outcome-to-weight table of a provider result payload.

    A payload whose values are not numbers is refused rather than read as
    outcomes: a job descriptor that happens to be returned alongside the result
    must not be tallied as if its fields were bitstrings.
    """

    current = _outcome_collection(payload, keys=keys)
    if not isinstance(current, Mapping) or not current:
        raise RuntimeError(f"{provider} result response does not contain outcomes.")
    table = dict(current)
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in table.values()
    ):
        raise RuntimeError(f"{provider} result response does not contain outcomes.")
    return table


def _tally_outcomes(
    outcomes: Sequence[Any],
    *,
    provider: str,
    outcome_key: Callable[[Any], str],
) -> dict[str, int]:
    """Tally a per-shot outcome list exactly as the vendor returned it.

    A vendor that reports one entry per shot has already ordered the register,
    and this does not reorder it: a bitstring silently reversed or sorted here
    would be indistinguishable from a correct result.
    """

    counts: dict[str, int] = {}
    for outcome in outcomes:
        key = outcome_key(outcome)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise RuntimeError(f"{provider} result response does not contain outcomes.")
    return counts


def _probability_counts(
    probabilities: Mapping[Any, Any],
    shots: int,
    *,
    provider: str,
    outcome_key: Callable[[Any], str],
) -> dict[str, int]:
    """Convert sampled probabilities to counts by largest remainder.

    A vendor that returns probabilities has not returned shots, so the counts
    this produces are an allocation of ``shots`` that preserves the reported
    probabilities to within one shot. The alternative, rounding each outcome
    independently, can disagree with the shot count the vendor reported.
    """

    weighted: list[tuple[str, float, int, float]] = []
    for outcome, raw_probability in probabilities.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0:
            raise RuntimeError(f"{provider} result contains an invalid probability")
        exact = probability * shots
        base = math.floor(exact)
        weighted.append((outcome_key(outcome), probability, base, exact - base))
    total_probability = sum(item[1] for item in weighted)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"{provider} probabilities do not sum to one")
    remaining = shots - sum(item[2] for item in weighted)
    ranked = sorted(
        range(len(weighted)), key=lambda index: (-weighted[index][3], index)
    )
    increments = set(ranked[:remaining])
    return {
        outcome: base + (index in increments)
        for index, (outcome, _probability, base, _remainder) in enumerate(weighted)
    }


def _outcome_counts(
    payload: Mapping[str, Any],
    *,
    provider: str,
    keys: Sequence[str],
    shots: int,
    outcome_key: Callable[[Any], str],
) -> dict[str, int]:
    """Read a provider result as counts.

    A payload that reports one entry per shot is tallied, and a payload that
    reports weights is read as counts when those weights are integral and as
    probabilities otherwise.
    """

    collection = _outcome_collection(payload, keys=keys)
    if isinstance(collection, Sequence) and not isinstance(collection, str | bytes):
        counts = _tally_outcomes(collection, provider=provider, outcome_key=outcome_key)
        if sum(counts.values()) != shots:
            raise RuntimeError(
                f"{provider} returned {sum(counts.values())} outcomes for a "
                f"{shots}-shot job"
            )
        return counts
    table = _outcome_table(payload, provider=provider, keys=keys)
    numeric = {outcome: float(value) for outcome, value in table.items()}
    if all(value.is_integer() and value >= 0 for value in numeric.values()):
        counts = {outcome_key(key): int(value) for key, value in numeric.items()}
        if sum(counts.values()) == shots:
            return counts
    return _probability_counts(table, shots, provider=provider, outcome_key=outcome_key)
