"""Pinned four-data-qubit molecular LCU coefficients and Pauli words.

Words are ordered by logical data index (least significant bit first).
The unused selector leaf 15 dispatches leaf 14, preserving the source tree.
"""

PROBABILITIES = (
    0.045630072820696,
    0.086739383197585,
    0.086739383197585,
    0.113725586193483,
    0.113725586193483,
    0.085098942097053,
    0.022786476693496,
    0.022786476693496,
    0.022786476693496,
    0.022786476693496,
    0.060910951530297,
    0.083697428223792,
    0.083697428223792,
    0.060910951530297,
    0.087978380017955,
    0.0,
)

PAULI_WORDS = (
    "IIII",
    "ZIII",
    "IIZI",
    "IZII",
    "IIIZ",
    "ZIZI",
    "YYYY",
    "YYXX",
    "XXYY",
    "XXXX",
    "ZZII",
    "ZIIZ",
    "IZZI",
    "IIZZ",
    "IZIZ",
    "IZIZ",
)

SELECTOR_PHASES = (
    3.141592653589793,
    0.0,
    0.0,
    3.141592653589793,
    3.141592653589793,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def selection_members(axis: str) -> tuple[int, ...]:
    """Group all selectors sharing one Pauli axis on each logical data bit."""
    groups = [
        [index for index, word in enumerate(PAULI_WORDS) if word[bit] == axis]
        for bit in range(4)
    ]
    if axis not in {"X", "Y", "Z"} or any(len(group) > 5 for group in groups):
        raise ValueError("Expected the pinned four-bit Pauli selection groups")
    if axis in {"X", "Y"} and any(len(group) != 2 for group in groups):
        raise ValueError("The X/Y selector groups must each contain exactly two leaves")
    # Selector 16 is outside the entire four-bit input domain, hence contributes
    # false to each padded disjunction. No physical selector leaf is removed.
    return tuple(index for group in groups for index in group + [16] * (5 - len(group)))
