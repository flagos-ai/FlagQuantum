# Signed register encodings

The register compiler converts a big-endian signed word in place.
The leading sign remains on its original qubit. The caller lends
`n - 2` clean carry qubits for a width-`n` word and one reusable
classical slot when `n > 2`. Both directions restore the carry qubits.
The width-two circuit has a single payload bit and needs no
measurement or carry.

## Two input contracts

The locked [Qualtran conversion](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/ones_complement_to_twos_complement.py)
declares a `QIntOnesComp` input. That [encoding](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py)
already complements the magnitude for negative values. The native
`ones_complement_to_twos_ir` adds the sign to the low word modulo
`2**(n-1)`. For example, four-bit negative one maps from 1110 to 1111.
Applying the source comment's sign-controlled complement first would
give 1010, which encodes negative six.

The separate `sign_magnitude_to_twos_ir` follows the construction in
[the cited paper, p. 24](https://arxiv.org/pdf/2105.12767#page=24).
It complements the magnitude on negative input and then adds the sign.
The inverse subtracts first and then restores the magnitude.

Both converters are reversible on the full word space. Positive zero
remains zero. The duplicate negative-zero encoding, all ones for one's
complement and sign-only for sign-magnitude, maps to the minimum
two's-complement word. This extension preserves bijectivity. For every
integer in the common numeric domain the decoded value is preserved.

## Gate counts and verification

One temporary AND per carry computes whether all lower payload bits
are ones (increment) or zeros (decrement) and the sign is negative.
Apply the carries from most significant to least significant, erasing
each before changing its input digit. The shared AND and erase gates,
including control polarity, feed-forward CZ and reset, match the
unmodified Qualtran leaf decomposition. The top-level conversion
exposes only a signature and cost model.

| Width | Native T + T† | CX, one's complement | CX, sign-magnitude | Measurements | Work qubits | Qualtran model Toffoli |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 1 | 2 | 0 | 0 | 0 |
| 5 | 12 | 22 | 26 | 3 | 3 | 3 |
| 8 | 24 | 43 | 50 | 6 | 6 | 6 |

Counts are per direction. The inverse additionally emits four X gates
per carry to implement the negative controls. Tests compare the full
locked numerical domain and exercise all raw words, the inverse,
complex amplitudes with a spectator, and every raw measurement branch
at widths two, three and five through the append entry point.
