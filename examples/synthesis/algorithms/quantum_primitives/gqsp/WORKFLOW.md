# Native generalized quantum signal processing

This program preserves the seven original triples of real phase parameters,
the seven-bit unsigned fraction x and negative_power=3. The fraction declares
x=word/128. A uniform preparation is followed by the signal rotation sequence;
the output includes both x and the indicator, so complete complex amplitudes
and their relative phases remain observable.

For this original generator u(x)=phase(2*pi*x), a coherent positive operation
on indicator0 is written as `phase(x*(1-ind), 2*pi)`. Its inverse on indicator1
is `phase(x*ind, -2*pi)`. These are quantum expressions compiled by synth,
not values read by host Python. A small borrowed-qbit helper expresses the
original X/U/X/Z rotation using within/apply and the public U operation.
This is a native expression of the fixed source generator, rather than an
implementation of arbitrary higher-order GQSP callbacks.

Users provide classical phase lists, bounded negative_power0..7 and an optional
initial indicator0/1. The example retains the source's exact loop endpoints:
when negative_power=7, the negative loop includes row0 again after the initial
row0 rotation. The independent Circuit constructs all128 independent2x2 blocks
using complex matrix products. Tests compare the unchanged source and default
native result, then two further nonzero phase models across indicator0/1 and
negative_power0/3/7 without global phase alignment. Width, operation, depth,
memory and nonfinite parameter rejection are checked. The program uses8bits
and zero work; the default native form has31operations/depth24, versus the
source41operations/depth34. These are measured resources for this example,
not a general optimization or performance claim.

Run the native main or comparisons.circuit module from this directory's Python
module path shown in README. Users do not supply physical wire numbers or
decode bitstrings. This accepts the quantum program, not a complete Notebook
workflow, polynomial approximation certificate or general GQSP library API.
