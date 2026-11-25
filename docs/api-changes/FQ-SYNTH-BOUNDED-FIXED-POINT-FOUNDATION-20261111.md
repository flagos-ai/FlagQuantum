# Shared bounded fixed-point evaluation foundation

Status: internal replacement contract. Native coverage remains 208/216.
Classification: `single_device_fast_path`.

The Brownian original has 23 physical bits, 15 signed output bits for G with eight
fractional bits, 32 prepared branches and 24 distinct output labels. Its existing
Qmod adapter evaluates a distinct bounded quantized numeric profile. It first
truncates literal values toward zero at twenty fractional bits, then rounds down
to eight fractional bits. Arithmetic products and powers round down at their
admitted result scale. This differs from the exact dyadic word compiler, whose
refusal of non-binary decimals remains correct and must remain unchanged.

A new independent Fraction recurrence has checked all 32 original support words,
including both clean work bits, all output raw labels and absolute amplitudes.
The full original 23-bit vector has exactly the expected 32 occupied words above
1e-12, maximum support error below 6e-15 and probability mass within 6e-14 of one.
The native expression `-6.283185307179586*(x-0.5)` still refuses non-binary decimal
capture. These observations establish a dependency, not native acceptance.

First extract the existing bounded scalar value domain and expression arithmetic
from `ecosystem/_qmod/quantized_numeric.py` into Compiler. Preserve its exact
rounding, interval inference, signed width, precision and bounded branch rules.
Source-independent callbacks may provide expression text, child nodes, literal
values and rejection locations, following the exact word-expression foundation.
Ecosystem retains Qmod parsing, literal token recognition, lexical helper and
ownership planning, reachable-definition checks, allocation/drop proofs and the
external source profile choice. Compiler must not import Ecosystem syntax or
execute Python functions. Reuse existing HybridProgram/qint/register records;
these bounded values are inference data, not a second circuit IR.

Demonstrate replacement of the Qmod evaluator and a separate Python-AST test
consumer of the shared arithmetic before exposing native syntax. Cover positive
and negative rounding ties, literal truncation near dyadic boundaries, signed
bounds, fractional alignment, multiplication, powers, overflow and branch limits.
Keep source-located refusals and the original 23-bit/1403-operation baseline.
All 216 original hashes, outcomes and resource rows must remain unchanged.

Native public admission is a subsequent explicit semantic contract. Reuse QNum
sign/scale annotations and `Program.numeric_probabilities`; no manual bit-string
decoder is needed. Quantization must be visible in the native program or its
explicit numeric policy. Do not turn the existing exact arithmetic entry into an
implicit eight-bit approximation. Do not certify Brownian with integer-only
payoffs, erase its discarded branch environment, reduce its original logical
scope or substitute a precomputed final state. Native helper composition,
quantized intermediate values, original output formats, coherent amplitudes,
reduced output states and independent Circuit validation remain required before
crediting the original. The other three finance programs require a fresh complete
dependency review; common function names alone do not prove they are unlocked.
