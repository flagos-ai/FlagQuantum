# Phase block encoding

G2-BE02 composes the locked Phase block wrapper in algorithms/primitives.
The wrapper forwards alpha, system/ancilla/resource widths and signal state;
epsilon is child epsilon plus the supplied phase precision. A native bound
child uses the existing Circuit append protocol. The caller supplies the child
in the requested direction; the wrapper does not invert measured instructions.
This works for differently bound unitary, SELECT and LCU children without a
second binding schema or a change to protected BlockEncoding APIs.

A private structural metadata view supplies the properties already present on
these three native wrappers. The protected spectral BlockEncoding protocol does
not describe resource/precision/signal metadata and is not changed. No public
exports or new quantum IR are introduced.

The source scalar GlobalPhase is a zero-wire atom. Native execution reuses
existing global_phase_ir on an explicitly chosen existing anchor. Nonzero
phases emit X, phase(theta), X, phase(theta); this multiplies every anchor state
by exp(i*pi*phi), including when it is entangled. The anchor is preserved and
requires no clean work. Zero phase emits nothing. These four instructions are
an explicit representation adaptation, not four source gates. Model GlobalPhase
has zero QEC categories and emitted primitive counts remain separate.

The private metadata view and the existing append protocol are exercised with
UnitaryBlockEncoding, SelectBlockEncoding, LCUBlockEncoding and nested phase
wrappers without modifying those consumers. Binding remains caller-owned: pass
the child already bound in the requested direction. Forward order is scalar
then child; reverse order is child adjoint then scalar adjoint. Invalid scalar
exponents, direction and carrier indices fail before child append.

Sixty source configurations cover T, CNOT and two-bit Add, five exponents
(including zero, negative, integer and nonbinary-fraction values), both
directions and auxiliary-port layouts. Each complete unmodified source Phase
is contracted to its dense tensor, not compared modulo an arbitrary scalar.
All native small measurement branches match that complete complex matrix and
clear child work while preserving coherent auxiliary inputs. The child gate
stream independently matches the untouched source child; the zero-wire atom's
four-instruction carrier expansion is tested through its numerical action.
Both QEC profiles match the full source category vector and logical peak in
all60 configurations. Two nested-phase cases additionally verify separate
scalars, accumulated precision and original-source matrix/model behavior.

Four SELECT/LCU consumer scenarios expose an original-source limitation:
Phase's packed system/ancilla/resource signature does not match those source
children's named selection/target signature. The full source tensor raises
BloqError and QubitCount raises RuntimeError. Both failures are reproduced
without changing the source. Native generic composition is checked against the
unchanged child tensor times the original GlobalPhase coefficient in every
small measurement branch; QEC categories match the still-valid source call
graph. This proves the composition semantics and existing-consumer replacement,
not an executable whole-source Phase path or source peak-width parity for these
failing combinations. The identity PREPARE fixture declares normalization1 and
uses the SELECT's exact register names; its inherited identity gates are unchanged.

Run `python -m examples.algorithms.phase_block_encoding`. It appends a phased
Add block and its exact inverse, preserving arbitrary coherent auxiliary
states and clean child work. The forward report is deliberately split:

| Quantity | Source logical model | Emitted IR |
|---|---:|---:|
| T equivalents / explicit T instructions | 4 | 4 |
| Scalar atom QEC categories / carrier instructions | 0 | 2 X + 2 phase |
| Measurements | 1 | 1 |
| Logical peak / reserved wires | 7 | 7 |

The two phase instructions are unsynthesized; no synthesis or physical cost is
inferred from the zero-cost logical atom. Symbolic phase/precision metadata and
model queries are supported; numerical binding refuses a symbolic exponent.
There are75 new tests and165 concentrated passes without skips. Runtime append,
full phase-sensitive round trip, existing unitary/prepare/SELECT/LCU consumers
and shared global-phase regressions pass. Strict mypy, Black, Ruff, scope and
architecture checks pass. Source blob d7e6b03237f63238e511a4e7aa5056808a646593
matches the pinned tree and installed reference. No protected API or source is
modified. Zero-wire numerical execution, generic controlled-phase wrappers and
whole-source-file acceptance remain open G2 items.
