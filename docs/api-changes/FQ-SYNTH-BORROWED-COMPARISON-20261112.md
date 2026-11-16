# Low-work unsigned comparison candidate

The user authorized completing all 216 native programs with independent Circuit
acceptance. The source-frozen audit on runtime implementation `589d67070` timed
out the full Rainbow integration comparison after 1800 seconds. Its `k=0` and
`k=1` comparisons passed before the timeout; `k=2` did not complete. Preserve
that failure. Neither a longer timeout nor a smaller source workload establishes
acceptance.

This Compiler-owned optimization changes no public symbol, signature, IR opcode,
execution backend, input precision or resource ceiling. Frozen verification
sources must remain unchanged until their existing processes terminate.

## Proposed implementation

Reuse the majority sweep from the existing pure reversible arithmetic kernels.
For equal-width, physically disjoint unsigned words, compute the carry of
`~a + b` from a zero carry, XOR it into an arbitrary target, and reverse the
entire sweep. The carry is exactly `a < b`. Restore both operands and the carry
on every input, including superpositions and nonzero targets. The kernel uses
only X, CX and CCX, without a relative-phase approximation.

The arithmetic scheduler may use this kernel only after checking equal widths,
distinct operand locations and disjoint target/scratch storage. Allocate and
record its actual one clean carry. Keep the existing prefix-equality kernel for
equality, constants, unequal widths and overlapping/aliased operands. Signed
and scaled arithmetic must continue through the existing checked unsigned-word
lowering and sign/scale alignment; the new kernel cannot reinterpret them.

An input is borrowed during the complete compute/copy/uncompute sequence; its
temporary gate-level changes must not escape the sequence. Coherent controls
must retain existing disjointness and inverse rules. No source-specific name,
width, financial parameter or truth table may select the optimization.

## Acceptance

- Exhaustively compare all width-one through width-five word pairs and both
  target values against an independent integer predicate. Check exact operand
  restoration, zero carry, and the inverse sequence.
- Compare a complex coherent state, including nonzero/coherent target values,
  against an independent permutation without phase alignment. Check invalid
  widths, repeated wires, overlaps, negative wires and noninteger bindings.
- Verify scheduler resource events count the actual carry and preserve both
  retain and recompute schedules, arbitrary XOR targets and phase composition.
  Keep constant, unequal-width, alias, signed and fixed-point regressions.
- Run the full unchanged Rainbow integration module at `k=0/1/2`, plus the
  existing named model-parameter, marker, environment and budget tests. Compare
  all logical complex amplitudes, absolute phase and the complete work tail.
  Keep original widths, preparation parameters, iteration counts and caps.
- Replay every frozen original and all six approved repairs. Preserve raw
  pre-optimization resources and identify any actual changed resource rows;
  never overwrite a baseline or count an original semantic refusal as a pass.
- After the final implementation, run every documented native comparison on
  one pinned source version. The two missing comparison launchers and all
  named autocallable scenarios remain independent gates. No integration PR or
  remote full CI before complete native-corpus acceptance.

## Evidence boundary

The external in-process prototype is diagnostic evidence only. It does not edit
the frozen runtime, replace the original audit result or count as production
acceptance. Its small-input checks and Rainbow measurements determine whether
to implement the candidate after the frozen runs finish. Track all five
categories and Notebook/Python workflows separately throughout delivery.
