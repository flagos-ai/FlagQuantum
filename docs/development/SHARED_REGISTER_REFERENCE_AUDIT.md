# Shared register source-comparison acceptance

This is verification infrastructure for existing G2 register implementations,
not a new arithmetic or execution module. It finishes the previously staged
comparison batch without broadening its scientific scenarios or changing native
semantics. Pending protected-contract proposals remain separate.

## Complete comparison responsibilities

The existing files cover raw binary-field storage, classical word inputs and
feedback, destructive measurement in both bases, parity, input-pool ownership,
tensor indices and superoperators, Program lowering and workspace retirement.
Their direct shared callbacks retain actual source owners, exact incoming and
outgoing labels, trace-marker types, array values and source exceptions.
Numerical branch, phase, cleanup and resource assertions remain unchanged.

Defining imports replace package reexports only after checking Python object
identity. Scientific AST comparison normalizes those identical names and the
validated SDK callbacks; it preserves original operands, calculations, feedback
and assertions. A DiscardInd reexport and its defining import are treated
symmetrically after verifying that both name the same original class.
All direct source modules are checked against their frozen inventory Git blobs
at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. The installed Kickmix 0.1 native
extension and Simulator are checked before reference execution.

Optional dependency absence uses the existing shared boundary. Normal optional
collection may skip explicitly; required FTQC mode raises the actual missing
dependency. No source algorithm, dtype input, array layout or reference result
is replaced to make the comparison run. No cast, ignore or checker exemption is
introduced.

## Verification and retained boundaries

The batch reuses earlier complete source evidence and performs concentrated
acceptance against the current production head. Required-mode pytest, strict
typing, format, lint, source hashes, scientific-body comparison, team scope,
language, architecture, IR and API checks retain their own receipts under
`g2-register-ownership-*-20261127` in the reference environment. The previous
three new callback-test failures concerned fresh private UUIDs and a wrong
source module, not a changed native quantum result; their corrections retain
the actual original label connectivity and Discard owner.

The existing user workflow remains `circuit.append(op)`. For an executable
measurement, feedback and ownership example, run:

```bash
FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES=1 python -m pytest -q \
  tests/unit/test_register_classical_discard.py \
  tests/unit/test_register_owned_measurement.py
```

This acceptance does not close whole CI/source-role parents, change the held
API proposals, accept upstream defects or claim new numerical functionality.
Remaining reference-guard findings are recorded separately from algorithm
coverage and from accepted source-file totals.
