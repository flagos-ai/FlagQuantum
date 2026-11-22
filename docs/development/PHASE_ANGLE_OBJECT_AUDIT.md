# Complete exact phase object conversion audit

## Ownership and interface before implementation

Frozen angle.pybind.cc/.h are K-0255/K-0256 at
9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453. Their complete single converter
accepts an external Python value, constructs Fraction, normalizes modulo two
half turns, rounds to the 128-bit full-turn ring using Python ties-to-even and
returns the low/high words. Both the builder z_pow and interactive
write_shot_phase clients use this function.

Ecosystem owns external value conversion in _phase_angle.py and immediately
returns the existing Compiler-owned PhaseAngle128. Compiler retains raw word
representation, exact arithmetic, KMX/KMB serialization and phase lowering.
Simulation retains packed phase state. One half turn is 2**127 units and
rounded 2**128 wraps to zero. A failed conversion leaves caller-owned phase
and circuit data intact. The complete helper has one input and one immutable
output and owns zero workspace. User operations retain circuit.append(op).

The existing double and decimal constructors retain their distinct contracts.
This bridge follows the original object converter, including Fraction conversion
errors and differences between decimal strings and binary floating-point inputs.
The full builder and interactive wrappers retain their own acceptance scopes.

## Complete source disposition and acceptance

The single function and header declaration map to phase_angle_from_half_turns.
Fraction admission, modulo, round and both word extractions are preserved through
the existing owned ring factory. Seven original blobs include the complete
helper/declaration, angle value dependency, both actual client files and their
two original test files. The original z_pow and two read/write phase tests pass.

All 190 frozen observations execute both the unchanged builder and shot-phase
clients. They cover every exact rotation level, positive and negative tie points,
wraparound, huge integers/fractions, Decimal, float, strings, booleans, smallest
binary subnormal and conversion errors. Both source consumers agree on errors
and retain their prior circuit/phase state. Native words, canonical instruction
text, binary round trips, phase composition and complete complex append outputs
agree. The fixture regenerates byte-exactly.

The concentrated phase/statistics/feedback regression passes 595 items and the
shared construction/coherent/trajectory integration passes 121. Strict typing
passes all three touched Python files. Helper statement coverage is 100%, with
zero control-flow branches. Black, Ruff, team scope, architecture, IR and API
checks pass. Dependency policy retains the same 720 findings and zero touched
files. Reference receipts use the g2-angle-object-20261122 prefix.

This scalar converter owns zero primitives, measurements and workspace. Its
accepted consumer emits one stored Z_POW, including a zero angle. Actual executed
counts remain separate from the analytical estimate field, which is None.
The source has zero measurement branches in this helper. Existing shared phase
and erasure branch evidence remains under its original owners. Full builder and
interactive wrapper files remain separately open. The native helper returns an
immutable value, and the existing caller-owned packed state is used to verify
exact composition. Pending Proposals 116–119 and prior source defects remain in
the main ledger.
