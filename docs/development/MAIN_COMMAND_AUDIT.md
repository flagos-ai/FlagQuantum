# Complete physical command dispatch audit

## Ownership and interface before implementation

Frozen Kickmix main.cc is K-0167. Ecosystem owns the raw KMX/KMB command boundary.
The dispatcher selects the existing owning entry point and forwards its arguments.
Compiler owns conversion/counting, Drawer owns generation/drawing and existing
Ecosystem sampling commands retain numerical execution ownership. Imports occur
only after selection. This adds no command registry or alternate execution path.

The existing owner main functions accept arguments without executable/subcommand.
The dispatcher removes those two source argv slots. The shared converter instead
receives its existing direction positional argument. Embedded NULs are normalized
to C argv-visible prefixes before selection/forwarding. Native help uses the actual
python -m flagquantum.ecosystem.kmx_command invocation. Source invalid_argument becomes a native ValueError boundary.
Original command-owner argparse and Python filesystem error policies are retained.

The user quantum workflow continues through circuit.append(op). This raw-format
entry point is also available as python -m flagquantum.ecosystem.kmx_command.

## Complete source disposition

The full original `main` definition is mapped, including missing-command and
unknown-command diagnostics, all eight branches, argument count/pointer
forwarding, returned status and the `invalid_argument` catch. The source file is
compiled unchanged with only its main symbol renamed. The oracle defines command
probes at the existing declared boundaries, preserving complete dispatch logic.
The generator verifies 28 original source/header Git blobs against the immutable
source inventory. Compiler-owned `immintrin.h` and `arm_neon.h` remain external
system headers and are recorded individually.

The 38 original observations cover missing/unknown/case-sensitive commands,
empty and UTF-8 arguments, equals syntax, separators, negative values, newlines,
nonzero owner statuses and owner exceptions. The frozen JSON regenerates
byte-exactly. Probe tests replace the real leaf modules at their existing `main`
boundaries. Separate end-to-end execution runs all eight actual native owners,
including KMX/KMB round trips and failure without overwriting caller output.
A coherent complex input passes through the converted circuit and normal
`circuit.append(op)` execution with the expected full X-state amplitudes.

Fresh-process inspection proves unknown dispatch imports no command owner and
real counting loads only its selected leaf. Module invocation, default argv and
C-visible NUL handling are exercised independently. The owner CLI grammar and
filesystem error behavior are reused and retain their prior acceptance scope.
Source help replaces only the actual executable invocation prefix.

## Acceptance and resources

The concentrated eight-owner regression passes 380 items. The existing physical
phase/overlap integration passes 55 items. Dispatcher statement and branch
coverage is 100%. Strict typing covers dispatcher, source generator and dispatch
workflow tests. Black, Ruff, team scope, architecture, IR layering and protected
API checks pass. Dependency policy retains 720 existing findings with zero touched
files. Receipts are under the reference environment using the
`g2-main-command-*-20261122` prefix.

The dispatcher emits zero additional quantum primitives, measurements or work
bits. It has no analytical quantum resource model. Actual source and bound
primitive counts, complex-state/measurement feedback and workspace evidence
remain owned by the accepted generation/conversion/sampling/counting suites.
Timing reports retain the separate measured-duration and source-goal categories.
The added converted one-qubit workflow contains one X, zero measurements and no
workspace. Pending Proposals 116–119 and previously recorded source defects remain
independent of this complete command-entry closure.
