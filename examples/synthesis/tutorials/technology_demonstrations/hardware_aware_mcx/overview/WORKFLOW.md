# Source mapping and acceptance

The pinned source Notebook has three code cells. Cell1 constructs15 controls
and one target, requests depth optimization under width18, and synthesizes
CX/U circuits for line and all-to-all connectivity. Native code retains these
parameters and prints actual executable costs. Source cells3 and5 print the
all-to-all and line depths respectively; the native Notebook does the same.
The hosted circuit viewer becomes local resource summaries, connectivity
inspection and named results. It is not a reproduced interactive hardware UI.
The source's unspecified random seed is replaced by deterministic construction.

Acceptance includes all original zero-input scenarios and supplementary
coherent, fully activated target-zero/target-one and near-active scenarios in
both topologies. Independent Circuit preparation plus an explicit index
permutation supplies complete complex reference amplitudes. Compare absolute
phase with tolerance2e-11 and total dirty work probability below1e-22. Verify
CX/U-only leaves, line adjacency, physical width, nonzero-target XOR and budget
refusals. No gate decomposition is shared with the truth oracle. Unit tests also
enumerate all logical basis columns at3..15 controls and arbitrary dirty-borrowed
inputs at3..7 controls, and compare actual random complex Circuit states.

Runtime optimization preserves universal U spelling and complete CX routing
SWAP witnesses. Witness validation refuses incomplete, misdirected, overlapping
or mismatched-source triples. Other unsupported basis conversions and phase
profiles remain checked refusals. No export, signature, IR opcode or public
storage type is added. The original216 quantum counterparts remain frozen.
Coverage credit requires fresh-kernel Notebook execution, focused quality gates,
216+6 replay and a terminal default-tier comparison against the previous batch.
