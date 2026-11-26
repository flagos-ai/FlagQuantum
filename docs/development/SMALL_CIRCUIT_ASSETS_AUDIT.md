# Complete small original circuit assets

## Ownership and interfaces before implementation

K-0330 through K-0335 are the six remaining small frozen Kickmix testdata KMX
assets at 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453. Verification owns their
byte-preserved test fixtures and source observations. Existing Compiler owns
KMX/KMB parsing, registers, exact angles and physical binding. Simulation owns
packed physical execution. Existing circuit.append(op) remains the coherent
execution entry. No new production representation is needed.

Physical IDs and mixed register member order are source-defined. Increment
registers and constant-modular multiplication use little-endian source ports.
The classical-add asset has eight data wires, two cleared wires and six dirty
borrowed wires. The increment-eight asset has five clean work wires q12..q16
and four idle spectators q8..q11. Measurement Fourier owns five cleared targets
and five classical outputs. Full variable modular multiplication retains its
source work/condition behavior; its complete conditional channel is verified
in the follow-up below.

Verify source Git blobs before copying the assets. Canonical text and full KMB
bytes come from the unchanged compiled reference. Packed branch observations
record exact per-lane output bits and 128-bit phases. Small coherent checks use
the actual physical binder and circuit.append(op), full unnormalized complex
branches and clean-work checks. Source concrete opcode counts and bound primitive
counts are distinct from existing analytical estimates. These raw assets declare
no independent analytical model.

## Complete asset disposition

| Parent | Frozen asset | Verified execution and disposition |
| --- | --- | --- |
| K-0330 | cmul16_mod_21.kmx | All 64 basis columns, arbitrary complex superposition and both control values are exact. Original asset multiplies valid residues by 4, despite its 16 filename. Retained source discrepancy keeps this parent open. |
| K-0331 | iadd8_qc_2clean_6dirty.kmx | Complete 8-bit classical-add asset accepted. All 256 addends, 256 target values, 64 dirty words and 64 erasure histories have the exact permutation, positive branch phase, clean q8/q9 and restored dirty q10..q15. |
| K-0332 | inc3.kmx | All eight operator columns and a complex superposition match little-endian increment. No work or measurements. Accepted. |
| K-0333 | inc8.kmx | All 32 erasure branches match increment on an arbitrary complex input including four entangled idle spectators. All five clean work wires return to zero. Accepted. |
| K-0334 | mul16_mod_21.kmx | Complete conditional-channel proof verifies all 42 valid logical columns, every allowed instrument history, absolute complex phase and clean work q0..q7. Existing exact source storage/record/phase observations, 256 packed families and four dense branches are reused. Accepted. |
| K-0335 | qft_measurement.kmx | All 32 complex branches match the independent dense Fourier projection, including output bit order, phases and all targets cleared. Accepted. |

The six raw files preserve their original bytes and frozen Git blobs. The
fixtures retain source copyright under Apache-2.0. The compiled locked source
produces their canonical text and complete KMB bytes. Two seeds per asset give
768 basis observations with exact final quantum/classical masks and all 128-bit
phase words. A streamed original-source execution is checked against the full
unaltered source circuit before its measurement transcript is recorded. The
native packed owner replays these same outcomes and matches every output and
phase. Reset draws are recovered from their observable sign on basis-one lanes;
on basis-zero lanes the draw is unobservable and zero is retained. This is
basis-state transcript evidence, with coherent evidence supplied separately.

## Full-complex and cleanup proof

Unmeasured assets run every operator column and a seeded arbitrary complex state
through the actual Circuit.append(FieldOperation(...)) entry and owned physical
binder. Increment-eight evaluates every unnormalized measurement branch with
entangled idle spectators, retaining the common 1/sqrt(32) Kraus factor.
Fourier measurement compares every full branch directly with a dense DFT in the
source little-endian input/output convention. No global phase is discarded.

Classical addition executes a packed exhaustive basis domain through the real
Simulation owner. Linearity proves its coherent positive permutation for all
addends and dirty words. Six selected addends additionally run every erasure
branch through the bound primitive IR on a dense complex state over all target
and dirty wires. Each of the six explicit reset targets is proved initially zero
or previously cleared by HMR before that reset. Both primitive reset Kraus maps
on zero are independently checked as identity/sqrt(2). Thus all 64 reset histories
share each of the 64 erasure maps, including normalization and positive phase.
The dense erasure comparisons retain their 1/sqrt(64) factors. Actual full reset
histories multiply this by a further 1/sqrt(64). This is an exact reset-domain
reduction, while source records and stored assets remain unchanged.

The larger modular asset has 46 reset and 46 measurement sites with conditional
activity. Its four dense checks retain activity-dependent unnormalized factors,
including inactive-outcome zero branches and all work wires. The 256 packed
branch families cover all 32 values and both controls, with clean-work and
positive-phase assertions scoped to valid residues 0..20. The original batch
did not prove every conditional history; the follow-up below closes that gap.
Invalid-residue
storage/source replay is retained separately without an arithmetic promise.

## Complete conditional modular channel (2026-11-27)

The complete unchanged asset has fourteen quantum wires: little-endian data
q8..q12, control q13 and clean work q0..q7. Valid inputs are all residues 0..20
under both control values. Its 46 resets and 46 HMR operations expose 92
instrument sites, of which 32 are conditional. This is 46 physical measurement
instructions plus 46 unfolded reset effects, not 92 emitted measurements.

The shared verification certificate consumes the actual bound primitive IR.
It reuses the existing reduced Boolean support owner, tracks quantum basis
functions, sign phase, classical XOR assignments and guards, and checks the
literal owned CCZ expansion. That expansion has unconditional H gates around
a guarded CCX; it must not be replaced by a different assumed guard pattern.
Each recognized H/measure/X-clear block is checked with its actual target,
result slot and clearing condition. Inactive result slots are proved initially
zero. Unknown atoms, matrix overrides and failed obligations refuse proof.

An active X-basis effect on a basis bit x contributes
`(-1)^(h*x) / sqrt(2)` and clears that bit. An inactive effect has only canonical
identity label zero; label one has zero operator. The certificate proves that
activity depends only on earlier outcomes, never logical input. Thus for every
allowed complete instrument history h, the Kraus operator on the valid input
subspace is `K_h = 2^(-A(h)/2) U`, where A(h) counts active effects and U is
identity for control zero and multiplication by sixteen modulo twenty-one for
control one. It proves the exact returned basis map, zero total sign phase and
zero work for every allowed history. Linearity includes arbitrary complex
superpositions of all 42 columns and external entanglement. Causal activity and
the independently checked complete local Kraus pairs establish total probability
one. A single common factor across different histories is not assumed.

There are exactly 623824686318171615643303936 allowed canonical histories. The
proof uses 3761 Boolean nodes rather than enumerating those histories. This
count is not a claim of exhaustive RNG-seed coverage: the original simulator
consumes RNG draws even when masked, while these ignored draws are summed into
the inactive identity channel. Original packed RNG/transcript observations
remain separately retained and unchanged.

An independent scan of the raw source effects finds sixty unconditional sites,
eight controlled by b27, sixteen by b28 and eight by b29. Each selector is sampled
once unconditionally; temporary inversions are proved restored before every
effect reads it. Consequently `A(h) = 60 + 8*b27 + 16*b28 + 8*b29`, and the
history count is `2^57 * (1 + 2^8)^2 * (1 + 2^16)`, equal to the Boolean graph's
count. Each of the eight selector choices has total probability exactly 1/8;
exact rational arithmetic independently verifies their sum is one.

Independent local checks use the locked Kickmix 0.1 native Simulator for both
R and HMR, with both input bits, active/inactive guards and both observed active
outcomes. Identical seeds expose R's hidden first draw through HMR instead of
inferring it from R's phase. Direct dense projection additionally checks every
local Kraus column and a complex superposition. A two-effect conditional example
has three allowed histories with factors 1/sqrt(2), 1/2 and 1/2; all columns,
an entangled spectator and total probability are checked. Full complex CCZ
columns check both guard polarities. Removing phase feedback from the small
example or the actual modular asset fails the certificate; stale inactive
result slots also fail.

The whole unexpanded fourteen-wire asset also runs through normal
`DynamicCircuit.append(op)` and native trajectory execution with foreign SDK
imports blocked. Its full complex state and clean work agree with the same
controlled modular permutation. This runtime sample checks execution-path
integration; exhaustive history coverage comes from the algebraic proof.

K-0334 now closes without changing source bytes, gates, compiler/runtime code,
reference observations or the established valid-input promise. K-0330 remains
open for its separate original filename/multiplier discrepancy. No source
defect waiver or invalid-residue arithmetic guarantee is inferred.

The follow-up passes 325 concentrated cases and 145 local integration cases.
All three changed Python files pass strict typing, Black and Ruff; Verification
scope, English, links, architecture, internal IR and protected API checks pass.
The asset, source simulator and source RNG blobs match the frozen CSV. The
673 source records and 920 ordinary bound instructions retain the prior exact
inventories. Exposing reset outcomes produces 982 verification instructions;
this is proof instrumentation, not a change to the emitted user circuit.

## Actual records and bound primitives

These are syntactic emitted counts before branch selection. Classical setup
slots are included explicitly in bound counts. The raw assets declare no
independent analytical cost model. Existing component-model estimates retain
their separate owners and are not substituted for these actual counts.

| Asset | Actual source records | Bound primitive histogram |
| --- | --- | --- |
| cmul16_mod_21 | 8 X, 21 CX, 41 CCX | 8 x, 21 cx, 41 ccx |
| iadd8_qc_2clean_6dirty | 147 records including 6 R, 6 HMR and one NEG | 81 x, 20 cx, 16 ccx, 4 swap, 12 h, 6 reset, 6 measure, 15 z, 1 cz, 22 classical_xor |
| inc3 | 1 X, 1 CX, 1 CCX | 1 x, 1 cx, 1 ccx |
| inc8 | 6 CCX, 6 CX, 5 HMR, 5 CZ, 1 X | 6 ccx, 6 cx, 5 h, 5 measure, 6 x, 5 cz, 1 classical_xor |
| mul16_mod_21 | 673 records including 46 R, 46 HMR and 4 CCZ | 53 cx, 48 ccx, 322 x, 100 h, 46 reset, 46 measure, 34 cz, 49 z, 222 classical_xor |
| qft_measurement | 2 SWAP, 5 HMR, 10 Z_POW | 2 swap, 5 h, 5 measure, 5 x, 10 phase, 5 classical_xor |

Every bound histogram is checked against the independently enumerated source
primitive expansion. Source opcode histograms remain separately stored.
No decomposition gate-count parity is inferred from an analytical model.

## Acceptance evidence and reproduction

The original 285-test concentrated batch and 55 local register integration tests pass.
Strict mypy passes the complete three-file changed Python surface. Black, Ruff,
Verification scope, architecture, internal IR and protected API baseline pass.
Native code and documentation pass whitespace checks. The two original modular
assets deliberately retain their trailing blank lines for exact Git-blob parity. The shared asset binder helper has 100% statement
coverage. All six assets and the complete observation fixture regenerate
byte-for-byte. Existing dependency-policy findings are unchanged because this
batch adds no production imports or numerical implementation.

```bash
python tests/unit/fixtures/generate_small_circuit_assets.py \
  --source-root /path/to/frozen/kickmix \
  --asset-directory /tmp/circuit_assets \
  --output /tmp/small_circuit_assets.json
cmp tests/unit/fixtures/small_circuit_assets.json /tmp/small_circuit_assets.json
python -m pytest tests/unit/test_small_circuit_assets.py \
  tests/unit/test_conditional_erasure_certificate.py \
  tests/unit/test_kickmix_qft_measurement.py \
  tests/unit/test_kmx_generation_command.py -q
```

The generator requires the locked compiled Kickmix 0.1 Simulator and refuses a
missing native extension. It performs no silent source-reference skip. Raw
intake, source observations, resource histograms and regression receipts live
beside the main ledgers under ftqc-reference-env/g2-small-circuit-assets-*.
