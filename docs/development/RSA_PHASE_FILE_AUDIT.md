# RSA phase estimation and package declarations

The locked reference is Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers
`qualtran/bloqs/cryptography/rsa/__init__.py`, `rsa.ipynb`,
`rsa_phase_estimate.py` and `rsa_phase_estimate_test.py`.

## Source declaration and bound circuit

The source header documents RSA factoring and re-exports its two
operations, ModExp and RSAPhaseEstimate. The native modules expose those
operations as ModularExponentiation and RSAPhaseEstimation through their
algorithm implementation paths. Both are used in the native tutorial.

The unsupplied factoring base and the concrete and symbolic modulus
sizing share a single helper with native modular exponentiation. The
empty source signature and the symbolic call graph are modeled
independently of the executable binding. The source graph declares
one `MeasureQFT(n)` and `2n` modular multipliers. The frozen composite
prepares `2n` PlusState bits, initializes the modular target to 1,
performs its ordered controlled multiplication chain and passes
all `2n` bits to MeasureQFT. It dirty-releases the target. The native
append binding uses the same composition and exposes independent
result slots and the caller's retained modular target.

The tests resolve the source's original input and returned Soquets, the
squared constants and the readout destinations. The native prefix
matches the verified modular exponentiation gate order; the QFT
transform and output measurements follow the original physical wire
order on permuted bindings. Independent DFT sums compare the full
complex state for all 16 terminal results at modulus 3 and all 64 at
modulus 5, including zero-probability outcomes, correlated modular
targets and all clean scratch. Balanced arithmetic erasure histories
are independently covered by the shared modular exponentiation tests.

| Width and modulus | Native primitive counts | Frozen declared model |
|---|---|---|
| 2, 3 | 824 T/T†, 2,096 CX, 600 H, 6 CP, 2 SWAP, 196 measurements, 19 bound wires | 193 AND, 8 CSwap, 1,397 Clifford, 195 measurements |
| 3, 5 | 2,574 T/T†, 6,804 CX, 1,884 H, 15 CP, 3 SWAP, 618 measurements, 25 bound wires | 1 T, 615 AND, 18 CSwap, 4,545 Clifford, 618 measurements |

The declared model has the smaller Fourier child and excludes the
explicit initializer and PlusState preparation. Its QubitCount query
fails at the unused source measurement results. Matching measurement
totals in the second fixture combine different arithmetic and Fourier
counting rules. The executable binding inherits the modulus-fit and
coprime admission conditions. A factory request for modulus 8 keeps
the source exponent width of six and has no admitted lowering with
the three-bit result.

## Tutorial and test disposition

The 22-cell source notebook ran in its source kernel with the original
test file, which reported six passes, four skips and one expected
failure. The [native RSA notebook](../../examples/notebooks/register_rsa_phase_estimation.ipynb)
preserves the three modular exponentiation instances, both phase
instances, symbolic and concrete signatures, and direct call graphs.
It adds a small concrete append example with explicit terminal slots.
The six native code cells and four SVGs ran in a clean native kernel;
the previous worked factoring tutorial also passed again.

Q-0468, Q-0470 and Q-0474 close. Q-0473 remains open for the recorded
source width/model and inherited modulus-fit boundaries, and the
unverified large physical circuits.

## Complete source model identities and factor generalization

The source file's constructor, empty signature, factory, physical composite,
multiplier declaration, grouped call graph, `_K` and `_generalize_k` behavior,
and two example factories now have native owners or explicit dispositions.
The constructor and factory reuse `shor_factoring_parameters`; physical
construction continues through `ModularExponentiation` and `MeasureQFT`.
`source_callees(allocator)` retains the exact generic multiplier key, including
unsigned width, newly allocated factor and modulus. Its allocator belongs to
the caller. `source_call_graph()` now retains the full root identity and these
parameterized children. The example notebook displays those parameters.

`generalize_rsa_multiplier_call` changes only the multiplier factor to the
source symbol `k_exp`. It uses the existing symbolic graph engine to combine
identical generalized calls while retaining different widths and moduli.
The regression compares both original example factories and preallocated
symbol names, then generalizes the eight actual multiplier calls from the
width-four, modulus-15 source composite. A second consumer verifies separate
width/modulus leaves and their additive cost. No new registry or execution
protocol is introduced. The original documentation specification maps to this
audit and the existing runnable notebook.

The immutable source file matches inventory blob
`9416620d05f37aa4fde717d498c185ff2d9eb8d0`. Original non-notebook tests report
five passes, four skips and one expected failure. The skipped lanes cover
symbolic decomposition, source counts availability and type incompatibility
between the initializer's QAny and the multiplier's QUInt. Prior full notebook
evidence remains recorded above. These source outcomes are not native passes.

The source's declared `n`-bit Fourier model remains separate from its physical
`2n`-bit readout. The new parameterized model does not resolve that discrepancy
or change physical primitive counts. Fresh resource receipts reproduce the
19-wire and 25-wire native circuits and both original gate-cost declarations
in the table above. The existing full-complex phase projections and modular
measurement-channel regressions are rerun for this batch. Q-0473 remains open
for the recorded source and physical-width boundaries.

Batch validation on 2026-11-21 passed 45 concentrated cases, 29 integration
cases and the six-cell, four-SVG native notebook in an isolated kernel with
reference imports blocked. Strict typing for all three Python files, Black,
Ruff, Algorithms scope, architecture, IR and public API checks passed. The
Cirq 1.6.1 lane passed 11 model and physical cases with native Kickmix present.

## Stored constructor interface recorded before implementation

Ecosystem owns the exact RSAPhaseEstimate and MeasureQFT identifiers and reuses
the existing RSA parameter validation and Fourier-measurement signature.
Compiler owns the literal 4n+3-call root: 2n PlusState calls, IntState(1),
2n controlled modular multipliers in reverse exponent order, MeasureQFT(2n),
and dirty Free of the modular result. Initial states and multipliers reuse
their current typed factories. All private arithmetic work remains with those
owners. The modular result is deliberately discarded dirty, not asserted zero.

Unmodified MeasureQFT cannot finalize its unused MeasureZ CBit outputs. Its
exact constructor therefore remains a bodyless LEFT-array declaration; the
new RSA root must refuse physical binding at that leaf. Preserve width-zero
QFT failure and concrete deeper-library failure, and symbolic RSA decomposition
refusal. Do not substitute the Kickmix demolition QFT or silently repair source
finalization. Existing native explicit-readout execution is a separate proven
path. Root signature remains empty, with no added public result or entry point.
The original n-width model stays separate from actual 2n readout.


### Exact root/shim constructor closure, November 27

Both original identifiers now resolve through
`ecosystem/_rsa_component_library.py`. The Compiler root resides in
`registers/phase/rsa_phase_component.py`. Native primitive, modular arithmetic
and backend owners are unchanged. Concrete widths two/four, the original
eight-bit example, and fixed width with symbolic modulus/base preserve all
source named input producers, port/array identities and ordered outputs.
Level-zero/one library round trips preserve source fields, and complete Programs
are identical between levels. Repeated equal multiplier constructors share one
native callee. Each distinct child definition equals the existing registered
CModMulK factory, with no modified helper body. Source caller-visible returns
remain empty, and the final release retains its dirty flag.

Symbolic-width metadata survives all three library levels and remains a
nonexecutable declaration. Original MeasureQFT shallow metadata is preserved
for widths zero/one/two. Its zero-width rejected-QFT-argument failure and its
positive-width unused-CBit finalization failures are asserted unchanged for
decomposition and deeper libraries. Concrete RSA level-two serialization
reaches that same original BloqError. Missing physical bodies explicitly refuse
binding. This closes the constructor/root wiring gap without inventing a
successful original terminal measurement graph or closing Q-0473.

The actual first multiplier of the width-two root is extracted with the
existing reachable-definition utility. Its primitive operands and feedback
match unmodified source expansion. Six valid input basis columns and a complex
superposition reproduce source classical permutation with absolute phases and
zero work amplitudes. A local erasure certificate covers all 2^48 raw histories;
none are sampled. Invalid modular input codes are not included in the promised
clean-output subspace. All scratch leases return. Existing native RSA readout
and parameter/model regressions also pass, retaining their separate output API.

The selected multiplier emits 1154 primitive instructions, including 206
T-family gates, 48 measurements and 48 resets, with thirteen private wires at
peak (three logical wires plus scratch, sixteen total). Its independent source model is
two CSwap, 48 And, 348 Clifford and 48 measurements, with peak sixteen. Model
T conversion uses seven T per CSwap; these model categories are not primitive
counts. The width-two root has eleven calls and four actual terminal inputs;
its independent original model still uses a two-bit terminal and declares
eight CSwap,193 And,1397 Clifford and195 measurements. Source whole QubitCount
fails at MeasureQFT; whole emitted inventory is unavailable, not zero.
External raw receipts are `g2-rsa-resources-20261127.json` and
`g2-rsa-constructor-probe-20261127.json`.

Both actual Cirq1.7.0/Qiskit2.5.0 and Cirq1.6.1/Qiskit2.0.0 lanes pass59
concentrated cases without skips, then the strengthened six-column child check
passes independently in each lane. Local integration passes145. Actual
Qualtran0.7.0 and Kickmix0.1 Simulator are asserted. Strict three,Black/Ruff,
Compiler/Ecosystem scopes,English/links,architecture,IR and API baselines pass.
Three frozen source/test/shim blobs remain unchanged. Original suite results
are5pass/4source skips/1source xfail/1notebook deselected; prior notebook evidence
is reused. The initial diagnostic was explicitly stopped after identifying
repeated child expansion; final bounded runs reuse unique child evidence and
retain the full original example and sharing checks. No source failure is
converted into a skip or a native physical success.
