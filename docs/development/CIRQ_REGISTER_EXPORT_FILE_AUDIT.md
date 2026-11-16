# Cirq register export source audit

The locked reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The Cirq package, exporter,
exporter tests, and tutorial blobs are pinned in
`tests/test_cirq_register_export.py`.

## Native ownership

The optional Cirq boundary places a validated physical circuit onto explicit
external qubits. `export_register_cirq` takes the native signature and the
physical wires of each port. It retains each register's shape with the bit
axis last. Supplied left, through, and right-only ports retain their qubits.
Omitted ports receive fresh names. Caller-owned qubits are reserved before
generating any port or work names. The result reports right and through ports
in signature order. Unbound temporary wires remain separate from the ports.
The exporter remaps the result of the existing static converter, so operation
order, measurement key grouping and conversion diagnostics use the same
implementation. The register adapter adds exact S-dagger and T-dagger
conversion for decomposed register operations. The public static converter
retains its v1 gate partition. No external qubit object enters native IR.

The public static exporter rejects instructions with classical conditions.
Its explicit lossy mode reports and drops the conditional operation. The
private register adapter can opt into strict feedback export with a caller
supplied key prefix. It supports single-wire measurements and resets,
classical XOR assignments, and conditions that read defined slots. Each
measurement receives a key derived from its instruction index. Overwriting
a slot preserves the earlier measurement record and points subsequent
conditions at the latest key. Callers supply distinct prefixes for circuits
whose measurement records will be composed.

Each classical XOR assignment snapshots a Boolean expression over immutable
measurement keys. Conditional assignment selects between the new expression
and the old destination. A later measurement overwrite cannot alter either
expression. The assignment emits zero Cirq operations. The exporter records
the emitted operation offset of each native instruction, so lease events
still occur at the correct point when a classical assignment emits no gate.

A guarded reset uses a Cirq classical condition directly. A guarded
measurement uses one dedicated clean work wire. It emits conditional CX
to that wire, measures it and resets it. The new classical value selects
the work measurement on the active branch and the old slot on the
inactive branch. No operation on the original data wire runs on the
inactive branch. The direct register adapter takes a caller-owned work
wire outside the ports. A wrapped program provides a clean lease for
the work wire and releases it after the expanded effect.

| Source responsibility | Native support |
| --- | --- |
| Source port order, qubit arrays and named bits | Concrete quantum register layouts, row-major words and explicit physical wire binding. Qid dimensions, aliases, shape and ownership are checked. |
| Through, left and right port placement | Through ports preserve identity. Left-only ports are excluded from output. Right-only ports use fresh names or caller-owned qubits. The adapter rejects aliasing across supplied ports. |
| Component conversion and graph order | The adapter consumes native physical IR. Native append and Compiler own decomposition and allocation before conversion. |
| Source Cirq gate wrappers and tensor fallback | `RegisterCirqGate` adapts an owned bound body. Through-only static bodies expose the native matrix after verifying any private workspace. A tensor-backed body can supply a matrix without Cirq leaf decomposition. Workspace-free gates retain integer repetition and adjoint behavior. |
| Workspace allocation and port transfer | The adapter replays Compiler lifetime events in the Cirq decomposition context. A left-only input is released under explicit ownership. Clean and borrowed private wires may be released and reused. The final clean lease of a right-only output is rebound to the caller's Qid, and its temporary reservation is released in register order. |
| Diagram symbols | Primitive circuits use Cirq's native exchange and Toffoli rendering. Wrapped register gates label operands in declaration order with the native register name and optional element index. |

## Verification

The locked original suite including its tutorial passed 16 cases. Native tests
compare original one- through five-bit word swaps, the named two-bit diagram,
a square of the wrapper's swap, and both Toffoli diagram modes. The full
unitaries use a caller-selected qubit order, including original disjoint
external line ranges. A separate shaped register checks the ordered
terminal measurement operands and exact key.

For one- through four-bit word swaps, the bound gate agrees with both the
locked decomposable and tensor-only wrappers. The bound native circuit
emits the source's swaps on the same ordered operands. A separate native
matrix body has no Cirq leaf decomposition and supplies the same complete
unitary through the wrapper. A phase-sensitive H/T body checks signed
integer powers and the adjoint gate, with an idle declared operand. The
right-only witness preserves supplied qubits in its decomposition and
returns no direct matrix hook.

The original four-control MultiAnd produces seven through/result wires.
Register export binds the original right-only junk and target qubits and
matches all 39 primitives with operands in the original decomposition order.
The exported circuit matches the full seven-wire complex matrix. Native
execution also matches the original coherent clean-input result with both
junk values and the target. The emitted sequence contains 18 CX, six H, six T,
six T-dagger, and three S. The source composite model declares three And
calls. The native measured compute/uncompute restores a coherent input and
zero scratch. With register feedback enabled, the positive and mixed control
variants of two consecutive And compute/uncompute pairs match the locked
source primitive sequence, including conditional CZ and reset. The two
measurements share a native slot and receive separate Cirq keys. All four
unnormalized complex branch vectors match, including a spectator and
zeroed scratch. A disjunction witness covers both classical polarities.
The public static contract continues to diagnose the same feedback body.
Each positive pair emits six CX, one conditional CZ, three H, one S,
two T, two T-dagger, one measurement and one reset. The mixed pair adds
four X. The separate locked model gives one And for compute and one
Clifford plus one measurement for erase in both variants.

The native XOR witness checks all eight inputs of the conditional update
`c ← c XOR (a AND b)` with complex spectator amplitudes. It overwrites
the native slot holding `a` before reading the assigned result. The Cirq
condition still references the original measurement. The test compares the
full native and Cirq branch vectors, including a work wire allocated after
the classical-only instruction and cleared before release.

The guarded-effect witness checks the four possible branch labels with
complex amplitudes and a spectator. Three labels have support. On the
inactive branch the old slot controls the result and the data retains its
coherence. On active branches the measured data resets, with the
unmodified complex amplitudes in each projected sector. All branches have
zero work. A single guarded measurement emits one conditional CX, one
measurement and one reset on the work wire. Its native IR declares one
guarded measurement. The separate guarded reset emits one conditional
reset. These are physical counts for the adapted witness.

The [native tutorial](../../examples/notebooks/cirq_register_interop.ipynb)
uses the original seeded Cirq circuit, rewrites its iSWAP into exact S, CZ and
SWAP operations, and checks the complete matrix through import and export.
It demonstrates named placement, shaped registers, fresh output placement and
append. It keeps the original modular exponentiation dimensions and call
declaration, and runs the adapter diagnostic on a smaller native compiled
instance. The original five-qubit composite QFT and modular exponentiation
roundtrip remain open. The tutorial's MultiAnd export cell still needs to use
the newly verified register adapter.

The locked ModExp diagram fixture uses base three, modulus four and a
two-bit output. One-step source decomposition binds `x0` and `x1` through
initialization and all four multiplier calls in exponent order 3, 2, 1, 0.
Full source flattening raises `Too-large classical value encountered in val`.
Native execution refuses a modulus that exceeds the word encoding. Valid
modulus fixtures already compare the native measured execution channel.

For the original controlled preparation of three values, source Cirq uses
3, 5, 7, and 7 wires at its opaque, decomposed, once-flattened, and fully
flattened levels. The final native body uses seven wires with four clean
work wires. Six measurements reuse two classical slots. Forward and
adjoint native runs compare all 64 measurement histories with the original
complex state, including a spectator and zeroed work wires. Equal final
width does not establish equality of the external allocation event order.

The adapter uses the compiler's emitted instruction offset for each workspace
event. The expanded operation position also includes non-emitting register
operations. An exact lifetime witness performs two sequential leases on one
private wire. Its clean and borrowed versions each preserve the source's
allocation/free order and four CX operations. The complete complex state
matches with a spectator. Clean work returns to zero, and borrowed work
returns its joint state with the spectator. Through wires retain their input
mapping. The private feedback path records the mapping from native instruction
offsets to emitted Cirq operations for measured workspace bodies.

For the original two-bit `IntState(1, 2)`, both wrappers emit one X on the
same caller-supplied least-significant bit and give the full `|01⟩` result.
The source allocates two separate bits. The native typed component allocates
one two-bit word. Each releases the same temporary Qids in bit order after
binding the outputs. The source model counts one arbitrary Clifford
component. An explicit replay fixture first leases the same native physical
ID as private work. Its two X gates stay on scratch and erase it before
the final lease is mapped to the caller's output.

Static through-unitary inspection checks that every clean wire exits in zero
and the admitted matrix factors as identity on borrowed wires. The
workspace witness matches the full logical matrix for clean and borrowed
leases. A leaked clean CX and a phase entangling borrowed work both refuse
matrix inspection. A separate complex phase fixture covers interleaved
clean and borrowed axes. Inspection uses the native dense width limit.

The source X/Free composite and the native owned left-only body each emit
one X on the caller's input before releasing that input. Neither passes the
caller Qid to the underlying qubit manager. The source model contains one
Clifford. A separate ordered replay reuses that physical wire number for a
right-only port and emits the second X on the distinct output Qid. The
spectator's complex amplitudes are preserved and the consumed input is
zero. The locked source `IntEffect` still refuses Cirq export of its basis
effect. The source witness uses explicit X and Free operations.

## Remaining files

- Q-0714 awaits the package-entry export map.
- Q-0715 retains dense inspection of dynamic workspace bodies, source
  wrapper value identity and custom glyph callbacks. Guarded measurement
  requires a dedicated clean wire in the export scope.
- Q-0721 retains the source composite QFT, MultiAnd export and ModExp
  roundtrip as open tutorial sections. The adaptation shows their diagnostics.

## Export test family acceptance

Q-0716 covers the complete frozen test file, including its notebook receipt.
The native witnesses exercise the original swap and tensor-only widths,
integer power admission, Toffoli glyphs, MultiAnd output binding, measured
cleanup, private workspace, and register-construction rejection cases.

The original modulus-four ModExp diagram is a one-level wrapper witness.
Its full decomposition retains the locked width refusal described above.
An additional valid ModExp with base two, modulus three, one exponent bit
and two output bits checks the original one-level operand schedule. Full
source Cirq export of the valid graph refuses the source basis effect.
The native path exports 1,155 operations with exact operands and
measurement keys. A complete 17-wire complex trajectory, including a
spectator, matches the source classical output isometry and leaves all
13 work wires clear. The 48 measurement outcomes are retained
individually. The source model separately gives 206 T-equivalent gates,
48 measurements and a peak of 16 wires. The full physical gate comparison
uses the compiler's source-derived primitive trace. The numerical witness
samples one trajectory of this larger circuit.
