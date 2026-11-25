# Bounded native literal powers

Follow #995, merged at 7c5d6e4b896957df4c820782717990c5c01f99d4 after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-NATIVE-POWERS-20261101.md).

## Complete user workflow

```python
import flagquantum as fq
from flagquantum import synth

source = """qfunc main(output x:qnum<2>,output result:qnum) {
    allocate(x); hadamard_transform(x); result=x**3;
}"""
program = synth.synthesize(source, source_format="qmod", max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "result"))))
```

The named pairs (0,0), (1,1), (2,8) and (3,27) each have probability 1/4.
Run `python -m examples.synthesis.tutorials.qmod.qmod_powers.main`. Users need no multiplication
chain, intermediate register, gate sequence, scratch assignment or bit decoding.
Classical powers fold; quantum-dependent constants retain their dependencies.

## Architecture and limits

Only two existing Ecosystem production files change. Parser admits positive
literal exponents 1..8. Adapter infers the complete unsigned power domain and
emits left-associated existing qint.multiply operations with exact intermediate
widths; exponent one returns the already emitted operand. At most seven
multiplications implement one admitted power. Existing **2 graphs retain the
same multiplication shape and aliases. No new API, IR, opcode, kernel, executor,
dependency or ownership mechanism is introduced.

Compiler remains authoritative for multiplication selection, aliasing, scratch
scheduling, local inverse cleanup, target conversion and complete resources.
This is not optimal addition-chain selection. Zero, negative, fractional,
symbolic, computed and larger exponents remain refused. The operator stays
right-associative: x**2**2 is refused as a computed exponent; (x**2)**2 works.
The 256-bit domain, existing expansion/node and resource guards remain in force.
Syntax admission does not imply that every higher power fits the CPU memory
ceiling; larger circuits may be refused before execution.

## Executed evidence

55 new cases pass: all eight exponents on both one-bit basis values under both
objectives, nontrivial coherent two-bit cubes and explicit-product replacement,
complex external references in phase and value outputs, parenthesized powers,
invalid forms before planning, classical folding, width, expansion and workspace
limits. The corrected old negative retains an unsupported **9 exponent rather
than treating the now-admitted **3 spelling as invalid.

The broad native-source regression plus API candidates passes 876 cases in
88.90 seconds. Both production modules cover 550/550 statements and 240/240
branch exits. One additional quantum-dependent constant-power case then passes
with the full 55-case new suite in 20.33 seconds, proving initialization reads
are retained alongside coherent output. Sets overlap and are not added. Strict mypy 2.x passes 879
production/example sources; Black 26, Ruff, architecture, public API snapshot,
generated docs and maturity checks pass. Original source/license digests remain
unchanged; this synthetic probe does not increase the four-original cohort.

On CPU complex128 / Torch 2.13.0, both width/depth objectives select 21 allocated
and used qubits, 14 hidden work bits, 269 operations (2 H, 133 CX, 134 CCX),
depth 206 for the cube example. These are implementation costs, not an optimum,
performance or hardware claim. Source/result widths are two/five bits.

The default smoke/unit run is recorded separately in the PR. The preceding
local baseline had three unchanged QFT fusion assertion failures; no Simulation
implementation or assertion is modified to make this feature pass. A temporary
verification wrapper initially had a quoting SyntaxError, corrected before
starting the actual default run; it is not counted as a test result.

No wider classical language, arbitrary exponent, signed/fixed-point arithmetic,
Input/free/drop, general control/inversion or hardware admission. Independent
developer trials remain 0/2 and release_authorized=false.
