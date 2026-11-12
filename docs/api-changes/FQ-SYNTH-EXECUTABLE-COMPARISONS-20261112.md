# Executable native comparison entries

The user authorized completing all 216 native quantum-program counterparts and
their independent Circuit acceptance. This integration contract governs the
development catalog and example comparison launchers. It changes no Stable Core
API, native language, compiler budget, numerical kernel or source parameter.

## Observed gap

On implementation `589d67070`, the documented comparison modules for
`whitebox_fuzzing` and `grover_graph_coloring` define independent reference
Circuits but have no module entry point. Running either module exits zero
without comparing a native program. Their existing integration tests contain
real full-state checks, but those tests do not make the documented module
command executable. A zero process exit is insufficient acceptance evidence.

The ongoing audit retains its exact source pins and raw exits. These two exits
must be classified as missing comparisons rather than successful native
acceptance, even if the underlying integration tests pass.

## Integration boundary

`tools/synthesis_catalog.py` remains the single catalog validator. Before a
native comparison can appear as runnable, validation must find its standard
`if __name__ == "__main__"` guard containing a call. A reference-only module or
an empty guard must fail catalog validation with a specific diagnostic. Add
negative tests using isolated temporary comparison files, without modifying
the real examples or bypassing source/provenance checks.

This structural check is necessary but not a proof of meaningful assertions,
independent semantics or numerical agreement. Source review and execution of
the actual comparison remain required. Imported shared runners are allowed
for byte-identical programs when the native entries share the same builder;
they do not create independent algorithm coverage.

This protected integration change precedes the Compiler-owned example changes.
All commits remain local until the complete native-corpus gate passes.

## Compiler-owned example acceptance

Add callable runners and module guards to both existing reference modules.
Reuse their independent CNF/truth enumeration and explicit Toffoli-ladder
Circuits, and the existing native builders. Compare every complex logical
amplitude without phase alignment, and independently check zero workspace on
both producers. Honor declared native register order. Different scratch widths
are compared by their proven-zero embeddings, not by dropping dirty amplitudes.

Run both programs at `r=0/1/2`. Also run whitebox at its unchanged original
default `r=10`; graph coloring retains its original default `r=1`. No width,
iteration, tolerance, workspace, memory or operation cap may be relaxed to
obtain acceptance. Retain the existing source/native/Circuit regression suite,
including arbitrary phase, zero-angle reads, aliases and actual-budget refusal.

After the source-frozen audits terminate, land the example changes and rerun
the complete documented native-module audit on the final sources. Keep all
earlier exits and failed or incomplete checks in immutable evidence. A mixed
collection of runtime versions is not a fresh full-corpus certification.

## Delivery gates

- The catalog rejects reference-only and empty-guard comparisons; ordinary
  executable comparison modules continue to validate.
- Both documented module commands perform their full-state assertions and
  retain original default workloads.
- Named Grover scenario/refusal tests pass. Complete autocallable scenarios,
  source replay, quality checks and the final native216 audit remain separate
  requirements; this change cannot waive any of them.
- Preserve five-category counts and separate Notebook/Python workflow status.
  Keep six approved source repairs distinct from the 210 unchanged accepted
  originals. Retain licenses, hashes and Qmod sources exclusively in fixtures.
- Update the delivery ledger and generated catalog after verification. Do not
  open an integration PR or start remote full CI before native216 acceptance.

## Supported user workflow

The native builders already expose the original high-level programs:

```python
import flagquantum as fq
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.main import (
    build_program,
)

program = build_program()  # Original twelve data bits and ten Grover iterations.
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

The repaired documented comparison command must execute and validate that same
default workload:

```sh
python -m examples.synthesis.applications.cybersecurity.whitebox_fuzzing.comparisons.circuit
python -m examples.synthesis.tutorials.basics.grover_graph_coloring.comparisons.circuit
```
