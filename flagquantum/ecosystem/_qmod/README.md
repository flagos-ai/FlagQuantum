# Qmod frontend

This package translates bounded Qmod source into FlagQuantum Compiler-owned
programs. The public entry is `synth.synthesize(Path("source.qmod"))`, or source
text with `source_format="qmod"`. Python qfunc enters through the Compiler's
Python frontend. Both paths reuse register verification, lowering and Runtime.

## Pipeline and ownership

```text
source.py                bounded UTF-8 acquisition and conversion diagnostics
    |
syntax.py                tokens, declarations and parser state
expression_parser.py     expression syntax
statement_parser.py      statement syntax
    |
adapter.py               source version, constants, rewrites, helper compilation
    |
context.py               one function's bindings, views, scopes and effect chain
    |
adapter_statements.py    ordered structural and statement dispatch
    +-- adapter_assignments.py   values, XOR and expression phases
    +-- adapter_storage.py       allocation, preparation and storage lifetime
    +-- adapter_gates.py         primitive and algorithm operations
    |
Compiler HybridProgram   source-local syntax stops here
    |
Compiler verification -> constrained lowering -> CircuitIR -> Runtime
```

`source.py` owns file acquisition and public conversion reports. `adapter.py`
owns source orchestration and creates one `_Adapt` context for each compiled
function or proof probe. It checks all helper definitions and rejects recursion.

`context.py` owns the mutable state used by an adaptation. Its core methods
resolve registers, emit checked effects and assemble the function. Existing
binding, arithmetic, control and evolution functions are exposed as typed context
methods. Statement families receive the context explicitly; they retain no state
between functions. The dispatcher preserves structural precedence: callback and
scope forms are handled before token-based assignment, storage and gate calls.

| State | Owner and lifetime |
| --- | --- |
| Function definitions, top-level constants and source records | Parse orchestration supplies read-only mappings to each context |
| Classical values, arrays, matrices, integers and operator bindings | The current function or specialized helper binding scope |
| Parameters, widths, signs, fractional bits and inferred record layouts | The current function's source symbols |
| Registers, views, suspended names, dropped names and certified workspace | The current function's quantum storage and ownership scope |
| `entry`, `effect`, `counter`, `operations` and expression functions | The current function's ordered Compiler model construction |
| Helper stack, loop indices and expansion counters | Checked static expansion, including scope restoration and limits |

Context methods and handlers must restore temporary helper and binding scopes.
They emit owned Compiler records and keep original source locations. The Compiler
independently verifies operation attributes, initialization, aliases, effect order
and cleanup. Source proof helpers do not replace Compiler verification.

## Supported source families

Support is bounded by each function's validated profile and budgets. Combining
individually supported forms may still require a separate proof or be refused.
The following table locates the existing implementations and detailed profiles.

| Source family | Implementation | Detailed profile |
| --- | --- | --- |
| Scalars, static arrays, qstruct fields and views | `syntax_model.py`, `adapter_views.py`, `bindings.py` | [Records and arrays](PROFILE_HISTORY.md) |
| Classical values, integer loops and helper specialization | `integers.py`, `adapter_calls.py`, `quantum_callbacks.py` | [Classical binding profiles](PROFILE_HISTORY.md) |
| Arithmetic, in-place words and predicate phases | `adapter_arithmetic.py`, `adapter_assignments.py`, `predicates.py` | [Numeric and predicate profiles](PROFILE_HISTORY.md) |
| Allocate, prepare, drop and proven cleanup | `adapter_storage.py`, `preparation.py`, `local_cleanup.py` | [Storage and preparation profiles](PROFILE_HISTORY.md) |
| Coherent control, within/apply and inverse composition | `adapter_controls.py`, `scope_control.py`, `unitary_effects.py` | [Composition profiles](PROFILE_HISTORY.md) |
| Pauli operators and time evolution | `indexed_pauli.py`, `adapter_evolution.py`, `pauli_schedule.py` | [Evolution profiles](PROFILE_HISTORY.md#sparse-pauli-evolution) |
| Grover, amplification and phase estimation | `oracles.py`, `amplification.py`, `qpe.py` | [QPE profiles](PROFILE_HISTORY.md#bounded-ordinary-qpe-composition) |
| Block encodings, QSVT and checked fixed-point regions | `qsvt.py`, `fixed_regions.py`, `scoped_amplification.py` | [QSVT profiles](PROFILE_HISTORY.md#coherent-even-and-odd-qsvt) |
| Private measurement/reset and bounded classical optimization | `deferred_measurement.py`, `cscope.py` | [Execution profiles](PROFILE_HISTORY.md) |

`syntax_model.py` is the authority for lexical and parsing budgets.
`adapter_model.py` owns adapter constants and the private handoff. Compiler
construction budgets and Runtime memory ceilings are checked at their existing
boundaries. The source version remains the pinned Classiq commit
`2bfd835e73b6882fcc515a725098b04766c238e7`; declaring that context does not
authenticate source bytes.

External classical bindings and explicit Qmod precision overrides are currently
used by the internal corpus tooling. The public synthesis signature is protected
by the existing API contract.

## Making a change

1. Extend syntax in the parser only when the source form needs it.
2. Add source adaptation to the appropriate statement family or existing domain
   helper. Keep scope restoration, source diagnostics and bounded expansion there.
3. Change Compiler verification and lowering in Compiler when the owned operation
   semantics require it. Reuse existing gates and numerical implementations.
4. Run accepted and rejected scenarios, then compare complex output states,
   workspace and resources through the existing execution path.

For a typical local change, begin with these existing tests.

```sh
python -m pytest tests/team/ecosystem/test_qmod_parser.py tests/team/ecosystem/test_qmod_source.py -q
python -m pytest tests/team/compiler/test_construction_registers.py tests/integration/test_synthesis_python_program.py -q
```

Run the [native examples and independent Circuit comparisons](../../../examples/synthesis/README.md)
from the repository root. The [public synthesis guide](../../synth/README.md)
explains Program execution and named readout.

## Corpus and evidence

The byte-identical pinned corpus has 210 exact CPU execution smoke passes and
six source-semantic refusals. A separately approved numeric overlay gives a
216-input acceptance revision. The [source fixtures](../../../tests/fixtures/classiq_qmod)
retain original licenses and hashes; the [overlay instructions](../../../tests/fixtures/classiq_qmod/accepted_revision/README.md)
reproduce the separate acceptance corpus. Individual full-state references and
resource evidence remain distinct from the batch execution smoke results.

## Historical profiles

[PROFILE_HISTORY.md](PROFILE_HISTORY.md) retains the detailed incremental
profiles, restrictions, examples and validation notes. Early sections describe
the narrower scope at that stage. Later sections record the subsequent supported
profiles. Use this README for current module ownership and contributor navigation.

First/second-order Pauli product-formula scheduling is shared with the existing
algorithm Circuit builder through Compiler's private `_pauli_schedule.py`.
The adapter still owns source parsing, static operator data, coefficient/time
binding, repetition/resource bounds and commutation compaction. It emits the
same checked register effects and retains term order, identity phase and source
locations. No native evolution entry, external type leak or new IR is introduced.

Real phase polynomial expansion is owned by Compiler `_phase_polynomial.py`.
The local `phase.py` retains Qmod's syntax callbacks, diagnostics and compatibility
imports; it must not own a second expansion algorithm. Native Python supplies
a different syntax implementation to the same compiler boundary. Qmod signed,
fractional, predicate and factor-sharing behavior remains unchanged.
