# CS07 public source implementation candidate

Date: 2026-10-31. Development evidence, unreleased. Independent developer trials
remain 0/2; release authorization remains false. The user authorized implementation
following the merge of proposal #969 as c994934998eb3a6dbe983e008c8ad7fe8875bfca.
See [authorization and wording corrections](../api-changes/FQ-SYNTH-QMOD-IMPLEMENTATION-20261031.md).
Historical proposal bytes and API baselines are preserved.

## User journey and ownership

synth.synthesize accepts either the existing Python function with declared UInt
inputs, a local PathLike[str] .qmod file, or explicit text with source_format="qmod".
Python retains its preparation semantics. Native input supplies initialization
and names, refuses mixed Python options, and uses the existing Ecosystem adapter,
Compiler register profile and Services complete Program factory. Synthesis plans
once, then fq.run(program) executes the same bound plan. probabilities reads
named results with its existing identity and amplitude-budget checks. There is
no new Program wrapper, executor, schema, registry or root export.

The normal source workflow needs no physical numbers, gates or bit decoding.
Run python -m examples.algorithms.synthesis_source_trial, optionally followed by
a local source path. See [usage and exact bounds](../../flagquantum/synth/README.md).

## Source and memory contracts

Strings are source text only. PathLike wins over callable dispatch; fspath must
return str and the suffix must be .qmod. Local regular files are bounded at
262144 bytes, decoded strictly as UTF-8 and closed after reading; parsing does not
execute source or access the network. Diagnostics retain filename/line/column and
reuse InteropConversionError. Unknown version context fails before file I/O.
The default declared version is 2bfd835e73b6882fcc515a725098b04766c238e7; it is
not an authenticity claim. Original manifests retain separate byte provenance.

Native inputs default to a 512 MiB planned-state buffer ceiling; explicit positive
memory_limit_bytes applies to either profile. Python's prior default is retained.
A refused plan raises flagquantum.errors.PlanningError before execution. This
bounds the planned state buffer, not total process peak memory. Reader amplitude
limits are separate and unchanged.

Common reports retain source_resources for the entire construction and resources
for the actual transformed circuit, register_widths, source_locations, source_format
and effective memory limit/origin. Native reports retain source_name/source_version
and omit arithmetic_resources and the private output_widths alias. Python's
existing arithmetic_resources diagnostic is preserved.

## Unchanged-original acceptance

Three original sources, each executed through both public file and text forms:

| Original | Total qubits | Workspace | Source operations | Actual operations | Actual depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| X.qmod | 1 | 0 | 1 | 1 | 1 |
| synthesis_tutorial.qmod | 23 | 14 | 379 | 379 | 288 |
| multiplication_2vars_example.qmod | 24 | 12 | 279 | 275 | 214 |

All six cases passed complete complex-state reference comparison and clean
workspace checks at tolerance 1e-12. State buffers are respectively 32 bytes,
128 MiB and 256 MiB. The acceptance harness separately preflights up to four
state buffers within 1 GiB and explicitly authorizes the larger named read.
Original dimensions, phases, manifests, licenses and fixtures are unchanged.
Six input-form cases still represent three distinct originals, not six originals.

## Verification and remaining scope

Executable contracts cover public Python/text/file flow, coherent phase and
mutation, line routing and both schedules, early mixed-option refusal, memory
admission, preparation-only whole costs, named-port errors, source context,
file changes after synthesis and unchanged Python import behavior. Strict typing
checks valid forms and refuses invalid input combinations. The API guard pins
the historical proposal hash and rejects unauthorized signature, report, memory,
namespace, approval, serialization or release changes. Historical run/read
signatures and seven synth exports remain unchanged.

This admits one main, fresh unsigned scalar outputs and the verified bounded
arithmetic/preparation subset. It does not admit general arrays/views, Input/Const,
free/drop, the entire standard library or all 216 originals. Human trial records,
release review and any additional language/profile admission remain separate.

### Local verification record

- Public input/contracts/typing/source acquisition: 74 passed.
- Existing synthesis/compiler-synthesis/parser regressions: 614 passed,
  two optional dependency skips.
- Existing construction semantics, schedules, selection and execution: 385 passed.
- Unchanged-original public file/text acceptance: 6 passed in 129.49 seconds.
- Source-loader coverage run: 139 passed; 53 statements and 20 branches, 100%.
- Whole-package strict mypy: 876 source files passed.
- Current and minimum Black 24.1.0, Ruff, public API baseline, architecture,
  dependency, vocabulary, repository language, maturity, generated documentation,
  local documentation links and ownership classification passed.
- Both Python and native-source executable examples completed with named results.
