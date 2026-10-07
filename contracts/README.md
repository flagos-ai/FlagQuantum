# Machine-readable contracts

This directory is the stable home for capability-specific and interoperability
contracts consumed by FlagQuantum tests and repository tools.

## What belongs here

Contracts in this directory must describe a compatibility surface that code,
tests, packaging, or an external implementation consumes. They fall into three
durable groups:

- public API and serialization baselines or candidates;
- hardware, numerical, and interoperability conformance contracts;
- repository architecture and migration guardrails enforced by tooling.

Review packets, approval commands, implementation authorizations, completion
records, and hashes whose only purpose is to authenticate another repository
file do not belong here. Git history retains those decisions; executable tests
must validate the resulting behavior directly.

## Contract families

| Family | Purpose |
| --- | --- |
| `*-interop-contract.toml` | Version lanes and semantic mappings for external frameworks. |
| `interop-capability-gap-matrix.toml` | Evidence-linked comparison of adapter coverage and prioritized gaps. |
| `cudaq-parity-matrix.toml` | CUDA-Q product capability baseline with a per-row FlagQuantum verdict, dependency class, and priority. |
| `qec-cudaq-alignment-checklist.toml` | Symbol-level CUDA-Q QEC alignment, one row per upstream surface item, with the evidence for each claim. |
| `split-real-imag-statevector-*-contract.toml` | Statevector representation, precision, device, and training acceptance boundaries. |
| `double-single-contract.toml` | Shared double-single arithmetic and conformance requirements. |
| `domestic-single-card-certification-contract.toml` | Domestic accelerator certification matrix and evidence requirements. |
| `circuit-composition-contract.toml` | The construction-time composition surface: what `Circuit.compose` and `Circuit.adjoint` guarantee, every way they refuse, and the operations of that family that do not exist yet. |
| `vn-entropy-output-contract.toml` | The von Neumann entropy output kind: the recorded entropies for pure and mixed programs, the complement identity that makes the selection a measurement, the execution modes that must agree on them, the named-base arithmetic, the subsystem-sized cost bound, and every refusal sentence. |
| `density-matrix-output-contract.toml` | The density-matrix output kind: the recorded matrices, the execution modes that must agree on them, the caller-ordered basis permutation, the statevector cost bound, and every refusal sentence. |
| `parameter-shift-coverage-contract.toml` | Which opcodes `batched_parameter_shift_gradient` can differentiate from one evaluation pair per parameter, measured against the opcode declaration for every registered opcode. |
| `primitives-admission-contract.toml` | The admission rule of `flagquantum/algorithms/primitives`: every export with the admission basis it was admitted on, the consumer that grounds it, its distribution semantics, and its differentiability shape. |
| `public-api-v0.2-baseline.json` | Pre-open-source exports, signatures, defaults, and dataclass fields used as the API convergence baseline. |
| `public-api-v1-candidate.json` | Proposed disposition of every baseline root export for the first public alpha. |
| `legacy-root-api-test-debt.json` | Zero baseline preventing legacy root API references from returning to tests. |
| `gradient-api-v1-candidate.json` | Authorized additive Stable Core contract for the `gradient` root export, its accepted method values, and the method it refuses. |
| `openqasm-import-v1-candidate.json` | Authorized additive Stable Core contract for the `from_openqasm` root export, its refusal vocabulary, and the accepted OpenQASM versions. |
| `execution-options-v1-candidate.json` | Proposed, not-yet-authorized Stable Core contract for `ExecutionOptions`. |

Validate the API migration baseline with
`python tools/public_api_snapshot.py`. It is not the final Stable Core contract
and must not be regenerated merely to make a check pass.

An authorized contract is read by a gate of its own, so a claim with no consumer
is a defect rather than decoration. `tools/check_openqasm_import_contract.py`
reads `openqasm-import-v1-candidate.json` against the shipped importer and
requires each of its declared rules to name an existing check or witness test.

The v1 candidate remains a proposal until API Change Proposal 001 is approved.
Candidate validation does not authorize changing the current public API.

Repository-wide policy files remain at the repository root because they are
entry points for CI and maintainers:

- `architecture.toml`
- `capability-maturity.toml`
- `dependency-policy.toml`

Contract filenames and schemas are compatibility surfaces. Move or rename a
contract only with all tool, test, and documentation references updated in the
same change. New capability contracts belong here, not at the repository root.

`double-single-contract.toml` is also shipped byte-for-byte as
`flagquantum/simulation/numerics/double-single-contract.toml` so installed conformance
checks do not depend on a repository checkout. Unit and distribution-artifact
checks reject drift or omission of that packaged mirror.

`circuit-composition-contract.toml` is the only contract that describes the
construction-time composition surface. The approval record is not restated here:
`docs/api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md` and
`docs/api-changes/FQ-CIRCUIT-ADJOINT-20261003.md` own the decision and the
authorization for the two methods, this contract owns their guaranteed behavior,
the refusal vocabulary that callers may rely on, and the fact that `control` and
`power` are unplanned rather than partial. Its conformance test in
`tests/unit/test_circuit_composition_contract.py` checks the contract against the
implementation, so a change to a refusal message or to an operation's presence in
that family is a contract change rather than a private detail.

`parameter-shift-coverage-contract.toml` records a measurement rather than a
decision: `batched_parameter_shift_gradient` sends one evaluation pair per input
parameter, so it can differentiate exactly the gate parameters whose declared
rule has two terms. Which opcodes those are is owned by
`flagquantum/core/operator_schema.py`, so the contract records what the
declaration says for every registered opcode and
`tools/check_parameter_shift_coverage_contract.py` re-derives it, censuses the
implementation for opcode names written in code, drives the profile for every
opcode, and builds every refusal row. The decision and the authorization are in
`docs/api-changes/FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md`; this contract
owns the measured coverage table. Its conformance test in
`tests/unit/test_parameter_shift_coverage_contract.py` mutates the contract's
clauses and requires the gate to name each one.

Three contracts describe CUDA-Q. `cudaq-export-contract.toml` owns the adapter
surface, `interop-capability-gap-matrix.toml` owns the adapter's per-format
coverage, and `cudaq-parity-matrix.toml` owns the product capability comparison.
They answer different questions and deliberately do not restate each other. The
version pin is shared: `cudaq-parity-matrix.toml` must declare the same
`cudaq_versions` as `cudaq-export-contract.toml`, and the gap matrix must keep
pointing its `cudaq` framework at that export contract, or
`python tools/parity_matrix.py --check` fails. That pin describes CUDA-Q core.
One parity domain is read against CUDA-Q QEC, which ships on its own release
line, so `cudaq-parity-matrix.toml` also records that line, the component version
its QEC rows were read at, and the limit that follows from the two lines being
disjoint: the core pin does not reach the QEC component, and a QEC row must not
be read as a statement about the pinned core versions.

`qec-cudaq-alignment-checklist.toml` zooms into that one QEC domain. It is not a
fourth opinion about the CUDA-Q surface: `cudaq-parity-matrix.toml` owns the
capability inventory, the priority, and the join to `capability-maturity.toml`,
and the checklist owns the symbol-level diff and the upstream attribution *within*
those rows. A checklist row naming a matrix row must agree with that row's status,
priority, and maturity entries, so the two cannot drift apart silently. Read it
through `docs/development/QEC_CUDAQ_ALIGNMENT.md`, which explains the rows and
must mention every one of them. It is checked by
`python tools/check_qec_cudaq_alignment.py`, run in CI's `quality` job, and the
check is itself mutation-tested by `tests/unit/test_qec_cudaq_alignment_check.py`.
