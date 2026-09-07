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
| `split-real-imag-statevector-*-contract.toml` | Statevector representation, precision, device, and training acceptance boundaries. |
| `double-single-contract.toml` | Shared double-single arithmetic and conformance requirements. |
| `domestic-single-card-certification-contract.toml` | Domestic accelerator certification matrix and evidence requirements. |
| `public-api-v0.2-baseline.json` | Pre-open-source exports, signatures, defaults, and dataclass fields used as the API convergence baseline. |
| `public-api-v1-candidate.json` | Proposed disposition of every baseline root export for the first public alpha. |
| `legacy-root-api-test-debt.json` | Zero baseline preventing legacy root API references from returning to tests. |
| `execution-options-v1-candidate.json` | Proposed, not-yet-authorized Stable Core contract for `ExecutionOptions`. |

Validate the API migration baseline with
`python tools/public_api_snapshot.py`. It is not the final Stable Core contract
and must not be regenerated merely to make a check pass.

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
