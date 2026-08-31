# Machine-readable contracts

This directory is the stable home for capability-specific and interoperability
contracts consumed by FlagQuantum tests and repository tools.

## Contract families

| Family | Purpose |
| --- | --- |
| `*-interop-contract.toml` | Version lanes and semantic mappings for external frameworks. |
| `split-real-imag-statevector-*-contract.toml` | Statevector representation, precision, device, and training acceptance boundaries. |
| `double-single-contract.toml` | Shared double-single arithmetic and conformance requirements. |
| `domestic-single-card-certification-contract.toml` | Domestic accelerator certification matrix and evidence requirements. |
| `public-api-v0.2-baseline.json` | Pre-open-source exports, signatures, defaults, and dataclass fields used as the API convergence baseline. |

Validate the API migration baseline with
`python tools/public_api_snapshot.py`. It is not the final Stable Core contract
and must not be regenerated merely to make a check pass.

Repository-wide policy files remain at the repository root because they are
entry points for CI and maintainers:

- `architecture.toml`
- `capability-maturity.toml`
- `dependency-policy.toml`

Contract filenames and schemas are compatibility surfaces. Move or rename a
contract only with all tool, test, and documentation references updated in the
same change. New capability contracts belong here, not at the repository root.

`double-single-contract.toml` is also shipped byte-for-byte as
`flagquantum/numerics/double-single-contract.toml` so installed conformance
checks do not depend on a repository checkout. Unit and distribution-artifact
checks reject drift or omission of that packaged mirror.
