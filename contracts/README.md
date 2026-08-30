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

Repository-wide policy files remain at the repository root because they are
entry points for CI and maintainers:

- `architecture.toml`
- `capability-maturity.toml`
- `dependency-policy.toml`

Contract filenames and schemas are compatibility surfaces. Move or rename a
contract only with all tool, test, and documentation references updated in the
same change. New capability contracts belong here, not at the repository root.
