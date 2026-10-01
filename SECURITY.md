# Security policy

## Reporting a vulnerability

Report privately to **flagquantum@gmail.com**. This is the address recorded in
`pyproject.toml`, and it is the only private channel this repository has today.

Do not open a public issue for a suspected vulnerability. If you have already opened
one, say so in your report and it will be handled from there.

Please include what you have:

- the affected version or commit (`flagquantum.version.__version__`, or a commit hash);
- what an attacker gains, and what they must already control to get it;
- the smallest reproduction you can produce, including the execution target and any
  remote provider involved;
- whether you intend to publish, and on what schedule.

You do not need a working exploit or a proof of concept to report.

### What to expect

There is **one maintainer** and no security response team. Reports are handled on a
best-effort basis. This policy deliberately states no acknowledgement or fix deadline,
because a deadline the project cannot meet is worse than none. You will get a direct
answer about whether the report is accepted, and the response will say plainly if the
project cannot address it in the time you need.

Credit is given in the fixing commit and release notes unless you ask otherwise.

### The private channel is currently the only one

GitHub's built-in private vulnerability reporting is **disabled** on this repository, so
the "Report a vulnerability" button does not appear. Enabling it is a maintainer settings
change and would be a second private channel; it is recorded below rather than assumed.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.2.x` | Yes |
| earlier | No |

Version `0.1.0` is the historical FlagOS 2.1 development snapshot, not a released line
(`docs/development/RELEASE_POLICY.md`). While the project is below `1.0`, fixes land on
the current minor line; there are no parallel maintained branches to backport to.

## Scope

In scope:

- **Credential handling.** Remote-provider adapters read credentials from the
  environment and from provider configuration. A path that transmits, logs, serializes,
  or embeds a credential where it should not is a vulnerability.
- **Deserialization.** Execution plans, checkpoints, training state, and FlagQuantum IR
  are serialized and reloaded. A payload that causes code execution, or that is accepted
  after failing a validation check, is a vulnerability.
- **Fail-open validation.** A guard that a caller can skip, or a validation that accepts
  a value it documents as rejected, is a vulnerability when the consequence is silent
  unsafe behavior. This project states fail-closed behavior as a design rule, so a
  divergence between the stated and actual contract is in scope.
- **Supply chain.** A compromised, typosquatted, or wrongly pinned dependency reachable
  through the package's dependency policy.

Out of scope:

- Vulnerabilities in PyTorch, Triton, CUDA, JAX, Qiskit, or a quantum-cloud provider's
  own service. Report those upstream; a report here about an adapter's *handling* of
  such a failure is in scope.
- Resource exhaustion from a program the user wrote and chose to run.
- Findings that require an attacker to already control the user's Python environment or
  local files.
- Anything that depends on a disabled platform feature as its only mitigation.

## Known limitations of this policy

This policy does not claim, and must not be read to claim:

- a bug bounty, a payment, or any compensation;
- a response-time guarantee, a fix deadline, or a coordinated-disclosure timeline;
- CVE issuance, an advisory process, or a security advisory list;
- a security review of the codebase, a threat model, or a claim that released code has
  been audited. None of those exists.

## Measured state of the platform features

Checked against the GitHub API on 2026-10-01. Recorded so the gap between the policy
above and the platform's capabilities is visible rather than implied.

| Feature | State |
| --- | --- |
| Private vulnerability reporting | disabled |
| Dependabot security updates | disabled |
| Secret scanning | disabled |
| Secret scanning push protection | disabled |
| Validity checks | disabled |

The only enforcement statement anywhere in the repository is
`dependency-policy.toml`, which sets `security_updates = "expedited"` under `[updates]`.
That is a policy for how quickly a dependency bump is merged, not a detection or
reporting mechanism.

## For maintainers

A new dependency's security posture is reviewed with its supply chain under
`docs/development/DEPENDENCY_POLICY.md`. A security fix is merged on the expedited
schedule in `dependency-policy.toml` rather than the monthly cadence, and it does not
wait for a documentation or capability update.

This file is a protected integration path (`team-ownership.toml`). Changing the reported
channel, the supported versions, or the scope above is an integration change.
