# Installed compiler wheel controls

The initial two immutable integrated FlagQuantum revisions pass the unchanged
SDKBench Linux installed-wheel conformance on attempt 1. The separate follow-up
below adds the current integrated source without replacing the initial results.
This verifies packaging and selected native semantics; it is not a resource-qualified benchmark, prospective
study, SDK ranking, build attestation or package release.

## Sources, environment and acceptance

The [retained evidence index](evidence/compiler-installed-wheel-controls.json)
records SDKBench source `9cb8f184825e658f7378763a1f3a39c57720233d` and
[run 44982483623](https://github.com/sdkbench-org/sdkbench/actions/runs/44982483623).
Both source revisions were reachable from FlagQuantum main before dispatch:

| Role | Source revision | Selected tests | Failures / errors / skips |
| --- | --- | ---: | ---: |
| First | `11745ab11070765974b90f5a8f512bf4c30576b1` | 12 | 0 / 0 / 0 |
| Second | `dce3948b946906875a759362e8570b22a8214d5f` | 12 | 0 / 0 / 0 |

The second includes the fixed 4-by-4 product and fixed-singleton matrix changes
from PRs #1276/#1288. It predates the separately merged telemetry and identity
restoration changes; this run does not certify those later revisions. The
comparison is between complete source revisions, not an isolated-pass proof.

Both retained source archives build with the same Python 3.12.14 and CPU PyTorch
2.10 build environment and 900-second build budget. The wheels are
`flagquantum-0.4.0-cp310-abi3-linux_x86_64.whl`, installed through two separate
interpreter invocations with identical interpreter executable bytes and one
shared dependency installation. Runtime CPU PyTorch is 2.14.0; common locked
inputs include NumPy 2.5.5, Qiskit 2.6.2 and Cirq 1.7.0. Comparator packages are
dependencies here, not measured participants.

The existing SDKBench workflow validates installed bytes and both restored-layout
registrations, then verifies matching non-frontend inputs. Runtime steps clear
the source-tree PYTHONPATH override and use the installed SDKBench wheel. The
retained joint development record explicitly states no entropy draw, no campaign
execution and publication eligibility false. This is development admission,
not a study registration freeze or independent attestation of the environment.

Each role runs ordinary restored and target-artifact compilation for one frozen
six-wire fixture plus QAOA, Ising, hopping, QFT and repetition six-wire line
applications. The unchanged tests reconstruct independent arbitrary-input
operators, enforce the existing semantic tolerance, require restored identity
placement and check native compilation/audit behavior. Target-artifact checks
also require a nonempty serialized artifact and valid delivered CZ accounting.
All 12 selected tests pass for each wheel; 48 other parameterized workflows are
outside this explicit selection. No test was skipped within either selection.
These checks do not certify symbolic/gradient behavior, larger widths, other
topologies, every artifact tamper boundary or full compiler parity.

## Archive verification and retention

The original GitHub artifact is 465,412,797 bytes. Its complete SHA-256 is
`3b5505505c90978a26057a6416dfcf99be15581756a35fc730b5f8979b8a549a`,
matching GitHub's digest. The offline audit additionally verifies both source
ZIP commit comments, both wheel and build-log hashes, all 1,342 RECORD-hashed
payload files in each wheel, and exact source-archive/wheel bytes for both
optimization modules. It checks the JUnit counts and cases, successful build
status, matched development record and interpreter identities. The exact audit
driver and raw small-record hashes are retained in the index. The audit does not
re-execute a Linux binary on the development machine or turn a build into an
independently attested build.

The first artifact transfer exceeded its 180-second transfer budget. Its partial
bytes were retained and resumed once with an explicit larger transfer budget;
the completed original archive passed the full expected digest. Neither build,
workflow nor conformance test was rerun. An initial offline source-member lookup
assumed a top-level directory; the git archive instead has root-relative paths.
Correcting that lookup did not change or rerun any retained result.

GitHub artifact expiry is recorded in the index; its retention is finite. The
complete original archive remains retained locally, while this repository keeps
the hashes and selected audit conclusions, not the full dependency wheelhouse.
Local retention does not establish indefinite public availability. The hosted
four-logical-CPU environment is not a qualified dedicated measurement host.
No formal time/memory, accelerator or scalability claim follows, and no machine,
service, kernel or driver was restarted or changed.

## Existing workflow

In an environment with the declared native dependencies installed:

```sh
SDKBENCH_NATIVE_CANDIDATE_TESTS=1 python -m pytest   tests/test_candidate_native_adapters.py   -k 'flagquantum-ordinary-restored or flagquantum-target-artifact' -q
```

This is an SDKBench-checkout workflow using each installed FlagQuantum wheel.
Expected: 12 selected tests pass for each declared installation. It invokes no
remote or paid quantum job. Reproducing the full build requires the workflow's
retained sources, build/runtime locks, wheels and separate interpreters.

## Evidence-origin integration correction

The first PR CI run failed the checked-in provenance-vocabulary check because
the initial index embedded an `external_dependency` origin. That value is
supported by the revision declaration table but not by the separate inline
provenance reader. The final repository index moves this origin to the existing
`evidence-revision-origins.toml` table and records the original audit-index hash
and the normalization explicitly. Sources, raw artifacts, hashes, numerical
evidence and acceptance results are unchanged. No gate or supported-origin
vocabulary was altered, and no scientific workflow was rerun.

All 58 evidence-revision tests pass in an isolated no-tags checkout containing
the identical changed bytes. The first local full-file attempt also exposed an
omitted example-manifest sparse fixture and 16 historical-origin conflicts
caused by the local-only `tn-calibration-baseline-20260731` tag. Restoring the
tracked fixture and verifying GitHub has no matching tag explains the setup
difference; the isolated clone excludes local tags. The source tag and historical
declarations remain unchanged. The corrected inline-origin test also passes in
the original worktree. These are provenance checks, not performance reruns.

## Current integrated source follow-up

On 2026-11-23, unchanged SDKBench workflow source
`9cb8f184825e658f7378763a1f3a39c57720233d` completed
[run 45110130991](https://github.com/sdkbench-org/sdkbench/actions/runs/45110130991)
on attempt 1. The first role retains the previous second source as a control;
the second role adds the integrated source with the later telemetry,
identity-restoration and pipeline changes. This is complete-source conformance,
not attribution of an individual pass's effects.

| Role | Integrated source revision | Selected tests | Failures / errors / skips |
| --- | --- | ---: | ---: |
| Control | `dce3948b946906875a759362e8570b22a8214d5f` | 12 | 0 / 0 / 0 |
| Current | `f88ebc88f8584cd580802cd219ba16d935fa6715` | 12 | 0 / 0 / 0 |

The workflow, Python 3.12.14, declared build dependencies, shared runtime lock,
separate interpreter invocations, native adapter selection and independent
operator/layout acceptance are unchanged. Both wheels build successfully;
every selected test passes. The actual dependency inventory and hashed
wheelhouse remain in the raw artifact rather than being inferred from package
version names. These inspected six-wire line inputs remain development controls;
there was no entropy draw, new application campaign or formal resource study.

The full original artifact is 469,694,894 bytes, with SHA-256
`0b7ef353d45635b9cb2de14efcbb7cbf4842875a8a483091a0813ef8b13f48ca`.
It was downloaded once within a 1,200-second transfer budget and matches GitHub's
recorded digest. The offline audit verifies both archive commit comments, all
source/wheel/build-log hashes, all 1,342 control and 1,343 current wheel RECORD
payloads, and every packaged compiler Python source against its retained source
archive. It additionally checks the 12+12 JUnit results, matched non-frontend
inputs, distinct interpreter invocations and identical interpreter executable
bytes. The evidence index's `followup_controls` preserves every initial index
field and adds this run's records, raw small-record hashes and exact audit driver.
The external workflow origin uses the existing artifact declaration; the raw
audit's digest records that index normalization explicitly.

GitHub retention expires on 2027-02-21. Complete original bytes are retained
locally; finite artifact retention is not indefinite public availability.
Passing packaging and these semantic controls does not certify all compiler
behavior, larger widths/topologies, symbolic differentiation, unseen real-world
programs, qualified latency, aggregate peak memory, build attestation or a
package release. Later source revisions require a separate check or explicit
byte-identity evidence. This workflow did not upgrade a kernel or driver or
restart a machine or service.


Local repository validation of this evidence update selected 21,528 passing
smoke/unit tests and 796 skips, with one failure and two setup errors. The
failure reports the same 100 historical origin conflicts associated with local
refs; all 94 affected artifacts and the three provenance control files are
byte-identical to the previously diagnosed baseline. The two checklist-test
setups failed while copying the repository because the local disk ran out of
space. Neither outcome is reported as a passing default tier. The changed
installed-wheel index passes the authoritative revision/origin check directly;
no origin declaration, validator, contract or threshold was altered. The Linux
24-test installed-wheel result above remains a separate successful execution.
