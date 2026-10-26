# QEC 0.4.0 retirement: mainline acceptance

Owner: QEC maintainers. Recorded on 2026-10-24 (Asia/Shanghai).
This is QEC-FUNCTIONAL-001 in the
[functional development plan](../roadmap/QEC_FUNCTIONAL_DEVELOPMENT_PLAN.md).
The plan's next deliverable is a contract proposal, not activation of new APIs.

## Delivered implementation and acceptance revisions

[PR #766](https://github.com/flagos-ai/FlagQuantum/pull/766) was merged by the
`FlagQuantum` account as `d286e4172cb9af4d1553a79199ee71f919af8c7a`.
The delivery agent did not perform that merge. The old
[structural acceptance record](QEC_RESTRUCTURE_ACCEPTANCE.md) describes #749's
pre-retirement compatibility boundary and remains historical evidence.

The installation/source acceptance revision is main
`85ed052b5cbab4871d7341e774a8a88f6a3fe86d`, fetched before this documentation
branch was created. Its QEC sources, QEC tests, retirement contract and QEC
examples are identical to the reviewed #766 head
`0ae3c722eaba4432c0de44d8a2df12ccec4776bc`.

Main advanced to `0d9cb06d1dd1b256f112b56092ed87e7d58e8ab1` before the
manual full-CI dispatches resolved. That intervening change touches only the
train/deploy example and its tests, not QEC. Record both revisions explicitly;
the full runs below are not described as executions at `85ed052b`.

## Required remote evidence

- [Main smoke at the installation revision](https://github.com/flagos-ai/FlagQuantum/actions/runs/40049520987).
- [Full main CI at 0d9cb06d](https://github.com/flagos-ai/FlagQuantum/actions/runs/40050300005).
- [Required single- and two-GPU main checks at 0d9cb06d](https://github.com/flagos-ai/FlagQuantum/actions/runs/40050303106).
- [Development container builds at 85ed052b](https://github.com/flagos-ai/FlagQuantum/actions/runs/40049520985).
- The documentation PR's own complete CI and required GPU checks at its reviewed
  head, linked by GitHub on that PR.

Run links are authoritative for current completion/conclusion. Starting a run
is not a passing result. Do not close this tranche or begin implementation
activation while any required check is failed, cancelled or incomplete. The PR
handoff records final conclusions and any pre-existing conditional skips.

## Source, installation and scientific boundaries

The accepted retirement keeps all 108 ordered root exports, one registered
`RotatedSurfaceCode`, the existing registries, surviving pickle identities and
numerical behavior. The 23 historical flat import and pickle paths are removed
under the approved 0.4.0 contract; no import hook or conversion unpickler is added.
The six top-level Python files include active shared implementations and the
root API facade. They are not unclassified historical modules.

All 15 original restructuring fixtures remain unchanged. Original JSON numerical
oracles and separately provenanced surviving-instance fixtures remain in place.
The retirement gate and 499 QEC alignment checks pass at this acceptance revision.

Source validation runs:

```bash
PYTHONPATH=. python tools/check_qec_retirement.py --phase retired
PYTHONPATH=. python tools/check_qec_cudaq_alignment.py
PYTHONPATH=. python -m pytest tests/api_contract/test_qec_retirement_contract.py tests/qec -q
```

The gate's expected output is:

```text
QEC retirement contract passed: retired, 23 modules, 108 exports
```

Rebuild the sdist and build the wheel from that sdist, using the declared
isolated build dependencies. Both artifacts contain exactly 50 QEC Python
members whose bytes match the acceptance checkout:

| Local macOS arm64 artifact | SHA-256 |
| --- | --- |
| `flagquantum-0.4.0.tar.gz` | `a4d2ac83418f6ed81b3690a7f1fae73226a5dcf2243c1098ee0da63ba4c92845` |
| `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl` | `300082dc584df93c24140bf7e9369aa25804a3aed4517912021afb898f01959f` |

Installation validation covers Python 3.10, 3.11 and 3.12 with core, Stim,
PyMatching and both optional extras: twelve profiles. Verify dependency
consistency, the native extension, a Bell-state smoke, the retired-path gate and
the six existing restructuring suites. Run outside the checkout with
`PYTHONPATH` unset and assert imports resolve to that environment's
`site-packages`. Run the existing user examples with exact stdout comparison;
the Stim example belongs only to an environment with the Stim extra.
This local platform evidence makes no Linux/Windows wheel or GPU performance
claim; remote platform checks and actual GPU jobs provide their own evidence.

## Runnable user example

With the built wheel installed, run the example by absolute path outside the
checkout, with `PYTHONPATH` unset. From a normal checkout the equivalent is:

```bash
python examples/qec/installed_imports.py
```

Expected output:

```text
Public exports: 108
Registered class: True
Code pickle path: flagquantum.qec.codes
Correction: (0,)
Logical error rate: 0.0
```

## Cleanup and continuation gate

After required remote checks and local installation/example checks pass, remove
only clean QEC temporary worktrees and branches proven merged into main. Preserve
original fixtures, retained identity metadata helpers, artifact hashes and test
evidence. Do not remove another task's active checkout or this unmerged review
branch. Keep unmerged work and report why it was retained.

Proceed next with QEC-FUNCTIONAL-002 as a separate proposal PR directly targeting
main. Implementation of a new protected public API waits for the API owner's
approval under [public API protection](PUBLIC_API_PROTECTION.md). No release,
tag, package publication, new feature activation, change to #731 or agent merge
operation is included in this acceptance tranche.
