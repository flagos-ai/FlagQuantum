# Sharded versus unsharded MPS on CPU

Part of #899: representation agreement was extended to state and gradient
conformance by #942, but that grid compares numerical paths on one device. This
compares *placements* of the same path — the question a user with a sharded
deployment asks — and keeps the two things a sharded run can change apart.

Payloads `world-2.json`, `world-3.json` and `world-4.json` were produced by
`benchmarks/mps_placement_conformance.py` from source commit
`4f91e4bbc8762d928193c44e0e350e8321cb4727` with `gloo` on one host, complex128
amplitudes, three angles drawn from one seed and shared by every rank. The
launcher is described under Method.

## Method

Each world size runs two arms of the same program:

| Arm | `max_bond` | `gradient_policy` | What a disagreement would mean |
| --- | --- | --- | --- |
| exact | 64 | `exact` | placement only; the cap cannot discard weight |
| truncating | 2 | `approximate` | placement and truncation together |

The unsharded reference is `run_mps` on rank 0 with the same circuit, the same
angles and the same cap, and the sharded run is the rank-owned reverse executor.
`relative_l2_error` is `‖g − g_ref‖ / ‖g_ref‖`; `cosine_similarity` is the
direction agreement.

The launcher matters for reproduction. `torchrun --standalone` needs a TCP
rendezvous that resolves this host's own hostname, which it does not here, so the
local runs used a file-backed `gloo` store with one process per rank. CI uses
`--standalone`, which is what the pytest wrapper calls.

## Result

| World | Width | Cap | Truncated | Discarded weight | Value rel. error | Gradient rel. error | Cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 6 | 64 | false | 0.000e+00 | 0.000e+00 | 2.201e-16 | 1.000000000000000 |
| 2 | 6 | 2 | true | 3.457e-04 | 1.118e-16 | 3.653e-16 | 1.000000000000000 |
| 3 | 8 | 64 | false | 0.000e+00 | 1.117e-16 | 4.685e-16 | 1.000000000000000 |
| 3 | 8 | 2 | true | 6.349e-04 | 5.590e-16 | 5.981e-16 | 1.000000000000000 |
| 4 | 10 | 64 | false | 0.000e+00 | 5.587e-16 | 4.750e-16 | 1.000000000000000 |
| 4 | 10 | 2 | true | 9.601e-04 | 2.237e-16 | 3.836e-16 | 1.000000000000000 |

Moving the program from one device to 2, 3 or 4 ranks changes the value and the
gradient by at most `6e-16` relative — machine epsilon for complex128. That holds
in both regimes, so the placement error is below the measurement's resolution
and the discarded weight in the truncating arm is attributable to the cap rather
than to sharding.

The factorization split is visible in the payloads and is what distinguishes the
arms: the exact arm runs 14/20/26 QR splits and no SVD, the truncating arm runs
3/5/7 SVD splits beside 11/15/19 QR splits, and it saves an order of magnitude
less forward state (1568/928/928 bytes against 9632/5536/5536).

## What this does not establish

- **Not scalability evidence.** These are CPU semantics on one host with a
  synthetic communication path. The payload's own `claim_boundary` says so, and
  a test asserts that string is present. No capacity, speed, efficiency or
  accelerator claim follows, and the two-node certificate's scope is untouched.
- **Three world sizes on a small program.** Widths 6-10 with caps of 2 and 64.
  A cap that truncates harder, a nonlocal observable, or a width where the
  sharding cut falls on a different bond were not measured.
- The truncating arm authorizes the retained-subspace rule explicitly, because
  the default refuses a gradient through a truncated split (#943). This measures
  that rule's placement invariance; it does not certify the rule, and #941
  remains the record of its difference from the derivative of the forward map.
- No checkpoint recovery, no optimizer step, no multi-node transport, and no
  gradient comparison against an independent oracle — that is
  `test_representation_agreement_sweep.py`'s job on one device, and #902's for
  training across steps.
