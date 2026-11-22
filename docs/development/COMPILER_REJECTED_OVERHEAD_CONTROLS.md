# Rejected compiler overhead controls

Two private overhead proposals were evaluated against integrated main
`4c8805d2882c4593e5ddfed95d03a7a353d02fd4` and rejected. Neither proposal
changes production code or the benchmark selector in this branch. This record
retains the negative evidence rather than presenting reduced operation counts
or isolated helper work as a demonstrated workflow improvement.

Both experiments use the unchanged clean SDKBench checkout
`9aa9260920984e2ca05bcb293abb885fccd9c144`, the previously inspected nonce,
one numerical thread and a separate 600-second process-group supervisor with
TERM/KILL cleanup. SDKBench stage timers, fixtures, acceptance tolerances and
budgets remain unchanged. The path is `single_device_fast_path`; neither study
is prospective or host-qualified. Temporary paths in the archived drivers must
be adjusted elsewhere.

## Deferred experimental preview realization

The original experimental native-CZ selector realizes its incumbent before
estimating alternatives. The private proposal first compares alternative Weyl
estimates with the incumbent's unreduced CZ count, while retaining the initial
native growth-budget check. It realizes the incumbent only if an alternative
can pass that preliminary filter, then retains the existing delivered CZ/depth
acceptance. This is a change to an already disabled experiment, not a production
routing optimization. The unchanged adapter budget of 20,000 native added
operations is retained and checked against its declared configuration.

Sixty paired target-artifact workflows give 120 independent acceptances, with
identical complete normalized output/layout hashes, quality metrics and adapter
audits. Forty six-wire executions use full operators up to phase; 80 larger
executions use three probes. That finite agreement does not establish universal
selector equivalence or unchanged failure behavior for arbitrary inputs.

Separate repeated target-artifact observations use two warmups and nine
alternating retained samples per variant. The timed region is complete adapter
execution; fixture generation, bundle loading and independent evaluation are
outside it. Separate Python allocation observations include the whole observation,
including setup and independent acceptance, and are not compiler-only peaks or
process RSS.

| Inspected case | Existing / proposed median | Whole-observation Python traced peak, existing / proposed |
| --- | ---: | ---: |
| 6-wire qft, grid | 202.421 / 206.001 ms | 35,364,633 / 35,364,447 bytes |
| 6-wire repetition, ring | 202.150 / 214.797 ms | 35,365,252 / 35,369,438 bytes |
| 12-wire qft, grid | 659.488 / 661.228 ms | 35,364,945 / 35,365,480 bytes |

[The exact archive](evidence/compiler-deferred-preview-rejected.json) retains the proposed source,
measurement driver, every paired acceptance, source pins, raw timing samples and
allocation observations. Both studies ran once, without case redraw or timing
retry. No local test suite overlapped the measurements. On this shared macOS
host, observed regressions are grounds to decline adoption, not proof of a
universal slowdown. There is no consistent demonstrated end-to-end benefit.

## Exact built-in angle fast path

The private proposal adds exact built-in float/int branches to the existing
`_polar_angle` reader before generic trainable-value and scalar extraction checks.
It leaves numeric subclasses and tensor objects on the existing path. No source
change was adopted, and no broader supported-input claim is made from these
numeric fixtures.

Ordinary restored and target-artifact workflows give 120 pairs and 240
independent acceptances, with identical complete normalized output/layout hashes,
quality metrics and adapter audits. Eighty six-wire executions use full operators
up to phase; 160 larger executions use three probes. This does not independently
certify symbolic, trainable or arbitrary subclass behavior for the proposal.

Separate repeated target-artifact observations use two warmups and nine
alternating retained samples per variant. The timed region is complete adapter
execution; fixture generation, bundle loading and independent evaluation are
outside it. Separate Python allocation observations include the whole observation,
including setup and independent acceptance, and are not compiler-only peaks or
process RSS.

| Inspected case | Existing / proposed median | Whole-observation Python traced peak, existing / proposed |
| --- | ---: | ---: |
| 6-wire qft, grid | 90.423 / 103.811 ms | 35,363,782 / 35,364,688 bytes |
| 6-wire repetition, ring | 116.715 / 116.946 ms | 35,363,967 / 35,364,921 bytes |
| 12-wire qft, grid | 561.775 / 558.735 ms | 35,364,605 / 35,365,897 bytes |

[The exact archive](evidence/compiler-primitive-angle-rejected.json) retains the proposed source,
measurement driver, every paired acceptance, source pins, raw timing samples and
allocation observations. Both studies ran once, without case redraw or timing
retry. No local test suite overlapped the measurements. On this shared macOS
host, observed regressions are grounds to decline adoption, not proof of a
universal slowdown. There is no consistent demonstrated end-to-end benefit.

## Decision and existing user workflow

Keep both implementations unchanged. These controls do not close the separate
native-CZ selector's production-resource admission gap, do not override earlier
prospective quality evidence, and do not support an SDK ranking, significance,
accelerator or scalability claim. Formal host-qualified SDKBench remains deferred;
no machine, service or kernel was restarted or changed.

Users retain the existing compiler entry point:

```python
import flagquantum as fq
from flagquantum.compiler import optimize

source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. This documentation-only branch
adds no implementation or public contract. It preserves the existing routing
replay, numerical proof and artifact acceptance requirements. Separate source
changes must demonstrate their own semantic and resource evidence before adoption.
