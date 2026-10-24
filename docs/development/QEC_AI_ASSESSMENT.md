# Prospective AI decoder study (roadmap PR 4)

Protocol date: 2026-10-24. Source baseline: main
`a5fc00898af53ab85037bec10c2db7579c68823b`, after #780 merged.
This protocol is fixed before running the full study. Smoke runs test software
only and may not support a scientific or performance claim. No existing public
API, decoder registry, deployment format or CI rule changes.

## Frozen first study

- CPU, Z-memory, independent data-X and check-measurement flips at stated p.
  This is phenomenological noise, not circuit depolarizing noise.
- Train at distance 3, rounds 3, p=0.02, 200 Adam steps, batch 128, LR=0.001.
- Three independent model seeds: 11, 22, 33. Training sampler seed = model seed
  +104729. Each variant trains separately from the same model seed.
- Variants: MLP, CNN, CNN with X/Z check-type input channels zeroed during both
  training and inference. The latter retains the validity mask and pooling.
- Validation at distance 3, rounds 3, p=0.006, 16384 shots; sampler seed = model
  seed +1000000. Select among thresholds 0.5, 0.9, 1.0 by lowest hybrid failure
  count; ties choose the larger threshold. No test data select the threshold.
- Test cells, in fixed order: (3,3,0.001), (3,3,0.003), (3,3,0.006),
  (5,5,0.003), (7,7,0.003), (5,15,0.003). Each tuple is distance/rounds/p.
  Each cell has 16384 shots. Seed = model seed +2000000 +10000*cell index.
- Test all variants on identically generated paired samples within each seed/cell.
  Transfer only learned head/encoder weights; retain target DEM/prior/layout.
- Primary endpoint: distance 5, rounds 5, p=0.003. Report every seed individually,
  not a pooled binomial interval that hides model-training variation.

Primary quality target is an upper confidence bound on hybrid/baseline LER <=0.9.
Bonferroni alpha=0.05/9 across three variants and three seeds at the primary
endpoint. Within each comparison split alpha across the two rate bounds before
taking their ratio. Distribution-free Hoeffding bounds are intentionally
conservative; zero/all-failure endpoints use exact binomial tail bounds. A zero
baseline lower bound gives an unbounded ratio, represented by JSON null.

A model-level primary claim would require all three seed-level primary bounds
to pass. The report does not automatically promote capabilities. Secondary
cells use unadjusted descriptive intervals and cannot be mined for a primary
success claim. Exact paired discordant-count p-values are additional diagnostics,
not permission to ignore the predeclared ratio bound or multiplicity.

This fixed sample budget may be insufficient at low LER. Record inconclusive,
not zero risk or success. A larger-budget follow-up must be declared separately;
do not repeatedly extend this run until a favorable result appears.

## Timing and evidence boundaries

On the primary cell, time 128 batch-one whole-block decodes after three warmups,
alternating baseline/hybrid order. Report nearest-rank P50/P95/P99 microseconds.
Pure baseline calls the global decoder directly, so no neural or Hc/Lc work is
charged to it. Hybrid includes neural input preparation, residual update,
conversion, global decode and logical composition. Both use the same PyMatching
configuration and process/CPU resources. Exclude data generation, construction
and training; record this scope with results. These timings are descriptive,
not statistically certified speedups or per-round feedback latency.

Use the optional PyMatching extra already supported by the project. No scipy or
new production dependency is required. The tool records config/hash/environment
and all raw failure/discordant counts. Training, validation and test seed domains
are disjoint; independent random streams need not yield disjoint physical errors.

```bash
PYTHONPATH=. python examples/qec/assess_predecoder.py --smoke --output /tmp/study-smoke.json
PYTHONPATH=. python examples/qec/assess_predecoder.py --output /tmp/study-full.json
python -m pytest tests/qec/test_predecoder_assessment.py -q
```

Outputs use exclusive creation; an existing record is never silently replaced.
The smoke mode uses one seed, two training steps, 32 shots/cell and eight timing
repetitions; its report is explicitly marked smoke-only. Do not compare its
quality or latency to a full study as evidence of improvement.

## Acceptance and remaining scope

Tests must detect incorrect multi-observable/paired counting, treating no observed
failures as no risk, accepting an imprecise point estimate, and accidentally
running the neural network on the pure baseline. Replay and timing scope checks
must pass, as must the complete vocabulary/format/type gates that caught earlier
PR omissions. Remote default/integration CI remains the merge gate.

This is the first Z-memory study and executable assessment foundation. X-memory,
more realistic noise, correlated/stronger decoder configurations, mask and
receptive-field ablations beyond check-type channels, model checkpoint archives,
training-seed uncertainty estimation and a powered performance study remain
follow-up scope. No accelerator migration is justified solely by a smoke result
or falling training loss. If benefit is not established, improve model/loss and
sampling precision before prioritizing PR 5 optimization or PR 7 hardware work.
