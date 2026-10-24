# Private spacetime predecoder (roadmap PR 3)

Based on main `288fd3f0b65712a9887193812a78654693d4dafa`, after #775 merged.
This change adds `_ai.spacetime.SpacetimePredecoder` behind the existing private
training, sampling and hybrid-decoding interfaces. Public exports, registry,
MLP defaults and checkpoint format are unchanged. It is a CPU offline research
model, not a streaming decoder or an accelerator performance claim.

## Geometry and model

Data wire `j*side+i` explicitly denotes square-grid coordinate `(i,j)`.
Check centers use doubled coordinates `(min(x)+max(x), min(y)+max(y))`, so
boundary checks and bulk checks share an integer grid without rounding.
Detector sites derive from their existing measurement references and check
ancillas. Their time is the latest syndrome round, or `rounds` for terminal
readout. Unresolvable ancillas, mixed checks, non-square data numbering and
colliding detector sites are refused. The row-major convention is supplied
geometry, not inferred physical-device coordinates or arbitrary-code support.

Inputs are `[batch, 4, rounds+1, 2*side-1, 2*side-1]`: syndrome, valid detector,
X-check and Z-check channels. Invalid sites carry no syndrome; masks distinguish
missing boundary detectors from measured zero. Two padded kernel-3 Conv3D/ReLU
layers produce eight local features. Features are gathered in original detector
order and mean-pooled over each original DEM mechanism's incident detectors.
A shared head combines those features, active-detector count, degree and prior.
It returns one logit per original mechanism, including boundary mechanisms.

Hc residual updates and Lc logical contributions remain in the existing hybrid
implementation. CNN inference never consumes latent fault or observable labels.
Dense DEM matrices and whole-history convolutions remain small-workload limits.
The model uses future context, so per-round amortized time is not feedback
latency. Existing refusal of invisible mechanisms/hyperedges still applies.

## Workflow and evidence

```bash
PYTHONPATH=. python examples/qec/ai_predecoder.py --model cnn \
  --steps 20 --shots 256 --checkpoint /tmp/cnn.pt
PYTHONPATH=. python examples/qec/ai_predecoder.py --model cnn --resume \
  --steps 20 --shots 256 --checkpoint /tmp/cnn.pt
PYTHONPATH=. python examples/qec/ai_predecoder.py --model cnn --evaluate-only \
  --shots 256 --checkpoint /tmp/cnn.pt
python -m pytest tests/qec/test_spacetime_predecoder.py \
  tests/qec/test_neural_predecoder.py tests/qec/test_neural_training_data.py -q
```

Omitting `--model` keeps the existing MLP. Select the same model family when
restoring; incompatible state dictionaries are rejected, not converted. CNN
geometry buffers are included in its state dictionary. This does not create a
portable deployment format or checkpoint compatibility guarantee.

Tests cover X/Z CSS memories at one/three rounds (using the explicit validated
logical-X construction from PR 2), valid/type masks and terminal timing,
mechanism-column ordering, nonzero Hc/Lc composition, gradients updating CNN
weights, exact Adam/RNG resume, and held-out evaluation. Native rotated-surface
records still declare Z only. There is no native X API promotion.

A CPU smoke run of two training steps and sixteen held-out samples produced
loss 0.566082 -> 0.558938, and one logical failure on each baseline/hybrid path.
This establishes execution only. PR 4 must compare CNN and MLP with predeclared
seeds/grids, uncertainty, ablations and matched timing before claiming benefit.
No threshold, cross-distance generalization, quantization or realtime evidence
is established by this PR. Full repository CI remains the merge gate.
