# Complete oblivious amplitude amplification workflow

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.overview.main
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.overview.comparisons.circuit
```

The [Notebook](walkthrough.ipynb) retains seed-one random input preparation,
both experiments, the good-block tables and success-probability calculation.
It reuses the existing [native qfunc composition](../main.py) and public named
probabilities. Users need no physical wires or bitstring decoding.
Exact local probabilities replace cloud sampling; no finite-shot or hardware
claim is made. Resource reports replace the cloud circuit viewer.

The first experiment puts one quarter of the probability in block zero.
One oblivious amplification iteration raises that probability to one. The
reflection acts on the block; the initial data preparation is excluded.
The four equal-weight Pauli terms encode one half of the unitary H tensor H.

The source's final assertion sums every row and checks normalization. This
migration explicitly corrects it to check the block-zero marginal, while
checking normalization separately. Independent Circuit comparison retains
absolute phase, data/block order and clean work. The robust near-unitary
extension remains explanatory background, not a tested general guarantee.

[Acceptance details](WORKFLOW.md). [Source provenance](provenance.json).
[Third-party license](../../../../CLASSIQ_LICENSE.txt).
