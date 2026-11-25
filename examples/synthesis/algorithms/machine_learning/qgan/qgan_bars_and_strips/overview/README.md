# Bars-and-stripes GAN and quantum GAN

The classical workflow regenerates 1,000 two-by-two images and runs 10
epochs. The hybrid workflow regenerates 250 images and runs one epoch.
Both use the original batch size of 64, model widths and Adam/BCE recipe.

Download the two original checkpoints into a `resources` directory and open
[walkthrough.ipynb](walkthrough.ipynb). You can specify another directory
with the `FQ_QGAN_ASSETS` environment variable. The workflow checks the
pinned hashes and loads tensor weights on CPU.

- [Classical generator checkpoint](https://raw.githubusercontent.com/Classiq/classiq-library/b69c4893d8264d1e59ce0909df2bd88efed25e00/algorithms/QML/qgan/resources/generator_trained_model.pth)
- [Hybrid generator checkpoint](https://raw.githubusercontent.com/Classiq/classiq-library/b69c4893d8264d1e59ce0909df2bd88efed25e00/algorithms/QML/qgan/resources/q_generator_trained_model.pth)

Run the complete workflow from the repository root.

```bash
python -m examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.overview.main \
    --assets resources --output qgan-output
```

The hybrid layer binds each minibatch to a native four-qubit Circuit using
PyTorch tensors. It retains the source's eight weights, including the second
coupling on the first pair. Two complete complex states are checked against
both the native qfunc and an independent Circuit. The local runtime draws
4,096 bitstrings per input row and obtains the four commuting projector
estimates from the same bitstrings. The derivative of the exact expectation
is used in the autodiff path before the source's final rounding.

Both original generator architectures use `round` in the training forward
path. The local run measures zero generator gradients and unchanged
checkpoint parameters after the optimizer steps. The discriminators update.
The report preserves the source's diagonal heuristic alongside direct
bars-and-stripes membership and a distinct-pattern count. Perfect membership
can coexist with a generator that always emits the same image.
