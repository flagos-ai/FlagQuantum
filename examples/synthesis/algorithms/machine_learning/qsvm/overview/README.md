# Quantum support vector machines

Train a one-qubit Bloch kernel on three generated data sets of 40 points each.
Then train Bloch and two-layer Z/ZZ kernels on all 100 originally selected Iris
rows and compare them with a classical RBF SVC on the same splits.

The local Notebook environment needs NumPy, Matplotlib, and scikit-learn in
addition to FlagQuantum. Install scikit-learn into the environment used for this
example, then run:

```bash
python -m pip install scikit-learn
python -m examples.synthesis.algorithms.machine_learning.qsvm.overview.main
```

Open [the Notebook](walkthrough.ipynb) to inspect both quantum feature maps,
the kernel readout, every class prediction, and the generated plots.
