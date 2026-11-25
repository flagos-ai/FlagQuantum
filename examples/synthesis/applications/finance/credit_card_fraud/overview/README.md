# Credit-card fraud classification

Run the original 125 training, 60 test, and 60 prediction transactions through
three-feature PCA preprocessing, the native three-qubit Pauli kernel, and the
four classical SVC comparisons.

Save the [pinned 948-row example CSV](https://raw.githubusercontent.com/Classiq/classiq-library/b69c4893d8264d1e59ce0909df2bd88efed25e00/applications/finance/resources/creditcard.csv)
as `creditcard.csv` beside the Notebook, or pass its path to the CLI. The data
come from the [Machine Learning Group at ULB credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
collected and analyzed with Worldline. The publisher lists Open Database and
Database Contents terms.

The local example environment needs NumPy, Matplotlib, and scikit-learn.

```bash
python -m pip install scikit-learn
python -m examples.synthesis.applications.finance.credit_card_fraud.overview.main --data creditcard.csv
```

Open [the Notebook](walkthrough.ipynb) for both t-SNE projections, the complete
quantum overlap checks, the observed classifications, and the classical scores.
