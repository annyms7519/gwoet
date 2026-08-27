# gnbt

`gnbt` is a Python package for **Discrete Naive Bayes (DNB)** and the
**Graphical WoE Transformer (GWoET)**.

GWoET transforms a dataset using dependencies among explanatory variables and
can be combined with downstream classifiers such as discrete naive Bayes,
logistic regression, and other machine-learning models.

> **Status:** Research software under active development. The public API may
> change before version 1.0.0.

## Package structure

```text
gwoet/
├── pyproject.toml
├── README.md
├── LICENSE
├── gwoet/
│   ├── __init__.py
│   ├── dnb.py
│   └── gwoet.py
├── tests/
└── examples/
```

## Installation

During development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/annyms7519/gwoet.git
cd gwoet
pip install -e .
```

After the repository is public, it can also be installed directly from GitHub:

```bash
pip install git+https://github.com/annyms7519/gwoet.git
```

The GitHub URL above is a placeholder until the repository is created.

## Requirements

The package depends on:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- seaborn
- NetworkX

Python 3.9 or later is currently specified in `pyproject.toml`.

## Discrete Naive Bayes

A basic DNB workflow is:

```python
from gwoet import discrete_naive_bayes, predict_dnb

model = discrete_naive_bayes(X_train, y_train)
pred = predict_dnb(model, X_test)
```

A scikit-learn-style estimator is also available:

```python
from gnbt import DiscreteNaiveBayes

model = DiscreteNaiveBayes()
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)
```

## Graphical WoE Transformer

The transformer can be used directly:

```python
from gwoet import GraphicalWoETransformer

tfm = GraphicalWoETransformer()
tfm.fit(X_train, y_train)

X_train_new = tfm.transform(X_train)
X_test_new = tfm.transform(X_test)
```

The transformed features can then be supplied to a downstream classifier.

## Main API

### Discrete Naive Bayes

- `DiscreteNaiveBayes`
- `discrete_naive_bayes`
- `is_discrete_naive_bayes`
- `predict_dnb`
- `generate_disc`
- `apply_disc`
- `get_accuracy_score`
- `get_iv`
- `plot_iv`
- `plot_woe`

### Graphical WoE Transformer

- `GraphicalWoETransformer`
- `graphical_woe_transformer`
- `is_graphical_woe_transformer`
- `transform_dataset`
- `gen_graph_data`
- `draw_graphical_model`
- `make_l1_coefficient_dict`
- `plot_l1_coefficients`
- `plot_l1_feature_coefficients`

## Reproducibility

For experiments used in papers, it is recommended to record the exact package
version or Git commit used to generate the results. Stable experimental
snapshots can be identified with Git tags or releases.

## Citation

A formal citation will be added when the corresponding paper or software
release is available.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

annyms7519
Email: annyms7519@gmail.com
