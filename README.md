# gwoet

`gwoet` is a Python package for **Discrete Naive Bayes (DNB)** and the
**Graphical Weight-of-Evidence Transformer (GWoET)**.

GWoET constructs interaction features from dependencies among explanatory
variables. The dependency structure is estimated from Weight-of-Evidence (WoE)
representations using Graphical Lasso and EBIC, with optional edge thresholding
and percentage pruning. The transformed dataset can then be supplied to DNB,
logistic regression, or other downstream classifiers.

Predictive feature selection based on L1-penalized logistic regression is kept
separate from GWoET structural learning in `selection.py`.

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
│   ├── gwoet.py
│   └── selection.py
├── tests/
│   ├── test_dnb.py
│   ├── test_gwoet.py
│   └── test_selection.py
└── examples/
```

## Installation

During development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/<username>/gwoet.git
cd gwoet
pip install -e .
```

It can also be installed directly from GitHub:

```bash
pip install git+https://github.com/<username>/gwoet.git
```

Replace `<username>` with the GitHub account containing the repository.

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
from gwoet import DiscreteNaiveBayes

model = DiscreteNaiveBayes()
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)
```

## Graphical WoE Transformer

GWoET can be used through the scikit-learn-compatible transformer:

```python
from gwoet import GraphicalWoETransformer

tfm = GraphicalWoETransformer(
    gamma=1.0,
    ebic_threshold=False,
    pruning_ratio=0.0,
    random_state=13,
)

tfm.fit(X_train, y_train)

X_train_new = tfm.transform(X_train)
X_test_new = tfm.transform(X_test)
```

The transformed features can then be supplied to a downstream classifier.

The functional API is also available:

```python
from gwoet import graphical_woe_transformer, transform_dataset

gwt = graphical_woe_transformer(
    X_train,
    y_train,
    gamma=1.0,
    ebic_threshold=True,
    pruning_ratio=0.2,
    random_state=13,
)

X_train_new = transform_dataset(gwt, X_train)
X_test_new = transform_dataset(gwt, X_test)
```

### Structure estimation

The current GWoET implementation uses the following sequence:

1. Discretize variables when required.
2. Calculate training WoE values, optionally using cross-fitting.
3. Construct a similarity matrix from WoE vectors.
4. Estimate a sparse precision matrix using Graphical Lasso.
5. Select the Graphical Lasso regularization strength using EBIC.
6. Optionally apply `ebic_threshold` to weak precision-matrix entries during
   EBIC evaluation.
7. Convert the selected precision matrix to partial correlations.
8. Optionally remove a fraction of the weakest remaining edges using
   `pruning_ratio`.
9. Construct interaction variables from the retained edges.
10. Apply IV-based feature pruning.

L1 logistic feature selection is intentionally **not** part of this structural
learning procedure. It is provided separately in `selection.py`.

### Main GWoET parameters

#### `gamma`

`gamma` is the high-dimensional penalty parameter in EBIC. Larger values
generally favor sparser Graphical Lasso solutions.

```python
tfm = GraphicalWoETransformer(gamma=1.0)
```

#### `ebic_threshold`

When `ebic_threshold=True`, weak off-diagonal precision-matrix entries are
thresholded before EBIC evaluation using the implemented threshold

```text
log(p * (p - 1) / 2) / sqrt(n)
```

where `p` is the number of graph nodes and `n` is the sample size.

```python
tfm = GraphicalWoETransformer(ebic_threshold=True)
```

#### `pruning_ratio`

`pruning_ratio` removes a fraction of the weakest nonzero partial-correlation
edges after EBIC estimation and optional EBIC thresholding.

```python
tfm = GraphicalWoETransformer(pruning_ratio=0.2)
```

For example, `pruning_ratio=0.2` removes approximately the weakest 20% of the
remaining edges. The valid range is `[0, 1)`. The default value `0.0` performs
no percentage pruning.

`pruning_ratio` can be used independently of `ebic_threshold`:

```python
# EBIC + percentage pruning
tfm = GraphicalWoETransformer(
    ebic_threshold=False,
    pruning_ratio=0.2,
)

# EBIC threshold + percentage pruning
tfm = GraphicalWoETransformer(
    ebic_threshold=True,
    pruning_ratio=0.2,
)
```

### Inspecting the fitted structure

The fitted transformer can be inspected directly:

```python
gwt = tfm.get_transformer()
pcor = tfm.get_pcor()
```

The transformer dictionary contains:

```text
disc     discretization information
pcor     full EBIC-estimated partial-correlation matrix
numeric  numerical-variable handling flag
nodes    retained node variables
edges    retained interaction edges
```

`pcor` stores the full EBIC-estimated partial-correlation matrix before
percentage pruning and IV pruning. Therefore, the final `nodes` and `edges`
may represent a smaller structure than the nonzero entries in `pcor`.

### Graph visualization

The fitted graphical model can be converted to a NetworkX graph or drawn
directly:

```python
G = tfm.gen_graph_data()
G, args = tfm.draw()
```

The functional equivalents are:

```python
from gwoet import gen_graph_data, draw_graphical_model

G = gen_graph_data(gwt)
G, args = draw_graphical_model(gwt)
```

## Predictive feature selection

Predictive feature selection and logistic-regression coefficient diagnostics
are implemented separately in `selection.py`. This separation keeps the GWoET
graph structure distinct from downstream predictive feature selection.

### L1 logistic feature selection

`select_variables_l1_logistic` accepts the original feature matrix `X` and the
binary target `y`. It internally discretizes the input features, calculates
Weight-of-Evidence (WoE) representations, and applies L1-penalized logistic
regression. Candidate values of `C` are compared by BIC, calculated from the
unpenalized log-likelihood of each fitted model.

Cross-fitted WoE values are used by default during feature selection.

```python
from gwoet import select_variables_l1_logistic

result = select_variables_l1_logistic(
    X_train,
    y_train,
    Cs=20,
    random_state=13,
)
```

The returned dictionary contains:

```text
model               fitted LogisticRegression model on WoE features
coef                coefficients for all WoE features
selected_variables  original variable names selected by nonzero coefficients
selected_coef       nonzero coefficients
best_C              C selected by BIC
best_bic             minimum BIC
bic_path             results for all candidate C values
n_iter               solver iteration count
```

This selection result should be interpreted as **predictive feature selection**,
not as an estimate of the GWoET dependency structure.

### Scikit-learn-compatible L1 feature selector

`L1LogisticSelector` provides the same WoE + L1/BIC selection logic as a
scikit-learn-compatible transformer. During `fit`, it estimates the selected
variables from the training data. During `transform`, it returns the selected
columns from the **original input `X`**, not the WoE values.

This makes it suitable as a filtering step in a `Pipeline` and ensures that,
when used with cross-validation, feature selection is re-estimated separately
inside each training fold.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from gwoet import L1LogisticSelector

pipeline = Pipeline([
    (
        "selection",
        L1LogisticSelector(
            Cs=20,
            random_state=13,
        ),
    ),
    (
        "model",
        LogisticRegression(max_iter=1000),
    ),
])

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
)
```

If the selected original features contain categorical variables, place an
appropriate downstream preprocessing step, such as a `ColumnTransformer`,
between `L1LogisticSelector` and the prediction model.

After fitting, the selector also provides standard feature-selection helpers:

```python
selector = L1LogisticSelector(Cs=20, random_state=13)
selector.fit(X_train, y_train)

selected = selector.get_feature_names_out()
support = selector.get_support()
X_train_selected = selector.transform(X_train)
```

### Logistic coefficient visualization

The package also provides utilities for organizing and plotting fitted logistic
regression coefficients:

- `make_logistic_coefficient_dict`
- `plot_logistic_coefficients`
- `plot_logistic_feature_coefficients`

These functions are diagnostics for downstream logistic models and are not used
to determine the GWoET graph.

## Main API

### Discrete Naive Bayes

- `DiscreteNaiveBayes`
- `discrete_naive_bayes`
- `is_discrete_naive_bayes`
- `predict_dnb`
- `make_y_bool`
- `make_sample_weight`
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

### Logistic feature selection and diagnostics

- `select_variables_l1_logistic`
- `L1LogisticSelector`
- `make_logistic_coefficient_dict`
- `plot_logistic_coefficients`
- `plot_logistic_feature_coefficients`

## Testing

From the repository root:

```bash
python -m pytest -q
```

The tests are separated according to module responsibility:

```text
test_dnb.py        DNB and shared WoE/discretization functionality
test_gwoet.py      GWoET structure estimation and transformation
test_selection.py  L1 selection, Pipeline/CV integration, and coefficient diagnostics
```

## Reproducibility

For experiments used in papers, record the exact package version or Git commit
used to generate the results. Stable experimental snapshots can be identified
with Git tags or releases.

Because GWoET can use cross-fitting and downstream procedures may involve
randomness, set `random_state` explicitly when reproducibility is required.

## Citation

A formal citation will be added when the corresponding paper or software
release is available.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

annyms7519
Email: annyms7519@gmail.com
