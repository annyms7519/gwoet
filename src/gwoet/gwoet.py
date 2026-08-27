"""
gwoet.py

This module provides functions and a class for Graphical Weight-of-Evidence
Transformer (GWoET).


Functions
---------
graphical_woe_transformer:
  Generate a graphical WoE transformer from training data.

is_graphical_woe_transformer:
  Check whether a model is a graphical WoE transformer.

transform_dataset:
  Transform a dataset using a fitted graphical WoE transformer.

gen_graph_data:
  Generate a NetworkX Graph from a fitted graphical naive Bayes transformer.

draw_graphical_model:
  Draw an undirected graphical model (Markov random field).

make_l1_coefficient_dict:
  Extract L1 logistic regression coefficients grouped by original variable.

plot_l1_coefficients:
  Plot L1 logistic regression coefficients by original variable.

plot_l1_feature_coefficients:
  Plot L1 logistic regression coefficients by transformed feature.

Example
-------
gwt = graphical_woe_transformer(X_train, y_train)

is_gwt = is_graphical_woe_transformer(gwt)

G = gen_graph_data(gwt)
G, args = draw_graphical_model(gwt)

X_train2 = transform_dataset(gwt, X_train)
X_test2 = transform_dataset(gwt, X_test)

dnb = discrete_naive_bayes(X_train2, y_train)
pred = predict_dnb(dnb, X_test2)

Class
-----
GraphicalWoETransformer:
  Scikit-learn-compatible transformer for graphical WoE transformation.

GraphicalWoETransformer methods
-------------------------------
fit:
  Fit the transformer to the training data.

transform:
  Transform a dataset using the fitted transformer.

gen_graph_data:
  Generate a NetworkX Graph from the fitted transformer.

draw:
  Draw an undirected graphical model.

get_transformer:
  Return the fitted graphical WoE transformer as a dictionary.

get_pcor:
  Return the partial correlation matrix as a DataFrame.

Example
-------
tfm = GraphicalWoETransformer()
tfm.fit(X_train, y_train)

G = tfm.gen_graph_data()
G, args = tfm.draw()

X_train2 = tfm.transform(X_train)
X_test2 = tfm.transform(X_test)

gwoet = tfm.get_transformer()
pcor = tfm.get_pcor()


Author: annyms7519
Created: 2024-07-05
Last modified: 2026-08-27
Version: 0.9.0
License: MIT
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.covariance import graphical_lasso
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_consistent_length

from .dnb import (
    apply_disc,
    calculate_cross_fitted_woe,
    calculate_woe,
    generate_disc,
    make_y_bool,
    make_sample_weight,
)

MIN_IV = 0.02   # Minimum IV used for variable pruning.
MAX_DEPTH = 10  # Maximum decision tree depth when target_degree="adaptive".

def graphical_woe_transformer(
    X,
    y,
    weight=None,
    discretizer='bic',
    cross_fitting=True,
    n_splits=5,
    shuffle=True,
    l1_selection=False,
    target_degree='adaptive',
    min_iv=MIN_IV,
    numeric=True,
    random_state=None,
    verbose=False,
):
    """
    Generate a graphical WoE transformer from training data.

    Args:
      X:
        DataFrame containing numerical or categorical variables.
      y:
        One-dimensional array-like target variable with the same number
        of observations as X.
      weight:
        None or one-dimensional array-like sample weights with the same
        number of observations as X. If None, all sample weights are
        treated as 1.0.
      discretizer:
        Discretization method. Must be "bic" or "mdlp".
      cross_fitting:
        Whether to use cross-fitted Weight of Evidence (WoE) values for
        the training data. If True, each observation is transformed using
        WoE values estimated without using the fold containing that
        observation.
      n_splits:
        Number of folds used for cross-fitting. Must be at least 2.
        This argument is used only when cross_fitting is True.
      shuffle:
        Whether to shuffle observations before constructing the
        cross-fitting folds. This argument is used only when
        cross_fitting is True.
      l1_selection:
        Whether to select variables using L1-penalized logistic regression.
      target_degree:
        Target average degree of the graph. If "adaptive", the target
        degree is determined automatically. If None, the full graph is used.
      min_iv:
        Minimum information value (IV) used for variable pruning.
        If less than or equal to 0, IV-based pruning is not performed.
      numeric:
        Whether to keep numerical variables as numerical variables.
        If False, numerical variables are discretized.
      random_state:
        Random seed, RandomState instance, or None.
      verbose:
        Whether to print detailed information.

    Returns:
      Dict containing the following keys:
        disc:
          Dictionary of discretizers whose keys are variable names.
        pcor:
          DataFrame representing the partial correlation matrix.
        numeric:
          Whether numerical variables are kept as numerical variables.
        nodes:
          List of original variables retained after variable pruning.
        edges:
          Dictionary of interaction terms and their component variable
          names retained after variable pruning.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if len(y) != len(X):
        raise ValueError("y must have the same length as X.")

    if discretizer not in ("bic", "mdlp"):
        raise ValueError("discretizer must be 'bic' or 'mdlp'.")

    if not isinstance(cross_fitting, bool):
        raise TypeError("cross_fitting must be a boolean.")

    if not isinstance(n_splits, int) or isinstance(n_splits, bool):
        raise TypeError("n_splits must be an integer.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")

    if not isinstance(l1_selection, bool):
        raise TypeError("l1_selection must be a boolean.")

    if not (
        target_degree == "adaptive"
        or (
            isinstance(target_degree, int)
            and not isinstance(target_degree, bool)
            and target_degree > 0
        )
        or target_degree is None
    ):
        raise ValueError(
            "target_degree must be 'adaptive', a positive integer, or None."
        )

    if not isinstance(min_iv, (int, float)) or isinstance(min_iv, bool):
        raise TypeError("min_iv must be numeric.")

    if not isinstance(numeric, bool):
        raise TypeError("numeric must be a boolean.")

    if (
        random_state is not None
        and not isinstance(
            random_state,
            (int, np.random.RandomState),
        )
    ):
        raise TypeError(
            "random_state must be an integer, RandomState instance, or None."
        )

    if (
        isinstance(random_state, int)
        and not isinstance(random_state, bool)
        and random_state < 0
    ):
        raise ValueError("random_state must be non-negative.")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")

    weight = make_sample_weight(weight, len(X))

    disc = generate_disc(
        X,
        y,
        discretizer=discretizer,
        weight=weight,
    )

    X_d = apply_disc(
        disc,
        X,
    )

    if cross_fitting:
        woe_result = calculate_cross_fitted_woe(
            X_d,
            y,
            weight,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        woe_result = calculate_woe(
            X_d,
            y,
            weight,
        )

    woe_table = woe_result['woe_table']
    y_bool = woe_result['y_bool']

    # Romove variables of zero norm.
    norm = np.linalg.norm(woe_table, axis=0)
    woe_table = woe_table[(woe_table.columns)[norm > 0]]
    X_d = X_d[woe_table.columns]

    if len(woe_table.columns) == 0:
        print("All variables were pruned. The original features are used instead.")

        nc = len(X.columns)
        pcor = pd.DataFrame(
            np.zeros((nc, nc)),
            columns=X.columns,
        )

        transformer = {
            'disc': disc,
            'pcor': pcor,
            'numeric': numeric,
            'degree': 0,
            'nodes': list(X.columns),
            'edges': {},
        }
        return transformer

    if l1_selection:
        l1_result = _select_variables_l1_logistic(
            X_woe=woe_table,
            y=y,
            weight=weight,
            random_state=random_state,
        )
        nodes = l1_result["selected_variables"]
        coef = l1_result["selected_coef"]
        best_C = l1_result["best_C"]
        if verbose:
            print(f"best_C={best_C:.3f}")
            print(f"Nodes: {len(nodes)}")

    else:
        nodes = list(woe_table.columns)
        if verbose:
            print(f"Nodes: {len(nodes)}")

    random_state = check_random_state(random_state)

    if target_degree != 'adaptive':
        td = target_degree

    else:
        td = MAX_DEPTH

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )

        dt0 = _decision_tree(
            max_depth=1,
            random_state=random_state,
        )

        auc_dt0 = cross_val_score(
            dt0,
            X,
            y,
            cv=cv,
            scoring='roc_auc'
        ).mean()

        for i in range(2, MAX_DEPTH + 1):
            dt1 = _decision_tree(
                max_depth=i,
                random_state=random_state,
            )

            auc_dt1 = cross_val_score(
                dt1,
                X,
                y,
                cv=cv,
                scoring='roc_auc'
            ).mean()

            if auc_dt1 - auc_dt0 < 0:
                td = i - 1
                break

            else:
                auc_dt0 = auc_dt1

        if verbose:
            print(f"target_degree = 'adaptive' -> {td}")

    mat = _gen_cosine_similarity_matrix(woe_table)

    pcor = _calc_partial_correlations(
        mat,
        X.shape[0],
        td,
        verbose,
    )

    pcor = pd.DataFrame(
        pcor,
        columns=woe_table.columns,
    )

    edges = _get_interaction_terms(
        X_d,
        y,
        weight,
        pcor,
    )

    if min_iv > 0:
        unnecessary_vars = _filter_variables_with_min_iv(
            X_d,
            y,
            weight,
            nodes,
            edges,
            min_iv,
            cross_fitting,
            n_splits,
            shuffle,
            random_state,
        )

        # Remove unnecessary nodes.
        nodes = [node for node in nodes if node not in unnecessary_vars]
        if verbose:
            print(f"Nodes after pruning: {len(nodes)}")

        # Remove unnecessary edges.
        edges = {
            edge_name: edge_nodes
            for edge_name, edge_nodes in edges.items()
            if edge_name not in unnecessary_vars
        }
        if verbose:
            print(f"Edges after pruning: {len(edges)}")

    if len(nodes) == 0 and len(edges) == 0:
        print("All variables were pruned. The original features are used instead.")

        nc = len(X.columns)
        pcor = pd.DataFrame(
            np.zeros((nc, nc)),
            columns=X.columns,
        )

        transformer = {
            'disc': disc,
            'pcor': pcor,
            'numeric': numeric,
            'degree': 0,
            'nodes': list(X.columns),
            'edges': {},
        }
    else:
        transformer = {
            'disc': disc,
            'pcor': pcor,
            'numeric': numeric,
            'degree': td,
            'nodes': nodes,
            'edges': edges,
        }

    return transformer

def _decision_tree(
    max_depth,
    random_state,
):
    """
    Decision tree implementation for graphical_woe_transformer().
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude=("object", "category"))),
            ("cat", categorical_transformer, selector(dtype_include=("object", "category"))),
        ]
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("transformer", preprocessor),
            ("estimator", model),
        ]
    )

    return pipeline

def is_graphical_woe_transformer(
    model,
):
    """
    Check whether a model is a graphical WoE transformer.
    """
    return (
        isinstance(model, dict)
        and {'disc', 'pcor', 'numeric', 'degree', 'nodes', 'edges'} <= model.keys()
    )

def transform_dataset(
    transformer,
    X,
):
    """
    Transform a dataset using a fitted graphical WoE transformer.

    The original row index of X is preserved during the transformation.

    Args:
      transformer:
        Fitted graphical WoE transformer.
      X:
        DataFrame containing numerical or categorical variables.

    Returns:
      Transformed DataFrame with the same index as X.
    """
    if not is_graphical_woe_transformer(transformer):
        raise ValueError(
            "transformer must be a fitted graphical WoE transformer."
        )

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    disc = transformer["disc"]
    nodes = transformer["nodes"]
    edges = transformer["edges"]

    X_d = apply_disc(disc, X)

    assert X_d.index.equals(X.index), \
        "apply_disc must preserve the index of X."

    if transformer["numeric"]:
        X_new = X[nodes].copy()

        for inter_name, edge_nodes in edges.items():
            name1, name2 = edge_nodes

            if (
                X[name1].dtype.kind in "iufc"
                and X[name2].dtype.kind in "iufc"
            ):
                X_new[inter_name] = (
                    X[name1] * X[name2]
                ).astype(float)

            else:
                X_new[inter_name] = _make_categorical_interaction(
                    X_d[name1],
                    X_d[name2],
                )

    else:
        X_new = X_d[nodes].copy()

        for inter_name, edge_nodes in edges.items():
            name1, name2 = edge_nodes

            X_new[inter_name] = _make_categorical_interaction(
                X_d[name1],
                X_d[name2],
            )

    return X_new

def _make_categorical_interaction(
    col1,
    col2,
):
    """
    Create a categorical interaction Series.

    Missing values are represented by np.nan.
    The original index is preserved.
    """
    assert col1.index.equals(col2.index), (
        "col1 and col2 must have the same index."
    )

    valid = col1.notna() & col2.notna()

    result = pd.Series(
        np.nan,
        index=col1.index,
        dtype=object,
    )

    result.loc[valid] = (
        col1.loc[valid].astype(str)
        + "*"
        + col2.loc[valid].astype(str)
    )

    return result

def _select_variables_l1_logistic(
    X_woe,
    y,
    weight=None,
    Cs=None,
    class_weight=None,
    max_iter=1000,
    optimization_tol=1e-3,
    tol_coef=1e-8,
    random_state=None,
):
    """
    Select variables using L1-penalized logistic regression and BIC.

    For each candidate C, an L1-penalized logistic regression model is fitted.
    BIC is calculated from the unpenalized log-likelihood of the fitted model,
    and the C with the minimum BIC is selected.

    Args:
      X_woe:
        DataFrame containing WoE features.
      y:
        Binary target variable.
      weight:
        Optional one-dimensional array-like of non-negative sample weights.
      Cs:
        Candidate inverse regularization strengths.
        If None, five values logarithmically spaced from 1e-3 to 1e1
        are used. If an integer, that number of logarithmically spaced
        values over the same range is used.
      class_weight:
        Class weights passed to LogisticRegression.
      max_iter:
        Maximum number of solver iterations.
      optimization_tol:
        Stopping tolerance used by the optimizer.
      tol_coef:
        Coefficient threshold used for variable selection and effective
        degrees of freedom.
      random_state:
        Random state.

    Returns:
      Dict containing the fitted model, coefficients, selected variables,
      selected coefficients, selected C, BIC, and the BIC path.
    """
    if not isinstance(X_woe, pd.DataFrame):
        raise TypeError("X_woe must be a pandas DataFrame.")

    if X_woe.shape[1] == 0:
        raise ValueError("X_woe must contain at least one variable.")

    if not X_woe.columns.is_unique:
        raise ValueError("X_woe must have unique column names.")

    if not isinstance(tol_coef, (int, float)) or isinstance(tol_coef, bool):
        raise TypeError("tol_coef must be numeric.")

    if tol_coef < 0:
        raise ValueError("tol_coef must be non-negative.")

    if (
        not isinstance(optimization_tol, (int, float))
        or isinstance(optimization_tol, bool)
    ):
        raise TypeError("optimization_tol must be numeric.")

    if optimization_tol <= 0:
        raise ValueError("optimization_tol must be positive.")

    check_consistent_length(X_woe, y)

    if weight is not None:
        check_consistent_length(X_woe, weight)

        weight = np.asarray(weight, dtype=np.float64)

        assert weight.ndim == 1, (
            "weight must be one-dimensional."
        )
        assert np.all(np.isfinite(weight)), (
            "weight must contain only finite values."
        )
        assert np.all(weight >= 0), (
            "weight must contain only non-negative values."
        )
        assert np.any(weight > 0), (
            "At least one sample weight must be positive."
        )

    # C-contiguous float64 array avoids repeated input conversions.
    X_values = np.asarray(
        X_woe.to_numpy(dtype=np.float64, copy=False),
        order="C",
    )

    assert np.all(np.isfinite(X_values)), (
        "X_woe must contain only finite numeric values."
    )

    y_bool = np.asarray(
        make_y_bool(y),
        dtype=bool,
    )

    if Cs is None:
        #Cs = np.logspace(-3, 1, 5)
        Cs = np.logspace(-3, 1, 20)

    elif isinstance(Cs, int):
        assert Cs >= 1, (
            "Cs must be a positive integer or a one-dimensional array."
        )

        Cs = np.logspace(
            -3,
            1,
            Cs,
        )

    else:
        Cs = np.asarray(
            Cs,
            dtype=np.float64,
        )

        assert Cs.ndim == 1, (
            "Cs must be one-dimensional."
        )
        assert len(Cs) > 0, (
            "Cs must contain at least one value."
        )
        assert np.all(np.isfinite(Cs)), (
            "Cs must contain only finite values."
        )
        assert np.all(Cs > 0), (
            "All values in Cs must be positive."
        )

    # BIC penalty uses the number of observations.
    if weight is None:
        n_bic = len(y_bool)
    else:
        n_bic = np.sum(weight)

    results = []
    best_model = None
    best_bic = np.inf

    for C in Cs:
        model = LogisticRegression(
            C=float(C),
            penalty="l1",
            solver="saga",
            class_weight=class_weight,
            fit_intercept=True,
            max_iter=max_iter,
            tol=optimization_tol,
            random_state=random_state,
        )

        model.fit(
            X_values,
            y_bool,
            sample_weight=weight,
        )

        coef_values = model.coef_.ravel()

        # Number of effectively estimated parameters:
        # non-zero coefficients + intercept.
        n_nonzero = np.sum(
            np.abs(coef_values) > tol_coef
        )
        n_params = int(n_nonzero + 1)

        # Predicted probability for the positive class.
        prob = model.predict_proba(X_values)[:, 1]

        # Avoid log(0).
        eps = np.finfo(np.float64).eps

        prob = np.clip(
            prob,
            eps,
            1.0 - eps,
        )

        log_likelihood_i = (
            y_bool * np.log(prob)
            + (~y_bool) * np.log(1.0 - prob)
        )

        if weight is None:
            log_likelihood = np.sum(
                log_likelihood_i
            )
        else:
            log_likelihood = np.sum(
                weight * log_likelihood_i
            )

        bic = (
            -2.0 * log_likelihood
            + n_params * np.log(n_bic)
        )

        results.append(
            {
                "C": float(C),
                "bic": float(bic),
                "log_likelihood": float(log_likelihood),
                "n_variables": int(n_nonzero),
                "n_params": n_params,
                "n_iter": model.n_iter_.copy(),
            }
        )

        if bic < best_bic:
            best_bic = bic
            best_model = model

    coef = pd.Series(
        best_model.coef_.ravel(),
        index=X_woe.columns,
        name="coefficient",
    )

    selected_coef = (
        coef[coef.abs() > tol_coef]
        .sort_values(
            key=np.abs,
            ascending=False,
        )
    )

    bic_path = pd.DataFrame(results)

    return {
        "model": best_model,
        "coef": coef,
        "selected_variables": selected_coef.index.tolist(),
        "selected_coef": selected_coef,
        "best_C": float(best_model.C),
        "best_bic": float(best_bic),
        "bic_path": bic_path,
        "n_iter": best_model.n_iter_.copy(),
    }

def _gen_cosine_similarity_matrix(
    woe_table,
):
    """
    Generate a cosine similarity matrix from a WoE table.

    This internal function is used by graphical_woe_transformer().

    Args:
      woe_table:
        DataFrame containing Weight of Evidence (WoE) features.

    Returns:
      Square ndarray containing pairwise cosine similarities between
      variables. All values are between -1 and 1, and all diagonal
      elements are 1.
    """
    assert isinstance(woe_table, pd.DataFrame), (
        "woe_table must be a pandas DataFrame."
    )

    woe_array = woe_table.to_numpy().T
    mat = cosine_similarity(
        woe_array,
        woe_array,
    )
    np.fill_diagonal(mat, 1.0)

    return mat

def _calc_partial_correlations(
    mat,
    n_samples,
    target_degree,
    verbose,
):
    """
    Calculate a partial correlation matrix whose graph has approximately
    the specified average degree.

    Alpha is selected by logarithmic binary search.

    Args:
      mat:
        Square ndarray representing a cosine similarity matrix.
      n_samples:
        Number of observations used to calculate mat.
      target_degree:
        Target average degree of the estimated graph.
        If None, full graph is adopted.
      verbose:
        Whether to print detailed information.

    Returns:
      Partial correlation matrix.
    """
    assert (
        isinstance(mat, np.ndarray)
        and mat.ndim == 2
        and mat.shape[0] == mat.shape[1]
        and np.isfinite(mat).all()
    ), "mat must be a finite square NumPy ndarray."

    assert (
        isinstance(n_samples, (int, np.integer))
        and not isinstance(n_samples, bool)
        and n_samples > 0
    ), "n_samples must be a positive integer."

    assert (
        target_degree is None
        or (
            isinstance(target_degree, (int, np.integer))
            and not isinstance(target_degree, bool)
            and target_degree > 0
        )
    ), "target_degree must be None or a positive integer."

    assert isinstance(verbose, bool), "verbose must be a boolean."

    p = len(mat)

    if target_degree is not None:
        target_degree = float(target_degree)

        if target_degree > p - 1:
            target_degree = float(p - 1)

        if p == 1 or target_degree == 0.0:
            return np.identity(p)

    best_precision = _calc_partial_correlations_ebic(
        mat,
        n_samples,
        verbose,
    )

    if target_degree is None:
        return best_precision

    else:
        return _select_edges_by_target_degree(
            best_precision,
            target_degree,
        )

def _calc_partial_correlations_ebic(
    mat,
    n_samples,
    verbose,
    gamma=0.5,
    alpha_ratios=(0.01, 0.03, 0.1, 0.3, 0.7),
    edge_threshold=1e-8,
    max_iter=100,
    tol=1e-3,
):
    """
    Select alpha from a small candidate set using EBIC and calculate
    the corresponding partial correlation matrix.

    Args:
      mat:
        Square empirical covariance or correlation-like matrix.
      n_samples:
        Number of observations used to calculate mat.
      verbose:
        Whether to display candidate results.
      gamma:
        EBIC high-dimensional penalty parameter.
      alpha_ratios:
        Candidate alpha values expressed as ratios of alpha_max.
      edge_threshold:
        Threshold for treating a precision-matrix element as an edge.
      max_iter:
        Maximum number of Graphical Lasso iterations.
      tol:
        Convergence tolerance.

    Returns:
      Partial correlation matrix.
    """
    assert (
        isinstance(mat, np.ndarray)
        and mat.ndim == 2
        and mat.shape[0] == mat.shape[1]
    ), "mat must be a square NumPy ndarray."

    assert (
        isinstance(n_samples, (int, np.integer))
        and not isinstance(n_samples, bool)
        and n_samples >= 1
    ), "n_samples must be a positive integer."

    assert isinstance(verbose, bool), "verbose must be a boolean."

    assert (
        isinstance(gamma, (int, float))
        and gamma >= 0.0
    ), "gamma must be non-negative."

    covariance = np.asarray(
        mat,
        dtype=np.float64,
    ).copy()

    covariance = (
        covariance + covariance.T
    ) / 2.0

    if not np.isfinite(covariance).all():
        raise ValueError(
            "mat must contain only finite values."
        )

    p = covariance.shape[0]

    if p == 1:
        return np.identity(1)

    iu = np.triu_indices(
        p,
        k=1,
    )

    alpha_max = float(
        np.max(
            np.abs(covariance[iu])
        )
    )

    if alpha_max <= 0.0:
        return np.identity(p)

    alphas = (
        alpha_max
        * np.asarray(
            alpha_ratios,
            dtype=np.float64,
        )
    )

    best_ebic = np.inf
    best_alpha = None
    best_precision = None
    best_edges = None
    best_degree = None

    for alpha in alphas:
        try:
            _, precision = graphical_lasso(
                emp_cov=covariance,
                alpha=float(alpha),
                max_iter=max_iter,
                tol=tol,
            )

            if not np.isfinite(
                precision
            ).all():
                continue

            sign, logdet = np.linalg.slogdet(
                precision
            )

            if sign <= 0:
                continue

            # Gaussian log-likelihood excluding constants:
            #
            # log L = n / 2 *
            #         (log det(Theta) - tr(S Theta))
            log_likelihood = (
                n_samples
                / 2.0
                * (
                    logdet
                    - np.trace(
                        covariance @ precision
                    )
                )
            )

            n_edges = int(
                np.count_nonzero(
                    np.abs(precision[iu])
                    > edge_threshold
                )
            )

            # EBIC:
            #
            # -2 log L
            # + |E| log(n)
            # + 4 gamma |E| log(p)
            ebic = (
                -2.0 * log_likelihood
                + n_edges * np.log(n_samples)
                + 4.0
                * gamma
                * n_edges
                * np.log(p)
            )

            mean_degree = (
                2.0 * n_edges / p
            )

            if verbose:
                print(
                    f"alpha={alpha:.6g}, "
                    f"edges={n_edges}, "
                    f"mean_degree={mean_degree:.3f}, "
                    f"EBIC={ebic:.3f}"
                )

            if ebic < best_ebic:
                best_ebic = ebic
                best_alpha = float(alpha)
                best_precision = precision.copy()
                best_edges = n_edges
                best_degree = mean_degree

        except Exception as error:
            if verbose:
                print(
                    f"alpha={alpha:.6g} failed: "
                    f"{error}"
                )

    if best_precision is None:
        if verbose:
            print(
                "All alpha candidates failed. "
                "Returning identity."
            )

        return np.identity(p)

    if verbose:
        print(
            "Selected by EBIC: "
            f"alpha={best_alpha:.6g}, "
            f"edges={best_edges}, "
            f"mean_degree={best_degree:.3f}, "
            f"EBIC={best_ebic:.3f}"
        )

    return _precision_to_partial_corr(
        best_precision
    )

def _precision_to_partial_corr(
    Theta,
):
    """
    Calculate a partial correlation matrix from a precision matrix.

    This internal function is used by _calc_partial_correlations().

    Args:
      Theta:
        Precision matrix.

    Returns:
      Partial correlation matrix.
    """
    d = np.sqrt(np.diag(Theta))
    rho = -Theta / np.outer(d, d)
    np.fill_diagonal(rho, 1.0)

    return rho

def _select_edges_by_target_degree(
    partial_corr,
    target_degree,
):
    """
    Select edges with the largest absolute partial correlations so that
    the average degree is approximately equal to target_degree.

    Args:
      partial_corr:
        Square partial correlation matrix.
      target_degree:
        Target average degree.

    Returns:
      Sparse partial correlation matrix.
    """
    assert isinstance(partial_corr, np.ndarray), (
        "partial_corr must be a NumPy ndarray."
    )
    assert partial_corr.ndim == 2, (
        "partial_corr must be two-dimensional."
    )
    assert partial_corr.shape[0] == partial_corr.shape[1], (
        "partial_corr must be square."
    )
    assert isinstance(target_degree, (int, float)), (
        "target_degree must be numeric."
    )
    assert 0 <= target_degree <= partial_corr.shape[0] - 1, (
        "target_degree must be between 0 and n_variables - 1."
    )

    p = partial_corr.shape[0]

    # Number of edges to retain.
    n_edges = int(round(target_degree * p / 2))

    # Upper triangular entries.
    i, j = np.triu_indices(p, k=1)
    w = np.abs(partial_corr[i, j])

    # Indices of largest absolute partial correlations.
    order = np.argsort(w)[::-1]

    keep = order[:n_edges]

    selected = np.zeros_like(partial_corr)
    selected[i[keep], j[keep]] = partial_corr[i[keep], j[keep]]
    selected[j[keep], i[keep]] = partial_corr[j[keep], i[keep]]

    np.fill_diagonal(selected, 1.0)

    return selected

def _get_interaction_terms(
    X,
    y,
    weight,
    pcor,
):
    """
    Get interaction terms based on a partial correlation matrix.

    This internal function is used by graphical_woe_transformer().

    Args:
      X:
        DataFrame containing categorical variables.
        Numerical variables must be discretized beforehand.
      y:
        One-dimensional array-like target variable aligned with X.
      weight:
        One-dimensional ndarray of non-negative sample weights aligned with X.
      pcor:
        DataFrame representing a square partial correlation matrix.

    Returns:
      Dict of interaction terms and their component variable names.
    """
    assert isinstance(X, pd.DataFrame), (
        "X must be a pandas DataFrame."
    )
    assert len(y) == len(X), (
        "y must have the same length as X."
    )
    assert isinstance(weight, np.ndarray), (
        "weight must be a NumPy ndarray."
    )
    assert len(weight) == len(X), (
        "weight must have the same length as X."
    )
    assert (
        isinstance(pcor, pd.DataFrame)
        and pcor.shape[0] == pcor.shape[1]
    ), "pcor must be a square pandas DataFrame."

    names = pcor.columns
    abs_pcor = np.abs(pcor.to_numpy())

    # List the links in the upper triangle.
    n = abs_pcor.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    weights = abs_pcor[i_idx, j_idx]
    links = np.stack([i_idx, j_idx, weights], axis=1)

    # Extract links with minimum weight or more.
    links = links[links[:, 2] >= 1e-8]
    if len(links) == 0:
        return {}

    # Generate output.
    edges = {}
    for i, j, wt in links:
        i, j = int(i), int(j)
        inter_name = names[i] + '*' + names[j]
        edges[inter_name] = [names[i], names[j]]

    return edges

def _filter_variables_with_min_iv(
    X_d,
    y,
    weight,
    nodes,
    edges,
    min_iv,
    cross_fitting,
    n_splits,
    shuffle,
    random_state,
):
    """
    Identify variables whose information value (IV) is below a threshold.

    This internal function constructs a DataFrame containing the specified
    node and interaction variables, transforms them into Weight of Evidence
    (WoE) values, and identifies variables with IV values below min_iv.

    This function is used by graphical_woe_transformer().

    Args:
      X_d:
        DataFrame containing categorical or discretized variables.
      y:
        One-dimensional array-like binary target variable.
      weight:
        Optional one-dimensional array-like sample weights.
      nodes:
        List of node variable names to include.
      edges:
        Dictionary whose keys are interaction variable names and whose
        values are pairs of node variable names.
        For example, {"a*b": ["a", "b"]}.
      min_iv:
        Minimum IV required to retain a variable.
      cross_fitting:
        Whether to calculate WoE using cross-fitting.
      n_splits:
        Number of folds used for cross-fitting.
      shuffle:
        Whether to shuffle observations before creating folds.
      random_state:
        Random seed used when shuffle is True.

    Returns:
      List of variable names whose IV values are below min_iv.
    """
    if not isinstance(X_d, pd.DataFrame):
        raise TypeError("X_d must be a pandas DataFrame.")

    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list.")

    if not isinstance(edges, dict):
        raise TypeError("edges must be a dictionary.")

    if not isinstance(min_iv, (int, float)) or isinstance(min_iv, bool):
        raise TypeError("min_iv must be numeric.")

    if min_iv < 0:
        raise ValueError("min_iv must be non-negative.")

    # Construct a new DataFrame containing the node variables.
    X_new = X_d[nodes].copy()

    # Add an interaction variable for each edge.
    for inter_name, edge_nodes in edges.items():
        if not isinstance(edge_nodes, (list, tuple)) or len(edge_nodes) != 2:
            raise ValueError(
                "Each value in edges must contain exactly two variable names."
            )

        name1, name2 = edge_nodes

        col1 = X_d[name1].astype("string")
        col2 = X_d[name2].astype("string")

        X_new[inter_name] = (
            (col1 + "*" + col2)
            .where(col1.notna() & col2.notna())
            .astype("category")
        )

    # Calculate WoE values for all node and interaction variables.
    if cross_fitting:
        woe_result = calculate_cross_fitted_woe(
            X_new,
            y,
            weight,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        woe_result = calculate_woe(
            X_new,
            y,
            weight,
        )

    woe_table = woe_result["woe_table"]
    y_bool = woe_result["y_bool"]

    if not isinstance(woe_table, pd.DataFrame):
        raise TypeError("woe_table must be a pandas DataFrame.")

    if not isinstance(y_bool, pd.Series):
        raise TypeError("y_bool must be a pandas Series.")

    if y_bool.dtype != bool:
        raise TypeError("y_bool must be a boolean Series.")

    # Identify variables whose IV values are below the threshold.
    unnecessary_vars = []

    for var in woe_table.columns:
        woe_vec = woe_table[var]

        iv = float(
            woe_vec[y_bool].mean()
            - woe_vec[~y_bool].mean()
        )

        if iv < min_iv:
            unnecessary_vars.append(var)

    return unnecessary_vars

def gen_graph_data(
    transformer,
):
    """
    Generate a NetworkX Graph from a fitted graphical WoE transformer.

    This function is used by draw_graphical_model().

    Args:
      transformer:
        Fitted graphical naive Bayes transformer.

    Returns:
      NetworkX Graph representing the graphical model.
    """
    if not is_graphical_woe_transformer(transformer):
        raise ValueError(
            "transformer must be a fitted graphical WoE transformer."
        )

    def _get_flatten(nested_list):
        return [item for sublist in nested_list for item in sublist]

    pcor = transformer['pcor']
    nodes = transformer['nodes']
    edges = transformer['edges']

    names = list(pcor.columns)
    pcor = pcor.to_numpy()

    # Initialize the graph.
    G = nx.Graph()

    # Add nodes.
    graph_nodes = list(set(nodes) | set(_get_flatten(edges.values())))
    for v in graph_nodes:
        G.add_node(v)

    # Add edges.
    for key in edges:
        v1 = edges[key][0]
        v2 = edges[key][1]
        try:
            i = names.index(v1)
        except ValueError:
            continue
        try:
            j = names.index(v2)
        except ValueError:
            continue
        G.add_edge(v1, v2, weight=pcor[i][j])

    return G

def draw_graphical_model(
    obj,
    pos=None,
    width_scale=None,
    node_color=None,
    edge_colors=None,
    **kwds,
):
    """
    Draw an undirected graphical model (Markov random field).

    Args:
      obj:
        Fitted graphical WoE transformer or NetworkX Graph.
      pos:
        Dictionary specifying node positions.
        If None, networkx.spring_layout() is used.
      width_scale:
        Positive scale factor applied to edge widths based on the
        absolute partial correlation coefficients.
        If None, the default edge width is used for all edges.
      node_color:
        Color used for nodes.
        If None, the default node color is used.
      edge_colors:
        Color or pair of colors used for edges.
        If a list or tuple is provided, the first color is used for
        edges with positive weights and the second for edges with
        negative weights. If None, the default edge color is used.
      **kwds:
        Additional keyword arguments passed to networkx.draw_networkx().

    Returns:
      Tuple containing:
        G:
          NetworkX Graph representing the graphical model.
        args:
          Dictionary of arguments used by networkx.draw_networkx().
    """
    if pos is not None and not isinstance(pos, dict):
        raise TypeError("pos must be a dictionary or None.")

    if width_scale is not None:
        if (
            not isinstance(
                width_scale,
                (int, float, np.integer, np.floating),
            )
            or isinstance(width_scale, bool)
        ):
            raise TypeError("width_scale must be numeric or None.")

        if width_scale <= 0:
            raise ValueError("width_scale must be positive.")

    if node_color is not None and not isinstance(node_color, str):
        raise TypeError("node_color must be a string or None.")

    if edge_colors is not None:
        if isinstance(edge_colors, str):
            pass

        elif isinstance(edge_colors, (list, tuple)):
            if (
                len(edge_colors) != 2
                or not all(isinstance(c, str) for c in edge_colors)
            ):
                raise ValueError(
                    "edge_colors must contain exactly two color strings."
                )

        else:
            raise TypeError(
                "edge_colors must be None, a string, "
                "or a list or tuple of two strings."
            )

    if is_graphical_woe_transformer(obj):
        G = gen_graph_data(obj)

    elif isinstance(obj, nx.Graph):
        G = obj

    else:
        raise TypeError(
            "obj must be a fitted graphical WoE transformer "
            "or a NetworkX Graph."
        )

    args = _draw_network_graph(
        G,
        pos=pos,
        width_scale=width_scale,
        node_color=node_color,
        edge_colors=edge_colors,
        **kwds,
    )

    return G, args

def _draw_network_graph(
    G,
    pos=None,
    width_scale=None,
    node_color=None,
    edge_colors=None,
    **kwds,
):
    """
    Draw a network graph from a NetworkX Graph.

    This internal function is used by draw_graphical_model().

    Args:
      G:
        NetworkX Graph.
      pos:
        Dictionary specifying node positions.
        If None, networkx.spring_layout() is used.
      width_scale:
        Positive scale factor applied to edge widths based on the
        absolute partial correlation coefficients.
        If None, the default edge width is used for all edges.
      node_color:
        Color used for nodes.
        If None, the default node color is used.
      edge_colors:
        Color or pair of colors used for edges.
        If a list or tuple is provided, the first color is used for
        edges with positive weights and the second for edges with
        negative weights. If None, the default edge color is used.
      **kwds:
        Additional keyword arguments passed to networkx.draw_networkx().

    Returns:
      Dictionary of arguments used by networkx.draw_networkx().
    """
    assert isinstance(G, nx.Graph), "G must be a NetworkX Graph."

    if pos is None:
        pos = nx.spring_layout(G)

    args = {'pos': pos, 'with_labels': True, **kwds}

    if width_scale is not None:
        width_scale = float(width_scale)
        edge_width = [
            abs(edge["weight"]) * width_scale
            for edge in G.edges.values()
        ]
        args['width'] = edge_width

    if node_color is not None:
        args['node_color'] = node_color

    if edge_colors is not None:
        if type(edge_colors) is str:
            args['edge_color'] = edge_colors
        else:
            if len(edge_colors) > 1:
                p_edge_color = edge_colors[0]
                n_edge_color = edge_colors[1]
                edge_color = [
                    p_edge_color
                    if edge["weight"] > 0
                    else n_edge_color
                    for edge in G.edges.values()
                ]
                args['edge_color'] = edge_color
            else:
                args['edge_color'] = edge_colors[0]

    nx.draw_networkx(G, **args)
    plt.show()

    return args

def make_l1_coefficient_dict(
    model,
    prep,
    X,
):
    """
    Extract L1 logistic regression coefficients grouped by original variables.

    This function is specific to the preprocessing pipeline used in this module
    and assumes the following preprocessing structure:

      prep: ColumnTransformer
       ├── "num": numerical transformer
       └── "cat": Pipeline
          └── "encoder": OneHotEncoder

    The fitted OneHotEncoder is used to recover the correspondence between
    original categorical variables and their encoded categories.

    Args:
      model:
        Fitted LogisticRegression model.
      prep:
        Fitted preprocessing pipeline used to transform X.
      X:
        DataFrame before preprocessing.

    Returns:
      Dict containing numerical and categorical coefficients grouped by
      original variables.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    num_features = [
        col
        for col in X.columns
        if X[col].dtype.kind in "iufc"
    ]

    cat_features = [
        col
        for col in X.columns
        if X[col].dtype.kind not in "iufc"
    ]

    coef = {
        "numeric": {},
        "categorical": {},
    }

    # Numerical features
    n_num = len(num_features)

    for i, var in enumerate(num_features):
        coef["numeric"][var] = float(
            model.coef_[0][i]
        )

    # Categorical features
    if cat_features:
        ohe = (
            prep
            .named_transformers_["cat"]
            .named_steps["encoder"]
        )

        offset = n_num

        for var, categories in zip(
            cat_features,
            ohe.categories_,
        ):
            n_categories = len(categories)

            values = model.coef_[0][
                offset:offset + n_categories
            ]

            coef["categorical"][var] = {
                str(category): float(value)
                for category, value in zip(
                    categories,
                    values,
                )
            }

            offset += n_categories

    return coef

def plot_l1_coefficients(
    coef,
    variables=None,
    exclude_zero=True,
):
    """
    Plot L1 logistic regression coefficients by original variable.

    Numerical variables are displayed in one plot. Each categorical
    variable is displayed in a separate plot.

    Args:
      coef:
        Coefficient dictionary generated by make_l1_coefficient_dict().
      variables:
        List of original variable names to display.
        If None, all variables are displayed.
        Variables that do not exist are ignored.
      exclude_zero:
        Whether to exclude zero coefficients.
    """
    if not isinstance(coef, dict):
        raise TypeError("coef must be a dictionary.")

    if variables is not None and not isinstance(variables, list):
        raise TypeError("variables must be a list or None.")

    if variables is not None:
        all_variables = (
            set(coef.get("numeric", {}))
            | set(coef.get("categorical", {}))
        )

        if not any(var in all_variables for var in variables):
            return None

    # Numerical features
    numeric_coef = coef.get("numeric", {})

    if variables is not None:
        numeric_coef = {
            key: value
            for key, value in numeric_coef.items()
            if key in variables
        }

    if exclude_zero:
        numeric_coef = {
            key: value
            for key, value in numeric_coef.items()
            if value != 0
        }

    if numeric_coef:
        plot_data = pd.DataFrame({
            "feature": list(numeric_coef.keys()),
            "coefficient": list(numeric_coef.values()),
        })

        plot_data["abs_coefficient"] = (
            plot_data["coefficient"].abs()
        )

        plot_data = plot_data.sort_values(
            "abs_coefficient",
            ascending=False,
        )

        fig, ax = plt.subplots(
            figsize=(
                7,
                max(3, 0.3 * len(plot_data)),
            )
        )

        sns.barplot(
            data=plot_data,
            x="coefficient",
            y="feature",
            order=plot_data["feature"],
            errorbar=None,
            ax=ax,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set(
            xlabel="Coefficient",
            ylabel="Numeric features",
        )

        sns.despine(
            ax=ax,
            left=True,
            bottom=True,
        )

        fig.tight_layout()
        plt.show()

    # Categorical features
    categorical_coef = coef.get(
        "categorical",
        {},
    )

    if variables is not None:
        categorical_coef = {
            key: value
            for key, value in categorical_coef.items()
            if key in variables
        }

    for var, categories in categorical_coef.items():
        if exclude_zero:
            categories = {
                key: value
                for key, value in categories.items()
                if value != 0
            }

        if not categories:
            continue

        plot_data = pd.DataFrame({
            "category": list(categories.keys()),
            "coefficient": list(categories.values()),
        })

        plot_data["abs_coefficient"] = (
            plot_data["coefficient"].abs()
        )

        plot_data = plot_data.sort_values(
            "abs_coefficient",
            ascending=False,
        )

        fig, ax = plt.subplots(
            figsize=(
                7,
                max(3, 0.3 * len(plot_data)),
            )
        )

        sns.barplot(
            data=plot_data,
            x="coefficient",
            y="category",
            order=plot_data["category"],
            errorbar=None,
            ax=ax,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set(
            xlabel="Coefficient",
            ylabel=var,
        )

        sns.despine(
            ax=ax,
            left=True,
            bottom=True,
        )

        fig.tight_layout()
        plt.show()

def plot_l1_feature_coefficients(
    coefficients,
    feature_names,
    top_n=None,
    exclude_zero=True,
):
    """
    Plot L1 logistic regression coefficients by transformed feature.

    Features are ordered by absolute coefficient magnitude.

    Args:
      coefficients:
        One-dimensional array-like coefficients.
      feature_names:
        One-dimensional array-like feature names corresponding to
        coefficients.
      top_n:
        Number of features to display, selected by absolute coefficient
        magnitude. If None, all features are displayed.
      exclude_zero:
        Whether to exclude features with zero coefficients.

    Returns:
      DataFrame containing the displayed features and coefficients.
    """
    coefficients = np.asarray(coefficients)
    feature_names = np.asarray(feature_names)

    if coefficients.ndim != 1:
        raise ValueError(
            "coefficients must be one-dimensional."
        )

    if feature_names.ndim != 1:
        raise ValueError(
            "feature_names must be one-dimensional."
        )

    if len(coefficients) != len(feature_names):
        raise ValueError(
            "coefficients and feature_names must have the same length."
        )

    if top_n is not None:
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            raise TypeError(
                "top_n must be an integer or None."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be positive."
            )

    if not isinstance(exclude_zero, bool):
        raise TypeError(
            "exclude_zero must be a boolean."
        )

    plot_data = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    if exclude_zero:
        plot_data = plot_data[
            plot_data["coefficient"] != 0
        ]

    plot_data["abs_coefficient"] = (
        plot_data["coefficient"].abs()
    )

    plot_data = plot_data.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    if top_n is not None:
        plot_data = plot_data.head(top_n)

    if plot_data.empty:
        return (
            plot_data
            .drop(columns="abs_coefficient")
            .reset_index(drop=True)
        )

    fig, ax = plt.subplots(
        figsize=(
            7,
            max(3, 0.3 * len(plot_data)),
        )
    )

    sns.barplot(
        data=plot_data,
        x="coefficient",
        y="feature",
        order=plot_data["feature"],
        errorbar=None,
        ax=ax,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        xlabel="Coefficient",
        ylabel=None,
    )

    sns.despine(
        ax=ax,
        left=True,
        bottom=True,
    )

    fig.tight_layout()
    plt.show()

    return (
        plot_data
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )

"""
scikit-learn compliant class that provides methods for graphical WoE transformer.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class GraphicalWoETransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        discretizer='bic',
        cross_fitting=True,
        n_splits=5,
        shuffle=True,
        l1_selection=False,
        target_degree='adaptive',
        min_iv=MIN_IV,
        numeric=True,
        random_state=None,
        verbose=False,
    ):
        self.discretizer = discretizer
        self.cross_fitting = cross_fitting
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.l1_selection = l1_selection
        self.target_degree = target_degree
        self.min_iv = min_iv
        self.numeric = numeric
        self.random_state = random_state
        self.verbose = verbose

    def __sklearn_tags__(
        self,
    ):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.input_tags.allow_nan = True
        return tags

    def fit(
        self,
        X,
        y,
        weight=None,
    ):
        ycat = pd.Categorical(y)
        self.classes_ = np.array(ycat.categories)
        self.coef_ = graphical_woe_transformer(
            X,
            y,
            weight=weight,
            discretizer=self.discretizer,
            cross_fitting=self.cross_fitting,
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            l1_selection=self.l1_selection,
            target_degree=self.target_degree,
            min_iv=self.min_iv,
            numeric=self.numeric,
            random_state=self.random_state,
            verbose=self.verbose
        )
        return self

    def transform(
        self,
        X,
    ):
        try:
            check_is_fitted(self, attributes=['coef_', 'classes_'])
        except NotFittedError as exc:
            print(f"The transformer is not fitted yet.")

        return transform_dataset(
            self.coef_,
            X,
        )

    def gen_graph_data(
        self,
    ):
        try:
            check_is_fitted(self, attributes=['coef_', 'classes_'])
        except NotFittedError as exc:
            print(f"The model is not fitted yet.")

        return gen_graph_data(self.coef_)

    def draw(
        self,
        G=None,
        pos=None,
        width_scale=None,
        node_color=None,
        edge_colors=None,
        **kwds
    ):
        try:
            check_is_fitted(self, attributes=['coef_', 'classes_'])
        except NotFittedError as exc:
            print(f"The model is not fitted yet.")

        if G is None:
            return draw_graphical_model(
                self.coef_,
                pos=pos,
                width_scale=width_scale,
                node_color=node_color,
                edge_colors=edge_colors,
                **kwds
            )

        else:
            assert type(G) is nx.Graph, "G must be a networkx.Graph."
            return draw_graphical_model(
                G,
                pos=pos,
                width_scale=width_scale,
                node_color=node_color,
                edge_colors=edge_colors,
                **kwds
            )

    def get_transformer(
        self,
    ):
        try:
            check_is_fitted(self, attributes=['coef_', 'classes_'])
        except NotFittedError as exc:
            print(f"The model is not fitted yet.")

        return self.coef_

    def get_pcor(
        self,
    ):
        try:
            check_is_fitted(self, attributes=['coef_', 'classes_'])
        except NotFittedError as exc:
            print(f"The model is not fitted yet.")

        return self.coef_['pcor']

