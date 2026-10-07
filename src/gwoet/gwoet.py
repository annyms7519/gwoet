"""gwoet.py

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
  Generate a NetworkX Graph from a fitted graphical WoE transformer.

draw_graphical_model:
  Draw an undirected graphical model (Markov random field).

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
Last modified: 2026-10-07
Version: 0.9.0.5
License: MIT
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import graphical_lasso
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


# Package import; a standalone copy can use a sibling dnb.py.
if __package__:
    from .dnb import (
        apply_disc,
        calculate_cross_fitted_woe,
        calculate_woe,
        generate_disc,
        make_sample_weight,
    )
else:
    from dnb import (
        apply_disc,
        calculate_cross_fitted_woe,
        calculate_woe,
        generate_disc,
        make_sample_weight,
    )


# constants
GAMMA = 1.0             # EBIC high-dimensional penalty parameter.
EBIC_THRESHOLD = False  # Whether to threshold weak precision-matrix elements.
PRUNING_RATIO = 0.0     # Fraction of weakest partial-correlation edges to prune.
ZERO_TOL = 1e-8         # Numerical tolerance for zero precision entries.
MIN_IV = 0.02           # Minimum IV used for variable pruning.


def _validate_real(
    value,
    name,
    *,
    minimum=0.0,
    strict=False,
):
    """
    Validate a finite real scalar.

    Boolean values are not treated as numeric values.
    The lower bound can be inclusive or exclusive.
    """
    if (
        not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            value,
            (bool, np.bool_),
        )
    ):
        raise TypeError(
            f"{name} must be a real number."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        relation = (
            "greater than" if strict else "at least"
        )
        raise ValueError(
            f"{name} must be finite and {relation} {minimum}."
        )

    return value


# ============================================================
# Graphical WoE transformer
# ============================================================

def graphical_woe_transformer(
    X,
    y,
    *,
    weight=None,
    discretizer='bic',
    cross_fitting=True,
    n_splits=5,
    shuffle=True,
    gamma=GAMMA,
    ebic_threshold=EBIC_THRESHOLD,
    pruning_ratio=PRUNING_RATIO,
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
      gamma:
        Non-negative EBIC high-dimensional penalty parameter used when
        selecting the Graphical Lasso regularization strength. Larger values
        favor sparser EBIC solutions.
      ebic_threshold:
        Whether to apply the Jankova-van de Geer threshold to
        precision-matrix elements before EBIC evaluation.
      pruning_ratio:
        Fraction of the weakest nonzero partial-correlation edges to remove
        after EBIC estimation and optional EBIC thresholding. Must be in
        [0, 1). A value of 0.2 removes approximately the weakest 20%.
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
          Full EBIC-estimated partial correlation matrix, before percentage
          pruning or IV pruning. Zero-norm WoE variables are excluded from
          estimation. This matrix may contain variables/edges absent from
          the final nodes/edges. If no WoE variable is estimable, a zero
          placeholder matrix is returned with the original-feature fallback.
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

    gamma = _validate_real(
        gamma,
        "gamma",
    )

    if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
        raise ValueError("X must have unique string column names.")
    if X.shape[1] == 0 or len(X) == 0:
        raise ValueError("X must contain observations and features.")
    if pd.isna(y).any() or len(pd.unique(y)) != 2:
        raise ValueError("y must contain exactly two nonmissing classes.")

    if not isinstance(ebic_threshold, bool):
        raise TypeError("ebic_threshold must be a boolean.")

    pruning_ratio = _validate_real(
        pruning_ratio,
        "pruning_ratio",
    )
    if pruning_ratio >= 1.0:
        raise ValueError("pruning_ratio must be less than 1.0.")

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

    woe_result = _calculate_training_woe(
        X_d,
        y,
        weight,
        cross_fitting=cross_fitting,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    woe_table = woe_result['woe_table']

    # Remove variables of zero norm.
    norm = np.linalg.norm(
        woe_table,
        axis=0,
    )
    woe_table = woe_table[(woe_table.columns)[norm > 0]]
    X_d = X_d[woe_table.columns]

    if len(woe_table.columns) == 0:
        if verbose:
            print("All variables were pruned. The original features are used instead.")
        return _fallback_transformer(
            X,
            disc,
            numeric,
        )

    nodes = list(woe_table.columns)
    if verbose:
        print(f"Nodes: {len(nodes)}")
    random_state = check_random_state(random_state)
    mat = _gen_cosine_similarity_matrix(woe_table)
    pcor_values = _calc_partial_correlations(
        mat,
        X.shape[0],
        gamma=gamma,
        ebic_threshold=ebic_threshold,
        verbose=verbose,
    )
    pcor = pd.DataFrame(
        pcor_values,
        index=nodes,
        columns=nodes,
    )

    # Optionally prune a fraction of the weakest EBIC partial-correlation edges.
    candidate_values = _prune_edges_by_ratio(
        pcor_values,
        pruning_ratio,
    )

    if verbose and pruning_ratio > 0.0:
        i_prune, j_prune = np.triu_indices(len(pcor_values), k=1)
        n_before = int(np.count_nonzero(
            np.abs(pcor_values[i_prune, j_prune]) >= ZERO_TOL
        ))
        n_after = int(np.count_nonzero(
            np.abs(candidate_values[i_prune, j_prune]) >= ZERO_TOL
        ))
        print(
            f"Percentage pruning: ratio={pruning_ratio:.3f}, "
            f"edges={n_before}->{n_after}"
        )

    candidate_pcor = pd.DataFrame(
        candidate_values,
        index=nodes,
        columns=nodes,
    )

    edges = _get_interaction_terms(
        X_d,
        y,
        weight,
        candidate_pcor,
    )

    selected_vars = _filter_variables_with_woe(
        X_d,
        y,
        nodes,
        edges,
        weight=weight,
        cross_fitting=cross_fitting,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        verbose=verbose,
    )

    if len(selected_vars) == 0:
        if verbose:
            print("All variables were pruned. The original features are used instead.")
        return _fallback_transformer(
            X,
            disc,
            numeric,
            pcor=pcor,
        )

    # Remove unnecessary edges.
    edges = {
        edge_name: edge_nodes
        for edge_name, edge_nodes in edges.items()
        if edge_name in selected_vars
    }
    if verbose:
        print(f"Edges after IV pruning: {len(edges)}")

    # Keep selected nodes + endpoints of selected edges.
    edge_nodes = {
        node
        for nodes_in_edge in edges.values()
        for node in nodes_in_edge
    }
    nodes = [
        node
        for node in nodes
        if node in selected_vars or node in edge_nodes
    ]
    if verbose:
        print(f"Nodes after IV pruning: {len(nodes)}")

    transformer = {
        'disc': disc,
        'pcor': pcor,
        'numeric': numeric,
        'nodes': nodes,
        'edges': edges,
    }
    return transformer


def _calculate_training_woe(
    X,
    y,
    weight,
    *,
    cross_fitting,
    n_splits,
    shuffle,
    random_state,
):
    """
    Calculate training WoE values.

    Cross-fitted WoE is used when cross_fitting is True.
    Otherwise, WoE is estimated using the full training dataset.
    """
    if cross_fitting:
        return calculate_cross_fitted_woe(
            X,
            y,
            weight,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    return calculate_woe(
        X,
        y,
        weight,
    )


def _fallback_transformer(
    X,
    disc,
    numeric,
    pcor=None,
):
    """
    Create a fallback transformer using the original features.

    Existing EBIC partial correlations are retained when available.
    The returned model contains no interactions.
    """
    return {
        "disc": disc,
        "pcor": (
            pcor if pcor is not None else pd.DataFrame(
                0.0,
                index=X.columns,
                columns=X.columns,
            )
        ),
        "numeric": numeric,
        "nodes": list(X.columns),
        "edges": {},
    }


def is_graphical_woe_transformer(
    model,
):
    """
    Check whether a model is a graphical WoE transformer.
    """
    return (
        isinstance(model, dict)
        and {'disc', 'pcor', 'numeric', 'nodes', 'edges'} <= model.keys()
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


# ============================================================
# Graph structure estimation
# ============================================================

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
    *,
    gamma=GAMMA,
    ebic_threshold=EBIC_THRESHOLD,
    verbose=False,
):
    """
    Return the full EBIC estimate; edge selection is a separate step.

    gamma controls the EBIC complexity penalty used to select the Graphical
    Lasso alpha. Percentage pruning is intentionally handled by the caller
    after this estimate has been obtained.
    """
    _validate_partial_corr_matrix(mat)

    if (
        not isinstance(n_samples, (int, np.integer))
        or isinstance(n_samples, (bool, np.bool_))
        or n_samples <= 0
    ):
        raise ValueError(
            "n_samples must be a positive integer."
        )

    gamma = _validate_real(
        gamma,
        "gamma",
    )

    if not isinstance(ebic_threshold, bool):
        raise TypeError(
            "ebic_threshold must be a boolean."
        )

    if not isinstance(verbose, bool):
        raise TypeError(
            "verbose must be a boolean."
        )

    if len(mat) == 1:
        return np.identity(1)

    return _calc_partial_correlations_ebic(
        mat,
        n_samples,
        gamma=gamma,
        ebic_threshold=ebic_threshold,
        verbose=verbose,
    )


def _calc_partial_correlations_ebic(
    mat,
    n_samples,
    *,
    gamma=GAMMA,
    alpha_ratios=(0.01, 0.03, 0.1, 0.3, 1.0),
    n_refine=7,
    min_alpha_ratio=0.001,
    ebic_threshold=False,
    zero_tol=ZERO_TOL,
    max_iter=100,
    tol=1e-3,
    verbose=False,
):
    """
    Select alpha using a two-stage EBIC search and calculate
    the corresponding partial correlation matrix.

    Optionally, weak off-diagonal precision-matrix elements are
    removed before EBIC evaluation using

        log(p * (p - 1) / 2) / sqrt(n).

    Stage 1:
      Evaluate a small set of coarse alpha ratios.

    Stage 2:
      Refine the search on a logarithmic grid around the best
      coarse alpha ratio.

      - If the smallest coarse ratio is selected, extend the
        search downward to min_alpha_ratio.
      - If the largest coarse ratio (normally 1.0) is selected,
        no further search is performed because alpha >= alpha_max
        normally corresponds to an empty graph.
      - Otherwise, search between the neighboring coarse ratios.

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
        Coarse candidate alpha values expressed as ratios of alpha_max.
      n_refine:
        Number of logarithmically spaced candidates used in the
        second-stage local search.
      min_alpha_ratio:
        Lower bound used when the smallest coarse alpha ratio
        is selected in the first stage.
      ebic_threshold:
        Whether to threshold weak off-diagonal precision-matrix
        elements before calculating EBIC.
      zero_tol:
        Numerical tolerance for treating a precision-matrix
        element as zero. This is separate from ebic_threshold.
      max_iter:
        Maximum number of Graphical Lasso iterations.
      tol:
        Convergence tolerance used by Graphical Lasso.

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

    assert isinstance(verbose, bool), (
        "verbose must be a boolean."
    )

    assert (
        isinstance(gamma, (int, float, np.integer, np.floating))
        and not isinstance(gamma, (bool, np.bool_))
        and np.isfinite(gamma)
        and gamma >= 0.0
    ), "gamma must be a finite non-negative number."

    assert isinstance(ebic_threshold, bool), (
        "ebic_threshold must be a boolean."
    )

    assert (
        isinstance(zero_tol, (int, float, np.integer, np.floating))
        and not isinstance(zero_tol, (bool, np.bool_))
        and np.isfinite(zero_tol)
        and zero_tol >= 0.0
    ), "zero_tol must be a finite non-negative number."

    assert (
        isinstance(n_refine, (int, np.integer))
        and not isinstance(n_refine, bool)
        and n_refine >= 2
    ), "n_refine must be an integer >= 2."

    assert (
        isinstance(min_alpha_ratio, (int, float))
        and not isinstance(min_alpha_ratio, bool)
        and 0.0 < min_alpha_ratio <= 1.0
    ), "min_alpha_ratio must be in (0, 1]."

    alpha_ratios = np.asarray(
        alpha_ratios,
        dtype=np.float64,
    )

    if (
        alpha_ratios.ndim != 1
        or alpha_ratios.size == 0
        or not np.isfinite(alpha_ratios).all()
        or np.any(alpha_ratios <= 0.0)
        or np.any(alpha_ratios > 1.0)
    ):
        raise ValueError(
            "alpha_ratios must be a one-dimensional sequence "
            "containing values in (0, 1]."
        )

    # Sort and remove duplicates.
    alpha_ratios = np.unique(
        alpha_ratios
    )

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

    # ---------------------------------------------------------
    # EBIC threshold.
    # ---------------------------------------------------------
    if ebic_threshold:
        ebic_threshold_value = (
            np.log(
                p * (p - 1) / 2.0
            )
            / np.sqrt(n_samples)
        )
    else:
        ebic_threshold_value = 0.0

    if verbose and ebic_threshold:
        print(
            "EBIC threshold: "
            f"{ebic_threshold_value:.6g}"
        )

    # ---------------------------------------------------------
    # Maximum Graphical Lasso alpha.
    # ---------------------------------------------------------
    alpha_max = float(
        np.max(
            np.abs(covariance[iu])
        )
    )

    if alpha_max <= 0.0:
        return np.identity(p)

    best_ebic = np.inf
    best_alpha = None
    best_ratio = None
    best_precision = None
    best_edges = None
    best_degree = None

    # Store evaluated ratios to avoid fitting the same alpha twice.
    evaluated_ratios = {}

    def count_edges(precision):
        """
        Count numerically nonzero off-diagonal precision elements.
        """
        return int(
            np.count_nonzero(
                np.abs(precision[iu]) > zero_tol
            )
        )

    def evaluate_ratio(
        ratio,
        stage,
    ):
        nonlocal best_ebic
        nonlocal best_alpha
        nonlocal best_ratio
        nonlocal best_precision
        nonlocal best_edges
        nonlocal best_degree

        ratio = float(ratio)

        # Avoid duplicate evaluations caused by overlapping grids.
        for previous_ratio in evaluated_ratios:
            if np.isclose(
                ratio,
                previous_ratio,
                rtol=1e-12,
                atol=1e-15,
            ):
                return evaluated_ratios[
                    previous_ratio
                ]

        alpha = alpha_max * ratio

        try:
            _, precision = graphical_lasso(
                emp_cov=covariance,
                alpha=float(alpha),
                max_iter=max_iter,
                tol=tol,
            )

            if not np.isfinite(precision).all():
                evaluated_ratios[ratio] = np.inf
                return np.inf

            # Ensure numerical symmetry.
            precision = (
                precision + precision.T
            ) / 2.0

            n_edges_before = count_edges(
                precision
            )

            # -------------------------------------------------
            # EBIC thresholding.
            # -------------------------------------------------
            if ebic_threshold:
                precision = precision.copy()

                weak_edges = (
                    np.abs(precision)
                    < ebic_threshold_value
                )

                # Never threshold diagonal elements.
                np.fill_diagonal(
                    weak_edges,
                    False,
                )

                precision[
                    weak_edges
                ] = 0.0

                # Preserve exact symmetry.
                precision = (
                    precision + precision.T
                ) / 2.0

            n_edges = count_edges(
                precision
            )

            # -------------------------------------------------
            # Check positive definiteness and calculate logdet.
            # -------------------------------------------------
            try:
                chol = np.linalg.cholesky(
                    precision
                )
            except np.linalg.LinAlgError:
                evaluated_ratios[ratio] = np.inf

                if verbose:
                    print(
                        f"{stage}: "
                        f"ratio={ratio:.6g}, "
                        f"alpha={alpha:.6g} failed: "
                        "precision matrix is not "
                        "positive definite."
                    )

                return np.inf

            logdet = (
                2.0
                * np.sum(
                    np.log(
                        np.diag(chol)
                    )
                )
            )

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

            evaluated_ratios[
                ratio
            ] = ebic

            if verbose:
                if ebic_threshold:
                    edge_text = (
                        f"edges="
                        f"{n_edges_before}->{n_edges}"
                    )
                else:
                    edge_text = (
                        f"edges={n_edges}"
                    )

                print(
                    f"{stage}: "
                    f"ratio={ratio:.6g}, "
                    f"alpha={alpha:.6g}, "
                    f"{edge_text}, "
                    f"mean_degree={mean_degree:.3f}, "
                    f"EBIC={ebic:.3f}"
                )

            if ebic < best_ebic:
                best_ebic = ebic
                best_alpha = float(alpha)
                best_ratio = ratio
                best_precision = precision.copy()
                best_edges = n_edges
                best_degree = mean_degree

            return ebic

        except Exception as error:
            evaluated_ratios[
                ratio
            ] = np.inf

            if verbose:
                print(
                    f"{stage}: "
                    f"ratio={ratio:.6g}, "
                    f"alpha={alpha:.6g} failed: "
                    f"{error}"
                )

            return np.inf

    # ---------------------------------------------------------
    # Stage 1: coarse search.
    # ---------------------------------------------------------
    if verbose:
        print(
            "EBIC coarse search:"
        )

    coarse_ebics = np.array(
        [
            evaluate_ratio(
                ratio,
                stage="coarse",
            )
            for ratio in alpha_ratios
        ],
        dtype=np.float64,
    )

    finite_mask = np.isfinite(
        coarse_ebics
    )

    if not np.any(finite_mask):
        if verbose:
            print(
                "All coarse alpha candidates failed. "
                "Returning identity."
            )

        return np.identity(p)

    coarse_best_idx = int(
        np.nanargmin(
            coarse_ebics
        )
    )

    # ---------------------------------------------------------
    # Stage 2: local logarithmic refinement.
    # ---------------------------------------------------------
    if coarse_best_idx == len(alpha_ratios) - 1:
        # Largest ratio selected.
        # Normally ratio=1.0 corresponds to alpha_max and
        # therefore an empty graph.
        if verbose:
            print(
                "Refinement skipped because the largest "
                "alpha ratio was selected."
            )

    else:
        if coarse_best_idx == 0:
            lower_ratio = min(
                min_alpha_ratio,
                alpha_ratios[0],
            )
            upper_ratio = alpha_ratios[1]

        else:
            lower_ratio = alpha_ratios[
                coarse_best_idx - 1
            ]
            upper_ratio = alpha_ratios[
                coarse_best_idx + 1
            ]

        if lower_ratio < upper_ratio:
            refine_ratios = np.geomspace(
                lower_ratio,
                upper_ratio,
                n_refine,
            )

            if verbose:
                print(
                    "EBIC refinement search:"
                )

            for ratio in refine_ratios:
                evaluate_ratio(
                    ratio,
                    stage="refine",
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
            f"ratio={best_ratio:.6g}, "
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


def _validate_partial_corr_matrix(
    mat,
):
    """
    Validate a nonempty finite symmetric square NumPy matrix.
    """
    if not isinstance(mat, np.ndarray):
        raise TypeError("The matrix must be a NumPy ndarray.")

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        raise ValueError("The matrix must be nonempty and square.")

    if not np.isfinite(mat).all() or not np.allclose(mat, mat.T):
        raise ValueError("The matrix must be finite and symmetric.")


def _prune_edges_by_ratio(
    partial_corr,
    pruning_ratio,
):
    """
    Remove a fraction of the weakest nonzero partial-correlation edges.

    Args:
      partial_corr:
        Symmetric partial correlation matrix.
      pruning_ratio:
        Fraction of nonzero edges to remove, in [0, 1).

    Returns:
      Pruned partial correlation matrix.
    """
    _validate_partial_corr_matrix(partial_corr)
    pruning_ratio = _validate_real(pruning_ratio, "pruning_ratio")
    if pruning_ratio >= 1.0:
        raise ValueError("pruning_ratio must be less than 1.0.")

    p = len(partial_corr)
    i, j = np.triu_indices(p, k=1)
    strength = np.abs(partial_corr[i, j])
    candidates = np.flatnonzero(strength >= ZERO_TOL)

    n_prune = int(np.floor(len(candidates) * pruning_ratio))
    if n_prune == 0:
        return partial_corr.copy()

    order = candidates[np.argsort(strength[candidates], kind="stable")]
    prune = order[:n_prune]

    selected = partial_corr.copy()
    selected[i[prune], j[prune]] = 0.0
    selected[j[prune], i[prune]] = 0.0
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
        if inter_name in X.columns or inter_name in edges:
            raise ValueError(f"Interaction name collision: {inter_name!r}.")
        edges[inter_name] = [names[i], names[j]]

    return edges


def _filter_variables_with_woe(
    X_d,
    y,
    nodes,
    edges,
    *,
    weight=None,
    cross_fitting=True,
    n_splits=5,
    shuffle=True,
    random_state=None,
    verbose=False,
):
    """
    Identify variables to retain based on information value (IV).

    This internal function constructs a DataFrame containing the specified
    node and interaction variables, transforms them into Weight of Evidence
    (WoE) values, and retains variables with IV larger than MIN_IV.
    """
    if not isinstance(X_d, pd.DataFrame):
        raise TypeError("X_d must be a pandas DataFrame.")
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list.")
    if not isinstance(edges, dict):
        raise TypeError("edges must be a dictionary.")

    X_new = X_d[nodes].copy()
    for inter_name, edge_nodes in edges.items():
        if not isinstance(edge_nodes, (list, tuple)) or len(edge_nodes) != 2:
            raise ValueError(
                "Each value in edges must contain exactly two variable names."
            )
        name1, name2 = edge_nodes
        X_new[inter_name] = _make_categorical_interaction(
            X_d[name1],
            X_d[name2],
        ).astype("category")

    woe_result = _calculate_training_woe(
        X_new,
        y,
        weight,
        cross_fitting=cross_fitting,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    woe_table = woe_result["woe_table"]
    y_bool = woe_result["y_bool"]

    if not isinstance(woe_table, pd.DataFrame):
        raise TypeError("woe_table must be a pandas DataFrame.")
    if not isinstance(y_bool, pd.Series):
        raise TypeError("y_bool must be a pandas Series.")
    if y_bool.dtype != bool:
        raise TypeError("y_bool must be a boolean Series.")
    if not woe_table.index.equals(y_bool.index):
        raise ValueError("WoE and y_bool indices must be aligned.")

    iv = woe_table.loc[y_bool].mean() - woe_table.loc[~y_bool].mean()
    selected_vars = iv.index[iv > MIN_IV].tolist()

    if verbose:
        print(f"selected variables by IV: {len(selected_vars)}")

    return selected_vars


# ============================================================
# Graph visualization
# ============================================================

def gen_graph_data(
    transformer,
):
    """
    Generate a NetworkX Graph from a fitted graphical WoE transformer.

    This function is used by draw_graphical_model().

    Args:
      transformer:
        Fitted graphical WoE transformer.

    Returns:
      NetworkX Graph representing the graphical model.
    """
    if not is_graphical_woe_transformer(transformer):
        raise ValueError(
            "transformer must be a fitted graphical WoE transformer."
        )

    pcor = transformer["pcor"]
    nodes = transformer["nodes"]
    edges = transformer["edges"]

    positions = {name: i for i, name in enumerate(pcor.columns)}
    values = pcor.to_numpy()

    # Initialize the graph.
    G = nx.Graph()

    # Add nodes.
    G.add_nodes_from(nodes)

    # Add edges.
    for v1, v2 in edges.values():
        if v1 not in positions or v2 not in positions:
            continue
        attributes = {"weight": values[positions[v1], positions[v2]]}
        G.add_edge(
            v1,
            v2,
            **attributes,
        )

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
        If None, networkx.kamada_kawai_layout() is used.
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
        #pos = nx.spring_layout(G)
        pos = nx.kamada_kawai_layout(G, weight=None)

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


"""
scikit-learn compliant class that provides methods for graphical WoE transformer.
"""
class GraphicalWoETransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        *,
        discretizer='bic',
        cross_fitting=True,
        n_splits=5,
        shuffle=True,
        gamma=GAMMA,
        ebic_threshold=EBIC_THRESHOLD,
        pruning_ratio=PRUNING_RATIO,
        numeric=True,
        random_state=None,
        verbose=False,
    ):
        self.discretizer = discretizer
        self.cross_fitting = cross_fitting
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.gamma = gamma
        self.ebic_threshold = ebic_threshold
        self.pruning_ratio = pruning_ratio
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
        fitted = graphical_woe_transformer(
            X,
            y,
            weight=weight,
            discretizer=self.discretizer,
            cross_fitting=self.cross_fitting,
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            gamma=self.gamma,
            ebic_threshold=self.ebic_threshold,
            pruning_ratio=self.pruning_ratio,
            numeric=self.numeric,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.coef_ = fitted
        self.classes_ = np.array(ycat.categories)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        return self

    def transform(
        self,
        X,
    ):
        check_is_fitted(
            self,
            attributes=['coef_', 'classes_'],
        )

        return transform_dataset(
            self.coef_,
            X,
        )

    def gen_graph_data(
        self,
    ):
        check_is_fitted(
            self,
            attributes=['coef_', 'classes_'],
        )

        return gen_graph_data(self.coef_)

    def draw(
        self,
        G=None,
        pos=None,
        width_scale=None,
        node_color=None,
        edge_colors=None,
        **kwds,
    ):
        check_is_fitted(
            self,
            attributes=['coef_', 'classes_'],
        )

        if G is None:
            return draw_graphical_model(
                self.coef_,
                pos=pos,
                width_scale=width_scale,
                node_color=node_color,
                edge_colors=edge_colors,
                **kwds,
            )

        else:
            assert type(G) is nx.Graph, "G must be a networkx.Graph."
            return draw_graphical_model(
                G,
                pos=pos,
                width_scale=width_scale,
                node_color=node_color,
                edge_colors=edge_colors,
                **kwds,
            )

    def get_transformer(
        self,
    ):
        check_is_fitted(
            self,
            attributes=['coef_', 'classes_'],
        )

        return self.coef_

    def get_pcor(
        self,
    ):
        check_is_fitted(
            self,
            attributes=['coef_', 'classes_'],
        )

        return self.coef_['pcor']

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        """
        Return output names in the same order as transform().
        """
        check_is_fitted(
            self,
            attributes=['coef_', 'feature_names_in_'],
        )
        if input_features is not None and not np.array_equal(
            np.asarray(
                input_features,
                dtype=object,
            ),
            self.feature_names_in_,
        ):
            raise ValueError("input_features must match the fitted column names.")
        return np.asarray(
            self.coef_['nodes'] + list(self.coef_['edges']),
            dtype=object,
        )

