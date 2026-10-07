"""selection.py

This module contains logistic-regression feature selection and coefficient
visualization utilities separated from the GWoET structural-learning module.

select_variables_l1_logistic() accepts raw X and y, performs WoE-based
selection internally, and L1LogisticSelector provides a scikit-learn-compatible
feature-filtering transformer for Pipeline and cross-validation workflows.


Author: annyms7519
Created: 2024-07-05
Last modified: 2026-10-07
Version: 0.9.0.5
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_consistent_length, check_is_fitted


# Package import; a standalone copy can use a sibling dnb.py.
if __package__:
    from .dnb import (
        apply_disc,
        calculate_cross_fitted_woe,
        calculate_woe,
        generate_disc,
        make_sample_weight,
        make_y_bool,
    )
else:
    from dnb import (
        apply_disc,
        calculate_cross_fitted_woe,
        calculate_woe,
        generate_disc,
        make_sample_weight,
        make_y_bool,
    )


def _validate_real(
    value,
    name,
    *,
    minimum=0.0,
    strict=False,
):
    """Validate a finite real scalar."""
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise TypeError(f"{name} must be a real number.")

    value = float(value)
    if (
        not np.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        relation = "greater than" if strict else "at least"
        raise ValueError(
            f"{name} must be finite and {relation} {minimum}."
        )
    return value


def select_variables_l1_logistic(
    X,
    y,
    *,
    weight=None,
    discretizer="bic",
    cross_fitting=True,
    n_splits=5,
    shuffle=True,
    Cs=None,
    class_weight=None,
    max_iter=1000,
    optimization_tol=1e-3,
    tol_coef=1e-8,
    random_state=None,
):
    """
    Select variables using WoE features and L1-penalized logistic regression.

    The input features are first discretized and transformed into Weight of
    Evidence (WoE) values. Cross-fitted WoE values are used by default for
    the training data. For each candidate C, an L1-penalized logistic
    regression model is then fitted to the WoE features. BIC is calculated
    from the unpenalized log-likelihood of each fitted model, and the C with
    the minimum BIC is selected.

    Args:
      X:
        DataFrame containing numerical or categorical variables.
      y:
        One-dimensional binary target variable with the same number of
        observations as X.
      weight:
        None or one-dimensional array-like sample weights with the same
        number of observations as X.
      discretizer:
        Discretization method. Must be "bic" or "mdlp".
      cross_fitting:
        Whether to use cross-fitted WoE values for the training data.
      n_splits:
        Number of folds used for cross-fitting. Must be at least 2.
      shuffle:
        Whether to shuffle observations before constructing cross-fitting
        folds.
      Cs:
        Candidate inverse regularization strengths. If None, 20 values
        logarithmically spaced from 1e-3 to 1e1 are used. If an integer,
        that number of values over the same range is used.
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
        Random seed, RandomState instance, or None.

    Returns:
      Dict containing the fitted logistic-regression model, coefficients,
      selected variables, selected coefficients, selected C, BIC, and the
      BIC path.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain observations and features.")

    if not X.columns.is_unique:
        raise ValueError("X must have unique column names.")

    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if len(y_array) != len(X):
        raise ValueError("y must have the same length as X.")
    if pd.isna(y_array).any() or len(pd.unique(y_array)) != 2:
        raise ValueError("y must contain exactly two nonmissing classes.")

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

    if (
        random_state is not None
        and not isinstance(random_state, (int, np.integer, np.random.RandomState))
    ):
        raise TypeError(
            "random_state must be an integer, RandomState instance, or None."
        )
    if (
        isinstance(random_state, (int, np.integer))
        and not isinstance(random_state, (bool, np.bool_))
        and random_state < 0
    ):
        raise ValueError("random_state must be non-negative.")
    if isinstance(random_state, (bool, np.bool_)):
        raise TypeError(
            "random_state must be an integer, RandomState instance, or None."
        )

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

    if not isinstance(woe_result, dict) or "woe_table" not in woe_result:
        raise ValueError("WoE calculation must return a dict containing 'woe_table'.")

    X_woe = woe_result["woe_table"]

    return _select_variables_l1_logistic(
        X_woe,
        y,
        weight=weight,
        Cs=Cs,
        class_weight=class_weight,
        max_iter=max_iter,
        optimization_tol=optimization_tol,
        tol_coef=tol_coef,
        random_state=random_state,
    )


def _select_variables_l1_logistic(
    X_woe,
    y,
    *,
    weight=None,
    Cs=None,
    class_weight=None,
    max_iter=1000,
    optimization_tol=1e-3,
    tol_coef=1e-8,
    random_state=None,
):
    """
    Select variables from a precomputed WoE table using L1 logistic regression.

    This is the low-level implementation used by
    select_variables_l1_logistic(). It assumes that WoE transformation has
    already been completed.
    """
    if not isinstance(X_woe, pd.DataFrame):
        raise TypeError("X_woe must be a pandas DataFrame.")

    if X_woe.shape[1] == 0:
        raise ValueError("X_woe must contain at least one variable.")

    if not X_woe.columns.is_unique:
        raise ValueError("X_woe must have unique column names.")

    tol_coef = _validate_real(tol_coef, "tol_coef")
    optimization_tol = _validate_real(
        optimization_tol,
        "optimization_tol",
        strict=True,
    )

    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer.")

    check_consistent_length(X_woe, y)

    if weight is not None:
        check_consistent_length(X_woe, weight)
        weight = np.asarray(
            weight,
            dtype=np.float64,
        )
        if weight.ndim != 1 or not np.isfinite(weight).all():
            raise ValueError("weight must be a finite one-dimensional array.")
        if (weight < 0).any() or not (weight > 0).any():
            raise ValueError(
                "weight must be nonnegative with a positive total."
            )

    X_values = np.asarray(
        X_woe.to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        order="C",
    )

    if not np.isfinite(X_values).all():
        raise ValueError("X_woe must contain only finite numeric values.")

    y_array = np.asarray(y)
    if (
        y_array.ndim != 1
        or pd.isna(y_array).any()
        or len(pd.unique(y_array)) != 2
    ):
        raise ValueError("y must contain exactly two nonmissing classes.")

    y_bool = np.asarray(
        make_y_bool(y),
        dtype=bool,
    )

    X_values = np.ascontiguousarray(X_values)

    if y_bool.ndim != 1 or len(np.unique(y_bool)) != 2:
        raise ValueError("y must contain exactly two classes.")

    if Cs is None:
        Cs = np.logspace(-3, 1, 20)
    elif (
        isinstance(Cs, (int, np.integer))
        and not isinstance(Cs, (bool, np.bool_))
    ):
        if Cs < 1:
            raise ValueError("Cs must be a positive integer.")
        Cs = np.logspace(-3, 1, int(Cs))
    else:
        if isinstance(Cs, (bool, np.bool_)):
            raise TypeError("Cs must not be boolean.")
        Cs = np.asarray(
            Cs,
            dtype=float,
        )
        if (
            Cs.ndim != 1
            or not Cs.size
            or not np.isfinite(Cs).all()
            or (Cs <= 0).any()
        ):
            raise ValueError(
                "Cs must contain finite positive values in a 1D sequence."
            )

    n_bic = len(y_bool) if weight is None else np.sum(weight)

    results = []
    best_model = None
    best_bic = np.inf

    for C in Cs:
        model = LogisticRegression(
            C=float(C),
            penalty="l1",
            solver="liblinear",
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
        n_nonzero = np.sum(
            np.abs(coef_values) > tol_coef
        )
        n_params = int(n_nonzero + 1)

        z = model.decision_function(X_values)
        log_likelihood_i = (
            y_bool.astype(np.float64) * z
            - np.logaddexp(0.0, z)
        )

        if weight is None:
            log_likelihood = np.sum(log_likelihood_i)
        else:
            log_likelihood = np.sum(weight * log_likelihood_i)

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


class L1LogisticSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible feature selector based on WoE + L1 logistic BIC.

    During fit, variables are selected by select_variables_l1_logistic().
    During transform, the selector returns the selected columns from the
    original input X; it does not return WoE values. This makes the selector
    suitable as a filtering step in a scikit-learn Pipeline before a separate
    preprocessing or prediction step.

    Parameters
    ----------
    discretizer : {"bic", "mdlp"}, default="bic"
        Discretization method used to construct WoE features for selection.
    cross_fitting : bool, default=True
        Whether to use cross-fitted WoE values during selection.
    n_splits : int, default=5
        Number of folds used for cross-fitted WoE.
    shuffle : bool, default=True
        Whether to shuffle observations before WoE cross-fitting.
    Cs : None, int, or 1D array-like, default=None
        Candidate inverse regularization strengths for L1 logistic regression.
    class_weight : dict, "balanced", or None, default=None
        Class weights passed to LogisticRegression.
    max_iter : int, default=1000
        Maximum number of solver iterations.
    optimization_tol : float, default=1e-3
        Stopping tolerance used by the optimizer.
    tol_coef : float, default=1e-8
        Coefficient threshold used to define selected variables.
    random_state : int, RandomState, or None, default=None
        Random state used for WoE cross-fitting and logistic regression.

    Notes
    -----
    In an outer cross-validation Pipeline, fit() is called only on each
    training fold. Therefore the variable selection is re-estimated within
    each fold rather than once on the full dataset.
    """

    def __init__(
        self,
        *,
        discretizer="bic",
        cross_fitting=True,
        n_splits=5,
        shuffle=True,
        Cs=None,
        class_weight=None,
        max_iter=1000,
        optimization_tol=1e-3,
        tol_coef=1e-8,
        random_state=None,
    ):
        self.discretizer = discretizer
        self.cross_fitting = cross_fitting
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.Cs = Cs
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.optimization_tol = optimization_tol
        self.tol_coef = tol_coef
        self.random_state = random_state

    def fit(self, X, y, weight=None):
        """Fit the selector and determine which original columns to retain."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain observations and features.")
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")

        result = select_variables_l1_logistic(
            X,
            y,
            weight=weight,
            discretizer=self.discretizer,
            cross_fitting=self.cross_fitting,
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            Cs=self.Cs,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            optimization_tol=self.optimization_tol,
            tol_coef=self.tol_coef,
            random_state=self.random_state,
        )

        selected_variables_ranked = list(result["selected_variables"])
        unknown = [
            var for var in selected_variables_ranked
            if var not in X.columns
        ]
        if unknown:
            raise ValueError(
                "Selected variables must be columns of X: "
                f"{unknown}."
            )

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.selection_result_ = result
        self.selected_variables_ranked_ = selected_variables_ranked
        selected_set = set(selected_variables_ranked)
        self.selected_variables_ = [
            name for name in self.feature_names_in_
            if name in selected_set
        ]
        self.support_ = np.asarray(
            [name in selected_set for name in self.feature_names_in_],
            dtype=bool,
        )

        # Convenient fitted attributes mirroring the selection result.
        self.model_ = result["model"]
        self.coef_ = result["coef"]
        self.selected_coef_ = result["selected_coef"]
        self.best_C_ = result["best_C"]
        self.best_bic_ = result["best_bic"]
        self.bic_path_ = result["bic_path"]
        self.n_iter_ = result["n_iter"]

        return self

    def transform(self, X):
        """Return the selected columns from the original feature matrix."""
        check_is_fitted(self, "support_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")

        expected = list(self.feature_names_in_)
        actual = list(X.columns)
        if set(actual) != set(expected):
            missing = [name for name in expected if name not in X.columns]
            extra = [name for name in actual if name not in self.feature_names_in_]
            raise ValueError(
                "X must contain the same feature names used during fit. "
                f"Missing: {missing}; extra: {extra}."
            )

        # Select by name so a harmless column-order change does not alter the
        # fitted support mask.
        return X.loc[:, self.selected_variables_].copy()

    def get_support(self, indices=False):
        """Return a boolean mask or integer indices for selected variables."""
        check_is_fitted(self, "support_")
        if not isinstance(indices, (bool, np.bool_)):
            raise TypeError("indices must be a boolean.")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        """Return the names of the selected original variables."""
        check_is_fitted(self, "support_")

        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if input_features.ndim != 1:
                raise ValueError("input_features must be one-dimensional.")
            if not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError(
                    "input_features must match the feature names used during fit."
                )

        return np.asarray(self.selected_variables_, dtype=object)


def make_logistic_coefficient_dict(
    model,
    prep,
    X,
):
    """
    Extract logistic regression coefficients grouped by original variables.

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


def plot_logistic_coefficients(
    coef,
    variables=None,
    exclude_zero=True,
):
    """
    Plot logistic regression coefficients by original variable.

    Numerical variables are displayed in one plot. Each categorical
    variable is displayed in a separate plot.

    Args:
      coef:
        Coefficient dictionary generated by
        make_logistic_coefficient_dict().
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

        if not any(
            var in all_variables
            for var in variables
        ):
            return None

    # Numerical features
    numeric_coef = coef.get(
        "numeric",
        {},
    )

    if variables is not None:
        numeric_coef = {
            key: value
            for key, value in numeric_coef.items()
            if key in variables
        }

    if numeric_coef:
        _plot_coefficient_bar(
            coefficients=list(numeric_coef.values()),
            labels=list(numeric_coef.keys()),
            ylabel="Numeric features",
            exclude_zero=exclude_zero,
        )

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
        _plot_coefficient_bar(
            coefficients=list(categories.values()),
            labels=list(categories.keys()),
            ylabel=var,
            exclude_zero=exclude_zero,
        )


def plot_logistic_feature_coefficients(
    coefficients,
    feature_names,
    top_n=None,
    exclude_zero=True,
):
    """
    Plot logistic regression coefficients by transformed feature.

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
    result = _plot_coefficient_bar(
        coefficients=coefficients,
        labels=feature_names,
        top_n=top_n,
        exclude_zero=exclude_zero,
    )

    return result.rename(
        columns={"label": "feature"}
    )


def _plot_coefficient_bar(
    coefficients,
    labels,
    *,
    ylabel=None,
    top_n=None,
    exclude_zero=True,
):
    """
    Plot coefficients as a horizontal bar chart.

    Coefficients are ordered by absolute magnitude.

    Args:
      coefficients:
        One-dimensional array-like coefficients.
      labels:
        One-dimensional array-like labels corresponding to coefficients.
      ylabel:
        Label for the y-axis.
      top_n:
        Number of coefficients to display.
        If None, all coefficients are displayed.
      exclude_zero:
        Whether to exclude zero coefficients.

    Returns:
      DataFrame containing the displayed labels and coefficients.
    """
    coefficients = np.asarray(coefficients)
    labels = np.asarray(labels)

    if coefficients.ndim != 1:
        raise ValueError(
            "coefficients must be one-dimensional."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if len(coefficients) != len(labels):
        raise ValueError(
            "coefficients and labels must have the same length."
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
        "label": labels,
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
        y="label",
        order=plot_data["label"],
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
        ylabel=ylabel,
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
