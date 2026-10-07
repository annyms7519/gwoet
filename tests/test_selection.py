"""
Test logistic feature selection and coefficient utilities.

Place this file in tests/ and run from the repository root:
    python -m pytest -q tests/test_selection.py
"""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from gwoet import (
    make_logistic_coefficient_dict,
    make_y_bool,
    plot_logistic_coefficients,
    plot_logistic_feature_coefficients,
    select_variables_l1_logistic,
)
from gwoet.selection import L1LogisticSelector

# Access private helpers in the actual implementation module.
s = import_module(select_variables_l1_logistic.__module__)


def test_public_selection_functions_come_from_selection_module():
    functions = (
        select_variables_l1_logistic,
        make_logistic_coefficient_dict,
        plot_logistic_coefficients,
        plot_logistic_feature_coefficients,
    )
    for func in functions:
        assert func.__module__.endswith(".selection")


@pytest.fixture
def woe_data():
    rng = np.random.RandomState(13)
    X_woe = pd.DataFrame(
        rng.normal(size=(250, 4)),
        columns=["a", "b", "a*b", "a*c"],
    )
    z = 1.4 * X_woe["a"] - 0.7 * X_woe["b"] + 0.9 * X_woe["a*c"]
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return X_woe, y


def test_public_selection_constructs_woe_before_l1(monkeypatch, woe_data):
    X_woe, y = woe_data
    X = pd.DataFrame({
        "a": np.arange(len(y), dtype=float),
        "b": np.arange(len(y), dtype=float) % 3,
        "a*b": np.arange(len(y), dtype=float) % 5,
        "a*c": np.arange(len(y), dtype=float) % 7,
    })
    weight = np.linspace(0.5, 1.5, len(X))

    calls = {}

    def fake_make_sample_weight(value, n):
        calls["weight_input"] = value
        calls["n"] = n
        return weight

    def fake_generate_disc(X_arg, y_arg, *, discretizer, weight):
        calls["generate_disc"] = (X_arg, np.asarray(y_arg), discretizer, weight)
        return {"disc": "ok"}

    def fake_apply_disc(disc, X_arg):
        calls["apply_disc"] = (disc, X_arg)
        return X_arg.copy()

    def fake_cross_fitted(X_d, y_arg, weight_arg, **kwargs):
        calls["cross_fitted"] = (X_d, np.asarray(y_arg), weight_arg, kwargs)
        return {
            "woe_table": X_woe.copy(),
            "y_bool": pd.Series(np.asarray(y_arg, dtype=bool), index=X_d.index),
        }

    def fake_low_level(X_arg, y_arg, **kwargs):
        calls["low_level"] = (X_arg, np.asarray(y_arg), kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(s, "make_sample_weight", fake_make_sample_weight)
    monkeypatch.setattr(s, "generate_disc", fake_generate_disc)
    monkeypatch.setattr(s, "apply_disc", fake_apply_disc)
    monkeypatch.setattr(s, "calculate_cross_fitted_woe", fake_cross_fitted)
    monkeypatch.setattr(s, "_select_variables_l1_logistic", fake_low_level)

    result = select_variables_l1_logistic(
        X,
        y,
        weight=weight,
        discretizer="mdlp",
        cross_fitting=True,
        n_splits=4,
        shuffle=False,
        Cs=[0.1, 1.0],
        class_weight="balanced",
        max_iter=500,
        optimization_tol=1e-4,
        tol_coef=1e-7,
        random_state=13,
    )

    assert result == {"sentinel": True}
    assert calls["n"] == len(X)
    assert calls["generate_disc"][2] == "mdlp"
    assert calls["generate_disc"][3] is weight
    assert calls["apply_disc"][0] == {"disc": "ok"}
    np.testing.assert_array_equal(calls["low_level"][0], X_woe)

    cross_kwargs = calls["cross_fitted"][3]
    assert cross_kwargs == {
        "n_splits": 4,
        "shuffle": False,
        "random_state": 13,
    }

    low_kwargs = calls["low_level"][2]
    assert low_kwargs["weight"] is weight
    assert low_kwargs["Cs"] == [0.1, 1.0]
    assert low_kwargs["class_weight"] == "balanced"
    assert low_kwargs["max_iter"] == 500
    assert low_kwargs["optimization_tol"] == pytest.approx(1e-4)
    assert low_kwargs["tol_coef"] == pytest.approx(1e-7)
    assert low_kwargs["random_state"] == 13




def _fake_selection_result(X, selected):
    coef = pd.Series(
        [1.0 if name in selected else 0.0 for name in X.columns],
        index=X.columns,
        name="coefficient",
    )
    selected_coef = coef.loc[selected]
    return {
        "model": SimpleNamespace(coef_=coef.to_numpy()[None, :]),
        "coef": coef,
        "selected_variables": list(selected),
        "selected_coef": selected_coef,
        "best_C": 0.3,
        "best_bic": 123.0,
        "bic_path": pd.DataFrame({"C": [0.3], "bic": [123.0]}),
        "n_iter": np.array([5]),
    }


def test_l1_logistic_selector_fit_transform_and_support(monkeypatch, woe_data):
    X, y = woe_data
    calls = []

    def fake_select(X_arg, y_arg, **kwargs):
        calls.append((X_arg.copy(), np.asarray(y_arg), kwargs))
        # Return ranked variables in an order different from X to verify that
        # transform preserves the original feature order.
        return _fake_selection_result(X_arg, ["a*c", "a"])

    monkeypatch.setattr(s, "select_variables_l1_logistic", fake_select)

    selector = L1LogisticSelector(
        discretizer="mdlp",
        cross_fitting=False,
        n_splits=4,
        shuffle=False,
        Cs=[0.1, 1.0],
        class_weight="balanced",
        max_iter=500,
        optimization_tol=1e-4,
        tol_coef=1e-7,
        random_state=13,
    )

    transformed = selector.fit_transform(X, y)

    assert list(transformed.columns) == ["a", "a*c"]
    assert transformed.index.equals(X.index)
    np.testing.assert_array_equal(
        selector.get_support(),
        [True, False, False, True],
    )
    np.testing.assert_array_equal(
        selector.get_support(indices=True),
        [0, 3],
    )
    np.testing.assert_array_equal(
        selector.get_feature_names_out(),
        ["a", "a*c"],
    )
    assert selector.selected_variables_ranked_ == ["a*c", "a"]
    assert selector.selected_variables_ == ["a", "a*c"]
    assert selector.best_C_ == pytest.approx(0.3)
    assert selector.best_bic_ == pytest.approx(123.0)

    assert len(calls) == 1
    kwargs = calls[0][2]
    assert kwargs["discretizer"] == "mdlp"
    assert kwargs["cross_fitting"] is False
    assert kwargs["n_splits"] == 4
    assert kwargs["shuffle"] is False
    assert kwargs["Cs"] == [0.1, 1.0]
    assert kwargs["class_weight"] == "balanced"
    assert kwargs["max_iter"] == 500
    assert kwargs["optimization_tol"] == pytest.approx(1e-4)
    assert kwargs["tol_coef"] == pytest.approx(1e-7)
    assert kwargs["random_state"] == 13


def test_l1_logistic_selector_clone(monkeypatch, woe_data):
    selector = L1LogisticSelector(
        discretizer="mdlp",
        cross_fitting=False,
        n_splits=3,
        Cs=7,
        random_state=13,
    )
    cloned = clone(selector)

    assert cloned.discretizer == "mdlp"
    assert cloned.cross_fitting is False
    assert cloned.n_splits == 3
    assert cloned.Cs == 7
    assert cloned.random_state == 13


def test_l1_logistic_selector_not_fitted(woe_data):
    X, _ = woe_data
    selector = L1LogisticSelector()

    with pytest.raises(NotFittedError):
        selector.transform(X)
    with pytest.raises(NotFittedError):
        selector.get_support()
    with pytest.raises(NotFittedError):
        selector.get_feature_names_out()


def test_l1_logistic_selector_rejects_feature_name_change(monkeypatch, woe_data):
    X, y = woe_data
    monkeypatch.setattr(
        s,
        "select_variables_l1_logistic",
        lambda X_arg, y_arg, **kwargs: _fake_selection_result(X_arg, ["a"]),
    )

    selector = L1LogisticSelector().fit(X, y)
    X_bad = X.drop(columns="b").assign(new=0.0)

    with pytest.raises(ValueError):
        selector.transform(X_bad)


def test_l1_logistic_selector_cross_validation_is_fold_local(monkeypatch):
    rng = np.random.RandomState(31)
    X = pd.DataFrame(
        rng.normal(size=(120, 4)),
        columns=["a", "b", "c", "d"],
        index=np.arange(120) * 2 + 3,
    )
    z = 1.2 * X["a"] - 0.8 * X["b"]
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))

    fit_indices = []

    def fake_select(X_arg, y_arg, **kwargs):
        fit_indices.append(tuple(X_arg.index))
        return _fake_selection_result(X_arg, ["a", "b"])

    monkeypatch.setattr(s, "select_variables_l1_logistic", fake_select)

    pipe = Pipeline([
        ("select", L1LogisticSelector(Cs=3, random_state=13)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
    scores = cross_val_score(pipe, X, y, cv=cv)

    assert len(scores) == 3
    assert np.isfinite(scores).all()
    assert len(fit_indices) == 3
    assert all(len(index) == 80 for index in fit_indices)
    assert all(set(index) != set(X.index) for index in fit_indices)

def test_public_selection_without_cross_fitting(monkeypatch, woe_data):
    X_woe, y = woe_data
    X = X_woe.copy()
    weight = np.ones(len(X))
    calls = {"full": 0, "cross": 0}

    monkeypatch.setattr(s, "make_sample_weight", lambda value, n: weight)
    monkeypatch.setattr(s, "generate_disc", lambda *a, **k: {})
    monkeypatch.setattr(s, "apply_disc", lambda disc, X_arg: X_arg)

    def fake_full(X_d, y_arg, weight_arg):
        calls["full"] += 1
        return {"woe_table": X_woe}

    def fake_cross(*args, **kwargs):
        calls["cross"] += 1
        return {"woe_table": X_woe}

    monkeypatch.setattr(s, "calculate_woe", fake_full)
    monkeypatch.setattr(s, "calculate_cross_fitted_woe", fake_cross)
    monkeypatch.setattr(
        s,
        "_select_variables_l1_logistic",
        lambda *a, **k: {"ok": True},
    )

    result = select_variables_l1_logistic(
        X,
        y,
        cross_fitting=False,
        Cs=3,
    )

    assert result == {"ok": True}
    assert calls == {"full": 1, "cross": 0}


@pytest.mark.parametrize("sample_weight", [False, True])
def test_low_level_l1_and_bic(woe_data, sample_weight):
    X_woe, y = woe_data
    weight = np.linspace(0.5, 2.0, len(X_woe)) if sample_weight else None

    result = s._select_variables_l1_logistic(
        X_woe,
        y,
        weight=weight,
        Cs=[0.03, 0.3, 3.0],
        random_state=13,
    )

    z = result["model"].decision_function(X_woe.to_numpy())
    z_from_coef = (
        X_woe.to_numpy() @ result["coef"].to_numpy()
        + result["model"].intercept_[0]
    )
    np.testing.assert_allclose(z, z_from_coef, atol=1e-12)

    assert result["best_bic"] == pytest.approx(result["bic_path"].bic.min())
    assert "penalty_weights" not in result
    assert "coef_scaled" not in result

    y_bool = np.asarray(make_y_bool(y), dtype=bool)
    reference = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=result["best_C"],
        tol=1e-3,
        max_iter=1000,
        random_state=13,
    ).fit(
        X_woe.to_numpy(),
        y_bool,
        sample_weight=weight,
    )
    np.testing.assert_allclose(reference.coef_, result["model"].coef_)

    loglik = y_bool.astype(float) * z - np.logaddexp(0.0, z)
    n_bic = len(y) if weight is None else weight.sum()
    ll = loglik.sum() if weight is None else (weight * loglik).sum()
    n_params = int((np.abs(result["coef"]) > 1e-8).sum()) + 1

    np.testing.assert_allclose(
        result["best_bic"],
        -2.0 * ll + n_params * np.log(n_bic),
    )


def test_low_level_output(woe_data):
    X_woe, y = woe_data
    result = s._select_variables_l1_logistic(
        X_woe,
        y,
        Cs=5,
        random_state=13,
    )

    assert set(result) == {
        "model",
        "coef",
        "selected_variables",
        "selected_coef",
        "best_C",
        "best_bic",
        "bic_path",
        "n_iter",
    }
    assert list(result["coef"].index) == list(X_woe.columns)
    assert set(result["selected_variables"]) <= set(X_woe.columns)
    assert list(result["selected_coef"].index) == result["selected_variables"]


@pytest.mark.parametrize(
    "Cs",
    [True, 0, [], [0.1, -1], [np.inf], [[1]], np.nan],
)
def test_invalid_Cs(woe_data, Cs):
    X_woe, y = woe_data
    with pytest.raises((TypeError, ValueError)):
        s._select_variables_l1_logistic(
            X_woe,
            y,
            Cs=Cs,
        )


@pytest.mark.parametrize(
    "tol_coef",
    [-1, True, np.bool_(True), np.nan, np.inf, "bad"],
)
def test_invalid_tol_coef(woe_data, tol_coef):
    X_woe, y = woe_data
    with pytest.raises((TypeError, ValueError)):
        s._select_variables_l1_logistic(
            X_woe,
            y,
            tol_coef=tol_coef,
        )


def test_low_level_rejects_nonfinite_woe(woe_data):
    X_woe, y = woe_data
    X_woe = X_woe.copy()
    X_woe.loc[0, "a"] = np.nan

    with pytest.raises(ValueError):
        s._select_variables_l1_logistic(
            X_woe,
            y,
            Cs=3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"discretizer": "bad"},
        {"cross_fitting": 1},
        {"n_splits": 1},
        {"n_splits": True},
        {"shuffle": 1},
        {"random_state": -1},
        {"random_state": True},
        {"random_state": "bad"},
    ],
)
def test_invalid_public_preprocessing_arguments(woe_data, kwargs):
    X, y = woe_data
    with pytest.raises((TypeError, ValueError)):
        select_variables_l1_logistic(
            X,
            y,
            **kwargs,
        )


def test_make_logistic_coefficient_dict():
    X = pd.DataFrame({
        "num": [0.0, 1.0, 2.0],
        "cat": ["x", "y", "x"],
    })

    model = SimpleNamespace(
        coef_=np.array([[1.5, -0.2, 0.3]])
    )
    encoder = SimpleNamespace(
        categories_=[np.array(["x", "y"], dtype=object)]
    )
    cat_pipeline = SimpleNamespace(
        named_steps={"encoder": encoder}
    )
    prep = SimpleNamespace(
        named_transformers_={"cat": cat_pipeline}
    )

    coef = make_logistic_coefficient_dict(
        model,
        prep,
        X,
    )

    assert coef == {
        "numeric": {"num": 1.5},
        "categorical": {
            "cat": {
                "x": -0.2,
                "y": 0.3,
            }
        },
    }


def test_make_logistic_coefficient_dict_numeric_only():
    X = pd.DataFrame({
        "a": [0.0, 1.0],
        "b": [1, 2],
    })
    model = SimpleNamespace(
        coef_=np.array([[0.5, -0.25]])
    )
    prep = SimpleNamespace(
        named_transformers_={}
    )

    coef = make_logistic_coefficient_dict(
        model,
        prep,
        X,
    )

    assert coef == {
        "numeric": {
            "a": 0.5,
            "b": -0.25,
        },
        "categorical": {},
    }


def test_plot_logistic_coefficients_dispatch(monkeypatch):
    coef = {
        "numeric": {
            "a": 1.0,
            "b": -0.5,
        },
        "categorical": {
            "cat": {
                "x": 0.2,
                "y": -0.3,
            },
        },
    }

    calls = []

    def fake_plot(*, coefficients, labels, ylabel=None, top_n=None, exclude_zero=True):
        calls.append({
            "coefficients": list(coefficients),
            "labels": list(labels),
            "ylabel": ylabel,
            "top_n": top_n,
            "exclude_zero": exclude_zero,
        })
        return pd.DataFrame({
            "label": labels,
            "coefficient": coefficients,
        })

    monkeypatch.setattr(s, "_plot_coefficient_bar", fake_plot)

    result = plot_logistic_coefficients(
        coef,
        variables=["a", "cat"],
        exclude_zero=False,
    )

    assert result is None
    assert calls == [
        {
            "coefficients": [1.0],
            "labels": ["a"],
            "ylabel": "Numeric features",
            "top_n": None,
            "exclude_zero": False,
        },
        {
            "coefficients": [0.2, -0.3],
            "labels": ["x", "y"],
            "ylabel": "cat",
            "top_n": None,
            "exclude_zero": False,
        },
    ]


def test_plot_logistic_coefficients_unknown_variables(monkeypatch):
    called = False

    def fake_plot(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(s, "_plot_coefficient_bar", fake_plot)

    result = plot_logistic_coefficients(
        {
            "numeric": {"a": 1.0},
            "categorical": {},
        },
        variables=["unknown"],
    )

    assert result is None
    assert called is False


def test_plot_logistic_feature_coefficients_order(monkeypatch):
    # Avoid depending on the active plotting backend / seaborn color defaults.
    monkeypatch.setattr(s.plt, "show", lambda: None)
    monkeypatch.setattr(s.sns, "barplot", lambda **kwargs: None)

    result = plot_logistic_feature_coefficients(
        coefficients=[0.1, -1.2, 0.0, 0.5],
        feature_names=["a", "b", "c", "d"],
        top_n=2,
        exclude_zero=True,
    )

    assert list(result["feature"]) == ["b", "d"]
    np.testing.assert_allclose(
        result["coefficient"].to_numpy(),
        [-1.2, 0.5],
    )

    s.plt.close("all")


@pytest.mark.parametrize(
    "top_n",
    [0, -1, True, 1.5, "2"],
)
def test_invalid_plot_top_n(top_n):
    with pytest.raises((TypeError, ValueError)):
        plot_logistic_feature_coefficients(
            coefficients=[1.0, -0.5],
            feature_names=["a", "b"],
            top_n=top_n,
        )


def test_plot_feature_coefficient_length_mismatch():
    with pytest.raises(ValueError):
        plot_logistic_feature_coefficients(
            coefficients=[1.0],
            feature_names=["a", "b"],
        )
