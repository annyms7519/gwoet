import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from gwoet import (
    DiscreteNaiveBayes,
    apply_disc,
    discrete_naive_bayes,
    generate_disc,
    get_iv,
    predict_dnb,
)

@pytest.fixture(scope="module")
def iris_binary_data():
    """Return the mixed-type binary Iris data used by the DNB tests."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    X["new_var1"] = pd.Series(["a"] * 50 + ["b"] * (len(X) - 50))
    X["new_var2"] = pd.Series([1] * 80 + [2] * (len(X) - 80))

    # Include missing values to verify that the public API handles them.
    X.iloc[1, :] = None
    X.iloc[2, 2] = np.nan
    X.iloc[2, 4] = None

    y = pd.Series(iris.target, name="target") == 2
    return X, y


def test_generate_disc(iris_binary_data):
    X, y = iris_binary_data
    disc = generate_disc(X, y)

    assert isinstance(disc, dict)
    assert set(disc) == set(X.columns)
    assert disc["new_var1"] is None

    np.testing.assert_allclose(
        disc["new_var2"],
        np.array([-np.inf, 1.5, np.inf]),
    )
    np.testing.assert_allclose(
        disc["petal length (cm)"],
        np.array([-np.inf, 4.75, 5.05, np.inf]),
    )
    np.testing.assert_allclose(
        disc["petal width (cm)"],
        np.array([-np.inf, 1.35, 1.75, np.inf]),
    )
    np.testing.assert_allclose(
        disc["sepal length (cm)"],
        np.array([-np.inf, 5.75, 7.05, np.inf]),
    )
    np.testing.assert_allclose(
        disc["sepal width (cm)"],
        np.array([-np.inf, 3.35, np.inf]),
    )


def test_apply_disc(iris_binary_data):
    X, y = iris_binary_data
    disc = generate_disc(X, y)
    X_d = apply_disc(disc, X)

    assert isinstance(X_d, pd.DataFrame)
    assert X_d.shape == X.shape
    assert X_d.index.equals(X.index)
    assert list(X_d.columns) == list(X.columns)

    # The original categorical variable should remain categorical in value.
    assert X_d.loc[0, "new_var1"] == "a"
    assert X_d.loc[149, "new_var1"] == "b"

    # Missing values inserted above should remain missing after discretization.
    assert X_d.loc[1].isna().all()
    assert pd.isna(X_d.loc[2, "petal length (cm)"])
    assert pd.isna(X_d.loc[2, "new_var1"])


def test_discrete_naive_bayes(iris_binary_data):
    X, y = iris_binary_data
    model = discrete_naive_bayes(X, y)

    assert isinstance(model, dict)
    assert set(model) == {"disc", "train"}

    train = model["train"]
    assert "probability" in train
    assert "raw_score" in train
    assert "woe" in train
    assert "woe_table" in train
    assert "y_bool" in train

    probability = np.asarray(train["probability"])
    raw_score = np.asarray(train["raw_score"])

    assert probability.shape == (len(X),)
    assert raw_score.shape == (len(X),)
    assert np.all(np.isfinite(probability))
    assert np.all((0.0 <= probability) & (probability <= 1.0))


def test_predict_dnb(iris_binary_data):
    X, y = iris_binary_data
    model = discrete_naive_bayes(X, y)
    pred = predict_dnb(model, X)

    assert isinstance(pred, dict)
    assert "probability" in pred
    assert "raw_score" in pred

    probability = np.asarray(pred["probability"])
    raw_score = np.asarray(pred["raw_score"])

    assert probability.shape == (len(X),)
    assert raw_score.shape == (len(X),)
    assert np.all(np.isfinite(probability))
    assert np.all((0.0 <= probability) & (probability <= 1.0))

    # Predicting the training data should reproduce the stored training scores.
    np.testing.assert_allclose(probability, model["train"]["probability"])
    np.testing.assert_allclose(raw_score, model["train"]["raw_score"])


def test_get_iv(iris_binary_data):
    X, y = iris_binary_data
    model = discrete_naive_bayes(X, y)
    iv = get_iv(model)

    assert isinstance(iv, dict)
    assert set(iv) == set(X.columns)
    assert all(np.isfinite(value) for value in iv.values())
    assert all(value >= 0.0 for value in iv.values())

    # Regression checks for representative variables from the original test.
    assert iv["new_var1"] == pytest.approx(1.8811522724412346)
    assert iv["petal width (cm)"] == pytest.approx(6.294723585979579)


def test_discrete_naive_bayes_estimator(iris_binary_data):
    X, y = iris_binary_data
    model = DiscreteNaiveBayes()
    model.fit(X, y)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    score = model.score(X, y)
    iv = model.get_iv()
    model_dict = model.get_model()

    assert np.asarray(y_pred).shape == (len(X),)
    assert np.asarray(y_pred).dtype == np.bool_

    assert proba.shape == (len(X), 2)
    assert np.all(np.isfinite(proba))
    assert np.all((0.0 <= proba) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    assert score == pytest.approx(0.98)

    assert isinstance(iv, dict)
    assert set(iv) == set(X.columns)

    assert isinstance(model_dict, dict)
    assert set(model_dict) == {"disc", "train"}


def test_estimator_matches_function_api(iris_binary_data):
    X, y = iris_binary_data

    function_model = discrete_naive_bayes(X, y)
    estimator = DiscreteNaiveBayes()
    estimator.fit(X, y)

    function_pred = predict_dnb(function_model, X)["probability"]
    estimator_pred = estimator.predict_proba(X)[:, 1]

    np.testing.assert_allclose(estimator_pred, function_pred)
    assert estimator.get_iv() == pytest.approx(get_iv(function_model))

