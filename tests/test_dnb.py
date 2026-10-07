"""
Test discrete naive Bayes through the actual GWoET package.

Place this file in tests/ and run from the repository root:
    python -m pytest -q tests/test_dnb.py

Public functions and the class are imported from gwoet.
Internal helpers are accessed through the actual implementation module.
"""
from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score


from gwoet import (
    DiscreteNaiveBayes,
    apply_disc,
    discrete_naive_bayes,
    generate_disc,
    get_accuracy_score,
    get_iv,
    is_discrete_naive_bayes,
    predict_dnb,
)


# Access helpers in the actual module defining the public training function.
d = import_module(discrete_naive_bayes.__module__)

RS = 13


@pytest.fixture
def mixed_data():
    """
    Create binary Iris data with numerical, categorical, and missing values.

    A nonconsecutive index detects accidental index replacement.
    The data are split deterministically within each class.
    """
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = (iris.target == 2).astype(int)

    X["new_var1"] = np.where(
        np.arange(len(X)) % 2 == 0,
        "a",
        "b",
    )
    X["new_var2"] = np.tile(
        [1, 2],
        len(X) // 2,
    )
    X["constant"] = 1.0
    X["all_missing"] = np.nan

    X.iloc[0] = np.nan
    X.iloc[2, 2] = np.nan
    X.loc[4, "new_var1"] = None

    index = pd.Index(
        np.arange(len(X)) * 3 + 7,
        name="row",
    )
    X.index = index
    y.index = index
    y.name = "target"

    test_mask = np.arange(len(X)) % 5 == 4

    return (
        X.loc[~test_mask].copy(),
        y.loc[~test_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[test_mask].copy(),
    )


@pytest.fixture
def categorical_data():
    """
    Create categories with unequal class frequencies and missing values.
    """
    X = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["a", "a", "a", "b", "b", "b", None, "a"],
                categories=["a", "b", "unused"],
            ),
        },
        index=pd.Index([8, 3, 14, 2, 20, 5, 11, 6]),
    )
    y = pd.Series(
        [0, 0, 1, 0, 1, 1, 1, 1],
        index=X.index,
        name="target",
    )
    weight = np.array(
        [1, 2, 3, 4, 5, 0, 10, 0],
        dtype=float,
    )

    return X, y, weight


@pytest.mark.parametrize(
    "discretizer",
    ["bic", "mdlp"],
)
def test_discretization_preserves_columns_index_and_missing(
    mixed_data,
    discretizer,
):
    """
    Fit discretizers and apply them consistently to training and test rows.
    """
    X, y, X_test, _ = mixed_data

    disc = generate_disc(
        X,
        y,
        discretizer=discretizer,
    )
    X_d = apply_disc(
        disc,
        X,
    )
    X_test_d = apply_disc(
        disc,
        X_test,
    )

    assert set(disc) == set(X.columns)
    assert disc["new_var1"] is None
    assert X_d.index.equals(X.index)
    assert X_test_d.index.equals(X_test.index)
    assert list(X_d) == list(X)
    pd.testing.assert_frame_equal(
        X_d.isna(),
        X.isna(),
    )

    for column in X:
        assert isinstance(
            X_d[column].dtype,
            pd.CategoricalDtype,
        )
        if disc[column] is not None:
            assert np.isneginf(disc[column][0])
            assert np.isposinf(disc[column][-1])
            assert np.all(np.diff(disc[column]) > 0)
            assert X_d[column].cat.categories.equals(
                X_test_d[column].cat.categories
            )

    for column in ["constant", "all_missing"]:
        np.testing.assert_array_equal(
            disc[column],
            [-np.inf, np.inf],
        )


@pytest.mark.parametrize(
    "discretizer",
    ["bic", "mdlp"],
)
def test_discretization_has_known_boundary(
    discretizer,
):
    """
    Recover the separating boundary of two strongly supported groups.
    """
    X = pd.DataFrame(
        {"x": [0.0] * 30 + [10.0] * 30},
    )
    y = np.array([0] * 30 + [1] * 30)

    disc = generate_disc(
        X,
        y,
        discretizer=discretizer,
    )

    np.testing.assert_allclose(
        disc["x"],
        [-np.inf, 5.0, np.inf],
    )


def test_apply_disc_boundary_unknown_and_missing():
    """
    Respect right-closed bins and preserve new categorical values.
    """
    X = pd.DataFrame(
        {
            "x": [-100.0, 0.5, 0.5001, 100.0, np.nan],
            "category": ["a", "new", "b", None, "a"],
        },
        index=[4, 9, 2, 7, 1],
    )
    disc = {
        "x": np.array([-np.inf, 0.5, np.inf]),
        "category": None,
    }

    result = apply_disc(
        disc,
        X,
    )

    np.testing.assert_array_equal(
        result["x"].cat.codes,
        [0, 0, 1, 1, -1],
    )
    assert result.loc[9, "category"] == "new"
    assert pd.isna(result.loc[7, "category"])
    assert result.index.equals(X.index)


@pytest.mark.parametrize(
    "laplace",
    [0.0, 0.5, 1.0],
)
def test_weighted_woe_matches_hand_calculation(
    categorical_data,
    laplace,
):
    """
    Check WoE against a manually calculated two-category contingency table.

    Missing values and zero-weight rows do not contribute to estimation.
    Known categories still receive their mappings on zero-weight rows.
    """
    X, y, weight = categorical_data

    result = d.calculate_woe(
        X,
        y,
        weight=weight,
        laplace=laplace,
    )

    # Negative category weights are [3, 4], positive weights are [3, 5].
    negative = np.array([3.0, 4.0]) + laplace
    positive = np.array([3.0, 5.0]) + laplace
    expected = np.log(
        (positive / positive.sum())
        / (negative / negative.sum())
    )

    assert result["count"]["group"] == {"a": 3, "b": 2}
    np.testing.assert_allclose(
        [result["woe"]["group"]["a"], result["woe"]["group"]["b"]],
        expected,
    )
    np.testing.assert_allclose(
        result["woe_table"]["group"],
        [expected[0]] * 3 + [expected[1]] * 3 + [0.0, expected[0]],
    )
    assert result["woe_table"].index.equals(X.index)
    assert result["y_bool"].index.equals(X.index)

    pd.testing.assert_frame_equal(
        d.apply_woe(
            X,
            result["woe"],
        ),
        result["woe_table"],
    )


def test_apply_woe_unknown_missing_and_reordered_categories():
    """
    Map by category value rather than category code.

    Missing and unseen categories receive a neutral WoE of zero.
    """
    X = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["b", "a", "new", None],
                categories=["new", "b", "a"],
            ),
        },
        index=[17, 4, 11, 9],
    )

    result = d.apply_woe(
        X,
        {"group": {"a": -0.75, "b": 1.25}},
    )

    np.testing.assert_allclose(
        result["group"],
        [1.25, -0.75, 0.0, 0.0],
    )
    assert result.index.equals(X.index)


@pytest.mark.parametrize(
    "shuffle",
    [False, True],
)
def test_cross_fitting_excludes_each_rows_category(
    shuffle,
):
    """
    Detect target leakage using a unique category for every observation.

    Each validation category is unseen in its training folds, so its
    cross-fitted WoE must be zero despite its nonzero full-data WoE.
    """
    X = pd.DataFrame(
        {
            "id": pd.Categorical([f"id_{i}" for i in range(40)]),
        },
        index=np.arange(40) * 5 + 3,
    )
    y = pd.Series(
        [0, 1] * 20,
        index=X.index,
    )

    result = d.calculate_cross_fitted_woe(
        X,
        y,
        n_splits=5,
        shuffle=shuffle,
        random_state=RS,
    )
    full = d.calculate_woe(
        X,
        y,
    )

    np.testing.assert_array_equal(
        result["woe_table"].to_numpy(),
        np.zeros((40, 1)),
    )
    assert result["woe"] == full["woe"]
    assert result["count"] == full["count"]
    assert np.all(np.abs(full["woe_table"]["id"]) > 0)
    assert result["fold"].index.equals(X.index)
    assert result["woe_table"].index.equals(X.index)
    assert set(result["fold"].unique()) == set(range(5))
    assert result["fold"].notna().all()

    for fold in range(5):
        labels = y.loc[result["fold"] == fold]
        assert len(labels) == 8
        assert labels.value_counts().to_dict() == {0: 4, 1: 4}


def test_cross_fitted_woe_matches_held_out_manual_calculation():
    """
    Check weighted fold mappings independently of the WoE implementation.
    """
    X = pd.DataFrame(
        {
            "group": pd.Categorical(["a", "a", "b", "b"] * 10),
        },
        index=np.arange(40) * 2 + 5,
    )
    y = pd.Series(
        [0, 1, 0, 1] * 10,
        index=X.index,
    )
    weight = np.linspace(0.5, 2.0, 40)

    result = d.calculate_cross_fitted_woe(
        X,
        y,
        weight=weight,
        n_splits=5,
        random_state=RS,
    )

    for fold in range(5):
        valid = result["fold"].to_numpy() == fold
        train = ~valid
        negative = []
        positive = []

        for category in ["a", "b"]:
            category_mask = X["group"].to_numpy() == category
            negative.append(
                weight[train & category_mask & (y.to_numpy() == 0)].sum() + 1.0
            )
            positive.append(
                weight[train & category_mask & (y.to_numpy() == 1)].sum() + 1.0
            )

        negative = np.asarray(negative)
        positive = np.asarray(positive)
        values = np.log(
            (positive / positive.sum())
            / (negative / negative.sum())
        )
        expected = np.where(
            X["group"].to_numpy()[valid] == "a",
            values[0],
            values[1],
        )

        np.testing.assert_allclose(
            result["woe_table"].iloc[np.flatnonzero(valid), 0],
            expected,
        )


@pytest.mark.parametrize(
    "discretizer",
    ["bic", "mdlp"],
)
@pytest.mark.parametrize(
    "cross_fitting",
    [False, True],
)
@pytest.mark.parametrize(
    "weighted",
    [False, True],
)
def test_function_and_class_end_to_end(
    mixed_data,
    discretizer,
    cross_fitting,
    weighted,
):
    """
    Train on mixed data and compare function and class predictions.

    Check calibration, probability normalization, labels, IV, and scoring.
    """
    X, y, X_test, y_test = mixed_data
    weight = (
        np.linspace(0.5, 2.0, len(X)) if weighted else None
    )

    model = discrete_naive_bayes(
        X,
        y,
        weight=weight,
        discretizer=discretizer,
        cross_fitting=cross_fitting,
        random_state=RS,
    )
    estimator = DiscreteNaiveBayes(
        discretizer=discretizer,
        cross_fitting=cross_fitting,
        random_state=RS,
    )
    assert estimator.fit(X, y, weight=weight) is estimator

    assert set(model) == {"disc", "train"}
    assert is_discrete_naive_bayes(model)
    assert set(model["disc"]) == set(X)

    train = model["train"]
    expected_keys = {
        "y_bool", "count", "woe", "woe_table", "raw_score",
        "probability", "c0", "c1", "iter",
    }
    assert expected_keys <= set(train)
    assert train["woe_table"].index.equals(X.index)
    assert train["y_bool"].index.equals(X.index)
    assert np.isfinite(train["woe_table"].to_numpy()).all()

    np.testing.assert_allclose(
        train["raw_score"],
        train["woe_table"].sum(axis=1),
    )
    np.testing.assert_allclose(
        train["probability"],
        1.0 / (1.0 + np.exp(-(train["c0"] + train["c1"] * train["raw_score"]))),
    )

    prediction = predict_dnb(
        model,
        X_test,
    )
    probability = estimator.predict_proba(X_test)

    assert set(prediction) == {"raw_score", "probability"}
    assert probability.shape == (len(X_test), 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        probability[:, 1],
        prediction["probability"],
    )
    np.testing.assert_array_equal(
        estimator.predict(X_test),
        prediction["probability"] >= 0.5,
    )
    np.testing.assert_array_equal(estimator.classes_, [0, 1])

    expected_accuracy = accuracy_score(
        y_test,
        prediction["probability"] >= 0.5,
    )
    assert estimator.score(X_test, y_test) == expected_accuracy
    assert get_accuracy_score(model, X_test, y_test) == expected_accuracy
    assert is_discrete_naive_bayes(estimator.get_model())

    iv = get_iv(model)
    assert set(iv) == set(X)
    for column, value in iv.items():
        expected_iv = (
            train["woe_table"].loc[train["y_bool"], column].mean()
            - train["woe_table"].loc[~train["y_bool"], column].mean()
        )
        assert value == pytest.approx(expected_iv)
    assert estimator.get_iv() == iv
    assert get_iv(model, ["absent"]) == {"absent": None}

    if cross_fitting:
        assert train["fold"].notna().all()
        assert train["fold"].index.equals(X.index)
    else:
        applied = d.apply_woe(
            apply_disc(model["disc"], X),
            train["woe"],
        )
        pd.testing.assert_frame_equal(applied, train["woe_table"])

    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()


def test_prediction_unknown_missing_and_extra_columns():
    """
    Give unseen and missing categories a neutral raw score.
    """
    X = pd.DataFrame(
        {"group": pd.Categorical(["a", "a", "b", "b"] * 10)},
    )
    y = np.array([0, 0, 1, 1] * 10)
    model = discrete_naive_bayes(
        X,
        y,
    )
    X_new = pd.DataFrame(
        {
            "group": pd.Categorical(["a", "b", "unseen", None]),
            "extra": [1, 2, 3, 4],
        },
        index=[10, 4, 22, 17],
    )

    prediction = predict_dnb(
        model,
        X_new,
    )

    np.testing.assert_array_equal(
        prediction["raw_score"][2:],
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        prediction["probability"][2:],
        1.0 / (1.0 + np.exp(-model["train"]["c0"])),
    )
    without_extra = predict_dnb(
        model,
        X_new[["group"]],
    )
    np.testing.assert_array_equal(
        prediction["probability"],
        without_extra["probability"],
    )


def test_zero_information_uses_weighted_prior():
    """
    Calibrate a constant raw score to the weighted class prior.
    """
    X = pd.DataFrame(
        {"constant": pd.Categorical(["same"] * 6)},
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    weight = np.array([1, 1, 1, 2, 2, 2], dtype=float)

    model = discrete_naive_bayes(
        X,
        y,
        weight=weight,
    )
    prediction = predict_dnb(
        model,
        X,
    )

    np.testing.assert_allclose(prediction["raw_score"], 0.0)
    np.testing.assert_allclose(prediction["probability"], 2.0 / 3.0)
    assert model["train"]["c1"] == 0.0


def test_sample_weights_validation_and_copy():
    """
    Validate weights without modifying the caller's array.
    """
    original = np.array([1.0, 0.0, 2.0])
    result = d.make_sample_weight(
        original,
        3,
    )

    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, original)
    np.testing.assert_array_equal(d.make_sample_weight(None, 3), np.ones(3))


@pytest.mark.parametrize(
    "weight",
    [[1, -1, 1], [1, np.nan, 1], [1, np.inf, 1], [1], [[1, 1, 1]]],
)
def test_invalid_sample_weights(
    weight,
):
    """
    Reject negative, nonfinite, misaligned, and multidimensional weights.
    """
    with pytest.raises(ValueError):
        d.make_sample_weight(
            weight,
            3,
        )


def test_make_y_bool_preserves_name_index_and_positive_category():
    """
    Treat the second sorted category as positive and preserve metadata.
    """
    y = pd.Series(
        ["good", "bad", "good", "bad"],
        index=[9, 3, 7, 1],
        name="target",
    )
    result = d.make_y_bool(y)

    np.testing.assert_array_equal(result, [True, False, True, False])
    assert result.index.equals(y.index)
    assert result.name == y.name
    assert result.dtype == bool


@pytest.mark.parametrize(
    "y",
    [[1, 1, 1], [0, 1, 2], [[0, 1], [1, 0]]],
)
def test_invalid_target(
    y,
):
    """
    Reject nonbinary and multidimensional targets.
    """
    with pytest.raises(ValueError):
        d.make_y_bool(y)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"discretizer": "invalid"}, ValueError),
        ({"cross_fitting": 1}, TypeError),
        ({"n_splits": 1}, ValueError),
        ({"n_splits": True}, TypeError),
        ({"shuffle": 1}, TypeError),
        ({"random_state": True}, TypeError),
    ],
)
def test_invalid_training_options(
    kwargs,
    error,
):
    """
    Reject unsupported training options before fitting.
    """
    X = pd.DataFrame(
        {"x": [0, 1, 0, 1]},
    )

    with pytest.raises(error):
        discrete_naive_bayes(
            X,
            [0, 1, 0, 1],
            **kwargs,
        )


def test_invalid_mappings_and_model():
    """
    Reject missing discretizers, missing WoE mappings, and invalid models.
    """
    X = pd.DataFrame(
        {"x": [0, 1]},
    )

    with pytest.raises(ValueError):
        apply_disc({}, X)
    with pytest.raises(ValueError):
        apply_disc({"x": None}, X)
    with pytest.raises(ValueError):
        d.apply_woe(X.astype("category"), {})
    with pytest.raises(TypeError):
        d.apply_woe(X, {"x": {"0": 0.0, "1": 1.0}})
    with pytest.raises(ValueError):
        predict_dnb({}, X)

    assert not is_discrete_naive_bayes(None)
    assert not is_discrete_naive_bayes({"disc": {}})


@pytest.mark.parametrize(
    "target_kind",
    ["string", "integer", "boolean", "ordered_category", "unused_category"],
)
def test_predict_preserves_original_class_labels(
    target_kind,
):
    """
    Require the scikit-learn prediction API to return original labels.

    Label order must agree with the negative and positive probability columns.
    Unused or differently ordered categorical labels must not change it.
    """
    X = pd.DataFrame(
        {"group": pd.Categorical(["a", "b"] * 20)},
    )
    if target_kind == "integer":
        y = np.array([-3, 8] * 20)
    elif target_kind == "boolean":
        y = np.array([False, True] * 20)
    elif target_kind == "ordered_category":
        y = pd.Categorical(
            ["negative", "positive"] * 20,
            categories=["positive", "negative"],
            ordered=True,
        )
    elif target_kind == "unused_category":
        y = pd.Categorical(
            ["negative", "positive"] * 20,
            categories=["unused", "positive", "negative"],
        )
    else:
        y = np.array(["negative", "positive"] * 20)

    estimator = DiscreteNaiveBayes().fit(X, y)
    prediction = estimator.predict(X)
    probability = estimator.predict_proba(X)

    assert set(prediction) <= set(estimator.classes_)
    assert len(estimator.classes_) == 2
    np.testing.assert_array_equal(
        prediction,
        estimator.classes_[(probability[:, 1] >= 0.5).astype(int)],
    )
    np.testing.assert_array_equal(
        prediction,
        np.asarray(y),
    )
    assert estimator.score(X, y) == accuracy_score(np.asarray(y), prediction)


@pytest.mark.parametrize(
    "method",
    [
        "predict", "predict_proba", "score", "get_iv",
        "get_model", "plot_iv", "plot_woe",
    ],
)
def test_unfitted_methods_raise_not_fitted_error(
    method,
):
    """
    Require unfitted public methods to raise NotFittedError.

    No prediction, model access, or plotting should continue before fitting.
    """
    estimator = DiscreteNaiveBayes()
    X = pd.DataFrame(
        {"x": [0, 1]},
    )

    with pytest.raises(NotFittedError):
        if method in ("predict", "predict_proba"):
            getattr(estimator, method)(X)
        elif method == "score":
            estimator.score(X, [0, 1])
        else:
            getattr(estimator, method)()


def test_failed_initial_fit_does_not_mark_estimator_fitted():
    """
    Set fitted attributes only after training succeeds.
    """
    estimator = DiscreteNaiveBayes()
    X = pd.DataFrame(
        {"x": [0, 1, 2]},
    )

    with pytest.raises(ValueError):
        estimator.fit(
            X,
            [0, 1, 2],
        )

    assert not hasattr(estimator, "classes_")
    assert not hasattr(estimator, "coef_")

    with pytest.raises(NotFittedError):
        estimator.predict(X)
