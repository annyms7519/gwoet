import numpy as np
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_iris

from gwoet import (
    GraphicalWoETransformer,
    graphical_woe_transformer,
    transform_dataset,
)

@pytest.fixture
def iris_binary():
    """Iris-based mixed-type binary-classification data used by the tests."""
    iris = load_iris()

    X = pd.DataFrame(
        data=iris["data"],
        columns=iris["feature_names"],
    )
    X["new_var1"] = pd.Series(["a"] * 50 + ["b"] * (len(X) - 50))
    X["new_var2"] = pd.Series([1] * 80 + [2] * (len(X) - 80))

    # Include missing values, as in the original regression test.
    X.iloc[1, :] = None
    X.iloc[2, 2] = np.nan
    X.iloc[2, 4] = None

    y = pd.Series(iris["target"], name="target") == 2

    return X, y


def test_graphical_woe_transformer_function(iris_binary):
    """The functional API should return a structurally valid transformer."""
    X, y = iris_binary

    model = graphical_woe_transformer(
        X,
        y,
        random_state=0,
        verbose=False,
    )

    assert isinstance(model, dict)

    expected_keys = {
        "degree",
        "disc",
        "edges",
        "nodes",
        "numeric",
        "pcor",
    }
    assert expected_keys.issubset(model)

    assert model["degree"] == 2
    assert model["numeric"] is True
    assert set(model["nodes"]).issubset(X.columns)
    assert isinstance(model["edges"], dict)
    assert isinstance(model["pcor"], pd.DataFrame)

    # The original test produced a 6 x 6 partial-correlation matrix.
    assert model["pcor"].shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(
        np.diag(model["pcor"].to_numpy()),
        np.ones(X.shape[1]),
    )


def test_transform_dataset_function(iris_binary):
    """transform_dataset() should preserve rows and append interaction features."""
    X, y = iris_binary

    model = graphical_woe_transformer(
        X,
        y,
        random_state=0,
        verbose=False,
    )
    X_new = transform_dataset(model, X)

    assert isinstance(X_new, pd.DataFrame)
    assert len(X_new) == len(X)
    assert X_new.index.equals(X.index)

    # Without L1 selection, all original variables should remain.
    assert set(X.columns).issubset(X_new.columns)

    # Each learned edge corresponds to one interaction feature.
    assert set(model["edges"]).issubset(X_new.columns)
    assert X_new.shape[1] == X.shape[1] + len(model["edges"])


@pytest.mark.parametrize(
    ("l1_selection", "target_degree"),
    [
        (False, "adaptive"),
        (False, 2),
        (True, "adaptive"),
        (True, 2),
    ],
)
def test_estimator_fit_transform(
    iris_binary,
    l1_selection,
    target_degree,
):
    """The sklearn-style transformer should work for the four original cases."""
    X, y = iris_binary

    tfm = GraphicalWoETransformer(
        l1_selection=l1_selection,
        target_degree=target_degree,
        random_state=0,
        verbose=False,
    )

    X_new = tfm.fit_transform(X, y)
    model = tfm.get_transformer()

    assert isinstance(X_new, pd.DataFrame)
    assert len(X_new) == len(X)
    assert X_new.index.equals(X.index)

    assert isinstance(model, dict)
    assert model["degree"] == 2
    assert set(model["nodes"]).issubset(X.columns)
    assert set(model["edges"]).issubset(X_new.columns)

    if not l1_selection:
        assert set(model["nodes"]) == set(X.columns)
        assert set(X.columns).issubset(X_new.columns)
    else:
        # L1 selection may prune original variables, but it must retain at
        # least one node and may not introduce unknown original variables.
        assert 1 <= len(model["nodes"]) <= X.shape[1]


def test_fit_then_transform_matches_fit_transform(iris_binary):
    """fit()+transform() and fit_transform() should produce the same result."""
    X, y = iris_binary

    params = dict(
        l1_selection=False,
        target_degree="adaptive",
        random_state=0,
        verbose=False,
    )

    tfm1 = GraphicalWoETransformer(**params)
    tfm1.fit(X, y)
    X_new1 = tfm1.transform(X)

    tfm2 = GraphicalWoETransformer(**params)
    X_new2 = tfm2.fit_transform(X, y)

    pd.testing.assert_frame_equal(X_new1, X_new2)


def test_functional_and_estimator_api_agree(iris_binary):
    """Functional and sklearn-style APIs should learn the same transformation."""
    X, y = iris_binary

    model = graphical_woe_transformer(
        X,
        y,
        l1_selection=False,
        target_degree="adaptive",
        random_state=0,
        verbose=False,
    )
    X_function = transform_dataset(model, X)

    tfm = GraphicalWoETransformer(
        l1_selection=False,
        target_degree="adaptive",
        random_state=0,
        verbose=False,
    )
    X_estimator = tfm.fit_transform(X, y)

    assert model["degree"] == tfm.get_transformer()["degree"]
    assert model["nodes"] == tfm.get_transformer()["nodes"]
    assert model["edges"] == tfm.get_transformer()["edges"]

    pd.testing.assert_frame_equal(X_function, X_estimator)

