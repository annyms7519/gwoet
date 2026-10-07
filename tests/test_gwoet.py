"""
Test Graphical WoE Transformer through the actual GWoET package.

Place this file in tests/ and run from the repository root:
    python -m pytest -q tests/test_gwoet.py

Public functions and the class are imported from gwoet.
Internal helpers are accessed through the actual implementation module.
"""
from importlib import import_module
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


from gwoet import (
    GraphicalWoETransformer,
    graphical_woe_transformer,
    transform_dataset,
)


# Access private helpers in the actual implementation module.
g = import_module(graphical_woe_transformer.__module__)


@pytest.fixture
def dense_pcor():
    rng = np.random.RandomState(13)
    mat = rng.uniform(-0.7, 0.7, (10, 10))
    mat = np.triu(mat, 1)
    mat += mat.T
    np.fill_diagonal(mat, 1.0)
    return mat


def _count_edges(mat):
    i, j = np.triu_indices(len(mat), k=1)
    return int(np.count_nonzero(np.abs(mat[i, j]) >= g.ZERO_TOL))


def test_removed_target_degree_and_l1_api_are_absent():
    """GWoET no longer exposes target-degree or logistic-selection controls."""
    removed_helpers = (
        "_normalize_target_degree",
        "_select_edges_by_target_degree",
        "_select_edges_by_target_degrees",
        "_make_edge_stage_dict",
        "_normalize_target_degree_weight",
        "_make_stage_penalty_map",
        "_make_l1_penalty_weights",
        "_select_variables_l1_logistic",
        "make_logistic_coefficient_dict",
        "plot_logistic_coefficients",
        "plot_logistic_feature_coefficients",
        "LogisticRegression",
    )
    for name in removed_helpers:
        assert not hasattr(g, name)

    assert not hasattr(g, "TARGET_DEGREE")

    public_params = inspect.signature(graphical_woe_transformer).parameters
    class_params = inspect.signature(GraphicalWoETransformer).parameters

    for name in ("target_degree", "target_degree_weight", "l1_selection", "l1_Cs"):
        assert name not in public_params
        assert name not in class_params

    assert "pruning_ratio" in public_params
    assert "pruning_ratio" in class_params


@pytest.mark.parametrize(
    "ratio,expected_count",
    [
        (0.0, 45),
        (0.1, 41),   # floor(45 * 0.1) = 4 removed
        (0.2, 36),   # floor(45 * 0.2) = 9 removed
        (0.5, 23),   # floor(45 * 0.5) = 22 removed
        (0.99, 1),   # floor(45 * 0.99) = 44 removed
    ],
)
def test_pruning_ratio_edge_count(dense_pcor, ratio, expected_count):
    pruned = g._prune_edges_by_ratio(dense_pcor, ratio)

    assert _count_edges(pruned) == expected_count
    np.testing.assert_allclose(pruned, pruned.T)
    np.testing.assert_array_equal(np.diag(pruned), np.ones(10))


def test_pruning_ratio_removes_weakest_edges():
    mat = np.eye(4)
    values = {
        (0, 1): 0.10,
        (0, 2): -0.20,
        (0, 3): 0.30,
        (1, 2): -0.40,
        (1, 3): 0.50,
        (2, 3): -0.60,
    }
    for (i, j), value in values.items():
        mat[i, j] = mat[j, i] = value

    # Six edges; ratio=1/3 removes floor(6/3)=2 weakest edges.
    pruned = g._prune_edges_by_ratio(mat, 1 / 3)

    assert pruned[0, 1] == 0.0
    assert pruned[0, 2] == 0.0
    assert pruned[0, 3] == pytest.approx(0.30)
    assert pruned[1, 2] == pytest.approx(-0.40)
    assert pruned[1, 3] == pytest.approx(0.50)
    assert pruned[2, 3] == pytest.approx(-0.60)


def test_pruning_ratio_zero_returns_equal_copy(dense_pcor):
    pruned = g._prune_edges_by_ratio(dense_pcor, 0.0)

    np.testing.assert_array_equal(pruned, dense_pcor)
    assert pruned is not dense_pcor


def test_pruning_ignores_tiny_edges():
    mat = np.eye(3)
    mat[0, 1] = mat[1, 0] = 0.6
    mat[0, 2] = mat[2, 0] = 1e-10

    # Only the 0.6 edge is a pruning candidate. floor(1 * 0.5) = 0.
    pruned = g._prune_edges_by_ratio(mat, 0.5)

    assert pruned[0, 1] == pytest.approx(0.6)
    assert pruned[0, 2] == pytest.approx(1e-10)
    assert pruned[1, 2] == 0.0


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.0,
        1.1,
        True,
        np.bool_(True),
        np.nan,
        np.inf,
        [],
        [0.2],
        (0.2,),
        np.array([0.2]),
        "bad",
    ],
)
def test_invalid_pruning_ratio_for_helper(value):
    with pytest.raises((TypeError, ValueError)):
        g._prune_edges_by_ratio(np.eye(3), value)


def test_single_variable_pruning():
    result = g._calc_partial_correlations(
        np.eye(1),
        50,
        gamma=0.5,
        ebic_threshold=False,
        verbose=False,
    )
    pruned = g._prune_edges_by_ratio(result, 0.9)

    np.testing.assert_array_equal(pruned, np.eye(1))


def test_gamma_and_threshold_are_forwarded_to_ebic(dense_pcor, monkeypatch):
    calls = []

    def estimate(mat, n_samples, **kwargs):
        calls.append((mat, n_samples, kwargs))
        return dense_pcor

    monkeypatch.setattr(g, "_calc_partial_correlations_ebic", estimate)

    result = g._calc_partial_correlations(
        np.eye(10),
        250,
        gamma=1.25,
        ebic_threshold=True,
        verbose=False,
    )

    assert len(calls) == 1
    _, n_samples, kwargs = calls[0]
    assert n_samples == 250
    assert kwargs["gamma"] == pytest.approx(1.25)
    assert kwargs["ebic_threshold"] is True
    assert kwargs["verbose"] is False
    np.testing.assert_array_equal(result, dense_pcor)


@pytest.mark.parametrize(
    "gamma",
    [-1, True, np.bool_(True), np.nan, np.inf, "0.5", [0.5]],
)
def test_invalid_gamma(gamma):
    with pytest.raises((TypeError, ValueError)):
        g._calc_partial_correlations(
            np.eye(2),
            50,
            gamma=gamma,
            ebic_threshold=False,
            verbose=False,
        )


def test_actual_ebic():
    mat = np.array([
        [1.0, 0.6, 0.2],
        [0.6, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])

    result = g._calc_partial_correlations(
        mat,
        250,
        gamma=0.5,
        ebic_threshold=False,
        verbose=False,
    )
    direct = g._calc_partial_correlations_ebic(
        mat,
        250,
        gamma=0.5,
        ebic_threshold=False,
        verbose=False,
    )

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_array_equal(np.diag(result), np.ones(3))
    np.testing.assert_allclose(result, direct)


@pytest.mark.parametrize("cross_fitting", [False, True])
@pytest.mark.parametrize("ebic_threshold", [False, True])
@pytest.mark.parametrize("pruning_ratio", [0.0, 0.2])
def test_pipeline_wiring(cross_fitting, ebic_threshold, pruning_ratio):
    rng = np.random.RandomState(31)
    X = pd.DataFrame(
        rng.binomial(1, 0.5, (120, 5)),
        columns=list("abcde"),
        index=np.arange(120) * 3 + 7,
    )
    y = ((X["a"] + X["b"] + rng.binomial(1, 0.3, len(X))) >= 2).astype(int)
    X["category"] = np.where(X["c"], "yes", "no")
    X.loc[X.index[0], "category"] = None

    tfm = GraphicalWoETransformer(
        gamma=0.75,
        ebic_threshold=ebic_threshold,
        pruning_ratio=pruning_ratio,
        cross_fitting=cross_fitting,
        random_state=13,
    )

    fitted = tfm.fit_transform(X, y)
    assert fitted.index.equals(X.index)
    assert list(fitted) == list(tfm.get_feature_names_out())

    model = tfm.get_transformer()
    assert model["pcor"].index.equals(model["pcor"].columns)
    assert set(model) == {"disc", "pcor", "numeric", "nodes", "edges"}
    assert all(
        set(pair) <= set(model["nodes"])
        for pair in model["edges"].values()
    )

    cloned = clone(tfm)
    assert cloned.gamma == tfm.gamma
    assert cloned.ebic_threshold == tfm.ebic_threshold
    assert cloned.pruning_ratio == tfm.pruning_ratio

    graph = tfm.gen_graph_data()
    assert set(graph.nodes) == set(model["nodes"])
    assert len(graph.edges) == len(model["edges"])
    for pair in model["edges"].values():
        np.testing.assert_allclose(
            graph.edges[tuple(pair)]["weight"],
            model["pcor"].loc[pair[0], pair[1]],
        )


def test_no_features_fallback(monkeypatch):
    X = pd.DataFrame({"a": [0, 1, 0, 1]}, index=[10, 20, 30, 40])

    def zero_woe(X, y, weight, **kwargs):
        return {
            "woe_table": pd.DataFrame(0.0, index=X.index, columns=X.columns),
            "y_bool": pd.Series([False, True, False, True], index=X.index),
        }

    monkeypatch.setattr(g, "_calculate_training_woe", zero_woe)

    model = graphical_woe_transformer(
        X,
        [0, 1, 0, 1],
        pruning_ratio=0.5,
    )

    assert set(model) == {"disc", "pcor", "numeric", "nodes", "edges"}
    assert model["edges"] == {}
    assert model["pcor"].index.equals(X.columns)
    assert transform_dataset(model, X).index.equals(X.index)


def test_not_fitted():
    tfm = GraphicalWoETransformer()
    methods = (
        lambda: tfm.transform(pd.DataFrame({"a": [1]})),
        tfm.get_transformer,
        tfm.get_pcor,
        tfm.gen_graph_data,
        tfm.draw,
        tfm.get_feature_names_out,
    )
    for method in methods:
        with pytest.raises(NotFittedError):
            method()


def test_category_interaction_missing_and_index():
    a = pd.Series(["x", None, "y"], index=[9, 4, 1])
    b = pd.Series(["a", "b", None], index=a.index)
    interaction = g._make_categorical_interaction(a, b)

    assert interaction.index.equals(a.index)
    assert interaction.iloc[0] == "x*a"
    assert interaction.iloc[1:].isna().all()


def test_endpoint_retention_after_iv_pruning(monkeypatch):
    X = pd.DataFrame({
        "a": [0, 1, 0, 1],
        "b": [1, 0, 0, 1],
        "c": [0, 1, 1, 0],
    })
    pcor = np.eye(3)
    pcor[0, 1] = pcor[1, 0] = 0.5

    def initial_woe(X, y, weight, **kwargs):
        y_bool = pd.Series(
            np.asarray(y) == np.unique(y)[-1],
            index=X.index,
        )
        values = np.tile(
            np.where(y_bool, 1.0, -1.0)[:, None],
            (1, len(X.columns)),
        )
        return {
            "woe_table": pd.DataFrame(values, index=X.index, columns=X.columns),
            "y_bool": y_bool,
        }

    monkeypatch.setattr(g, "_calculate_training_woe", initial_woe)
    monkeypatch.setattr(g, "_calc_partial_correlations", lambda *a, **k: pcor)
    monkeypatch.setattr(
        g,
        "_filter_variables_with_woe",
        lambda *args, **kwargs: ["a*b"],
    )

    model = graphical_woe_transformer(
        X,
        [0, 1, 0, 1],
        pruning_ratio=0.0,
    )

    assert model["nodes"] == ["a", "b"]
    assert model["edges"] == {"a*b": ["a", "b"]}
    np.testing.assert_array_equal(model["pcor"], pcor)


def test_discretized_transform_and_input_names():
    tfm = GraphicalWoETransformer(
        numeric=False,
        pruning_ratio=0.0,
    )
    X = pd.DataFrame(
        {"a": [0, 1] * 10, "b": [1, 0] * 10},
        index=np.arange(20) + 42,
    )

    out = tfm.fit_transform(X, [0, 1] * 10)

    assert out.index.equals(X.index)
    np.testing.assert_array_equal(
        tfm.get_feature_names_out(X.columns),
        list(out),
    )
    with pytest.raises(ValueError):
        tfm.get_feature_names_out(["other"])


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.0,
        1.5,
        True,
        np.bool_(True),
        np.nan,
        np.inf,
        "0.2",
        [0.2],
    ],
)
def test_invalid_public_pruning_ratio(value):
    X = pd.DataFrame({"a": [0, 1, 0, 1]})

    with pytest.raises((TypeError, ValueError)):
        graphical_woe_transformer(
            X,
            [0, 1, 0, 1],
            pruning_ratio=value,
        )

    with pytest.raises((TypeError, ValueError)):
        GraphicalWoETransformer(
            pruning_ratio=value,
        ).fit(
            X,
            [0, 1, 0, 1],
        )


@pytest.fixture
def graph_wiring(monkeypatch, dense_pcor):
    X = pd.DataFrame(
        np.arange(400).reshape(40, 10) % 3,
        columns=[f"v{i}" for i in range(10)],
    )
    y = np.array([0, 1] * 20)

    def training_woe(X, y, weight, **kwargs):
        y_bool = pd.Series(
            np.asarray(y) == np.unique(y)[-1],
            index=X.index,
        )
        values = np.tile(
            np.where(y_bool, 1.0, -1.0)[:, None],
            (1, len(X.columns)),
        )
        return {
            "woe_table": pd.DataFrame(values, index=X.index, columns=X.columns),
            "y_bool": y_bool,
        }

    monkeypatch.setattr(g, "_calculate_training_woe", training_woe)

    calls = []

    def estimate(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return dense_pcor.copy()

    monkeypatch.setattr(g, "_calc_partial_correlations_ebic", estimate)

    return X, y, dense_pcor.copy(), calls


@pytest.mark.parametrize(
    "ratio,count",
    [
        (0.0, 45),
        (0.1, 41),
        (0.2, 36),
        (0.5, 23),
    ],
)
def test_full_ebic_survives_percentage_and_iv_pruning(
    monkeypatch,
    graph_wiring,
    ratio,
    count,
):
    X, y, expected, calls = graph_wiring
    seen = []

    def selection(X, y, nodes, edges, *args, **kwargs):
        seen.append(edges.copy())
        return ["v0"] + list(edges)[:1]

    monkeypatch.setattr(g, "_filter_variables_with_woe", selection)

    model = graphical_woe_transformer(
        X,
        y,
        pruning_ratio=ratio,
        gamma=0.5,
    )

    assert len(calls) == 1
    assert len(seen[0]) == count
    assert len(model["edges"]) == min(1, count)
    assert len(model["nodes"]) < len(X.columns)

    # The stored pcor remains the full EBIC estimate before percentage pruning.
    np.testing.assert_array_equal(model["pcor"], expected)
    assert list(model["pcor"]) == list(X)

    graph = g.gen_graph_data(model)
    assert len(graph.edges) == min(1, count)


@pytest.mark.parametrize("ebic_threshold", [False, True])
def test_threshold_and_pruning_ratio_coexist(
    monkeypatch,
    graph_wiring,
    ebic_threshold,
):
    X, y, expected, calls = graph_wiring
    seen = []

    def selection(X, y, nodes, edges, *args, **kwargs):
        seen.append(edges.copy())
        return list(nodes) + list(edges)

    monkeypatch.setattr(g, "_filter_variables_with_woe", selection)

    model = graphical_woe_transformer(
        X,
        y,
        gamma=1.25,
        ebic_threshold=ebic_threshold,
        pruning_ratio=0.2,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["gamma"] == pytest.approx(1.25)
    assert calls[0]["kwargs"]["ebic_threshold"] is ebic_threshold

    # The mocked EBIC matrix has 45 nonzero edges; 20% pruning removes 9.
    assert len(seen[0]) == 36
    assert len(model["edges"]) == 36
    np.testing.assert_array_equal(model["pcor"], expected)


def test_post_iv_fallback_keeps_full_ebic(monkeypatch, graph_wiring):
    X, y, expected, calls = graph_wiring
    monkeypatch.setattr(
        g,
        "_filter_variables_with_woe",
        lambda *a, **k: [],
    )

    model = graphical_woe_transformer(
        X,
        y,
        pruning_ratio=0.5,
        gamma=0.75,
    )

    assert model["nodes"] == list(X)
    assert model["edges"] == {}
    assert set(model) == {"disc", "pcor", "numeric", "nodes", "edges"}
    np.testing.assert_array_equal(model["pcor"], expected)
    assert len(calls) == 1
    assert calls[0]["kwargs"]["gamma"] == pytest.approx(0.75)


def test_defaults_use_full_ebic_without_percentage_pruning(
    monkeypatch,
    graph_wiring,
):
    X, y, expected, calls = graph_wiring

    assert g.PRUNING_RATIO == pytest.approx(0.0)
    assert graphical_woe_transformer.__kwdefaults__["pruning_ratio"] == pytest.approx(
        g.PRUNING_RATIO
    )
    assert graphical_woe_transformer.__kwdefaults__["gamma"] == g.GAMMA
    assert (
        graphical_woe_transformer.__kwdefaults__["ebic_threshold"]
        is g.EBIC_THRESHOLD
    )

    tfm = GraphicalWoETransformer()
    assert tfm.pruning_ratio == pytest.approx(g.PRUNING_RATIO)
    assert tfm.gamma == g.GAMMA
    assert tfm.ebic_threshold is g.EBIC_THRESHOLD

    tfm.fit(X, y)

    assert len(calls) == 1
    assert calls[0]["kwargs"]["gamma"] == pytest.approx(g.GAMMA)
    assert calls[0]["kwargs"]["ebic_threshold"] is g.EBIC_THRESHOLD
    np.testing.assert_array_equal(tfm.get_pcor(), expected)
    assert set(tfm.get_transformer()) == {
        "disc",
        "pcor",
        "numeric",
        "nodes",
        "edges",
    }


def test_five_key_model_can_be_used_directly():
    X = pd.DataFrame(
        {
            "a": [0, 1] * 10,
            "b": [1, 0] * 10,
        },
        index=np.arange(20) + 4,
    )
    y = np.array([0, 1] * 10)

    model = graphical_woe_transformer(
        X,
        y,
        gamma=0.5,
        ebic_threshold=False,
        pruning_ratio=0.0,
        cross_fitting=False,
        random_state=13,
    )

    assert set(model) == {"disc", "pcor", "numeric", "nodes", "edges"}
    assert g.is_graphical_woe_transformer(model)

    transformed = transform_dataset(
        model,
        X,
    )

    assert list(transformed) == model["nodes"] + list(model["edges"])
    assert transformed.index.equals(X.index)
