"""
gwoet

A Python package for Discrete Naive Bayes (DNB) and Graphical WoE
Transformer (GWoET).

Package structure
-----------------
gwoet/
├── __init__.py
├── dnb.py
│   └── Functions and classes for Discrete Naive Bayes (DNB)
├── gwoet.py
│   └── Functions and classes for Graphical WoE Transformer (GWoET)
└── selection.py
     └── Functions for logistic feature selection and diagnostics
"""

from .dnb import (
    DiscreteNaiveBayes,
    apply_disc,
    discrete_naive_bayes,
    generate_disc,
    get_accuracy_score,
    get_iv,
    is_discrete_naive_bayes,
    make_sample_weight,
    make_y_bool,
    plot_iv,
    plot_woe,
    predict_dnb,
)
from .gwoet import (
    GraphicalWoETransformer,
    draw_graphical_model,
    gen_graph_data,
    graphical_woe_transformer,
    is_graphical_woe_transformer,
    transform_dataset,
)
from .selection import (
    L1LogisticSelector,
    select_variables_l1_logistic,
    make_logistic_coefficient_dict,
    plot_logistic_coefficients,
    plot_logistic_feature_coefficients,
)

__version__ = "0.9.0"
__author__ = "annyms7519"
__email__ = "annyms7519@gmail.com"

__all__ = [
    # Discrete Naive Bayes
    "DiscreteNaiveBayes",
    "discrete_naive_bayes",
    "is_discrete_naive_bayes",
    "predict_dnb",
    "make_y_bool",
    "make_sample_weight",
    "generate_disc",
    "apply_disc",
    "get_accuracy_score",
    "get_iv",
    "plot_iv",
    "plot_woe",

    # Graphical WoE Transformer
    "GraphicalWoETransformer",
    "graphical_woe_transformer",
    "is_graphical_woe_transformer",
    "transform_dataset",
    "gen_graph_data",
    "draw_graphical_model",

    # Logistic Feature Selection and Diagnostics
    "L1LogisticSelector",
    "select_variables_l1_logistic",
    "make_logistic_coefficient_dict",
    "plot_logistic_coefficients",
    "plot_logistic_feature_coefficients",
]
