# flake8: noqa: E501
"""Autograding script."""

import gzip
import json
import os
import pickle

import pandas as pd  # type: ignore

# ------------------------------------------------------------------------------
MODEL_FILENAME = "files/models/model.pkl.gz"
MODEL_COMPONENTS = [
    "OneHotEncoder",
    "SelectKBest",
    "MinMaxScaler",
    "LinearRegression",
]
SCORES = [
    -1.590,
    -2.429,
]
METRICS = [
    {
        "type": "metrics",
        "dataset": "train",
        "r2": 0.889,
        "mse": 5.950,
        "mad": 1.600,
    },
    {
        "type": "metrics",
        "dataset": "test",
        "r2": 0.728,
        "mse": 32.910,
        "mad": 2.430,
    },
]


# ------------------------------------------------------------------------------
#
# Internal tests
#
def _load_model():
    """Generic test to load a model"""
    assert os.path.exists(MODEL_FILENAME)
    with gzip.open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)
    assert model is not None
    return model


def _test_components(model):
    """Test components"""
    assert "GridSearchCV" in str(type(model))
    current_components = [str(model.estimator[i]) for i in range(len(model.estimator))]
    for component in MODEL_COMPONENTS:
        assert any(component in x for x in current_components)


def _load_grading_data():
    """Load grading data"""
    with open("files/grading/x_train.pkl", "rb") as file:
        x_train = pickle.load(file)

    with open("files/grading/y_train.pkl", "rb") as file:
        y_train = pickle.load(file)

    with open("files/grading/x_test.pkl", "rb") as file:
        x_test = pickle.load(file)

    with open("files/grading/y_test.pkl", "rb") as file:
        y_test = pickle.load(file)

    return x_train, y_train, x_test, y_test


def _test_scores(model, x_train, y_train, x_test, y_test):
    """Test scores"""
    assert model.score(x_train, y_train) < SCORES[0]
    assert model.score(x_test, y_test) < SCORES[1]


def _load_metrics():
    assert os.path.exists("files/output/metrics.json")
    metrics = []
    with open("files/output/metrics.json", "r", encoding="utf-8") as file:
        for line in file:
            metrics.append(json.loads(line))
    return metrics


def _test_metrics(metrics):

    for index in [0, 1]:
        assert metrics[index]["type"] == METRICS[index]["type"]
        assert metrics[index]["dataset"] == METRICS[index]["dataset"]
        assert metrics[index]["r2"] > METRICS[index]["r2"]
        assert metrics[index]["mse"] < METRICS[index]["mse"]
        assert metrics[index]["mad"] < METRICS[index]["mad"]


def test_homework():
    """Tests"""

    model = _load_model()
    x_train, y_train, x_test, y_test = _load_grading_data()
    metrics = _load_metrics()

    _test_components(model)
    _test_scores(model, x_train, y_train, x_test, y_test)
    _test_metrics(metrics)
