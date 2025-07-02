from unittest.mock import patch

import numpy as np
import pytest
from test_data_regression import (
    regression_test_data,
    small_regression_data,
    train_test_split,
)

from encoders.regression import pearsonr, pearsonr_scorer, ridge_regression, z_score, zs

# ruff: noqa: F811 - pytest fixtures are injected by name


def test_zs_1d():
    """Test z-score normalization for 1D arrays."""
    x = np.array([1, 2, 3, 4, 5])
    z = zs(x)

    assert np.allclose(z.mean(), 0, atol=1e-10)
    assert np.allclose(z.std(), 1)


def test_zs_2d():
    """Test z-score normalization for 2D arrays."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((10, 3))
    z = zs(x)

    assert z.shape == x.shape
    assert np.allclose(z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(z.std(axis=0), 1)


def test_z_score_with_given_stats():
    """Test z-score normalization with provided means and stds."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    means = np.array([3, 4])
    stds = np.array([2, 2])

    normalized = z_score(data, means, stds)
    expected = (data - means) / stds

    assert np.allclose(normalized, expected)


def test_pearsonr_1d():
    """Test Pearson correlation for 1D arrays."""
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([2, 4, 6, 8, 10])

    corr = pearsonr(x1, x2)

    assert isinstance(corr, (float, np.floating))
    assert np.allclose(corr, 1.0)


def test_pearsonr_2d():
    """Test Pearson correlation for 2D arrays."""
    rng = np.random.default_rng(42)
    x1 = rng.standard_normal((100, 3))
    x2 = x1 + rng.standard_normal((100, 3)) * 0.1

    corr = pearsonr(x1, x2)

    assert isinstance(corr, np.ndarray)
    assert corr.shape == (3,)
    assert np.all(corr > 0.5)


def test_pearsonr_scorer():
    """Test the scoring function for RidgeCV."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    y = rng.standard_normal((50, 2))

    estimator = Ridge()
    estimator.fit(X, y)

    score = pearsonr_scorer(estimator, X, y)

    assert isinstance(score, (float, np.ndarray))
    if isinstance(score, np.ndarray):
        assert score.shape == (2,)


def test_ridge_regression_basic(regression_test_data, train_test_split):
    """Test basic functionality of ridge_regression."""
    X_data_dict = regression_test_data["X_data_dict"]
    y_data_dict = regression_test_data["y_data_dict"]
    train_stories = train_test_split["train_stories"]
    test_stories = train_test_split["test_stories"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=train_stories,
        test_stories=test_stories,
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.logspace(0, 2, 5),
    )

    assert isinstance(scores, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert isinstance(best_alphas, np.ndarray)

    assert scores.shape == (3,)
    assert weights.shape == (3, 5)
    assert best_alphas.shape == (3,)


def test_ridge_regression_shapes(small_regression_data):
    """Test that ridge_regression produces correct output shapes."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0, 10.0]),
    )

    assert scores.shape == (2,)
    assert weights.shape == (2, 3)
    assert best_alphas.shape == (2,)


def test_ridge_regression_default_alphas(small_regression_data):
    """Test ridge_regression with default alpha values."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
    )

    assert scores.shape == (2,)
    assert weights.shape == (2, 3)
    assert best_alphas.shape == (2,)


def test_ridge_regression_none_alphas(small_regression_data):
    """Test ridge_regression with None alphas (should use default)."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=None,
    )

    assert scores.shape == (2,)
    assert weights.shape == (2, 3)
    assert best_alphas.shape == (2,)


def test_ridge_regression_multiple_train_stories(
    regression_test_data, train_test_split
):
    """Test ridge_regression with multiple training stories."""
    X_data_dict = regression_test_data["X_data_dict"]
    y_data_dict = regression_test_data["y_data_dict"]
    train_stories = train_test_split["train_stories"]
    test_stories = train_test_split["test_stories"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=train_stories,
        test_stories=test_stories,
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0, 100.0]),
    )

    assert len(train_stories) == 3
    assert len(test_stories) == 1
    assert scores.shape == (3,)


def test_ridge_regression_data_normalization(small_regression_data):
    """Test that data is properly normalized during ridge regression."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    with patch("encoders.regression.z_score") as mock_z_score:
        mock_z_score.side_effect = lambda data, means, stds: (data - means) / (
            stds + 1e-6
        )

        ridge_regression(
            train_stories=["train"],
            test_stories=["test"],
            X_data_dict=X_data_dict,
            y_data_dict=y_data_dict,
            score_fct=pearsonr,
            alphas=np.array([1.0]),
        )

        assert mock_z_score.call_count == 2


def test_ridge_regression_score_function_called(small_regression_data):
    """Test that the scoring function is called correctly."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    def mock_score(y_true, y_pred):
        return np.array([0.5, 0.6])

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=mock_score,
        alphas=np.array([1.0]),
    )

    assert np.allclose(scores, [0.5, 0.6])


def test_ridge_regression_single_alpha(small_regression_data):
    """Test ridge regression with a single alpha value."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0]),
    )

    assert scores.shape == (2,)
    assert weights.shape == (2, 3)
    assert best_alphas.shape == (2,)
    assert np.all(best_alphas == 1.0)


def test_ridge_regression_reproducibility(small_regression_data):
    """Test that ridge regression gives consistent results."""
    X_data_dict = small_regression_data["X_data_dict"]
    y_data_dict = small_regression_data["y_data_dict"]

    scores1, weights1, alphas1 = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0, 10.0]),
    )

    scores2, weights2, alphas2 = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0, 10.0]),
    )

    assert np.allclose(scores1, scores2)
    assert np.allclose(weights1, weights2)
    assert np.allclose(alphas1, alphas2)


def test_ridge_regression_with_zero_variance_features():
    """Test ridge regression handles zero-variance features properly."""
    rng = np.random.default_rng(42)

    X_data_dict = {
        "train": np.column_stack([rng.standard_normal((20, 2)), np.ones(20)]),
        "test": np.column_stack([rng.standard_normal((10, 2)), np.ones(10)]),
    }

    y_data_dict = {
        "train": rng.standard_normal((20, 2)),
        "test": rng.standard_normal((10, 2)),
    }

    scores, weights, best_alphas = ridge_regression(
        train_stories=["train"],
        test_stories=["test"],
        X_data_dict=X_data_dict,
        y_data_dict=y_data_dict,
        score_fct=pearsonr,
        alphas=np.array([1.0]),
    )

    assert scores.shape == (2,)
    assert weights.shape == (2, 3)
    assert best_alphas.shape == (2,)
