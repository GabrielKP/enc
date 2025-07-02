import numpy as np
import pytest


@pytest.fixture
def regression_test_data():
    """
    Create lightweight test data for regression testing.

    Returns:
    --------
    dict: Contains X_data_dict and y_data_dict with simulated data
          - 4 stories total (3 train, 1 test)
          - Each story has ~100 samples with 5-dimensional features
          - Each story has 3 response channels (simulating brain regions)
    """
    rng = np.random.default_rng(42)

    n_samples = 100
    n_features = 5
    n_channels = 3
    stories = ["story1", "story2", "story3", "story4"]

    X_data_dict = {}
    y_data_dict = {}

    for story in stories:
        X_data_dict[story] = rng.standard_normal((n_samples, n_features))

        true_weights = rng.standard_normal((n_features, n_channels)) * 0.5
        noise = rng.standard_normal((n_samples, n_channels)) * 0.3
        y_data_dict[story] = np.dot(X_data_dict[story], true_weights) + noise

    return {
        "X_data_dict": X_data_dict,
        "y_data_dict": y_data_dict,
        "stories": stories,
        "true_weights": true_weights,
    }


@pytest.fixture
def train_test_split():
    """
    Define train/test split mimicking cross-validation setup.

    Returns:
    --------
    dict: Contains train_stories and test_stories lists
    """
    return {"train_stories": ["story1", "story2", "story3"], "test_stories": ["story4"]}


@pytest.fixture
def small_regression_data():
    """
    Even smaller dataset for quick testing.

    Returns:
    --------
    dict: Contains X_data_dict and y_data_dict with minimal data
    """
    rng = np.random.default_rng(123)

    X_data_dict = {
        "train": rng.standard_normal((20, 3)),
        "test": rng.standard_normal((10, 3)),
    }

    y_data_dict = {
        "train": rng.standard_normal((20, 2)),
        "test": rng.standard_normal((10, 2)),
    }

    return {"X_data_dict": X_data_dict, "y_data_dict": y_data_dict}
