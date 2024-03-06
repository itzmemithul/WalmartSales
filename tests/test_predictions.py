"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from WalmartSales_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Assuming sample_input_data is a tuple where the first element is the DataFrame for prediction
    # and the second element (y_true) are the actual values for comparison.
    X_test, y_true = sample_input_data

    # Given
    expected_num_of_predictions = len(X_test)  # Adjust expectation based on the actual size of X_test

    # When
    result = make_prediction(input_data=X_test)
    print(result)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_num_of_predictions

    _predictions = list(predictions)

    r2 = r2_score(y_true, _predictions)
    mse = mean_absolute_error(y_true, _predictions)

    assert r2 > 0.8
    assert mse < 80000
