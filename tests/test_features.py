
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from WalmartSales_model.config.core import config
from WalmartSales_model.processing.features import OutlierHandler



def test_unemployment_variable_outlierhandler(sample_input_data):
    # Assuming sample_input_data is a DataFrame for simplification
    df = sample_input_data[0]  # Adjust based on your actual data structure

    # Given
    encoder = OutlierHandler(variable=config.model_config.unemployment_var)
    q1, q3 = np.percentile(df['Unemployment'], q=[25, 75])
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)

    # Ensure there's at least one outlier
    outliers_before = df[df['Unemployment'] > upper_bound]
    assert not outliers_before.empty, "No outliers found in the test data."

    # When
    subject = encoder.fit(df).transform(df)

    # Then
    # Ensure no 'Unemployment' values are above the upper bound after handling
    assert subject[subject['Unemployment'] > upper_bound].empty, "Outliers were not properly handled."


