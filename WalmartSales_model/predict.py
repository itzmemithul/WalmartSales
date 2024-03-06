import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from WalmartSales_model import __version__ as _version
from WalmartSales_model.config.core import config
from WalmartSales_model.processing.data_manager import load_pipeline
from WalmartSales_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
WalmartSales_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    print('input_data :', input_data)
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))

    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = WalmartSales_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {'Store':[39],'Date': ['07-11-2011'], 'Holiday_Flag': [0], 'Temperature': [38.6], 'Fuel_Price': [2.9], 'CPI': [211],
               'Unemployment': [7.899]}

    make_prediction(input_data = data_in)