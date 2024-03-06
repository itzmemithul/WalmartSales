import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from WalmartSales_model.config.core import config
from WalmartSales_model.processing.features import ColumnScaler
from WalmartSales_model.processing.features import OutlierHandler

WalmartSales_pipe = Pipeline([

    ######### Scaling ###########
    ('column_scaler_temp', ColumnScaler(columns_to_scale= config.model_config.temperature_var)), 
    ('column_scaler_fuelprice', ColumnScaler(columns_to_scale= config.model_config.fuel_price_var)),
    ('column_scaler_cpi', ColumnScaler(columns_to_scale= config.model_config.cpi_var)),
    ('column_scaler_unemployment', ColumnScaler(columns_to_scale= config.model_config.unemployment_var)),
    
    ######## Handle outliers ########
    ('handle_outliers_temp', OutlierHandler(variable = config.model_config.temperature_var)),
    ('handle_outliers_fuelprice', OutlierHandler(variable = config.model_config.fuel_price_var)),
    ('handle_outliers_cpi', OutlierHandler(variable = config.model_config.cpi_var)),
    ('handle_outliers_unemployment', OutlierHandler(variable = config.model_config.unemployment_var)),
    
    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config.n_estimators, 
                                       max_depth = config.model_config.max_depth,
                                      random_state = config.model_config.random_state))
    
    ])