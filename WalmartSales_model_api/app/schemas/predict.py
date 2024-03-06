from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from WalmartSales_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Store": 39,
                        "Date": "2012-11-05",  # Corrected to "Date" and kept as a string
                        "Holiday_Flag": 0,
                        "Temperature": 38.6,
                        "Fuel_Price": 2.9,
                        "CPI": 211,
                        "Unemployment": 7.899,
                        "Quarter": 4,
                    }
                ]
            }
        }
