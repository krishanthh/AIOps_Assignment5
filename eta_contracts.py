"""Shared request contract for the REST and Ray Serve applications."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURES = ['distance_km','n_stops','hour','weekday','traffic_index','rain','weight_kg']

class Delivery(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode='after')
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError('infeasible: distance above 150 km with more than 20 stops')
        return self

class PredictRequest(BaseModel):
    instances: Annotated[list[Delivery], Field(min_length=1,max_length=500)]

class CanaryRequest(PredictRequest):
    customer_id: str = Field(min_length=1,max_length=256)

def example():
    return dict(distance_km=20.0,n_stops=5,hour=12,weekday=2,traffic_index=0.3,rain=0,weight_kg=8.0)
