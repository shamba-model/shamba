import numpy as np

from ..soil_model_types import InverseSoilModelData, InverseSoilModelBaseSchema
from ..soil_model_params import ExampleSoilModelParams


def create(
    soil,
    climate,
    cover=np.ones(12),
    soil_model_params: ExampleSoilModelParams = ExampleSoilModelParams(),
) -> InverseSoilModelData:
    schema = InverseSoilModelBaseSchema()
    params = {}
    # This will fail because params is empty
    return schema.load(params)  # type: ignore
