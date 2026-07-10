from ..soil_model_types import ForwardSoilModelData, ForwardSoilModelBaseSchema
from ..soil_model_params import ExampleSoilModelParams


def create(
    soil,
    climate,
    cover,
    Ci,
    no_of_years,
    crop=[],
    tree=[],
    litter=[],
    fire=[],
    solve_to_value=False,
    soil_model_params: ExampleSoilModelParams = ExampleSoilModelParams(),
) -> ForwardSoilModelData:
    schema = ForwardSoilModelBaseSchema()
    params = {}
    # This will fail because params is empty
    return schema.load(params)  # type: ignore
