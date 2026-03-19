from enum import Enum
import numpy as np
from marshmallow import Schema, fields, post_load

from ..climate import ClimateDataSchema
from ..soil_params import SoilParamsSchema


class SoilModelType(Enum):
    ROTH_C = "RothC"
    EXAMPLE = "Example"


# NB. the format of the following 3 classes currently matches requirements for RothC,
# which is the only soil C model currently available.
class ForwardSoilModelData:
    """
    Forward RothC object

    Instance variables
    ----------------
    SOC            vector with soil distributions for each year
    soil_params    SoilParams object with soil params (porosity, field capacity, etc.)
    climate        Climate object with climate data (rain, evaporation, etc.)
    cover          vector with cover for each year
    k              vector with crop coefficient for each year
    inputs         vector with inputs for each year
    Cy0Year        vector with initial soil carbon for each year

    """

    def __init__(
        self,
        soil_params,
        climate,
        cover,
        k,
        SOC,
        inputs,
        Cy0Year,
    ):
        self.soil = soil_params
        self.climate = climate
        self.cover = cover
        self.k = k
        self.SOC = SOC
        self.inputs = inputs
        self.Cy0Year = Cy0Year


class BaseSoilModelData:
    """
    Object for RothC soil models.
    Includes methods and variables common to both forward and inverse.

    Instance variables
    ------------------
    k       rate constants for the 4 soil pools (with RMF)
    cover   vector with soil cover for each month (1 if covered, 0 else)

    """

    def __init__(
        self,
        soil_params,
        climate,
        cover,
        k,
    ):
        self.soil = soil_params
        self.climate = climate
        self.cover = np.array(cover)
        self.k = np.array(k)


class InverseSoilModelData(BaseSoilModelData):
    """
    Inverse RothC model. Extends BaseSoilModelData class.

    Instance variables
    ------------------
    eq_C     calculated equilibrium distribution of carbon
    input_C  yearly input to soil giving eq_C
    x       partitioning coefficients

    """

    def __init__(self, eq_C, input_C, x, **kwargs):
        super().__init__(**kwargs)
        self.eq_C = eq_C
        self.input_C = input_C
        self.x = x


class SoilModelBaseSchema(Schema):
    soil_params = fields.Nested(SoilParamsSchema, required=True)
    climate = fields.Nested(ClimateDataSchema, required=True)
    cover = fields.List(fields.Float, required=True)
    k = fields.List(fields.List(fields.Float), required=True)


class ForwardSoilModelBaseSchema(SoilModelBaseSchema):
    SOC = fields.List(fields.List(fields.Float), required=True)
    inputs = fields.List(fields.List(fields.Float), required=True)
    Cy0Year = fields.Float(required=True)

    @post_load
    def build_forward_soil_model(self, data, **kwargs):
        soil_c_data = {k: data[k] for k in SoilModelBaseSchema().fields.keys()}
        forward_data = {k: data[k] for k in ["SOC", "inputs", "Cy0Year"]}
        return ForwardSoilModelData(**forward_data, **soil_c_data)


class InverseSoilModelBaseSchema(SoilModelBaseSchema):
    eq_C = fields.List(fields.Float, required=True)
    input_C = fields.Float(required=True)
    x = fields.List(fields.Float, required=True)

    @post_load
    def build_inverse_soil_model(self, data, **kwargs):
        soil_c_data = {k: data[k] for k in SoilModelBaseSchema().fields.keys()}
        inverse_data = {k: data[k] for k in ["eq_C", "input_C", "x"]}
        return InverseSoilModelData(**inverse_data, **soil_c_data)
