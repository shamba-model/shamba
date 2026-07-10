import math

import numpy as np
from marshmallow import fields, post_load
from scipy import optimize

from ...common import csv_handler
from .roth_c import create as create_roth_c, dC_dt
from ..soil_model_params import RothCParams
from ..soil_model_types import (
    InverseSoilModelData,
    BaseSoilModelData,
    InverseSoilModelBaseSchema,
)


def create(
    soil, climate, cover=np.ones(12), soil_model_params: RothCParams = RothCParams()
) -> InverseSoilModelData:
    """Creates inverse rothc object.

    Args:
        soil: soil object with soil parameters
        climate: climate object with climate parameters
        cover: monthly soil cover vector
        soil_model_params: RothC rate-modifier and partitioning constants (defaults
            are RothC standard values). Generic naming because other soil models could be used.


    """
    roth_c = create_roth_c(soil, climate, cover, no_of_years=1, roth_c_params=soil_model_params)

    eq_C, input_C, x = solver(roth_c, roth_c_params=soil_model_params)

    params = {
        "soil_params": vars(roth_c.soil),
        "climate": vars(roth_c.climate),
        "cover": roth_c.cover,
        "k": roth_c.k,
        "eq_C": eq_C,
        "input_C": input_C,
        "x": x,
    }

    schema = InverseSoilModelBaseSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in InverseRothC data: {str(errors)}")

    return schema.load(params)  # type: ignore


def solver(roth_c, roth_c_params: RothCParams = RothCParams()):
    """Run RothC in 'inverse' mode to find inputs
    giving equilibrium and Carbon pool values at equilibrium.

    Args:
        roth_c_params: DPM/RPM partitioning constants (defaults are RothC standard values)

    Returns:
        eq_C: equilibrium distribution of carbon
        eq_input: yearly tree input giving equilibrium
        x: partitioning coefficient for pools

    """

    # Partioning coefficients
    x = get_partitions(roth_c, roth_c_params=roth_c_params)
    t = 0  # just needed to define for dC_dt function
    C0 = np.array([0.1, 10, 0, 0])  # initial ballpark guess

    # Keep track of difference b/t Ctot and Ceq
    previous_difference = 1000.0

    # Loop through range of inputs
    for input in np.arange(0.1, 10, 0.1):
        C = optimize.fsolve(dC_dt, C0, args=(t, x, roth_c.k[0], input))
        Ctot = np.sum(np.array(C)) + roth_c.soil.iom
        current_difference = math.fabs(Ctot - roth_c.soil.Ceq)

        if current_difference < previous_difference:
            previous_difference = current_difference
            input_C = input
            previous_C = C

        if previous_difference < current_difference:
            break

    eq_C = previous_C
    eq_input = input_C
    return eq_C, eq_input, x


def get_partitions(roth_c, roth_c_params: RothCParams = RothCParams()):
    """Calculate partitioning coefficients.

    Args:
        roth_c_params: DPM/RPM partitioning constants (defaults are RothC standard values)

    Returns:
        x: partitioning coefficient for the 4 pools

    """

    # Determine p_2 (fraction of input going to CO2) based on clay content
    z = 1.67 * (1.85 + 1.6 * math.exp(-0.0786 * roth_c.soil.clay))  # CO2/(BIO + HUM)
    p2 = z / (z + 1)  # CO2

    # p_3 is always 0.46 (see RothC user guide)
    p3 = 0.46  # BIO proportion of BIO+HUM

    # p_1 is dpm fraction of input:
    #   deciduous tropical woodland: dpm=dpm_frac_tree, rpm=1-dpm_frac_tree
    #   crops: dpm=dpm_frac_crop, rpm=1-dpm_frac_crop
    # no crops at equilibrium, so p1 is always dpm_frac_tree
    p1 = roth_c_params.dpm_frac_tree

    # DPM fraction, RPM fraction, BIO fraction, HUM fraction
    return np.array([p1, 1 - p1, p3 * (1 - p2), (1 - p2) * (1 - p3)])
