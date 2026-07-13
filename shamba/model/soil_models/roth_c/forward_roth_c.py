import math
import os

import matplotlib.pyplot as plt
import numpy as np
from marshmallow import fields, post_load
from scipy import integrate
from tabulate import tabulate
from typing import List, Tuple, Optional

from ... import configuration, emit
from ...common import csv_handler
from .roth_c import SoilModelBaseSchema
from .roth_c import create as create_roth_c
from .roth_c import dC_dt
from ..soil_model_params import RothCParams
from ..soil_model_types import ForwardSoilModelData, ForwardSoilModelBaseSchema


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
    soil_model_params: RothCParams = RothCParams(),
) -> ForwardSoilModelData:
    """Creates ForwardSoilModelData.

    Args:
        soil: soil object
        climate: climate object
        cover: soil cover vector
        Ci: initial carbon pools (for solver)
        crop: list of crop objects which provide carbon to soil
        tree: list of tree objects which provide carbon to soil
        litter: list of litter objects which provide carbon to soil
        solve_to_value: whether to solve to value (to Cy0) or by time
        soil_model_params: RothC rate-modifier and partitioning constants (defaults
            are RothC standard values). Generic naming because other soil models could be used.

    """
    roth_c = create_roth_c(soil, climate, cover, no_of_years, roth_c_params=soil_model_params)

    SOC, inputs, Cy0Year = solver(
        roth_c, Ci, no_of_years, crop, tree, litter, fire, solve_to_value,
        roth_c_params=soil_model_params,
    )

    params = {
        "soil_params": vars(roth_c.soil),
        "climate": vars(roth_c.climate),
        "cover": roth_c.cover,
        "k": roth_c.k,
        "SOC": SOC,
        "inputs": inputs,
        "Cy0Year": Cy0Year,
    }

    schema = ForwardSoilModelBaseSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in ForwardRothC data: {str(errors)}")

    return schema.load(params)  # type: ignore


def solver(
    roth_c, Ci, no_of_years, crop=[], tree=[], litter=[], fire=[], solve_to_value=False,
    roth_c_params: RothCParams = RothCParams(),
):
    """Run RothC in 'forward' mode;
    solve dC_dt over a given time period
    or to a certain value, given a vector with soil inputs.
    Use scipy.integrate.odeint as a Runge-Kutta solver.

    Args:
        crop: list of Crop objects (not reduced by fire)
        tree: list of Tree objects (not reduced by fire)
        solve_to_value: whether to solve to a particular value (Cy0)
                        as opposed to for a certain amount of time
    Returns:
        C: vector with yearly distribution of soil carbon
        inputs: yearly inputs to soil with 2 columns (crop,tree)
        year_target_reached: year that the target value
                (if solve_to_value==True) of Cy0 was reached
                since it will (probably) not be on nice round num.

    """

    # Reduce inputs due to fire
    soilIn_crop, soilIn_tree = emit.reduce_from_fire(
        no_of_years=no_of_years, crop=crop, tree=tree, litter=litter, fire=fire
    )

    # make input into array with 2 columns (soilIn_crop,soilIn_tree)
    inputs = np.column_stack((soilIn_crop, soilIn_tree))

    # Calculate yearly values of x based dpm:rpm ratio for each year
    x = get_partitions(roth_c, inputs, no_of_years, roth_c_params=roth_c_params)
    t = np.arange(0, 1, 0.001)
    C = np.zeros((no_of_years + 1, 4))
    C[0] = Ci
    Ctot = np.zeros(no_of_years + 1)

    # keep track of when target year reached
    year_target_reached = 0
    # To keep track of closest value to SOCf
    if solve_to_value:
        previous_difference_outer = 1000.0
        previous_difference_inner = 1000.0

    for i in range(1, no_of_years + 1):
        # Careful with indices - using 1-based for arrays here
        # since, e.g., C[2] should correspond to carbon after 2 years

        # Solve the diffEQs to get pools for year i
        Ctemp = integrate.odeint(
            dC_dt, C[i - 1], t, args=(x[i - 1], roth_c.k[i-1], inputs[i - 1].sum())
        )
        C[i] = Ctemp[-1]  # carbon pools at end of year

        # Check to see if close to target value
        if solve_to_value:
            j = -1
            for c in Ctemp:
                j += 1
                Ctot = c.sum() + roth_c.soil.iom
                current_difference = math.fabs(Ctot - roth_c.soil.Cy0)

                if current_difference - previous_difference_inner < 0.00000001:
                    previous_difference_inner = current_difference
                    c_inner = c
                    closest_j = j
                else:  # getting farther away
                    break

            if previous_difference_inner < previous_difference_outer:
                previous_difference_outer = previous_difference_inner
                c_outer = c_inner
                closest_i = i
            else:
                break

    if solve_to_value:
        C[closest_i] = c_outer
        C = C[0 : closest_i + 1]
        year_target_reached = float(closest_j) / len(t)
        inputs = inputs[0 : len(C[:, 0]) - 1]

    return C, inputs, year_target_reached


def get_partitions(
    roth_c, inputs, no_of_years, roth_c_params: RothCParams = RothCParams()
):
    """Calculate partitioning coefficients.

    Args:
        input: 2d array with crop_in,tree_in inputs as the columns
        roth_c_params: DPM/RPM partitioning constants (defaults are RothC standard values)
    Returns:
        x: partitioning coefficient (as a vector for each year)

    """

    # Determine p_2 (fraction of input going to CO2) based on clay content
    z = 1.67 * (1.85 + 1.6 * math.exp(-0.0786 * roth_c.soil.clay))
    p2 = z / (z + 1)

    # p_3 is always 0.46 (see RothC papaer
    p3 = 0.46

    # p_1 is dpm fraction of input:
    #   deciduous tropical woodland: dpm=0.2, rpm=0.8
    #   crops: dpm=0.59, rpm=0.41
    # So weigh p_1 according to amount of trees/crops for each year

    # Normalize
    i = 0
    norm_input = np.zeros((no_of_years, 2))
    for row in inputs:
        if math.fabs(float(row.sum())) < 0.00000001:
            norm_input[i] = 0
        else:
            norm_input[i] = inputs[i] / float(row.sum())
        i += 1

    # Weighted mean of dpm
    p1 = np.array(
        roth_c_params.dpm_frac_crop * norm_input[:, 0]
        + roth_c_params.dpm_frac_tree * norm_input[:, 1]
    )

    # Construct x
    # make arrays (p1 already array)
    x = np.column_stack(
        (
            p1,
            1 - p1,
            np.array(no_of_years * [p3 * (1 - p2)]),
            np.array(no_of_years * [(1 - p2) * (1 - p3)]),
        )
    )

    return x
