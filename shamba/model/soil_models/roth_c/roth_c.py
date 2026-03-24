import logging as log
import sys

import numpy as np
from marshmallow import Schema, fields

from ...climate import ClimateDataSchema
from ...soil_params import SoilParamsSchema
from ..soil_model_types import BaseSoilModelData, SoilModelBaseSchema


# Class variables (defaults)
K_BASE = np.array([10.0, 0.3, 0.66, 0.02])


def create(soil, climate, cover, no_of_years):
    """Creates rothc object.

    Args:
        soil: SoilParams object with soil parameters
        climate: Climate object with climate parameters
        cover: monthly cover vector (1=covered, 0=uncovered)

    """

    params = {
        "soil_params": vars(soil),
        "climate": vars(climate),
        "cover": cover,
        "k": get_rmf(climate=climate, cover=cover, soil=soil, no_of_years=no_of_years)[..., np.newaxis] * K_BASE,
    }

    schema = SoilModelBaseSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in RothC data: {str(errors)}")

    validated_data = schema.load(params)
    return BaseSoilModelData(**validated_data)  # type: ignore


# Rate modifying-factor function - needed in forward and inverse
def get_rmf(climate, cover, soil, no_of_years):
    """Calculate the rate modifying factor for each year based on climate and soil cover.

    Supports both single-pattern climate/cover (12 values, repeated for all years)
    and multi-year climate/cover (12 * no_of_years values).

    Returns:
        rmf: array of shape (no_of_years,) — yearly mean of the combined RMF
    """
    cc = soil.clay
    d = soil.depth
    rmf = np.zeros(no_of_years)

    multi_year_climate = len(climate.temperature) >= 12 * no_of_years
    multi_year_cover = len(cover) >= 12 * no_of_years

    for y in range(no_of_years):
        # Slice the appropriate 12-month window, or use the single pattern.
        if multi_year_climate:
            temp = climate.temperature[y*12:(y+1)*12]
            rain = climate.rain[y*12:(y+1)*12]
            evap = climate.evaporation[y*12:(y+1)*12]
        else:
            temp = climate.temperature
            rain = climate.rain
            evap = climate.evaporation

        cover_year = cover[y*12:(y+1)*12] if multi_year_cover else cover

        # Calculation of b (topsoil moisture deficit RMF)
        # Deficit is difference between rain and evaporation (pet/0.75)
        deficit = rain - evap
        m = get_first_pos_def(deficit)
        m, rain_always_exceeds_evaporation = get_first_neg_def(deficit, m)
        b = np.ones(12)
        if rain_always_exceeds_evaporation:
            rmf[y] = b.mean()
            continue

        # Rainfall < evap in month m, so start calculating SMD from month before m
        m -= 1

        max_smd = -(20 + 1.3 * cc - 0.01 * (cc**2)) * (d / 23.0)
        accumulator_tsmd = 0.0

        # Now define deficit as rain - pet
        deficit = rain - evap * 0.75

        # Loop through each month
        for i in range(12):
            accumulator_tsmd = get_acc_tsmd(accumulator_tsmd, deficit[m], cover_year[m], max_smd)
            if accumulator_tsmd >= 0.444 * max_smd:
                b[m] = 1
            elif accumulator_tsmd >= max_smd:
                b[m] = 0.2 + 0.8 * (max_smd - accumulator_tsmd) / ((1 - 0.444) * max_smd)
            else:
                log.error("DEFICIT = %5.2f" % accumulator_tsmd)
                sys.exit(1)

            m += 1
            if m > 11:
                m = 0

        # Temperature RMF (a)
        a = np.zeros(12)
        warm = temp > -5.0
        a[warm] = 47.91 / (1.0 + np.exp(106.06 / (temp[warm] + 18.27)))

        # Soil cover RMF (c)
        c = np.ones(12)
        c[cover_year == 1] = 0.6

        rmf[y] = (a * b * c).mean()

    return rmf


# Helper methods for finding b (topsoil moisture RMF)
# Find first month where deficit > 0
def get_first_pos_def(deficit):
    is_sane = False
    m = 0
    for i in np.where(deficit > 0):
        if any(i):  # could be empty list
            is_sane = True
            m = min(i)
            break

    if not is_sane:
        log.warning("EVAPORATION ALWAYS EXCEED RAINFALL")
        m = 0

    return m  # first month where deficit > 0


# Find first month after m where rainfall < evap (deficit<0)
def get_first_neg_def(deficit, m):
    rain_always_exceeds_evaporation = True
    for i in range(12):
        m += 1
        if m > 11:
            m = 0
        if deficit[m] < 0:
            rain_always_exceeds_evaporation = False
            break

    return m, rain_always_exceeds_evaporation


# Get accumulator_TSMD for a given month
def get_acc_tsmd(smd, def_m, cover_m, max):
    if def_m > 0:
        # Add excess rain to SMD
        smd += def_m
        if smd > 0:
            smd = 0
    else:
        # deficit < 0
        if cover_m == 1:
            # Crop present, so increase SMD
            smd += def_m
            if smd < max:
                smd = max
        else:
            # Crop not present (see BareSMD in RothC user guide)
            if smd < max / 1.8:
                pass
            else:
                # Increase SMD
                smd += def_m
                if smd < max / 1.8:
                    smd = max / 1.8
    # End if-else for deficit > 0
    return smd


def dC_dt(C, t, x, k, input):
    """Function for system of differential equations governing
    amounts of carbon in each pool (vector C).

    Args:
        C: vector of carbon pools
        t: time (not used, but needed for the scipy.optimize fit)
        x: partitioning coefficient
        k: rate constants
        input: soil input in the given year
    Returns:
        rhs: array of the RHS of dC/dt.

    """

    # carbon gain from decay (goes to BIO, HUM, CO2)
    bio_humin = C[0] * k[0] + C[1] * k[1] + C[2] * k[2] + C[3] * k[3]

    rhs = np.array(
        [
            input * x[0] - C[0] * k[0],
            input * x[1] - C[1] * k[1],
            bio_humin * x[2] - C[2] * k[2],
            bio_humin * x[3] - C[3] * k[3],
        ]
    )

    return rhs
