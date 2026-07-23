#!/usr/bin/python

"""Module holding Crop class for the crop model."""
from typing import Tuple, List

import numpy as np
from marshmallow import Schema, fields, post_load

from .common import csv_handler
from .common_schema import OutputSchema as ClimateDataOutputSchema

from .crop_params import (
    CropParamsData,
    CropParamsSchema,
)
import model.common.constants as CONSTANTS
from .crop_params import from_species_index as create_crop_params_from_species_index


class CropModelData:
    """
    Crop model object. Calculate residues and soil inputs
    for given parameters.

    Instance variables
    ------------------
    crop_params     CropParams object with crop params (slope, carbon, etc.)
    output          output to soil,fire in t C ha^-1
                    (dict with keys 'carbon','nitrogen','DMon','DMoff')
    """

    def __init__(self, crop_params, output):
        self.crop_params = crop_params
        self.output = output


class ClimateDataSchema(Schema):
    crop_params = fields.Nested(CropParamsSchema)
    output = fields.Nested(ClimateDataOutputSchema)

    @post_load
    def build(self, data, **kwargs):
        return CropModelData(**data)


def create(crop_params, no_of_years, crop_yield, left_in_field) -> CropModelData:
    """Args:
    crop_params: CropParams object with crop params
    crop_yield: dry matter yield of the crop in t C ha^-1
    left_in_field: fraction of residues left in field post-harvest
    """
    raw_crop_model_data = {
        "crop_params": vars(
            crop_params
        ),  # We need to convert this to a dictionary for validation
        "output": get_inputs(crop_params, no_of_years, crop_yield, left_in_field),
    }

    schema = ClimateDataSchema()
    errors = schema.validate(raw_crop_model_data)

    if errors != {}:
        print(f"Errors in crop model data: {str(errors)}")

    return schema.load(raw_crop_model_data)  # type: ignore


def get_inputs(crop_params, no_of_years, crop_yield, left_in_field):
    """Calculate and return soil carbon inputs, nitrogen inputs,
    on-farm residues, and off-farm residues from soil parameters.

    Use the IPCC slope and intercept values
    outlined in 2006 national GHG assessment guidelines (table 11.2)
    to model

    Args:
        crop_yield: yearly dry-matter crop yield (in t C ha^-1)
        left_in_field: fraction left in field after harvest
    Returns:
        output: dict with soil,fire inputs due to crop
                    (keys='carbon','nitrogen','DMon','DMoff')
    """

    # residues
    residue = crop_yield * crop_params.slope + np.where(crop_yield > 0, crop_params.intercept, 0)
    residue *= np.ones(no_of_years)  # convert to array
    residue_AG = residue * left_in_field
    residue_BG = crop_yield + residue
    residue_BG *= crop_params.root_to_shoot * CONSTANTS.CROP_ROOT_IN_TOP_30

    output = {}

    # Standard outputs - in tonnes of carbon and as vectors
    output["above"] = {
        "carbon": residue_AG * crop_params.carbon_above,
        "nitrogen": residue_AG * crop_params.nitrogen_above,
        "DMon": residue_AG,
        "DMoff": residue * (1 - left_in_field),
    }
    output["below"] = {
        "carbon": residue_BG * crop_params.carbon_below,
        "nitrogen": residue_BG * crop_params.nitrogen_below,
        "DMon": residue_BG,
        "DMoff": np.zeros(len(residue)),
    }

    return output


def save(crop_model, file="crop_model.csv"):
    """Save output of crop model to a csv file.
    Default path is in OUTPUT_DIR.

    Args:
        file: name or path to csv file
    """
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(crop_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)


def get_crop_models_and_crop_params(
    input_data, no_of_years, start_index, end_index, crop_getter
):
    results = [
        crop_getter(input_data, no_of_years, index)
        for index in range(start_index, end_index + 1)
    ]

    # Unzip the results into two separate lists
    crop_models, crop_params = zip(*results)

    # Convert the results to lists (as zip returns tuples)
    return list(crop_models), list(crop_params)


def get_crop_bases(
    input_data, no_of_years, start_index, end_index
) -> Tuple[List[CropModelData], List[CropParamsData]]:
    return get_crop_models_and_crop_params(
        input_data, no_of_years, start_index, end_index, get_crop_base
    )


def get_crop_projects(
    input_data, no_of_years, start_index, end_index
) -> Tuple[List[CropModelData], List[CropParamsData]]:
    return get_crop_models_and_crop_params(
        input_data, no_of_years, start_index, end_index, get_crop_project
    )


def get_crop_data(
    input_data, no_of_years, prefix, index
) -> Tuple[CropModelData, CropParamsData]:
    spp = int(input_data[f"{prefix}_spp{index}"])
    harvest_yield = np.zeros(no_of_years)
    harvest_yield[
        int(input_data[f"{prefix}_start{index}"]) : int(
            input_data[f"{prefix}_end{index}"]
        )
    ] = float(input_data[f"{prefix}_yd{index}"])
    harv_frac = float(input_data[f"{prefix}_left{index}"])
    if harvest_yield.sum() > 0 and spp == 0:
        raise ValueError(
            f"Crop yield is greater than 0 but species index is 0 for {prefix} index {index}. Please check your input data."
        )
    crop_params = create_crop_params_from_species_index(spp)
    crop_model = create(
        crop_params=crop_params,
        no_of_years=no_of_years,
        crop_yield=harvest_yield,
        left_in_field=harv_frac,
    )

    return crop_model, crop_params


def get_crop_base(
    input_data, no_of_years, index
) -> Tuple[CropModelData, CropParamsData]:
    return get_crop_data(input_data, no_of_years, "crop_base", index)


def get_crop_project(
    input_data, no_of_years, index
) -> Tuple[CropModelData, CropParamsData]:
    return get_crop_data(input_data, no_of_years, "crop_proj", index)
