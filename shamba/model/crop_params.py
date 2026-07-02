#!/usr/bin/python


import logging as log
import sys

import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler
import model.common.constants as CONSTANTS

# --------------------------
# Read species data from csv
# --------------------------

# Read csv file with default crop data
def load_crop_species_data(
    filename: str = "crop_params.csv",
) -> dict[int, dict]:
    """
    Load crop species data from CSV file, keyed by each row's own Sc
    (species code) column — not by row position. Each species' display name
    is read from the file's own Name column (index 1).

    Args:
        filename: Name of the CSV file to load (default: "crop_params.csv")

    Returns:
        Dictionary mapping species code (Sc) to its parameter dictionary.
    """
    resolved_path = csv_handler.resolve_csv_path(filename)
    data = np.atleast_2d(
        np.genfromtxt(resolved_path, skip_header=1, usecols=(0, 2, 3, 4, 5, 6, 7, 8), delimiter=",", comments="#")
    )
    # separate imports to 
    names = np.atleast_1d(
        np.genfromtxt(resolved_path, skip_header=1, usecols=(1,), dtype=str, delimiter=",", comments="#")
    )

    if np.isnan(data).any():
        bad_rows = [r + 2 for r in np.where(np.isnan(data).any(axis=1))[0]]
        bad_rows.append([r+2 for r in np.where(np.isnan(names).any(axis=1))[0]])
        raise ValueError(
            f"'{filename}' has a missing/blank numeric value in row(s) {bad_rows} "
            f"(counting the header as row 1). Every crop row must have a value "
            f"in every column (Sc, a, b, crop_bgn, crop_agn, crop_bgc, crop_agc, crop_rs)."
        )
    blank_name_rows = [r + 2 for r in np.where(names == "")[0]]
    if blank_name_rows:
        raise ValueError(f"'{filename}' has a missing Name in row(s) {blank_name_rows}.")


    sc_codes = data[:, 0]
    slope = data[:, 1]
    intercept = data[:, 2]
    nitrogenBelow = data[:, 3]
    nitrogenAbove = data[:, 4]
    carbonBelow = data[:, 5]
    carbonAbove = data[:, 6]
    rootToShoot = data[:, 7]

    species_data = {}
    for i, sc in enumerate(sc_codes):
        species_code = int(sc)
        if species_code in species_data:
            raise ValueError(
                f"'{filename}' has more than one row with species code (Sc) "
                f"{species_code} — each species code must appear exactly once."
            )
        species_data[species_code] = {
            "species": str(names[i]).strip(),
            "species_code": species_code,
            "slope": slope[i],
            "intercept": intercept[i],
            "nitrogen_below": nitrogenBelow[i],
            "nitrogen_above": nitrogenAbove[i],
            "carbon_below": carbonBelow[i],
            "carbon_above": carbonAbove[i],
            "root_to_shoot": rootToShoot[i],
        }
    return species_data


class CropParamsData:
    """
    Crop object to hold crop params. Can be initialised from species name,
    species index, csv file, or manually (callling __init__ with params
    in a dict)

    Instance variables
    ------------------
    species         crop species display name
    species_code     crop species code (Sc column in crop_params.csv)
    slope           crop IPCC slope
    intercept       crop IPCC y-intercept
    nitrogen_below   crop below-ground nitrogen content as a fraction
    nitrogen_above   crop above-ground nitrogen content as a fraction
    carbon_below     crop below-ground carbon content as a fraction
    carbon_above     crop above-ground carbon content as a fraction
    root_to_shoot     crop root-to-shoot ratio

    """

    def __init__(
        self,
        species,
        species_code,
        slope,
        intercept,
        nitrogen_below,
        nitrogen_above,
        carbon_below,
        carbon_above,
        root_to_shoot,
    ):
        self.species = species
        self.species_code = species_code
        self.slope = slope
        self.intercept = intercept
        self.nitrogen_below = nitrogen_below
        self.nitrogen_above = nitrogen_above
        self.carbon_below = carbon_below
        self.carbon_above = carbon_above
        self.root_to_shoot = root_to_shoot


class CropParamsSchema(Schema):
    species = fields.String(required=True)
    species_code = fields.Integer(required=True)
    slope = fields.Float(required=True)
    intercept = fields.Float(required=True)
    nitrogen_below = fields.Float(required=True)
    nitrogen_above = fields.Float(required=True)
    carbon_below = fields.Float(required=True)
    carbon_above = fields.Float(required=True)
    root_to_shoot = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return CropParamsData(**data)


def from_species_index(index) -> CropParamsData:
    """Construct Crop object from its species code (Sc column in crop_params.csv).

    Args:
        index: species code to look up
    Return:
        Crop object
    Raises:
        KeyError: if the species code isn't present in crop_params.csv

    """
    index = int(index)
    species_data = load_crop_species_data()
    if index not in species_data:
        raise KeyError(
            f"No crop species with code {index} found in crop_params.csv "
            f"(available codes: {sorted(species_data)})."
        )
    schema = CropParamsSchema()
    return schema.load(species_data[index])  # type: ignore


def save(crop_params, file="crop_params.csv"):
    """Save crop params in a csv.
    Default path is in OUTPUT_DIR with filename 'crop_params.csv'

    Args:
        file: name or path to csv file

    """
    data = [
        crop_params.species_code,
        crop_params.species,
        crop_params.slope,
        crop_params.intercept,
        crop_params.nitrogen_below,
        crop_params.nitrogen_above,
        crop_params.carbon_below,
        crop_params.carbon_above,
        crop_params.root_to_shoot,
    ]
    cols = [
        "Sc",
        "Name",
        "a",
        "b",
        "crop_bgn",
        "crop_agn",
        "crop_bgc",
        "crop_agc",
        "crop_rs",
    ]
    csv_handler.print_csv(file, data, col_names=cols)
