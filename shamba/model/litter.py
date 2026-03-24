import logging as log

import numpy as np
from marshmallow import Schema, fields, post_load, ValidationError

from .common import csv_handler
from .common_schema import OutputSchema as LitterDataOutputSchema
import model.common.constants as CONSTANTS
from .common.validations import validate_between_0_and_1


class LitterModelData:
    """
    Litter model object. Read litter params
    and calculate residues and inputs to the soil.

    Instance variables
    ------------------
    carbon      litter carbon content
    nitrogen    litter nitrogen content
    output      output to soil,fire (dict with keys 'DMon','DMoff,'carbon','nitrogen')
    """

    def __init__(self, carbon, nitrogen, output):
        self.carbon = carbon
        self.nitrogen = nitrogen
        self.output = output


class LitterDataSchema(Schema):
    carbon = fields.Float(required=True)
    nitrogen = fields.Float(required=True)
    output = fields.Nested(LitterDataOutputSchema)

    @post_load
    def build(self, data, **kwargs):
        return LitterModelData(**data)


def create(
    litter_params, litter_vector, nitrogen_vector=None
) -> LitterModelData:
    """Create LitterModelData object.

    Args:
        litter_params:      dict with litter params (keys='carbon','nitrogen')
        litter_vector:      annual vector of DM additions (t DM ha^-1)
        nitrogen_vector:    annual vector of N fractions; overrides computing N
                            from the scalar nitrogen fraction in litter_params
    Returns:
        LitterModelData: object containing litter parameters
    """
    validate_between_0_and_1(
        [litter_params["carbon"], litter_params["nitrogen"]]
    )

    carbon = litter_params["carbon"]
    nitrogen = litter_params["nitrogen"]
    params = {
        "carbon": carbon,
        "nitrogen": nitrogen,
        "output": get_inputs(
            carbon=carbon,
            nitrogen=nitrogen,
            litter_vector=litter_vector,
            nitrogen_vector=nitrogen_vector,
        ),
    }

    schema = LitterDataSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in litter model data: {str(errors)}")

    return schema.load(params)  # type: ignore


def from_defaults(litter_vector):
    """
    Same as create, but with default litter parameters.
    """
    params = {
        "carbon": CONSTANTS.ORGANIC_INPUT_C,
        "nitrogen": CONSTANTS.ORGANIC_INPUT_N,
    }
    return create(litter_params=params, litter_vector=litter_vector)


def synthetic_fertiliser(quantity_vector, nitrogen_vector):
    """Synthetic fertiliser (special case of litter).
    Be sure to keep separate though when passing a litter object to
    other methods/classes. (e.g. fert isn't an input to soil model)"""
    # carbon=0: synthetic fertiliser adds no carbon to soil
    # nitrogen=0: placeholder only — nitrogen_vector overrides it in get_inputs
    params = {"carbon": 0, "nitrogen": 0}

    return create(
        litter_params=params,
        litter_vector=quantity_vector,
        nitrogen_vector=nitrogen_vector,
    )


def get_inputs(carbon, nitrogen, litter_vector, nitrogen_vector=None):
    """Calculate and return DM, C, and N inputs to
    soil from additional litter.

    Args:
        carbon: litter carbon content
        nitrogen: litter nitrogen content
        litter_vector: annual vector of DM additions (t DM ha^-1)
        nitrogen_vector: annual vector of N fractions; overrides nitrogen if provided
    Returns:
        output: dict with soil,fire inputs due to litter (keys='carbon','nitrogen','DMon','DMoff')

    """
    DMinput = np.array(litter_vector)
    Cinput = DMinput * carbon
    if nitrogen_vector is not None:
        Ninput = DMinput * np.array(nitrogen_vector)
    else:
        Ninput = DMinput * nitrogen

    # Standard output (same as crop and tree classes)
    output = {}
    output["above"] = {
        "carbon": Cinput,
        "nitrogen": Ninput,
        "DMon": DMinput,
        "DMoff": np.zeros(len(Cinput)),
    }
    output["below"] = {
        "carbon": np.zeros(len(Cinput)),
        "nitrogen": np.zeros(len(Cinput)),
        "DMon": np.zeros(len(Cinput)),
        "DMoff": np.zeros(len(Cinput)),
    }

    return output


def save(litter_model, file="litter_model.csv"):
    """Save output to a csv. Default path is OUTPUT_DIR

    Args:
        file: name or path to csv

    """
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(litter_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)
