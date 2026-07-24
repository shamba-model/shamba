import logging as log

import numpy as np
from marshmallow import Schema, fields, post_load, ValidationError

from .common import csv_handler
from .common_schema import OutputSchema as LitterDataOutputSchema
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
    litter_params, litter_frequency, litter_quantity, no_of_years, litter_vector=None
) -> LitterModelData:
    """Create LitterModelData object.

    Args:
        litter_params:  dict with litter params(keys='carbon','nitrogen')
        litter_frequency:     frequency of litter application
        litter_quantity:      Quantity (in t C ha^-1) of litter when applied.
        litter_vector:   vector with custom litter additions (t DM / ha)
                        (e.g. for when litter isn't at regular freq.)
                        -> overrides any quantity and freq. info
    Returns:
        LitterModelData: object containing litter parameters
    """
    errors = validate_between_0_and_1(
        [litter_params["carbon"], litter_params["nitrogen"]]
    )

    if errors:
        raise ValidationError(errors)

    carbon = litter_params["carbon"]
    nitrogen = litter_params["nitrogen"]
    params = {
        "carbon": carbon,
        "nitrogen": nitrogen,
        "output": get_inputs(
            carbon=carbon,
            nitrogen=nitrogen,
            litter_frequency=litter_frequency,
            litter_quantity=litter_quantity,
            litter_vector=litter_vector,
            no_of_years=no_of_years,
        ),
    }

    schema = LitterDataSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in litter model data: {str(errors)}")

    return schema.load(params)  # type: ignore


def from_csv(
    litter_frequency,
    litter_quantity,
    no_of_years,
    filename="litter.csv",
    row=0,
    litter_vector=None,
):
    """
    Same as create, but with litter parameters from a csv file.
    """

    data = csv_handler.read_csv(filename)
    data = np.atleast_2d(data)
    try:
        params = {"carbon": data[row, 0], "nitrogen": data[row, 1]}
        litter = create(
            litter_params=params,
            litter_frequency=litter_frequency,
            litter_quantity=litter_quantity,
            no_of_years=no_of_years,
            litter_vector=litter_vector,
        )
    except IndexError:
        log.exception("Can't find row %d in %s" % (row, filename))
        raise

    return litter


def from_defaults(litter_frequency, litter_quantity, no_of_years, litter_vector=None):
    """
    Same as create, but with litter carbon/nitrogen content read from
    litter_params.csv (falls back to litter_params_defaults.csv if the
    project's input folder doesn't supply its own).
    """
    return from_csv(
        litter_frequency=litter_frequency,
        litter_quantity=litter_quantity,
        no_of_years=no_of_years,
        filename="litter_params.csv",
        litter_vector=litter_vector,
    )


def synthetic_fertiliser(frequency, quantity, nitrogen, no_of_years, vector=None):
    """Synthetic fertiliser (special case of litter).
    Be sure to keep separate though when passing a litter object to
    other methods/classes. (e.g. fert isn't an input to soil model)"""
    params = {"carbon": 0, "nitrogen": nitrogen}

    return create(
        litter_params=params,
        litter_frequency=frequency,
        litter_quantity=quantity,
        no_of_years=no_of_years,
        litter_vector=vector,
    )


def get_inputs(
    carbon, nitrogen, litter_frequency, litter_quantity, litter_vector, no_of_years
):
    """Calculate and return DM, C, and N inputs to
    soil from additional litter.

    Args:
        carbon: litter carbon content
        nitrogen: litter nitrogen content
        litter_frequency: frequency of litter application
        litter_quantity: amount of dry matter added to field when litter added in t DM ha^-1
        litter_vector: vector with custom litter additions (t DM / ha)
        no_of_years: number of years in the project
    Returns:
        output: dict with soil,fire inputs due to litter (keys='carbon','nitrogen','DMon','DMoff')

    """

    if litter_vector is None:
        # Construct vectors for DM, C, N
        Cinput = np.zeros(no_of_years)
        Ninput = np.zeros(no_of_years)
        DMinput = np.zeros(no_of_years)

        # Years when litter is applied.
        # Note: `litter_frequency == 0` means "no applications" 
        # (consistent with fire interval handling).
        if litter_frequency == 0:
            years = np.array([], dtype=int)
        else:
            years = np.arange(0, no_of_years, litter_frequency, dtype=int)

        DMinput[years] = litter_quantity
        Cinput[years] = litter_quantity * carbon
        Ninput[years] = litter_quantity * nitrogen
    else:
        # DM vector already specified
        DMinput = np.array(litter_vector)
        Cinput = DMinput * carbon
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
