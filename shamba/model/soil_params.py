from typing import Dict, Any, Tuple
import numpy as np
from marshmallow import Schema, fields, post_load, validates_schema, ValidationError

from model.common import csv_handler
from model.common.data_sources.soil import get_soil_data, SoilData


def validate_clay(value):
    return ["Clay value must be between 0 and 100."] if value < 0 or value > 100 else []


def validate_Cy0(value):
    return (
        ["Cy0 value must be between 0 and 10000."] if value < 0 or value > 10000 else []
    )


class SoilParamsData:
    def __init__(self, Cy0, clay, depth, Ceq, iom, Cy0_q05, Cy0_q95, clay_q05, clay_q95):
        self.Cy0 = Cy0
        self.clay = clay
        self.depth = depth
        self.Ceq = Ceq
        self.iom = iom
        # Quantile fields for Monte Carlo uncertainty.
        # When no uncertainty data is available, q05 = q95 = mean (deterministic).
        self.Cy0_q05 = Cy0_q05
        self.Cy0_q95 = Cy0_q95
        self.clay_q05 = clay_q05
        self.clay_q95 = clay_q95


class SoilParamsSchema(Schema):
    """
    Object to hold soil parameter information.

    Instance variables
    ------------------
    Cy0      soil carbon at start of project in t C ha^-1 (year 0)
    clay     soil clay content as percentage
    depth    depth of soil in cm (default=30)
    Ceq      soil carbon at equilibrium in t C ha^-1 (calculated from Cy0)
    iom      soil inert organic matter in t C ha^-1 (calculated from Ceq)
    Cy0_q05  Q0.05 quantile for Cy0 (for Monte Carlo; equals Cy0 when no uncertainty)
    Cy0_q95  Q0.95 quantile for Cy0
    clay_q05 Q0.05 quantile for clay
    clay_q95 Q0.95 quantile for clay
    """

    clay = fields.Float(required=True, validate=lambda v: validate_clay(v))
    Cy0 = fields.Float(required=True, validate=lambda v: validate_Cy0(v))
    depth = fields.Float(required=True)
    Ceq = fields.Float(required=True)
    iom = fields.Float(required=True)
    Cy0_q05 = fields.Float(required=True, validate=lambda v: validate_Cy0(v))
    Cy0_q95 = fields.Float(required=True, validate=lambda v: validate_Cy0(v))
    clay_q05 = fields.Float(required=True, validate=lambda v: validate_clay(v))
    clay_q95 = fields.Float(required=True, validate=lambda v: validate_clay(v))

    @validates_schema
    def validate_quantile_ordering(self, data, **kwargs):
        errors = []
        if not (data["Cy0_q05"] <= data["Cy0"] <= data["Cy0_q95"]):
            errors.append("Cy0_q05 must be <= Cy0 <= Cy0_q95.")
        if not (data["clay_q05"] <= data["clay"] <= data["clay_q95"]):
            errors.append("clay_q05 must be <= clay <= clay_q95.")
        if errors:
            raise ValidationError(errors)

    @post_load
    def build(self, data, **kwargs):
        return SoilParamsData(**data)


def create(soil_params: Dict[str, Any]) -> SoilParamsData:
    """Create soil data.

    Args:
        soil_params: dict with soil params. Required keys: 'Cy0', 'clay'.
            Optional quantile keys: 'Cy0_q05', 'Cy0_q95', 'clay_q05', 'clay_q95'.
            When quantile keys are absent, they default to the mean value,
            representing zero uncertainty (MC run degrades to deterministic).

    Returns:
        SoilParamsData: object containing soil parameters
    """
    Cy0 = soil_params["Cy0"]
    clay = soil_params["clay"]
    Ceq = 1.25 * Cy0  # this is an assumption detailed in the SHAMBA model description: Ceq is 25% higher than Cy0 (Cy0 + 0.25*Cy0 = 1.25* Cy0)

    params = {
        "Cy0": Cy0,
        "clay": clay,
        "depth": 30.0,
        "Ceq": Ceq,
        "iom": 0.049 * Ceq**1.139,
        "Cy0_q05":  soil_params.get("Cy0_q05",  Cy0),
        "Cy0_q95":  soil_params.get("Cy0_q95",  Cy0),
        "clay_q05": soil_params.get("clay_q05", clay),
        "clay_q95": soil_params.get("clay_q95", clay),
    }

    schema = SoilParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in soil params: {errors}")

    return schema.load(params)  # type: ignore


def get_soil_params(
    location: Tuple[float, float],
    use_api: bool,
    plot_index,
    plot_id,
    filename="soil-info.csv",
):
    """
    Construct SoilParamsData object from SoilGrids API or local CSV for given location.

    Args:
        location: Tuple of (latitude, longitude) coordinates
        use_api: Whether to use SoilGrids API or local CSV file
        plot_index: Index of the plot in the data (0-based)
        plot_id: Identifier of the plot
        filename: Name of local CSV file (default: "soil-info.csv")

    Returns:
        SoilParamsData: Soil parameters object
    """

    result: SoilData = get_soil_data(location, use_api, plot_index, plot_id, filename)

    if result is None:
        raise ValueError("Soil data not found in API or local file. Please provide a local soil file.")

    params = {
        "Cy0":      result.soc.mean,
        "clay":     result.clay.mean,
        "Cy0_q05":  result.soc.q05,
        "Cy0_q95":  result.soc.q95,
        "clay_q05": result.clay.q05,
        "clay_q95": result.clay.q95,
    }
    return create(params)


def print_to_stdout(soil_params):
    """Print soil information to stdout."""
    print("SOIL INFORMATION")
    print()
    print("================")
    print("Equilibrium C -", soil_params.Ceq)
    print("C at y=0  - - -", soil_params.Cy0)
    print("IOM - - - - - -", soil_params.iom)
    print("Clay  - - - - -", soil_params.clay)
    print("Depth - - - - -", soil_params.depth)
    print("")


def save(soil_params, file="soil_params.csv"):
    """Save soil params to a csv file. Default path is in OUTPUT_DIR
    with filename soil_params.csv

    Args:
        file: name or path to csv file. If path is not given
                (only name), put in OUTPUT_DIR for this program run.

    """
    data = np.array(
        [
            soil_params.Cy0,
            soil_params.clay,
            soil_params.Ceq,
            soil_params.iom,
            soil_params.depth,
        ]
    )
    cols = ["Cy0", "clay", "Ceq", "iom", "depth"]
    csv_handler.print_csv(file, data, col_names=cols)
