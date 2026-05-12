from typing import Tuple, List, Optional, NamedTuple, Any, Dict
import logging as log
from toolz import compose
import requests
import numpy as np
import socket
from enum import Enum
from copy import deepcopy
from model.common import csv_handler
from model.common.data_sources.helpers import return_none_on_exception


class SoilQuantiles(NamedTuple):
    """Depth-weighted mean plus Q0.05/Q0.95 quantiles for a single soil property."""
    mean: float
    q05: float
    q95: float


class SoilData(NamedTuple):
    """Soil organic carbon and clay content with quantiles."""
    soc: SoilQuantiles   # t C ha⁻¹
    clay: SoilQuantiles  # %


API_URL = "https://rest.isric.org/soilgrids/v2.0/"


def read_soil_table(filename: str, plot_index: int, plot_id: int) -> SoilData:
    """Read the soil data from user CSV file.

    Accepts two CSV formats:
    - 3-column: plot_name, Cy0, clay
      Quantiles are set equal to the mean (no uncertainty).
    - 7-column: plot_name, Cy0, clay, Cy0_q05, Cy0_q95, clay_q05, clay_q95
      Quantiles are read from columns 3–6.

    Data values are validated in soil_params.py.
    """
    data = csv_handler.read_csv(filename)
    data = np.atleast_2d(data)

    if int(data[plot_index, 0]) != int(plot_id):
        raise ValueError("Plot order in soil data does not match input data")

    cy0_mean = data[plot_index, 1]
    clay_mean = data[plot_index, 2]

    if data.shape[1] >= 7:
        cy0_q05  = data[plot_index, 3]
        cy0_q95  = data[plot_index, 4]
        clay_q05 = data[plot_index, 5]
        clay_q95 = data[plot_index, 6]
    else:
        # No quantile columns: represent zero uncertainty as q05 = q95 = mean.
        cy0_q05 = cy0_q95 = cy0_mean
        clay_q05 = clay_q95 = clay_mean

    return SoilData(
        soc=SoilQuantiles(mean=cy0_mean, q05=cy0_q05, q95=cy0_q95),
        clay=SoilQuantiles(mean=clay_mean, q05=clay_q05, q95=clay_q95),
    )


def get_soil_data(
    location_coordinates: Tuple[float, float],
    use_api: bool,
    plot_index: int,
    plot_id: int,
    filename: str,
) -> Optional[SoilData]:
    """Get soil data from soilgrids api or from local csv file."""
    api_response: Optional[Dict[str, Any]] = (
        None
        if not use_api
        else get_properties_from_soilgrids_api(
            location=Point(location_coordinates[1], location_coordinates[0])
        )
    )

    if api_response is None:

        try:
            return read_soil_table(
                plot_index=plot_index, plot_id=plot_id, filename=filename
            )
        except:
            raise ValueError(
                "Soil data not found in API or local file. Please provide local file."
            )

    return compose(
        get_soc_and_clay,
        process_data,
        convert_units_in_api_response,
    )(
        api_response
    )  # type: ignore


class SoilProperty(Enum):
    """Enum for Soil Grids soil properties.

    This class defines properties available in Soil Grids.

    """

    BULK_DENSITY_OF_FINE_FRACTION = "bdod"
    CATION_EXCHANGE_CAPACITY = "cec"
    VOLUMETRIC_FRACTION_OF_COARSE_FRACTION = "cfvo"
    PROPORTION_OF_CLAY_IN_FINE_FRACTION = "clay"
    TOTAL_NITROGEN = "nitrogen"
    PH = "phh2o"
    ORGANIC_CARBON_DENSITY = "ocd"
    ORGANIC_CARBON_STOCKS = "ocs"
    PROPORTION_OF_SAND_IN_FINE_FRACTION = "sand"
    PROPORTION_OF_SILT_IN_FINE_FRACTION = "silt"
    SOIL_ORGANIC_CARBON_CONTENT_IN_FINE_FRACTION = "soc"
    VOL_WATER_CONTENT_MINUS_10KPA = "wv0010"
    VOL_WATER_CONTENT_MINUS_33KPA = "wv0033"
    VOL_WATER_CONTENT_MINUS_1500KPA = "wv1500"


DEFAULT_SOIL_PROPERTIES = [
    SoilProperty.PROPORTION_OF_CLAY_IN_FINE_FRACTION,
    SoilProperty.ORGANIC_CARBON_STOCKS,
    SoilProperty.PROPORTION_OF_SILT_IN_FINE_FRACTION,
    SoilProperty.PROPORTION_OF_SAND_IN_FINE_FRACTION,
]


class Depth(Enum):
    """Enum for Soil Grids depths.

    This class defines depths available in Soil Grids.

    """

    ZERO_TO_FIVE_CM = "0-5cm"
    ZERO_TO_THIRTY_CM = "0-30cm"
    FIVE_TO_FIFTEEN_CM = "5-15cm"
    FIFTEEN_TO_THIRTY_CM = "15-30cm"
    THIRTY_TO_SIXTY_CM = "30-60cm"
    SIXTY_TO_HUNDRED_CM = "60-100cm"
    ONE_HUNDRED_TO_TWO_HUNDRED_CM = "100-200cm"


DEFAULT_DEPTHS = [
    Depth.ZERO_TO_FIVE_CM,
    Depth.FIVE_TO_FIFTEEN_CM,
    Depth.FIFTEEN_TO_THIRTY_CM,
    Depth.ZERO_TO_THIRTY_CM,
]


class Value(Enum):
    """Enum for Soil Grids values.

    This class defines values available in Soil Grids.

    """

    Q0_05 = "Q0.05"
    Q0_5 = "Q0.5"
    Q0_95 = "Q0.95"
    MEAN = "mean"
    UNCERTAINTY = "uncertainty"


DEFAULT_VALUES = [Value.MEAN, Value.Q0_05, Value.Q0_95]

# Conversion to conventional units from here:
# https://www.isric.org/explore/soilgrids/faq-soilgrids (properties table)
UNIT_CONVERSIONS = {
    SoilProperty.BULK_DENSITY_OF_FINE_FRACTION: 100,
    SoilProperty.CATION_EXCHANGE_CAPACITY: 10,
    SoilProperty.VOLUMETRIC_FRACTION_OF_COARSE_FRACTION: 10,
    SoilProperty.PROPORTION_OF_CLAY_IN_FINE_FRACTION: 10,
    SoilProperty.TOTAL_NITROGEN: 100,
    SoilProperty.PH: 10,
    SoilProperty.ORGANIC_CARBON_DENSITY: 10,
    SoilProperty.ORGANIC_CARBON_STOCKS: 1,  # Not converted: soil model wants t ha-1, not kg m-2
    SoilProperty.PROPORTION_OF_SAND_IN_FINE_FRACTION: 10,
    SoilProperty.PROPORTION_OF_SILT_IN_FINE_FRACTION: 10,
    SoilProperty.SOIL_ORGANIC_CARBON_CONTENT_IN_FINE_FRACTION: 10,
    SoilProperty.VOL_WATER_CONTENT_MINUS_10KPA: 1,
    SoilProperty.VOL_WATER_CONTENT_MINUS_33KPA: 1,
    SoilProperty.VOL_WATER_CONTENT_MINUS_1500KPA: 1,
}


# Type aliases for results
DepthValueDict = dict[str, dict[str, float]]
SoilPropertyDepthValueDict = dict[SoilProperty, DepthValueDict]


class Point(NamedTuple):
    longitude: float
    latitude: float


@return_none_on_exception(requests.RequestException, socket.gaierror)
def get_properties_from_soilgrids_api(
    location: Point,
    soil_properties: list[SoilProperty] = DEFAULT_SOIL_PROPERTIES,
    depths: list[Depth] = DEFAULT_DEPTHS,
    values: list[Value] = DEFAULT_VALUES,
) -> Optional[dict]:
    """Retrieve soil property data for a given location.

    This function queries the Soil Grids API to fetch soil properties based on specified
    coordinates, soil property types, depths, and value types.

    Args:
        location (Point): A Point object containing longitude and latitude coordinates.
        soil_properties (list[SoilProperty], optional): A list of soil properties to retrieve.
            Defaults to DEFAULT_SOIL_PROPERTIES.
        depths (list[Depth], optional): A list of depth levels to query.
            Defaults to DEFAULT_DEPTHS.
        values (list[Value], optional): A list of value types (e.g., mean, median) to retrieve.
            Defaults to DEFAULT_VALUES.

    Returns
    -------
        Optional[dict]: A dictionary containing the queried soil properties from the API.

    """
    query_url = API_URL + "properties/query"
    str_soil_properties = [soil_property.value for soil_property in soil_properties]
    str_depths = [depth.value for depth in depths]
    str_values = [value.value for value in values]
    params = {
        "lon": location.longitude,
        "lat": location.latitude,
        "property": str_soil_properties,
        "depth": str_depths,
        "value": str_values,
    }
    headers = {"accept": "application/json"}
    response = requests.get(query_url, params=params, headers=headers)
    return response.json()


def get_soc_and_clay(api_response: List[Tuple[str, SoilQuantiles]]) -> SoilData:
    soc = next((q for name, q in api_response if name == "soc"), None)
    if soc is None:
        raise ValueError(
            "SOC (soil organic carbon) property not found in API soil data, please provide csv soil data"
        )

    clay = next((q for name, q in api_response if name == "clay"), None)
    if clay is None:
        raise ValueError(
            "Clay content property not found in API soil data, please provide csv soil data"
        )

    return SoilData(soc=soc, clay=clay)


def process_data(api_response: Dict[str, Any]) -> List[Tuple[str, SoilQuantiles]]:
    """
    In the SoilGrids API response, data are either available for 0-5, 5-15, 15-30 cm OR for
    0-30 cm (SOC stocks only).
    This function calculates the depth-weighted values for 0-30cm for the mean, Q0.05, and Q0.95.
    """
    data = api_response["properties"]["layers"]

    def depth_weighted(depths: List[Any], value_key: str) -> float:
        return (
            sum(
                depth["values"][value_key]
                * (depth["range"]["bottom_depth"] - depth["range"]["top_depth"])
                for depth in depths
                if value_key in depth["values"]
            )
            / 30
        )

    return [
        (
            layer["name"],
            SoilQuantiles(
                mean=depth_weighted(layer["depths"], "mean"),
                q05=depth_weighted(layer["depths"], "Q0.05"),
                q95=depth_weighted(layer["depths"], "Q0.95"),
            ),
        )
        for layer in data
    ]


def convert_value(value: float, conversion_factor: float) -> float:
    return value / conversion_factor if value != 0 else value


def convert_units_in_api_response(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert units in the Soil Grids API response using predefined conversion factors.

    This function creates a new dictionary by applying unit conversions to soil property
    values based on predefined conversion factors.

    https://www.isric.org/explore/soilgrids/faq-soilgrids (properties table)

    Args:
        d (dict): The Soil Grids API response containing soil property data with unit-dependent values.

    Returns:
        dict: A new dictionary with converted values.
    """
    result = deepcopy(api_response)
    layers = result.get("properties", {}).get("layers", [])

    for layer in layers:
        name_enum = SoilProperty(layer.get("name", ""))
        conversion_factor = UNIT_CONVERSIONS.get(name_enum, 1)

        for depth in layer.get("depths", []):
            depth["values"] = {
                k: convert_value(v, conversion_factor)
                for k, v in depth.get("values", {}).items()
            }

    return result
