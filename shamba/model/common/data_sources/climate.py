import numpy as np
from datetime import datetime
from collections import defaultdict
import requests
import socket
from typing import List, Any, Dict, Optional, Tuple

from model.common.data_sources.helpers import return_none_on_exception

MONTHS_COUNT = 12

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def aggregate_daily_to_monthly(
    daily_values: np.ndarray,
    date_strings: List[str],
    aggregate_fn,
) -> np.ndarray:
    """Aggregate daily values to monthly scalars, returning a flat 12*n_years array.

    Values are ordered year-major: [Jan_y1, Feb_y1, ..., Dec_y1, Jan_y2, ...].

    Args:
        daily_values: 1-D array of daily values, aligned with date_strings.
        date_strings: list of ISO date strings (e.g. "1995-01-15") from the API.
        aggregate_fn: applied to daily values within each (year, month), e.g.
            np.mean for temperature, np.sum for rain/ET.

    Returns:
        Flat array of length 12 * n_years.
    """
    year_month_vals: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for date_str, val in zip(date_strings, daily_values):
        year, month = int(date_str[:4]), int(date_str[5:7])
        year_month_vals[(year, month)].append(float(val))

    years = sorted({y for y, m in year_month_vals})
    result = []
    for year in years:
        for month in range(1, 13):
            vals = year_month_vals.get((year, month), [])
            result.append(float(aggregate_fn(vals)) if vals else 0.0)

    return np.array(result)


def get_climate_data(
    longitude: float, latitude: float
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get climate data for a given location from the Open-Meteo archive API.

    Returns a 3-tuple of (temperature, rain, evap), each a flat array of length
    12 * 30 = 360, in year-major month order: [Jan_y1, Feb_y1, ..., Dec_y30].

    Returns None if the API call fails.

    Note: PET-to-evaporation conversion (/ 0.75) is applied in climate.py
    from_location(), so evap values here are in PET units.
    """
    current_year = datetime.now().year
    last_full_year = current_year - 1
    start_year = last_full_year - 29

    start_date = datetime(start_year, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime(last_full_year, 12, 31).strftime("%Y-%m-%d")

    api_response = get_weather_forecast(
        latitude=latitude,
        longitude=longitude,
        daily_params=[
            "temperature_2m_mean",
            "rain_sum",
            "et0_fao_evapotranspiration",
        ],
        start_date=start_date,
        end_date=end_date,
    )

    if api_response is None:
        return None

    daily_data = api_response["daily"]
    date_strings: List[str] = daily_data["time"]

    temp = aggregate_daily_to_monthly(
        np.array(daily_data["temperature_2m_mean"]), date_strings, np.mean
    )
    rain = aggregate_daily_to_monthly(
        np.array(daily_data["rain_sum"]), date_strings, np.sum
    )
    evap = aggregate_daily_to_monthly(
        np.array(daily_data["et0_fao_evapotranspiration"]), date_strings, np.sum
    )

    return temp, rain, evap


@return_none_on_exception(requests.RequestException, socket.gaierror)
def get_weather_forecast(
    latitude: float,
    longitude: float,
    daily_params: List[str],
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily_params),
        "models": "era5",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }

    response = requests.get(API_URL, params=params)
    return response.json()
