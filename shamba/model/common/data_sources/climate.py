import numpy as np
from datetime import datetime
from collections import defaultdict
import requests
import socket
from typing import List, Any, Dict, Optional, Tuple, NamedTuple

from model.common.data_sources.helpers import return_none_on_exception

MONTHS_COUNT = 12

API_URL = "https://archive-api.open-meteo.com/v1/archive"


class ClimateStats(NamedTuple):
    """Inter-annual mean and standard deviation for one calendar month."""
    mean: float
    std: float


def compute_monthly_mean_std(
    daily_values: np.ndarray,
    date_strings: List[str],
    aggregate_fn,
) -> List[ClimateStats]:
    """
    Compute inter-annual mean and standard deviation for each calendar month.

    For each calendar month (1–12):
      1. For each year, aggregate the daily values within that (year, month)
         using aggregate_fn — np.mean for temperature, np.sum for rain/ET.
      2. Collect the resulting annual scalars across all years (~30 values).
      3. Return ClimateStats(mean, std) across those annual scalars.

    std uses ddof=1 (sample standard deviation). Returns std=0.0 if fewer than
    two years of data are present for a month.

    Args:
        daily_values: 1-D array of daily values, aligned with date_strings.
        date_strings: list of ISO date strings (e.g. "1995-01-15") from the API.
        aggregate_fn: function applied to a list of daily values within one
            (year, month) to produce an annual scalar, e.g. np.mean or np.sum.

    Returns:
        List of 12 ClimateStats, one per calendar month (January first).
    """
    year_month_vals: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for date_str, val in zip(date_strings, daily_values):
        year, month = int(date_str[:4]), int(date_str[5:7])
        year_month_vals[(year, month)].append(float(val))

    result = []
    for month in range(1, 13):
        annual_scalars = [
            aggregate_fn(vals)
            for (y, m), vals in sorted(year_month_vals.items())
            if m == month
        ]
        if len(annual_scalars) > 1:
            result.append(ClimateStats(
                mean=float(np.mean(annual_scalars)),
                std=float(np.std(annual_scalars, ddof=1)),
            ))
        else:
            mean_val = float(np.mean(annual_scalars)) if annual_scalars else 0.0
            result.append(ClimateStats(mean=mean_val, std=0.0))

    return result


def get_climate_data(
    longitude: float, latitude: float
) -> Optional[Tuple[List[ClimateStats], List[ClimateStats], List[ClimateStats]]]:
    """
    Get climate data for a given location from the Open-Meteo archive API.

    Returns a 3-tuple of (temperature_stats, rain_stats, evap_stats), each a list
    of 12 ClimateStats objects (one per calendar month). Each ClimateStats holds
    the inter-annual mean and standard deviation across ~30 years of daily data.

    The standard deviation captures inter-annual variability: for each month, daily
    values within each (year, month) are aggregated first, and then std is taken
    across the ~30 resulting annual scalars.

    Returns None if the API call fails.

    Note: PET-to-evaporation conversion (/ 0.75) is applied in climate.py
    from_location(), so both mean and std are in PET units here.
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

    temp_stats = compute_monthly_mean_std(
        np.array(daily_data["temperature_2m_mean"]), date_strings, np.mean
    )
    rain_stats = compute_monthly_mean_std(
        np.array(daily_data["rain_sum"]), date_strings, np.sum
    )
    evap_stats = compute_monthly_mean_std(
        np.array(daily_data["et0_fao_evapotranspiration"]), date_strings, np.sum
    )

    return temp_stats, rain_stats, evap_stats


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
