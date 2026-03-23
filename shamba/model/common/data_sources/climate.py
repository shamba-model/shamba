import os
import numpy as np
import math
import calendar
from datetime import datetime, timedelta
from itertools import groupby
import requests
import socket
from typing import List, Any, Dict, Optional, Tuple

from model.common import csv_handler
from model.common.data_sources.helpers import return_none_on_exception

MONTHS_COUNT = 12

TEMPERATURE_BASENAME = "tmp_"
RAINFALL_BASENAME = "pre_"
PET_BASENAME = "pet_"
BASENAMES = [TEMPERATURE_BASENAME, RAINFALL_BASENAME, PET_BASENAME]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def generate_dates(start_year: int, end_year: int) -> np.ndarray:
    all_dates_in_range = []
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        year_dates = [
            start_date + timedelta(days=i) for i in range(365 + calendar.isleap(year))
        ]
        all_dates_in_range.extend(year_dates)
    return np.array(all_dates_in_range)


def pair_dates_with_values(dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Ensure dates and values have the same length
    min_length = min(len(dates), len(values))
    return np.column_stack((dates[:min_length], values[:min_length]))


def group_by_month(date_value_pairs: np.ndarray) -> list:
    months = np.vectorize(lambda d: d.month)(date_value_pairs[:, 0])
    return [(month, date_value_pairs[months == month]) for month in range(1, 13)]


def calculate_monthly_average(
    month_group: Tuple[int, np.ndarray],
) -> Tuple[int, np.floating[Any]]:
    month, values = month_group
    return month, np.mean(values[:, 1])


def calculate_monthly_sum(
    month_group: Tuple[int, np.ndarray],
) -> Tuple[int, np.floating[Any]]:
    month, values = month_group
    return month, np.sum(values[:, 1])


def segment_and_average_by_month(
    daily_values: np.ndarray, start_year: int, end_year: int
) -> np.ndarray:
    dates = generate_dates(start_year, end_year)
    date_value_pairs = pair_dates_with_values(dates, daily_values)
    grouped_by_month = group_by_month(date_value_pairs)
    monthly_averages = np.array(
        [calculate_monthly_average(group) for group in grouped_by_month]
    )
    return monthly_averages


def segment_and_sum_by_month(
    daily_values: np.ndarray, start_year: int, end_year: int
) -> np.ndarray:
    dates = generate_dates(start_year, end_year)
    no_of_years = end_year - start_year + 1
    date_value_pairs = pair_dates_with_values(dates, daily_values)
    grouped_by_month = group_by_month(date_value_pairs)
    monthly_sums = np.array(
        [calculate_monthly_sum(group) for group in grouped_by_month]
    )
    return monthly_sums / no_of_years


def get_climate_data(longitude: float, latitude: float) -> Optional[np.ndarray]:
    """
    Get climate data for a given location from the API.

    Each piece of climate data is a 12-month average. The data is grouped by month
    and then averaged over the months.

    Args:
        longitude (float): The longitude of the location.
        latitude (float): The latitude of the location.

    Returns:
        np.ndarray of shape (3, 12) with rows [temperature, rain, evapotranspiration],
        or None if the API call fails.
    """
    # TODO: extend to return 12*n_years climate data (one value per month per year) so that
    # each model year uses its own monthly climate rather than a 30-year average.
    # This should use a forecast API (not historical archive) to supply future-year climate
    # data for the duration of the project. n_years would be passed down from
    # handle_intervention via from_location().
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
    temperature = segment_and_average_by_month(
        np.array(daily_data["temperature_2m_mean"]),
        start_year=start_year,
        end_year=last_full_year,
    )
    rain = segment_and_sum_by_month(
        np.array(daily_data["rain_sum"]), start_year=start_year, end_year=last_full_year
    )
    evapotranspiration = segment_and_sum_by_month(
        np.array(daily_data["et0_fao_evapotranspiration"]),
        start_year=start_year,
        end_year=last_full_year,
    )

    result = np.vstack([temperature[:, 1], rain[:, 1], evapotranspiration[:, 1]])
    return result


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
        # "format": "json",
        # "timeformat": "unixtime",
    }

    response = requests.get(API_URL, params=params)
    return response.json()
