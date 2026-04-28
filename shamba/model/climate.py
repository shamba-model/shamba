#!/usr/bin/python

"""Module holding Climate class."""

import math
import calendar
import os
from model import configuration

import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler
from model.common.data_sources.climate import get_climate_data, ClimateStats


def validate_monthly_list_length(lst):
    return ["List length must be a non-zero multiple of 12"] if len(lst) == 0 or len(lst) % 12 != 0 else []


def validate_temperature(values):
    length_errors = validate_monthly_list_length(values)
    value_errors = [
        "Temperature out of expected range or is NaN"
        for val in values
        if val < -100.0 or val > 100.0 or np.isnan(val)
    ]
    return length_errors + value_errors


def validate_rain(values):
    length_errors = validate_monthly_list_length(values)
    value_errors = [
        "Rain out of expected range or is NaN"
        for val in values
        if val < 0 or val > 4000.0 or np.isnan(val)
    ]
    return length_errors + value_errors


def validate_evaporation(values):
    length_errors = validate_monthly_list_length(values)
    value_errors = [
        "Evaporation out of expected range or is NaN"
        for val in values
        if val < 0 or val > 4000.0 or np.isnan(val)
    ]
    return length_errors + value_errors


class ClimateData:
    def __init__(self, temperature, rain, evaporation,
                 temperature_std=None, rain_std=None, evaporation_std=None):
        self.temperature = np.array(temperature)
        self.rain = np.array(rain)
        self.evaporation = np.array(evaporation)
        # Std fields for Monte Carlo uncertainty.
        # Always 12 elements (one per calendar month) regardless of how many years
        # the mean arrays span. Default to zeros when not provided.
        self.temperature_std = np.zeros(12) if temperature_std is None else np.array(temperature_std)
        self.rain_std = np.zeros(12) if rain_std is None else np.array(rain_std)
        self.evaporation_std = np.zeros(12) if evaporation_std is None else np.array(evaporation_std)


class ClimateDataSchema(Schema):
    temperature = fields.List(
        fields.Float, validate=lambda values: validate_temperature(values)
    )
    rain = fields.List(fields.Float, validate=lambda values: validate_rain(values))
    evaporation = fields.List(
        fields.Float, validate=lambda values: validate_evaporation(values)
    )
    # Optional std fields for Monte Carlo uncertainty.
    # Not required — absent means zero uncertainty.
    temperature_std = fields.List(fields.Float, load_default=None)
    rain_std        = fields.List(fields.Float, load_default=None)
    evaporation_std = fields.List(fields.Float, load_default=None)

    @post_load
    def build(self, data, **kwargs):
        return ClimateData(**data)


def from_vectors(temperature, rain, evaporation) -> ClimateData:
    """Construct ClimateData directly from pre-validated arrays.

    Bypasses the length-12 schema check, so accepts multi-year arrays
    (e.g. length 12 * N_YEARS from split input files).
    Stds are not accepted here; they come only from the API or climate.csv.
    """
    return ClimateData(
        temperature=np.array(temperature),
        rain=np.array(rain),
        evaporation=np.array(evaporation),
    )


def from_location(location, use_api: bool, climate_vectors=None) -> ClimateData:
    """Construct Climate object for a given location.

    Priority order:
      1. Climate API (if use_api=True and call succeeds)
      2. climate_vectors from split input file (Temp/Rain/evap columns)
      3. Local climate.csv file

    Args:
        location: (latitude, longitude) tuple
        use_api: whether to attempt the climate API
        climate_vectors: optional tuple of (temperature, rain, evaporation)
            arrays from the split _climate_cover_data.csv file
    Returns:
        ClimateData object
    """
    latitude = location[0]
    longitude = location[1]

    if use_api:
        climate_result = get_climate_data(latitude=latitude, longitude=longitude)

        if climate_result is not None:
            temp_stats, rain_stats, evap_stats = climate_result
            # PET → evap conversion applies to both mean and std
            evap_stats = [ClimateStats(s.mean / 0.75, s.std / 0.75) for s in evap_stats]

            temperature    = [s.mean for s in temp_stats]
            rain           = [s.mean for s in rain_stats]
            evaporation    = [s.mean for s in evap_stats]
            temperature_std = [s.std for s in temp_stats]
            rain_std        = [s.std for s in rain_stats]
            evaporation_std = [s.std for s in evap_stats]

            params = {"temperature": temperature, "rain": rain, "evaporation": evaporation}
            schema = ClimateDataSchema()
            errors = schema.validate(params)
            if errors != {}:
                print(f"Errors in climate data: {str(errors)}")

            return ClimateData(
                temperature=temperature,
                rain=rain,
                evaporation=evaporation,
                temperature_std=temperature_std,
                rain_std=rain_std,
                evaporation_std=evaporation_std,
            )

        print("Climate API unavailable — falling back to local climate data.")

    if climate_vectors is not None:
        return from_vectors(*climate_vectors)

    try:
        return from_csv()
    except ValueError:
        raise ValueError("Climate data not found in API, split input file, or local climate.csv.")


def from_csv(filename="climate.csv") -> ClimateData:
    """Construct Climate object from a csv file.

    Args:
        filename: path to csv file containing climate data
    Returns:
        Climate object
    Raises:
        ValueError: if headers don't contain 'temp', 'rain', and either 'evap' or 'pet'

    """
    data = csv_handler.read_csv(filename)
    headers = np.genfromtxt(
        os.path.join(configuration.INPUT_DIR, filename),
        max_rows=1,
        delimiter=",",
        dtype=None,
        encoding=None,
    )
    headers = np.char.lower(headers)

    try:
        # Check if PET or open-pan evaporation data is present
        has_pet = "pet" in headers
        has_evap = "evap" in headers

        if has_pet and has_evap:
            print("Both 'evap' and 'pet' found in climate data — 'evap' will be used and 'pet' discarded.")
            has_pet = False
        elif not has_pet and not has_evap:
            raise ValueError("Climate data must contain either 'pet' or 'evap'")

        evap_col = "pet" if has_pet else "evap"

        def col(name):
            return data[:, np.where(headers == name)[0][0]].copy()

        temperature = col("temp")
        rain        = col("rain")
        evaporation = col(evap_col)

        if has_pet:
            evaporation /= 0.75

        # Optional std columns: temp_std, rain_std, evap_std
        has_std = "temp_std" in headers
        if has_std:
            temperature_std = col("temp_std")
            rain_std        = col("rain_std")
            evaporation_std = col("evap_std")
            if has_pet:
                evaporation_std /= 0.75
        else:
            temperature_std = np.zeros_like(temperature)
            rain_std        = np.zeros_like(rain)
            evaporation_std = np.zeros_like(evaporation)

        params = {"temperature": temperature, "rain": rain, "evaporation": evaporation}
        errors = ClimateDataSchema().validate(params)
        if errors != {}:
            raise ValueError(str(errors))

        return ClimateData(
            temperature=temperature,
            rain=rain,
            evaporation=evaporation,
            temperature_std=temperature_std,
            rain_std=rain_std,
            evaporation_std=evaporation_std,
        )
    except ValueError as e:
        raise ValueError(f"Error in climate data: {str(e)}")
    except IndexError:
        raise ValueError("Climate data file is not in the correct format.")


def plot(climate):
    """Plot climate data in a matplotlib figure."""

    x_axis = list(range(1, 13))
    fig, ax1 = plt.subplots()
    fig.suptitle("Climate data")

    ax1.bar(x_axis, climate.rain, align="center", ec="k", fc="w")
    ax1.plot(x_axis, climate.evaporation, "k--D")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Rain and evaporation (mm/month)")
    ax1.set_title("Monthly Climate Data")

    ax2 = ax1.twinx()
    ax2.plot(x_axis, climate.temperature, "b-o")
    ax1.set_xlim(0, 13)
    ax2.set_ylabel("Temperature (C)", color="b")

    # Set ax2 to blue to set apart from other axis
    for tl in ax2.get_yticklabels():
        tl.set_color("b")


def print_to_stdout(climate):
    """Print climate data to stdout using tabulate."""

    month_names = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]

    table_title = "CLIMATE DATA"

    # Prepare the data for tabulate
    table_data = [
        [month, f"{temp:.2f}", f"{rain:.2f}", f"{evap:.2f}"]
        for month, temp, rain, evap in zip(
            month_names, climate.temperature, climate.rain, climate.evaporation
        )
    ]

    # Define headers
    headers = ["Month", "Temp. (°C)", "Rain (mm)", "Evap. (mm)"]

    # Print the table using tabulate
    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(
        tabulate(table_data, headers=headers, numalign="center", tablefmt="fancy_grid")
    )


def save(climate, file="climate.csv"):
    """Save climate data to a csv file.
    Default path is in cfg.OUTPUT_DIR with filename 'climate.csv'.

    Args:
        file: name or path to csv file. If only name is given, file
                is put in cfg.INPUT_DIR.

    """
    temperature = climate.temperature
    rain = climate.rain
    evaporation = climate.evaporation
    csv_handler.print_csv(
        file,
        np.transpose([temperature, rain, evaporation]),
        col_names=["temp", "rain", "evap"],
    )
