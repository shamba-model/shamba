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
from model.common.data_sources.climate import get_climate_data


def validate_monthly_list_length(lst):
    return ["List must contain 12 elements"] if len(lst) != 12 else []


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
    def __init__(self, temperature, rain, evaporation):
        self.temperature = np.array(temperature)
        self.rain = np.array(rain)
        self.evaporation = np.array(evaporation)


class ClimateDataSchema(Schema):
    temperature = fields.List(
        fields.Float, validate=lambda values: validate_temperature(values)
    )
    rain = fields.List(fields.Float, validate=lambda values: validate_rain(values))
    evaporation = fields.List(
        fields.Float, validate=lambda values: validate_evaporation(values)
    )

    @post_load
    def build(self, data, **kwargs):
        return ClimateData(**data)


def from_location(location, use_api: bool) -> ClimateData:
    """Construct Climate object using CRU-TS
    dataset for a given location.

    Args:
        location
    Returns:
        Climate object
    """
    # Location stuff
    latitude = location[0]
    longitude = location[1]

    if use_api:
        climate_array = get_climate_data(latitude=latitude, longitude=longitude)

        if climate_array is not None:
            # pet given in OpenMeteo instead of evaporation, so convert
            climate_array[2] /= 0.75

            params = {
                "temperature": climate_array[0],
                "rain": climate_array[1],
                "evaporation": climate_array[2],
            }

            schema = ClimateDataSchema()
            errors = schema.validate(params)
            if errors != {}:
                print(f"Errors in climate data: {str(errors)}")
            return schema.load(params)  # type: ignore

        print("Climate API unavailable — falling back to local climate file.")

    try:
        return from_csv()
    except ValueError:
        raise ValueError("Climate data not found in API or local file.")


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

        # Set the correct order based on what's available
        if has_pet:
            correct_order = ("temp", "rain", "pet")
        else:
            correct_order = ("temp", "rain", "evap")

        # Create clim array with the correct rows
        climate_data = np.zeros((3, 12))
        for i in range(3):
            climate_data[i] = data[:, np.where(headers == correct_order[i])[0][0]]

        # Convert PET to open-pan evaporation if PET data was used
        if has_pet:
            climate_data[2] /= 0.75

        climate: ClimateData = ClimateDataSchema().load(
            {
                "temperature": climate_data[0],
                "rain": climate_data[1],
                "evaporation": climate_data[2],
            }
        )  # type: ignore
    except ValueError as e:
        raise ValueError(f"Error in climate data: {str(e)}")
    except IndexError:
        raise ValueError("Climate data file is not in the correct format.")

    return climate


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
