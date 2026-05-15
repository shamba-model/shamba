"""Integration tests for the split-file CSV input path.

Two tests:
  - Data pipeline: verify that reading the three split-file CSVs produces an
    equivalent dict to expand_single_row_data_input on the same scenario.
  - Full model run: verify that handle_intervention returns identical emissions
    for both input paths.

Fixtures are generated programmatically from WL_input.csv so the split-file
CSVs always stay in sync with the single-row fixture without manual maintenance.
"""

import csv
import os

import numpy as np
import pytest

from model import configuration
from model.common.calculate_emissions import handle_intervention
from model.common.data_handler import (
    expand_single_row_data_input,
    read_and_validate_timeseries_by_header,
)
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
from model.soil_models.soil_model_types import SoilModelType

FIXTURES_DIR = os.path.join(configuration.TESTS_DIR, "fixtures")
WL_SINGLE_ROW = os.path.join(FIXTURES_DIR, "WL_input.csv")

N_COHORTS = 1
ALLOMETRIC_KEYS = ["chave dry", "chave dry"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_vector_csv(path, data_dict):
    """Write a dict of numpy arrays as a column-per-key CSV.

    Each key becomes a column header; arrays of different lengths are aligned
    on row 0 and shorter columns are NaN-padded so the file is rectangular.
    read_and_validate_timeseries_by_header strips NaN values per-column, so
    each column is restored to its original length on read-back.
    """
    headers = list(data_dict.keys())
    max_len = max(np.atleast_1d(arr).size for arr in data_dict.values())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(max_len):
            row = []
            for h in headers:
                arr = np.atleast_1d(data_dict[h])
                row.append(float(arr[i]) if i < arr.size else float("nan"))
            writer.writerow(row)


def _write_split_files(tmp_path, scalar, tree_size, mgmt, cover):
    _write_vector_csv(tmp_path / "WL_plot_data.csv", scalar)
    _write_vector_csv(tmp_path / "WL_tree_size_data.csv", tree_size)
    _write_vector_csv(tmp_path / "WL_mgmt_data.csv", mgmt)
    _write_vector_csv(tmp_path / "WL_climate_cover_data.csv", cover)


def _read_split_files(tmp_path, n_years):
    prefix = str(tmp_path / "WL")
    scalar = read_and_validate_timeseries_by_header(
        f"{prefix}_plot_data.csv",
        permitted_vector_lengths=[1],
        target_vector_length=1,
    )
    mgmt = read_and_validate_timeseries_by_header(
        f"{prefix}_mgmt_data.csv",
        permitted_vector_lengths=[1, n_years, n_years + 1],
        target_vector_length=n_years,
    )
    tree_size = read_and_validate_timeseries_by_header(
        f"{prefix}_tree_size_data.csv",
        permitted_vector_lengths=list(range(5, n_years + 1)),
        target_vector_length=None,
    )
    cover = read_and_validate_timeseries_by_header(
        f"{prefix}_climate_cover_data.csv",
        permitted_vector_lengths=[1] + [i * 12 for i in range(1, n_years + 1)],
        target_vector_length=12 * n_years,
    )
    return scalar | mgmt | tree_size | cover


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_split_file_data_pipeline_matches_single_row(tmp_path):
    """Reading split-file CSVs must produce the same keys and values as
    expand_single_row_data_input for the same scenario."""
    scalar, tree_size, mgmt, cover = expand_single_row_data_input(WL_SINGLE_ROW)
    n_years = int(scalar["yrs_proj"].item())
    reference = scalar | mgmt | tree_size | cover

    _write_split_files(tmp_path, scalar, tree_size, mgmt, cover)
    split = _read_split_files(tmp_path, n_years)

    assert set(reference.keys()) == set(split.keys()), (
        f"Key mismatch.\n"
        f"Only in reference: {set(reference) - set(split)}\n"
        f"Only in split:     {set(split) - set(reference)}"
    )
    for key in reference:
        ref_arr = np.atleast_1d(reference[key])
        split_arr = np.atleast_1d(split[key])
        np.testing.assert_allclose(
            ref_arr, split_arr, rtol=1e-10,
            err_msg=f"Mismatch for key '{key}'"
        )


def test_split_file_full_run_matches_single_row(tmp_path, monkeypatch):
    """handle_intervention must return identical emissions for single-row and
    split-file input paths when both represent the same scenario.

    configuration.INPUT_DIR is pointed at the fixtures directory so that
    climate.csv and soil-info.csv are found without API access."""
    monkeypatch.setattr(configuration, "INPUT_DIR", FIXTURES_DIR)

    scalar, tree_size, mgmt, cover = expand_single_row_data_input(WL_SINGLE_ROW)
    n_years = int(scalar["yrs_proj"].item())
    single_row_dict = scalar | mgmt | tree_size | cover

    _write_split_files(tmp_path, scalar, tree_size, mgmt, cover)
    split_dict = _read_split_files(tmp_path, n_years)

    forward_model = ForwardSoilModule.get_soil_model(SoilModelType.ROTH_C)
    inverse_model = InverseSoilModule.get_soil_model(SoilModelType.ROTH_C)

    common_kwargs = dict(
        n_cohorts=N_COHORTS,
        plot_index=0,
        allometry=ALLOMETRIC_KEYS,
        use_climate_api=False,
        use_soil_api=False,
        create_forward_soil_model=forward_model.create,
        create_inverse_soil_model=inverse_model.create,
    )

    result_single = handle_intervention(intervention_input=single_row_dict, **common_kwargs)
    result_split = handle_intervention(intervention_input=split_dict, **common_kwargs)

    np.testing.assert_allclose(
        result_single.emit_base_emissions,
        result_split.emit_base_emissions,
        rtol=1e-10,
        err_msg="emit_base_emissions differ between single-row and split-file paths",
    )
    np.testing.assert_allclose(
        result_single.emit_project_emissions,
        result_split.emit_project_emissions,
        rtol=1e-10,
        err_msg="emit_project_emissions differ between single-row and split-file paths",
    )
