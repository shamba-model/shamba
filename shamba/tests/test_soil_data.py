"""Tests for soil uncertainty capture (Issue P1).

Covers:
- read_soil_table: 3-column CSV gives q05=q95=mean; 7-column CSV parsed correctly
- process_data: API response extracts all three quantiles with correct depth weighting
- get_soc_and_clay: missing SOC or clay raises plain-language errors
- SoilParamsData quantile fields populated correctly via get_soil_params / create
- Quantile ordering validation in SoilParamsSchema
"""
import numpy as np
import pytest

from model.common.data_sources.soil import (
    SoilData,
    SoilQuantiles,
    get_soc_and_clay,
    process_data,
    read_soil_table,
)
from model.soil_params import SoilParamsData, create


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_3COL = "shamba/tests/fixtures/soil-info.csv"
FIXTURE_7COL = "shamba/tests/fixtures/soil-info-quantiles.csv"


# ---------------------------------------------------------------------------
# read_soil_table — 3-column CSV
# ---------------------------------------------------------------------------

def test_read_soil_table_3col_returns_soil_data(tmp_path):
    """3-column CSV: returned SoilData has correct mean values, q05 and q95 
    both equal the mean (no uncertainty)."""
    csv = tmp_path / "soil.csv"
    csv.write_text("plot_name,Cy0,clay\n0,5.0,36.0\n")

    result = read_soil_table(str(csv), plot_index=0, plot_id=0)

    assert isinstance(result, SoilData)
    assert result.soc.mean == pytest.approx(5.0)
    assert result.clay.mean == pytest.approx(36.0)
    assert result.soc.q05 == pytest.approx(result.soc.mean)
    assert result.soc.q95 == pytest.approx(result.soc.mean)
    assert result.clay.q05 == pytest.approx(result.clay.mean)
    assert result.clay.q95 == pytest.approx(result.clay.mean)


# ---------------------------------------------------------------------------
# read_soil_table — 7-column CSV
# ---------------------------------------------------------------------------

def test_read_soil_table_7col_reads_quantiles(tmp_path):
    """7-column CSV: quantiles are read from columns 3–6."""
    csv = tmp_path / "soil.csv"
    csv.write_text(
        "plot_name,Cy0,clay,Cy0_q05,Cy0_q95,clay_q05,clay_q95\n"
        "0,5.0,36.0,3.0,7.5,28.0,44.0\n"
    )

    result = read_soil_table(str(csv), plot_index=0, plot_id=0)

    assert result.soc.mean == pytest.approx(5.0)
    assert result.soc.q05  == pytest.approx(3.0)
    assert result.soc.q95  == pytest.approx(7.5)
    assert result.clay.mean == pytest.approx(36.0)
    assert result.clay.q05  == pytest.approx(28.0)
    assert result.clay.q95  == pytest.approx(44.0)


def test_read_soil_table_plot_order_mismatch_raises(tmp_path):
    """Mismatched plot ID raises a plain-language ValueError."""
    csv = tmp_path / "soil.csv"
    csv.write_text("plot_name,Cy0,clay\n99,5.0,36.0\n")

    with pytest.raises(ValueError, match="Plot order"):
        read_soil_table(str(csv), plot_index=0, plot_id=0)


# ---------------------------------------------------------------------------
# process_data — API response parsing
# ---------------------------------------------------------------------------

def _make_api_response(name, depths):
    """Build a minimal API response dict with one layer."""
    return {
        "properties": {
            "layers": [
                {
                    "name": name,
                    "depths": depths,
                }
            ]
        }
    }


def _depth(top, bottom, mean, q05, q95):
    return {
        "range": {"top_depth": top, "bottom_depth": bottom},
        "values": {"mean": mean, "Q0.05": q05, "Q0.95": q95},
    }


def test_process_data_single_layer_depth_weighted():
    """process_data computes correct depth-weighted averages for mean and quantiles."""
    # One layer: 0–5 cm and 5–15 cm, both with known values.
    # Depth-weighted average over 0–30 cm denominator:
    #   mean  = (2.0*5 + 4.0*10) / 30 = (10 + 40) / 30 = 50/30 ≈ 1.6667
    #   q05   = (1.0*5 + 2.0*10) / 30 = 25/30 ≈ 0.8333
    #   q95   = (3.0*5 + 6.0*10) / 30 = 75/30 = 2.5
    response = _make_api_response(
        "soc",
        [
            _depth(0, 5, mean=2.0, q05=1.0, q95=3.0),
            _depth(5, 15, mean=4.0, q05=2.0, q95=6.0),
        ],
    )

    result = process_data(response)

    assert len(result) == 1
    name, quants = result[0]
    assert name == "soc"
    assert quants.mean == pytest.approx(50 / 30)
    assert quants.q05  == pytest.approx(25 / 30)
    assert quants.q95  == pytest.approx(75 / 30)


def test_process_data_missing_quantile_key_skipped():
    """Depths that lack Q0.05/Q0.95 keys are skipped for those quantiles."""
    response = _make_api_response(
        "clay",
        [
            # Only 'mean' present — Q0.05 and Q0.95 absent
            {"range": {"top_depth": 0, "bottom_depth": 30}, "values": {"mean": 30.0}},
        ],
    )

    result = process_data(response)
    name, quants = result[0]
    assert quants.mean == pytest.approx(30.0)
    # q05 and q95 will be 0.0 (no contributions to the weighted sum)
    assert quants.q05 == pytest.approx(0.0)
    assert quants.q95 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_soc_and_clay
# ---------------------------------------------------------------------------

def test_get_soc_and_clay_returns_soil_data():
    api = [
        ("soc",  SoilQuantiles(mean=5.0, q05=3.0, q95=7.0)),
        ("clay", SoilQuantiles(mean=36.0, q05=28.0, q95=44.0)),
    ]
    result = get_soc_and_clay(api)

    assert isinstance(result, SoilData)
    assert result.soc.mean  == pytest.approx(5.0)
    assert result.clay.mean == pytest.approx(36.0)


def test_get_soc_and_clay_missing_soc_raises():
    api = [("clay", SoilQuantiles(mean=36.0, q05=28.0, q95=44.0))]
    with pytest.raises(ValueError, match="SOC"):
        get_soc_and_clay(api)


def test_get_soc_and_clay_missing_clay_raises():
    api = [("soc", SoilQuantiles(mean=5.0, q05=3.0, q95=7.0))]
    with pytest.raises(ValueError, match="[Cc]lay"):
        get_soc_and_clay(api)


# ---------------------------------------------------------------------------
# create / SoilParamsData — quantile fields
# ---------------------------------------------------------------------------

def test_create_without_quantiles_defaults_to_mean():
    """create() with only Cy0/clay: quantile fields equal the mean."""
    soil = create({"Cy0": 5.0, "clay": 36.0})

    assert soil.Cy0_q05  == pytest.approx(5.0)
    assert soil.Cy0_q95  == pytest.approx(5.0)
    assert soil.clay_q05 == pytest.approx(36.0)
    assert soil.clay_q95 == pytest.approx(36.0)


def test_create_with_quantiles_stores_correctly():
    """create() with quantile keys: values stored in SoilParamsData."""
    soil = create({
        "Cy0": 5.0, "clay": 36.0,
        "Cy0_q05": 3.0, "Cy0_q95": 7.5,
        "clay_q05": 28.0, "clay_q95": 44.0,
    })

    assert soil.Cy0_q05  == pytest.approx(3.0)
    assert soil.Cy0_q95  == pytest.approx(7.5)
    assert soil.clay_q05 == pytest.approx(28.0)
    assert soil.clay_q95 == pytest.approx(44.0)


def test_create_quantile_ordering_violation_raises():
    """create() rejects q05 > mean or mean > q95 with a validation error."""
    with pytest.raises(Exception):
        create({
            "Cy0": 5.0, "clay": 36.0,
            "Cy0_q05": 8.0,  # q05 > mean — invalid
            "Cy0_q95": 10.0,
            "clay_q05": 28.0, "clay_q95": 44.0,
        })
