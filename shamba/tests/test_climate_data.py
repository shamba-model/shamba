"""Tests for climate uncertainty capture (Issue P2).

Covers:
- compute_monthly_mean_std: correct inter-annual mean and std for both
  mean (temperature) and sum (rain/ET) aggregation
- from_csv: 3-column CSV gives std=0; 6-column CSV populates std arrays
- ClimateData: std fields default to zero when not supplied
- Zero-std: all samples identical (tested via ClimateData construction)
"""
import numpy as np
import pytest
from datetime import date

from model.common.data_sources.climate import ClimateStats, compute_monthly_mean_std
from model.climate import ClimateData, from_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_date_strings(start_year, end_year):
    """Generate ISO date strings for every day from start_year to end_year."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            import calendar
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates


# ---------------------------------------------------------------------------
# compute_monthly_mean_std — temperature (mean aggregation)
# ---------------------------------------------------------------------------

def test_compute_monthly_mean_std_temperature_known_values():
    """
    Two years of synthetic data with known January daily values.
    Year 1: all Jan days = 10.0  → annual mean = 10.0
    Year 2: all Jan days = 20.0  → annual mean = 20.0
    Expected: mean = 15.0, std = std([10, 20], ddof=1) = 7.071...
    """
    date_strings = make_date_strings(2000, 2001)

    # Assign 10.0 to all days in 2000, 20.0 to all days in 2001
    values = np.array([
        10.0 if ds[:4] == "2000" else 20.0
        for ds in date_strings
    ])

    result = compute_monthly_mean_std(values, date_strings, np.mean)

    jan = result[0]  # index 0 = January
    assert jan.mean == pytest.approx(15.0)
    assert jan.std  == pytest.approx(np.std([10.0, 20.0], ddof=1))

    # Other months should have the same values (synthetic data is uniform per year)
    for month_stats in result:
        assert month_stats.mean == pytest.approx(15.0)
        assert month_stats.std  == pytest.approx(np.std([10.0, 20.0], ddof=1))


def test_compute_monthly_mean_std_returns_twelve_entries():
    date_strings = make_date_strings(2000, 2001)
    values = np.ones(len(date_strings))
    result = compute_monthly_mean_std(values, date_strings, np.mean)
    assert len(result) == 12


# ---------------------------------------------------------------------------
# compute_monthly_mean_std — rain (sum aggregation)
# ---------------------------------------------------------------------------

def test_compute_monthly_mean_std_rain_known_values():
    """
    Two years, January daily rain values:
    Year 1: 31 days × 2.0 mm → monthly sum = 62.0
    Year 2: 31 days × 4.0 mm → monthly sum = 124.0
    Expected: mean = 93.0, std = std([62, 124], ddof=1)
    """
    date_strings = make_date_strings(2000, 2001)

    values = np.array([
        2.0 if ds[:4] == "2000" else 4.0
        for ds in date_strings
    ])

    result = compute_monthly_mean_std(values, date_strings, np.sum)

    jan = result[0]
    jan_sum_2000 = 2.0 * 31  # January has 31 days
    jan_sum_2001 = 4.0 * 31
    assert jan.mean == pytest.approx((jan_sum_2000 + jan_sum_2001) / 2)
    assert jan.std  == pytest.approx(np.std([jan_sum_2000, jan_sum_2001], ddof=1))


def test_compute_monthly_mean_std_single_year_std_is_zero():
    """With only one year of data, std must be 0.0 (not NaN)."""
    date_strings = make_date_strings(2000, 2000)
    values = np.ones(len(date_strings)) * 5.0
    result = compute_monthly_mean_std(values, date_strings, np.mean)
    for stats in result:
        assert stats.std == pytest.approx(0.0)
        assert not np.isnan(stats.std)


# ---------------------------------------------------------------------------
# from_csv — 3-column (no std)
# ---------------------------------------------------------------------------

def test_from_csv_3col_std_is_zero(tmp_path, monkeypatch):
    """3-column climate CSV: std arrays are all zero."""
    csv = tmp_path / "climate.csv"
    csv.write_text(
        "temp,rain,evap\n"
        + "\n".join(f"20.0,80.0,60.0" for _ in range(12))
        + "\n"
    )

    import model.configuration as cfg
    monkeypatch.setattr(cfg, "INPUT_DIR", str(tmp_path))

    climate = from_csv(filename=str(csv))

    assert np.all(climate.temperature_std == pytest.approx(0.0))
    assert np.all(climate.rain_std        == pytest.approx(0.0))
    assert np.all(climate.evaporation_std == pytest.approx(0.0))


def test_from_csv_3col_mean_values_correct(tmp_path, monkeypatch):
    """3-column climate CSV: mean arrays read correctly."""
    csv = tmp_path / "climate.csv"
    csv.write_text(
        "temp,rain,evap\n"
        + "\n".join(f"{i}.0,{i*10}.0,{i*5}.0" for i in range(1, 13))
        + "\n"
    )

    import model.configuration as cfg
    monkeypatch.setattr(cfg, "INPUT_DIR", str(tmp_path))

    climate = from_csv(filename=str(csv))

    assert climate.temperature[0] == pytest.approx(1.0)
    assert climate.rain[0]        == pytest.approx(10.0)
    assert climate.evaporation[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# from_csv — 6-column (with std)
# ---------------------------------------------------------------------------

def test_from_csv_6col_std_populated(tmp_path, monkeypatch):
    """6-column climate CSV: std arrays populated from file."""
    csv = tmp_path / "climate.csv"
    rows = "\n".join(
        f"20.0,80.0,60.0,2.0,8.0,6.0" for _ in range(12)
    )
    csv.write_text("temp,rain,evap,temp_std,rain_std,evap_std\n" + rows + "\n")

    import model.configuration as cfg
    monkeypatch.setattr(cfg, "INPUT_DIR", str(tmp_path))

    climate = from_csv(filename=str(csv))

    assert np.all(climate.temperature_std == pytest.approx(2.0))
    assert np.all(climate.rain_std        == pytest.approx(8.0))
    assert np.all(climate.evaporation_std == pytest.approx(6.0))


# ---------------------------------------------------------------------------
# ClimateData — default std behaviour
# ---------------------------------------------------------------------------

def test_climate_data_std_defaults_to_zero():
    """ClimateData constructed without std args: std arrays are all zero."""
    climate = ClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[60.0] * 12,
    )

    assert np.all(climate.temperature_std == pytest.approx(0.0))
    assert np.all(climate.rain_std        == pytest.approx(0.0))
    assert np.all(climate.evaporation_std == pytest.approx(0.0))


def test_climate_data_std_stored_correctly():
    """ClimateData constructed with std args: values stored correctly."""
    climate = ClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[60.0] * 12,
        temperature_std=[2.0] * 12,
        rain_std=[8.0] * 12,
        evaporation_std=[6.0] * 12,
    )

    assert np.all(climate.temperature_std == pytest.approx(2.0))
    assert np.all(climate.rain_std        == pytest.approx(8.0))
    assert np.all(climate.evaporation_std == pytest.approx(6.0))
