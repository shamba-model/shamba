"""Tests for climate data handling.

Covers:
- aggregate_daily_to_monthly: correct values and year-major ordering
- get_climate_data: returns flat arrays via mocked API
- from_csv: 3-column CSV gives std=0; 6-column CSV populates std arrays
- ClimateData: std fields default to zero when not supplied
- from_vectors: means and stds computed correctly from multi-year input
"""
import numpy as np
import pytest
import calendar

from model.common.data_sources.climate import aggregate_daily_to_monthly, get_climate_data
from model.climate import ClimateData, from_csv, from_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_date_strings(start_year, end_year):
    """Generate ISO date strings for every day from start_year to end_year."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates


# ---------------------------------------------------------------------------
# aggregate_daily_to_monthly
# ---------------------------------------------------------------------------

def test_aggregate_daily_to_monthly_length():
    """Two years of data → 24 monthly values."""
    date_strings = make_date_strings(2000, 2001)
    values = np.ones(len(date_strings))
    result = aggregate_daily_to_monthly(values, date_strings, np.mean)
    assert len(result) == 24


def test_aggregate_daily_to_monthly_year_major_order():
    """Values are in year-major order: Jan_y1, Feb_y1, ..., Dec_y1, Jan_y2, ..."""
    date_strings = make_date_strings(2000, 2001)
    # Year 2000 days = 1.0, Year 2001 days = 2.0
    values = np.array([1.0 if ds[:4] == "2000" else 2.0 for ds in date_strings])
    result = aggregate_daily_to_monthly(values, date_strings, np.mean)
    # First 12 entries are 2000 (mean = 1.0), next 12 are 2001 (mean = 2.0)
    assert np.all(result[:12] == pytest.approx(1.0))
    assert np.all(result[12:] == pytest.approx(2.0))


def test_aggregate_daily_to_monthly_sum_aggregation():
    """Sum aggregation: January 2000 with 31 days × 2.0 mm → 62.0."""
    date_strings = make_date_strings(2000, 2000)
    values = np.full(len(date_strings), 2.0)
    result = aggregate_daily_to_monthly(values, date_strings, np.sum)
    jan_sum = 2.0 * 31  # January has 31 days
    assert result[0] == pytest.approx(jan_sum)


# ---------------------------------------------------------------------------
# get_climate_data — mocked API
# ---------------------------------------------------------------------------

def test_get_climate_data_returns_flat_arrays(monkeypatch):
    """get_climate_data returns (temp, rain, evap) as flat arrays."""
    import model.common.data_sources.climate as climate_mod

    date_strings = make_date_strings(2000, 2001)
    n_days = len(date_strings)
    fake_response = {
        "daily": {
            "time": date_strings,
            "temperature_2m_mean": [10.0] * n_days,
            "rain_sum": [2.0] * n_days,
            "et0_fao_evapotranspiration": [3.0] * n_days,
        }
    }
    monkeypatch.setattr(climate_mod, "get_weather_forecast", lambda **kwargs: fake_response)

    result = climate_mod.get_climate_data(latitude=0.0, longitude=0.0)

    assert result is not None
    temp, rain, evap = result
    assert len(temp) == 24  # 2 years × 12 months
    assert len(rain) == 24
    assert len(evap) == 24


def test_get_climate_data_year_major_ordering(monkeypatch):
    """Temperature values appear in year-major order in the returned array."""
    import model.common.data_sources.climate as climate_mod

    date_strings = make_date_strings(2000, 2001)
    n_days = len(date_strings)
    values = [10.0 if ds[:4] == "2000" else 20.0 for ds in date_strings]
    fake_response = {
        "daily": {
            "time": date_strings,
            "temperature_2m_mean": values,
            "rain_sum": [0.0] * n_days,
            "et0_fao_evapotranspiration": [0.0] * n_days,
        }
    }
    monkeypatch.setattr(climate_mod, "get_weather_forecast", lambda **kwargs: fake_response)

    temp, _, _ = climate_mod.get_climate_data(latitude=0.0, longitude=0.0)

    assert np.all(temp[:12] == pytest.approx(10.0))
    assert np.all(temp[12:] == pytest.approx(20.0))


def test_get_climate_data_returns_none_on_api_failure(monkeypatch):
    """get_climate_data returns None when the API call fails."""
    import model.common.data_sources.climate as climate_mod

    monkeypatch.setattr(climate_mod, "get_weather_forecast", lambda **kwargs: None)

    result = climate_mod.get_climate_data(latitude=0.0, longitude=0.0)
    assert result is None


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
    assert len(climate.temperature_std) == 12
    assert len(climate.rain_std)        == 12
    assert len(climate.evaporation_std) == 12


# ---------------------------------------------------------------------------
# ClimateData — default std behaviour
# ---------------------------------------------------------------------------

def test_climate_data_std_defaults_to_zero():
    """ClimateData constructed without std args: std arrays are all zero and 12 elements."""
    climate = ClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[60.0] * 12,
    )

    assert np.all(climate.temperature_std == pytest.approx(0.0))
    assert np.all(climate.rain_std        == pytest.approx(0.0))
    assert np.all(climate.evaporation_std == pytest.approx(0.0))
    assert len(climate.temperature_std) == 12
    assert len(climate.rain_std)        == 12
    assert len(climate.evaporation_std) == 12


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


# ---------------------------------------------------------------------------
# from_vectors — single year
# ---------------------------------------------------------------------------

def test_from_vectors_single_year_means_equal_input():
    """Single-year input: means equal the input values; stds all zero."""
    temp = [float(10 + m) for m in range(12)]
    rain = [float(80 + m) for m in range(12)]
    evap = [float(50 + m) for m in range(12)]
    climate = from_vectors(temp, rain, evap)

    np.testing.assert_array_almost_equal(climate.temperature, temp)
    np.testing.assert_array_almost_equal(climate.rain, rain)
    np.testing.assert_array_almost_equal(climate.evaporation, evap)
    assert np.all(climate.temperature_std == pytest.approx(0.0))
    assert np.all(climate.rain_std        == pytest.approx(0.0))
    assert np.all(climate.evaporation_std == pytest.approx(0.0))


# ---------------------------------------------------------------------------
# from_vectors — multi-year means
# ---------------------------------------------------------------------------

def test_from_vectors_multi_year_means_correct():
    """2-year input: monthly means are the per-month average across years."""
    # Year 1: all months = 10.0; Year 2: all months = 20.0 → mean = 15.0
    climate = from_vectors([10.0] * 12 + [20.0] * 12, [80.0] * 24, [50.0] * 24)

    assert np.all(climate.temperature == pytest.approx(15.0))
    assert len(climate.temperature) == 12


# ---------------------------------------------------------------------------
# from_vectors — multi-year stds
# ---------------------------------------------------------------------------

def test_from_vectors_multi_year_stds_correct():
    """2-year input with known values: stds match np.std(ddof=1) exactly."""
    climate = from_vectors([10.0] * 12 + [20.0] * 12, [80.0] * 24, [50.0] * 24)

    expected_std = np.std([10.0, 20.0], ddof=1)  # ≈ 7.071
    assert np.all(climate.temperature_std == pytest.approx(expected_std))
    assert len(climate.temperature_std) == 12


def test_from_vectors_identical_years_std_zero():
    """Multi-year input where all years are identical: stds are zero."""
    climate = from_vectors([15.0] * 36, [80.0] * 36, [50.0] * 36)

    assert np.all(climate.temperature_std == pytest.approx(0.0))
    assert len(climate.temperature_std) == 12
