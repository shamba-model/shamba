# Test coverage summary

---

## test_validation.py — data_handler.py validation functions

### `validate_required_mgmt_keys()`
| Test | Purpose |
|------|---------|
| `test_passes_when_all_required_keys_present` | All required keys present → no errors returned |
| `test_error_names_missing_keys_in_single_message` | Multiple missing keys are grouped into one error message (not one error per key) |

### `validate_species_data()`
| Test | Purpose |
|------|---------|
| `test_passes_when_age_and_diam_present_for_declared_species` | Valid species with age + diameter data → no errors |
| `test_error_when_size_data_missing_for_declared_species` | Declared species with no size data → errors naming the missing keys |
| `test_no_false_positive_when_only_one_key_missing` | Only the actually missing key is reported; present keys are not flagged |

### `resolve_evap_pet()`
| Test | Purpose |
|------|---------|
| `test_evap_wins_and_pet_discarded_when_both_present` | When both evap and PET supplied, evap is used and PET is discarded |
| `test_pet_converted_to_evap_via_open_pan_factor` | When only PET supplied, it is converted to evap via the open-pan factor (÷ 0.75) |
| `test_raises_when_neither_evap_nor_pet_present` | Neither key present → ValueError raised |

---

## test_emit.py — emit.py building-block functions

### `reduce_from_fire()`
| Test | Purpose |
|------|---------|
| `test_no_fire_returns_unmodified_above_plus_below` | Zero fire → output = above + below carbon, unmodified |
| `test_fire_reduces_crop_above_ground_by_combustion_factor` | Fire year reduces crop above-ground by the crop combustion factor (0.85); below-ground untouched |
| `test_fire_reduces_tree_above_ground_by_tree_combustion_factor` | Fire year reduces tree above-ground by the tree combustion factor (0.74); below-ground untouched |
| `test_empty_inputs_return_zeros` | No crop, tree, or litter inputs → zero output arrays |

### `soc_sink()`
| Test | Purpose |
|------|---------|
| `test_numeric_value_1_tC_increase` | 1 t C/ha increase → 44/12 t CO2/ha/yr (IPCC conversion factor) |
| `test_constant_soc_returns_zero` | Constant SOC → zero sink/source |

### `tree_sink()`
| Test | Purpose |
|------|---------|
| `test_numeric_value` | Known biomass increment → correct CO2e delta (2 t C/ha/yr → 2 × 44/12) |
| `test_multiple_trees_are_summed` | Multiple tree objects contribute additively to the total sink |

### Zero-input smoke tests
| Test | Purpose |
|------|---------|
| `test_fert_emit_zero_for_empty_inputs` | No fertiliser inputs → zero emissions |
| `test_nitrogen_emit_zero_for_empty_inputs` | No nitrogen inputs → zero emissions |

---

## test_allometrics.py — tree_growth.py allometric and growth curve functions

### Allometric equations
| Test | Purpose |
|------|---------|
| `TestRyanAllometric::test_known_value_at_dbh_10` | Numeric spot-check of Ryan (2010) formula at DBH=10 against hand calculation |
| `TestRyanAllometric::test_zero_dbh_returns_zero` | DBH=0 → AGB=0 (guard against log(0)) |
| `TestChaveDryAllometric::test_known_value_at_dbh_10_wd_0p6_carbon_0p48` | Numeric spot-check of Chave dry (2005) formula at DBH=10, WD=0.6, carbon=0.48 |
| `TestChaveDryAllometric::test_higher_wood_density_gives_higher_agb` | WD appears as a multiplicative factor; higher WD must give higher AGB |
| `TestTumwebazeMarkhamiaAllometric::test_known_value_at_dbh_10_carbon_0p48` | Numeric spot-check of Tumwebaze Markhamia (2013) formula at DBH=10, carbon=0.48 |

### Growth curve functions (Eqs. 6.1–6.4)
| Test | Purpose |
|------|---------|
| `test_linear_function` | f(x) = a·x: correct value and zero at x=0 |
| `test_exponential_1param_function` | f(x) = (1+a)^x − 1: zero at x=0; correct value at x=1, a=1 |
| `test_hyperbolic_function` | f(x) = a·(1 − exp(−b·x)): zero at x=0; approaches asymptote a at large x |
| `test_logistic_function` | f(x) = a/(1+exp(−b·(x−c))): equals a/2 at the inflection point x=c |
| `test_fit_produces_monotonically_increasing_hyperbolic_curve` | Fitting to monotonically increasing data yields a monotonically increasing hyperbolic curve |

### `TestFromCsvBiomassInput` — direct AGB input (bypasses allometrics)
| Test | Purpose |
|------|---------|
| `test_uses_provided_biomass_directly` | When `biomass_sp<n>` is present, growth uses those values not the allometric equation |
| `test_falls_back_to_allometry_when_biomass_not_provided` | Without biomass input, falls back to computing AGB from diameter via allometric key |
| `test_provided_biomass_independent_of_allometric_key` | Changing allometric key has no effect on growth when biomass is supplied directly |
| `test_age_and_biomass_only_no_diam` | `diam_sp<n>` may be absent when biomass is provided directly; no diameter stored |

---

## test_roth_c.py — RothC soil carbon model

### `get_rmf()` — Rate Modifying Factor
| Test | Purpose |
|------|---------|
| `test_rmf_equals_one_when_rain_always_exceeds_evap` | Always-wet climate triggers the short-circuit branch, returning b.mean()=1.0 for all years |
| `test_severe_drought_gives_lower_rmf_than_mild_drought` | Larger moisture deficit → smaller b factor → lower RMF (tests the full moisture calculation path) |
| `test_soil_cover_reduces_rmf_under_drought` | Soil cover applies c=0.6 instead of 1.0, reducing the RMF |

### `inverse_roth_c.create()`
| Test | Purpose |
|------|---------|
| `test_equilibrium_pools_sum_to_approximately_ceq` | Equilibrium pool totals (+ IOM) ≈ Ceq within 20% (coarse grid search tolerance) |
| `test_equilibrium_pools_are_non_negative` | All four equilibrium pool values are ≥ 0 |

### `forward_roth_c.create()`
| Test | Purpose |
|------|---------|
| `test_soc_has_correct_shape` | Output SOC array is (n_years+1) × 4 pools |
| `test_pools_do_not_go_substantially_negative` | ODE solver may undershoot near zero; no pool goes below −1e-6 t C/ha |
| `test_inverse_forward_round_trip_is_stable` | Equilibrium pools from inverse solver fed into forward solver produce < 30% total carbon change over 10 years |
| `test_forward_solver_numeric_regression` *(skipped)* | Exact pool values at year 5 to be confirmed against the official RothC Python reference repository before activation |

---

## test_crop_emissions.py — crop model emissions (baseline and project)

### `test_crop_model()` — parametrized: WL, testB
| Test case | Purpose |
|-----------|---------|
| WL | Constant single-crop baseline (all years equal); project switches to zero emissions after year 4 (crop removed) |
| testB | Mixed-crop baseline with two crop types across different year ranges; project has irregular non-zero windows from changing crop schedules |

Both cases call `get_crop_bases` / `get_crop_projects`, then `Emit.create(crop=..., fire=..., burn_off=...)` and assert against hardcoded per-year emission arrays (rel tolerance 1e-4).

---

## test_fertiliser_emissions.py — synthetic fertiliser emissions (baseline and project)

### `test_fertiliser_model()` — parametrized: WL, testB
| Test case | Purpose |
|-----------|---------|
| WL | No fertiliser in either baseline or project — all-zero expected arrays confirm no spurious emissions |
| testB | Periodic fertiliser applications in project only; expected array has non-zero values at each application year |

Calls `LitterModel.synthetic_fertiliser(frequency, quantity, nitrogen, no_of_years)`, then `Emit.create(fert=..., fire=..., burn_off=...)` and asserts against hardcoded per-year arrays (rel tolerance 1e-4).

---

## test_fire_emissions.py — fire (burn-off) emissions from crops

### `test_crop_fire_model()` — parametrized: WL, testB
| Test case | Purpose |
|-----------|---------|
| WL | No fire in either scenario — all-zero arrays confirm `fire_emit` is inert when fire is absent |
| testB | Project has fire events; expected arrays show non-zero fire emissions only at the scheduled fire years |

Calls `Emit.fire_emit(fire=..., crop=..., tree=[], litter=[], burn_off=..., gwp=GWP_AR6)` and asserts per-year arrays (rel tolerance 1e-5).

---

## test_litter_emissions.py — external litter input emissions (baseline and project)

### `test_litter_model()` — parametrized: WL, testB
| Test case | Purpose |
|-----------|---------|
| WL | No litter inputs in either scenario — all-zero arrays confirm no spurious emissions |
| testB | Periodic litter applications; baseline and project expected arrays show non-zero values at each litter application year, with project year-0 higher due to additional litter type |

Calls `LitterModel.from_defaults(litter_frequency, litter_quantity, no_of_years)`, then `Emit.create(litter=..., fire=..., burn_off=...)` and asserts per-year arrays (rel tolerance 1e-5).

---

## test_tree_emissions.py — tree model emissions (baseline and project)

### `test_tree_model()` — parametrized: WL, testB
| Test case | Purpose |
|-----------|---------|
| WL | Baseline has no trees (all zeros); project shows a carbon sink building over time with positive spikes at thinning/harvest events |
| testB | Baseline has a single pre-existing tree cohort (small but non-zero sink); project has multiple cohorts with staggered planting — irregular per-year sink values reflecting growth curves, thinning, and mortality |

Constructs tree objects via `TreeParams`, `TreeGrowth.get_growth` / `create_tree_growths`, `TreeModel.from_defaults` / `create_tree_projects`, then calls `Emit.create(tree=..., fire=..., burn_off=...)`. Asserts per-year arrays with relaxed tolerance (rel=1e-3) because expected values depend on curve-fitting results that may differ slightly from Excel reference calculations.

---

## test_integration_split_file.py — split-file CSV input path

| Test | Purpose |
|------|---------|
| `test_split_file_data_pipeline_matches_single_row` | Reading the three split-file CSVs (plot, mgmt, tree_size, climate_cover) produces a dict with identical keys and values to `expand_single_row_data_input` for the same scenario |
| `test_split_file_full_run_matches_single_row` | `handle_intervention` returns numerically identical baseline and project emissions for both input paths (rtol=1e-10) |

Fixtures are generated programmatically from `WL_input.csv`; no manually maintained split-file fixtures.

---

## test_soil_data.py — soil uncertainty capture

### `read_soil_table()`
| Test | Purpose |
|------|---------|
| `test_read_soil_table_3col_returns_soil_data` | 3-column CSV: q05 and q95 both equal the mean (no uncertainty) |
| `test_read_soil_table_7col_reads_quantiles` | 7-column CSV: q05/q95 columns parsed into correct fields |
| `test_read_soil_table_plot_order_mismatch_raises` | Mismatched plot ID raises a plain-language ValueError |

### `process_data()`
| Test | Purpose |
|------|---------|
| `test_process_data_single_layer_depth_weighted` | Depth-weighted averages for mean and quantiles computed correctly over 0–30 cm denominator |
| `test_process_data_missing_quantile_key_skipped` | Depths lacking Q0.05/Q0.95 keys contribute zero to those weighted sums |

### `get_soc_and_clay()`
| Test | Purpose |
|------|---------|
| `test_get_soc_and_clay_returns_soil_data` | Valid API payload returns a SoilData with correct mean values |
| `test_get_soc_and_clay_missing_soc_raises` | Missing SOC raises a ValueError mentioning "SOC" |
| `test_get_soc_and_clay_missing_clay_raises` | Missing clay raises a ValueError mentioning "clay" |

### `create()` / `SoilParamsData`
| Test | Purpose |
|------|---------|
| `test_create_without_quantiles_defaults_to_mean` | Only Cy0/clay supplied: q05 and q95 fields default to the mean value |
| `test_create_with_quantiles_stores_correctly` | All six quantile keys supplied: stored in correct SoilParamsData fields |
| `test_create_quantile_ordering_violation_raises` | q05 > mean raises a validation error |

---

## test_climate_data.py — climate data handling

### `aggregate_daily_to_monthly()`
| Test | Purpose |
|------|---------|
| `test_aggregate_daily_to_monthly_length` | Two years of daily data → 24 monthly values |
| `test_aggregate_daily_to_monthly_year_major_order` | Output is in year-major order: Jan_y1 … Dec_y1, Jan_y2 … Dec_y2 |
| `test_aggregate_daily_to_monthly_sum_aggregation` | Sum aggregation: January (31 days × 2.0 mm) = 62.0 |

### `get_climate_data()` — mocked API
| Test | Purpose |
|------|---------|
| `test_get_climate_data_returns_flat_arrays` | Returns (temp, rain, evap) as flat monthly arrays |
| `test_get_climate_data_year_major_ordering` | Temperature values appear in year-major order |
| `test_get_climate_data_returns_none_on_api_failure` | API returning None → function returns None |

### `from_csv()`
| Test | Purpose |
|------|---------|
| `test_from_csv_3col_std_is_zero` | 3-column CSV: all std arrays are zero |
| `test_from_csv_3col_mean_values_correct` | 3-column CSV: mean arrays read correctly |
| `test_from_csv_6col_std_populated` | 6-column CSV: std arrays populated from file columns |

### `ClimateData`
| Test | Purpose |
|------|---------|
| `test_climate_data_std_defaults_to_zero` | Constructed without std args: std arrays are all zero, length 12 |
| `test_climate_data_std_stored_correctly` | Constructed with std args: values stored in correct fields |

### `from_vectors()`
| Test | Purpose |
|------|---------|
| `test_from_vectors_single_year_means_equal_input` | Single-year input: means equal the input; stds are all zero |
| `test_from_vectors_multi_year_means_correct` | 2-year input: monthly means are the per-month average across years |
| `test_from_vectors_multi_year_stds_correct` | 2-year input: stds match `np.std(ddof=1)` exactly |
| `test_from_vectors_identical_years_std_zero` | All years identical: stds are zero |

---

## test_distribution_handler.py — distribution_handler.py

### Valid loading
| Test | Purpose |
|------|---------|
| `test_load_all_seven_distributions` | CSV with one row per distribution type loads without errors |
| `test_load_returns_correct_spec_values` | Parsed DistributionSpec contains the values from the CSV row |
| `test_load_min_abs_absent_is_none` | Empty min_abs cell → spec.min_abs is None |
| `test_load_min_abs_column_absent` | CSV without min_abs column is valid; spec.min_abs is None |
| `test_empty_spread_upper_copies_from_spread_lower` | Empty spread_upper treated as equal to spread_lower |

### Validation errors
| Test | Purpose |
|------|---------|
| `test_unknown_parameter_raises` | Parameter name not in input dict → ValueError naming the parameter |
| `test_unsupported_distribution_raises` | Unrecognised distribution name → ValueError naming the distribution |
| `test_zero_spread_lower_raises` | spread_lower = 0 → ValueError mentioning spread_lower |
| `test_negative_spread_upper_raises` | Negative spread_upper → ValueError mentioning spread_upper |
| `test_negative_min_abs_raises` | Negative min_abs → ValueError mentioning min_abs |
| `test_min_abs_on_skew_normal_raises` | min_abs with skew_normal → error mentioning skew_normal |
| `test_min_abs_on_beta_raises` | min_abs with beta → error mentioning beta |
| `test_asymmetric_spread_on_symmetric_distribution_raises` | spread_lower ≠ spread_upper on normal/truncated_normal/lognormal → error mentioning "symmetric" (parametrized over all three) |
| `test_beta_on_non_fraction_parameter_raises` | beta distribution on a parameter whose mean is outside [0,1] → error |
| `test_multiple_errors_reported_together` | Two bad rows: both error messages appear in the same exception |

### Warnings
| Test | Purpose |
|------|---------|
| `test_warning_base_zero_no_min_abs` | Parameter with base value 0 and no min_abs → warning naming the parameter |
| `test_warning_fraction_parameter_can_breach_unit_interval` | Fraction parameter with a non-beta distribution that can exceed [0,1] → warning |
| `test_warning_temp_without_min_abs` | `temp` parameter without min_abs → warning |
| `test_warning_skew_normal_nearly_symmetric` | skew_normal with nearly equal spreads → warning to use normal |
| `test_warning_lognormal_high_sigma` | lognormal with spread > 0.5 → warning about the large-sigma approximation |
| `test_no_error_raised_despite_warnings` | Rows triggering warnings but no errors still return a valid spec |

---

## test_sampler.py — sampler.py

### `draw_samples()` — output structure
| Test | Purpose |
|------|---------|
| `test_output_length` | Result list has exactly n_samples entries |
| `test_output_keys_match_base` | Each sample dict has the same keys as the base dict |
| `test_output_shapes_match_base` | Each sample array has the same shape as the corresponding base array |

### `draw_samples()` — values
| Test | Purpose |
|------|---------|
| `test_perturbed_values_differ_from_base` | With large spread, drawn values differ from base across N=100 samples |
| `test_vector_draws_are_element_wise` | Each element drawn independently: position with higher base value has higher std |
| `test_unperturbed_keys_unchanged` | Keys absent from distributions dict are copied unchanged |
| `test_base_dict_not_mutated` | Original base dict is not modified by draw_samples |
| `test_climate_perturbation_is_multiplicative` | Rain perturbation: ratio drawn/base is uniform across all months (scalar multiplier) |

### `draw_samples()` — distribution coverage and special cases
| Test | Purpose |
|------|---------|
| `test_all_distributions_produce_n_samples` | All seven distribution types produce the correct number of samples (parametrized) |
| `test_skew_normal_equal_spreads_centred_on_base` | skew_normal with equal spreads produces draws centred near the base mean |
| `test_min_abs_floor_normal` | Tiny base value: min_abs dominates the std and produces meaningful spread |
| `test_fraction_parameter_clamped_with_warning` | Fraction parameter breaching [0,1] is clamped and a warning is emitted |

### `sample_soil_params()`
| Test | Purpose |
|------|---------|
| `test_sample_soil_params_zero_uncertainty` | q05 = q95 = mean → all draws equal the mean |
| `test_sample_soil_params_nonzero_spread` | Non-zero quantiles produce draws with spread matching the expected σ |
| `test_sample_soil_params_cy0_clipped_to_nonnegative` | Cy0 draws are clipped to ≥ 0 |
| `test_sample_soil_params_clay_clipped_to_valid_range` | Clay draws are clipped to [0, 100] |
| `test_sample_soil_params_derived_fields_consistent` | Ceq and iom in each sample are computed from the drawn Cy0, not copied from the mean |

### `sample_climate_params()`
| Test | Purpose |
|------|---------|
| `test_sample_climate_params_zero_std` | All stds zero → all draws equal the means |
| `test_sample_climate_params_nonzero_std` | Non-zero std: draws match Normal(mean, std) in mean and spread |
| `test_sample_climate_params_rain_clipped` | Rain draws clipped to ≥ 0 |
| `test_sample_climate_params_evap_clipped` | Evaporation draws clipped to ≥ 0 |

### `sample_model_params()`
| Test | Purpose |
|------|---------|
| `test_sample_model_params_output_length` | Returns a list of length n_samples |
| `test_sample_model_params_output_type` | Each element is an EmissionFactors namedtuple |
| `test_sample_model_params_ef_burn_structure` | ef_burn dict has the four expected keys |
| `test_sample_model_params_combustion_factor_structure` | combustion_factor dict has keys 'crop' and 'tree' |
| `test_sample_model_params_scalar_fields_are_float` | ef_N_inputs and volatile_frac_* are plain floats |
| `test_sample_model_params_values_vary` | Default distributions produce draws that differ from the default EmissionFactors |
| `test_sample_model_params_no_distributions_returns_constant` | Empty distribution_dict → all samples equal the base EmissionFactors |
| `test_sample_model_params_seed_reproducibility` | Same seed produces identical draws on two separate calls |
| `test_sample_model_params_different_seeds_differ` | Different seeds produce different draws |

---

## test_runner.py — runner.py

### `run_monte_carlo()`
| Test | Purpose |
|------|---------|
| `test_run_monte_carlo_returns_n_samples` | Result list has exactly n_samples entries |
| `test_run_monte_carlo_deterministic_no_distributions` | With no distribution_dict, every handle_intervention call receives identical input dicts |

### `summarise_mc_results()`
| Test | Purpose |
|------|---------|
| `test_summarise_mc_results_column_names` | All expected `{pool}_{stat}` columns present for every scenario |
| `test_summarise_mc_results_all_scenarios_same_keys` | base, project, and diff dicts have identical column sets |
| `test_summarise_mc_results_array_length` | Each column array has length n_years |
| `test_summarise_mc_results_zero_std_when_identical` | Identical samples → std is zero in every scenario |
| `test_summarise_mc_results_known_mean` | Mean of emit_difference equals the known per-year mean across samples |
| `test_summarise_mc_results_q50_is_median` | q50 matches the expected median for a known set of samples |
| `test_summarise_mc_results_base_is_zero` | Baseline summary mean is zero when all sample baseline values are zero |

### `write_mc_summary_csv()`
| Test | Purpose |
|------|---------|
| `test_write_mc_summary_csv_headers` | CSV header row is 'year' followed by all summary column names |
| `test_write_mc_summary_csv_row_count` | CSV has exactly n_years data rows plus one header row |
| `test_write_mc_summary_csv_year_column` | Year column contains 1, 2, …, n_years |
| `test_write_mc_summary_csv_values_roundtrip` | Values read back from CSV match the summary dict to float precision |
