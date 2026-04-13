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
