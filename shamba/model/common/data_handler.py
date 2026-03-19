#!/usr/bin/python

"""Module for data related functions in the SHAMBA program."""

from marshmallow.validate import Range, OneOf
from marshmallow import Schema, ValidationError, fields
import numpy as np
import re
from typing import Optional

from model.common.legacy_adapter import rename_legacy_headers


REQUIRED_HEADER_DATATYPE = {
    "lat": "scalar float",
    "lon": "scalar float",
    "yrs_proj": "scalar integer",
    "yr_mon": "scalar integer",
    "analysis_no": "scalar integer",
    "plot_name": "scalar integer",
    "year": "integer",
    "Temp": "float",
    "Rain": "float",
    "evap": "float",
    "pet": "float",
    "base_cover": "binary",
    "proj_cover": "binary",
    "fire_on_base": "binary",
    "fire_on_proj": "binary",
    "fire_off_base": "binary",
    "fire_off_proj": "binary",
}

ANCHOR_HEADER_DATATYPE_PATTERNS = {
    r"^crop_(base|proj)_spp": "scalar integer",
    r"^(base|proj)_species": "scalar integer",
    r"^(base|proj)_sf_qty": "float",  # only SF not LIT here, as only SF needs a matching _n proportion
}

CROP_HEADER_DATATYPE_PATTERNS = {
    # Crops (baseline & project), any index
    r"^crop_(base|proj)_spp": "scalar integer",
    r"^crop_(base|proj)_yd": "float",
    r"^crop_(base|proj)_left": "proportion",}

SPECIES_HEADER_DATATYPE_PATTERNS = {
    r"^(age_sp)": "integer",
    r"^(diam_sp)": "float",
}

COHORT_HEADER_DATATYPE_PATTERNS = {
    # Cohort species, planting years & densities by cohort index
    r"^(base|proj)_species": "scalar integer",
    r"^(base|proj)_plant_yr": "scalar integer",
    r"^(base|proj)_plant_dens": "scalar integer",

    # Thinning percents by cohort index
    r"^thin_(base|proj)_cohort": "proportion",

    # Thinning fractions by pool, cohort index embedded
    r"^thin_(base|proj)_(br|st)_cohort": "proportion",

    # Mortality by cohort
    r"^(base|proj)_mort_cohort": "proportion",
    r"^mort_(base|proj)_(br|st)_cohort": "proportion",
}

FERT_HEADER_DATATYPE_PATTERNS = {
    # Synthetic fertiliser
    r"^(base|proj)_sf_qty": "float",
    r"^(base|proj)_sf_n": "proportion",
    }

LITTER_HEADER_DATATYPE_PATTERNS = {
    # Litter
    r"^(base|proj)_lit_qty": "float",
}


# Pattern-based types for optional headers (regex patterns as keys)
HEADER_DATATYPE_OPT_PATTERNS = CROP_HEADER_DATATYPE_PATTERNS | SPECIES_HEADER_DATATYPE_PATTERNS | COHORT_HEADER_DATATYPE_PATTERNS | FERT_HEADER_DATATYPE_PATTERNS | LITTER_HEADER_DATATYPE_PATTERNS

# Explicit anchor -> required groupings for validate_all_grouped_headers.
# Each entry is (anchor_pattern, [required_patterns]).
# For each index N where a header matching anchor_pattern+N exists, all
# required_patterns+N must also be present.
GROUPED_HEADER_VALIDATIONS = [
    # If fertiliser quantity is given for index N, nitrogen % is required.
    (r"^(base|proj)_sf_qty", [r"^(base|proj)_sf_n"]),
    # If a crop species is given for index N, yield and residue fraction are required.
    (r"^crop_(base|proj)_spp", [r"^crop_(base|proj)_yd", r"^crop_(base|proj)_left"]),
    # If a cohort species is given for index N, planting year and density are required.
    (r"^(base|proj)_species", [r"^(base|proj)_plant_yr", r"^(base|proj)_plant_dens"]),
]

def get_header_type(header: str) -> Optional[str]:
    # Exact match first
    if header in REQUIRED_HEADER_DATATYPE:
        return REQUIRED_HEADER_DATATYPE[header]
    # Pattern match
    for pattern, type_name in HEADER_DATATYPE_OPT_PATTERNS.items():
        if re.match(pattern+r"(\d+)$", header):
            return type_name
    return None

def make_field_for_type(type_name: str):
    if type_name == "scalar float":
        return fields.Float()
    if type_name == "scalar integer":
        return fields.Integer()
    if type_name == "scalar proportion":
        return fields.Float(validate=Range(min=0.0, max=1.0))
    if type_name == "scalar binary":
        return fields.Integer(validate=OneOf([0, 1]))
    if type_name == "float":
        return fields.List(fields.Float())
    if type_name == "integer":
        return fields.List(fields.Integer())
    if type_name == "proportion":
        return fields.List(fields.Float(validate=Range(min=0.0, max=1.0)))
    if type_name == "binary":
        return fields.List(fields.Integer(validate=OneOf([0, 1])))
    # fallback
    raise ValueError(f"Header type name '{type_name}' not linked to a field spec.")

def build_field_specs(headers):
    field_specs = {}
    errors = []
    for h in headers:
        t = get_header_type(h)
        if t is None:
            errors.append(f"Header '{h}' does not match any known type.")
            continue
        field_specs[h] = make_field_for_type(t)
    if errors:
        error_message = "Errors found in data header specifications:\n" + "\n".join(errors)
        raise ValueError(error_message)
    return field_specs

def broadcast_to_length(data: dict, target_length: int, keys_to_broadcast: list[str]) -> np.ndarray:
    """Broadcasts a 1D array to a specified target length by repeating its values as needed."""
    for key, arr in data.items():
        if key in keys_to_broadcast and arr.size < target_length:
            # repeat the array values to the target length
            data[key] = np.tile(arr, target_length // arr.size + 1)[:target_length]
        elif arr.size == target_length:
            pass  # No need to change
        else:
            raise ValueError(f"Cannot broadcast array of size {arr.size} for key '{key}' to target length {target_length}.")
    return data

def first_error_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        for item in x:
            s = first_error_text(item)
            if s:
                return s
    if isinstance(x, dict):
        for v in x.values():
            s = first_error_text(v)
            if s:
                return s
    return None

def read_and_validate_timeseries_by_header(file_path: str, permitted_vector_lengths: list[int], target_vector_length: int | None = None) -> dict[str, np.ndarray]:
    """Reads a CSV file and returns a validated dictionary where each key is a header and the value is a 
        numpy array of the corresponding column data.
        The intended use is to read timeseries data from a CSV file where the first row contains 
        headers and the subsequent rows contain numerical data for some timestep - e.g. each year of the project.

    Args:
        file_name (str): The name of the CSV file to read."""
    
    headers = np.genfromtxt(file_path, delimiter=",", max_rows=1, dtype=str, encoding = None)
    headers = np.char.strip(headers)  # Remove leading/trailing whitespace from headers

    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=float)
    data = np.atleast_2d(data)
    data_dict = {header: data[:, i] for i, header in enumerate(headers)}
    # remove Inf and NaN values from data_dict
    for header, values in data_dict.items():
        data_dict[header] = values[np.isfinite(values)]

    # Check all headers for uniqueness and data for permitted length, collect and then print error messages
    errors = []

    if len(headers) != len(set(headers)):
        errors.append("Duplicate headers found in the CSV file. All headers must be unique.")

    for header, values in data_dict.items():
        if values.size == 0:
            errors.append(f"No data found for header '{header}'.")
        if 1 in permitted_vector_lengths and values.size == 1:
            data_dict[header] = values.flatten()  # Convert to 1D array if it's a single value
        elif values.size not in permitted_vector_lengths:
            errors.append(f"Data for header '{header}' has {values.size} entries, but expected one of {permitted_vector_lengths}.")
    
    # Validate headers against expected types, raise error messages
    field_specs = build_field_specs(headers) # this will collect and raise errors
    
    # If field_specs creatd, validate data against the schema, collect and then print error messages
    InputSchema = Schema.from_dict(field_specs)
    # Convert numpy arrays to scalars or lists for validation
    data_for_validation = {}
    for header, arr in data_dict.items():
       type_name = get_header_type(header)
       if type_name and "scalar" in type_name:
           # enforce exactly one value for scalars
           if arr.size != 1:
               raise ValueError(f"Header '{header}' must have exactly one value, found {arr.size}.")
           data_for_validation[header] = arr.item()      # scalar
       else:
           # keep vectors as lists for schema
           data_for_validation[header] = arr.tolist()

    try:
        validated = InputSchema().load(data_for_validation)  # Validate data against the schema
        validated_data_dict = {h: np.array(validated[h]) for h in validated}  # Convert validated data back to numpy arrays
    except ValidationError as e:
        collapsed = {}
        for field, msgs in e.messages.items():
        # If there are any errors for this field, just emit one summary string
            error_message = first_error_text(msgs) # this extracts the first error message in a readable string
            collapsed[field] = f"Vector '{field}' {error_message}"

        errors.append("Validation errors:\n" + "\n".join(collapsed.values()))

    if errors:
        error_message = "Errors found in the input data:\n" + "\n".join(errors)
        raise ValueError(error_message)
    
    # Broadcast any keys that are marked for broadcasting in the field specs, and convert to final 2d array for model input
    if target_vector_length is not None:
        keys_to_broadcast = [h for h, spec in field_specs.items() if isinstance(spec, fields.List)] # checks the field specs for which keys are lists (i.e. vectors) and should be broadcast if they have only one value
        validated_data_dict = broadcast_to_length(validated_data_dict, target_length= target_vector_length, keys_to_broadcast=keys_to_broadcast)
    
    return validated_data_dict


def group_indices(headers, pattern):
    """
    Find all integer indices N such that some header matches
    `pattern + r"(\\d+)$"`. The base `pattern` may have other
    capturing groups; always take the LAST capturing group as the index.
    """
    return {
        int(m.group(m.lastindex))
        for h in headers
        if (m := re.match(pattern, h))
    }

def _pattern_to_example(pattern: str, index: int) -> str:
    """Convert a regex header pattern and index into a human-readable example string.

    For instance, r"^(base|proj)_sf_n" with index 1 becomes "{base/proj}_sf_n1".
    """
    p = pattern.lstrip("^").rstrip("$")
    p = re.sub(r"\(([^)]+)\)", lambda m: "{" + "/".join(m.group(1).split("|")) + "}", p)
    return f"{p}{index}"


def validate_grouped_headers(headers, anchor_pattern, required_patterns):
    """
    For each index N where a header matches `anchor_pattern + r"(N)$"`,
    require that for every pattern P in `required_patterns` there exists
    some header matching `P + r"(N)$"`.
    """
    errors = []

    anchor_indexes = group_indices(headers, anchor_pattern + r"(\d+)$")
    for i in anchor_indexes:
        for required_pattern in required_patterns:
            required_regex = required_pattern + rf"{i}$"
            if not any(re.match(required_regex, h) for h in headers):
                errors.append(
                    f"'{_pattern_to_example(required_pattern, i)}' is required "
                    f"because '{_pattern_to_example(anchor_pattern, i)}' is present"
                )
    return errors


def validate_all_grouped_headers(data):
    errors = []
    for anchor_pattern, required_patterns in GROUPED_HEADER_VALIDATIONS:
        errors.extend(
            validate_grouped_headers(
                list(data.keys()),
                anchor_pattern=anchor_pattern,
                required_patterns=required_patterns,
            )
        )
    return errors

def validate_species_data(data: dict) -> list[str]:
    """Check that every species code declared in (base|proj)_species{n} headers
    has corresponding age and diameter data in the same input dict.

    Species codes are the *values* of those scalar headers (e.g. base_species1 = 3
    means cohort 1 uses species type 3, so age_sp3 and diam_sp3 must be present).

    Returns a list of error strings (empty if valid).
    """
    errors = []
    species_codes = set()

    for key, arr in data.items():
        if re.match(r"^(base|proj)_species\d+$", key):
            try:
                species_codes.add(int(np.atleast_1d(arr)[0]))
            except (ValueError, IndexError):
                errors.append(f"Could not read species code from '{key}'.")

    for code in sorted(species_codes):
        for required in (f"age_sp{code}", f"diam_sp{code}"):
            if required not in data:
                errors.append(
                    f"'{required}' is required because species {code} is declared "
                    f"in the input data but has no corresponding size data."
                )

    return errors


def resolve_evap_pet(data_dict: dict) -> dict:
    """Ensure climate data contains exactly one evaporation column named 'evap'.

    Rules:
    - If both 'evap' and 'pet' are present, 'pet' is discarded and 'evap' is used.
    - If only 'pet' is present, it is converted to open-pan evaporation (evap = pet / 0.75)
      and stored under 'evap'.
    - If neither is present, raises ValueError.

    Returns the dict with 'pet' removed and 'evap' guaranteed to be present.
    """
    has_evap = "evap" in data_dict
    has_pet = "pet" in data_dict

    if not has_evap and not has_pet:
        raise ValueError("Climate data must contain either 'evap' or 'pet'.")

    if has_evap and has_pet:
        print("Both 'evap' and 'pet' found in climate data — 'evap' will be used and 'pet' discarded.")
        del data_dict["pet"]

    if not has_evap and has_pet:
        data_dict["evap"] = data_dict.pop("pet") / 0.75

    return data_dict


def _read_single_row_csv(file_path: str) -> dict[str, float]:
    """Read a single-row legacy CSV without validating header names.

    The legacy single-row format stores all parameters as scalar values
    in a single data row. Header names use the old naming convention and
    are not registered in the new-format type dicts; this reader bypasses
    that validation so the expand function can translate them.

    Returns:
        dict mapping each header string to its scalar float value.
    """
    headers = np.genfromtxt(file_path, delimiter=",", max_rows=1, dtype=str, encoding=None)
    headers = np.char.strip(headers)
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=float)
    data = np.atleast_1d(data.flatten())
    return {h: float(v) for h, v in zip(headers, data)}


def expand_single_row_data_input(file_path: str):
    """Transform a legacy single-row CSV into the standardised four-dict output.

    The single-row format stores all parameters as scalar values and uses
    interval/presence scalars to describe management schedules. This function
    converts those scalars into annual vectors using the new header naming
    convention, so downstream models receive the same shape regardless of
    which input format was used.

    Returns:
        scalar_input_data   dict of scalar site/project parameters
        tree_size_data      dict of variable-length age/diameter arrays per species
        mgmt_input_data     dict of annual (no_of_years,) management vectors
        cover_data          dict of monthly-annual (12 * no_of_years,) cover vectors
    """
    raw = rename_legacy_headers(_read_single_row_csv(file_path))

    def s(key, default=0.0) -> float:
        return float(raw.get(key, default))

    def si(key, default=0) -> int:
        return int(raw.get(key, default))

    no_of_years = si("yrs_proj")

    # --- Scalar pass-through ---
    scalar_input_data = {}
    for h in ("lat", "lon", "yrs_proj", "yr_mon", "analysis_no", "plot_name"):
        if h in raw:
            scalar_input_data[h] = np.atleast_1d(np.asarray(raw[h], dtype=float))

    # Species and planting parameters (already renamed to new convention by rename_legacy_headers)
    for key in raw:
        if any(re.match(p, key) for p in (
            r"^(base|proj)_species\d+$",
            r"^(base|proj)_plant_yr\d+$",
            r"^(base|proj)_plant_dens\d+$",
        )):
            scalar_input_data[key] = np.atleast_1d(np.asarray(raw[key], dtype=float))

    # --- Tree size data ---
    # Collect species numbers from the scalar pass-through
    sp_list = []
    for key, arr in scalar_input_data.items():
        if re.match(r"^(base|proj)_species\d+$", key):
            try:
                sp_list.append(int(arr[0]))
            except Exception:
                continue

    tree_size_data = {}
    age_base_keys = ["age1", "age2", "age3", "age4", "age5", "age6"]
    diam_base_keys = ["diam1", "diam2", "diam3", "diam4", "diam5", "diam6"]

    for spp_number in sorted(set(sp_list)):
        sp_prefix = "" if spp_number == 1 else f"sp{spp_number}_"
        age_vals = [s(f"{sp_prefix}{k}") for k in age_base_keys]
        diam_vals = [s(f"{sp_prefix}{k}") for k in diam_base_keys]

        # Sort measurements by age
        pairs = sorted(zip(age_vals, diam_vals))
        if pairs:
            ages, diams = zip(*sorted(pairs))
            tree_size_data[f"age_sp{spp_number}"] = np.array(ages)
            tree_size_data[f"diam_sp{spp_number}"] = np.array(diams)

    # --- Crop data ---
    crop_data = {}
    for index in range(1, 4):
        for prefix in ("crop_base", "crop_proj"):
            scalar_input_data[f"{prefix}_spp{index}"] = np.atleast_1d(
                np.asarray(si(f"{prefix}_spp{index}"), dtype=float)
            )
            start_year = si(f"{prefix}_start{index}")
            end_year = si(f"{prefix}_end{index}")
            harvest_yield = np.zeros(no_of_years)
            harv_frac = np.zeros(no_of_years)
            harvest_yield[start_year:end_year] = s(f"{prefix}_yd{index}")
            harv_frac[start_year:end_year] = s(f"{prefix}_left{index}")
            crop_data[f"{prefix}_yd{index}"] = harvest_yield
            crop_data[f"{prefix}_left{index}"] = harv_frac

    # --- Fertiliser ---
    fertiliser_data = {}
    for prefix in ("base", "proj"):
        qty_vec = np.zeros(no_of_years)
        n_vec = np.zeros(no_of_years)
        interval = si(f"{prefix}_sf_int")
        if interval > 0:
            qty_vec[::interval] = s(f"{prefix}_sf_qty")
            n_vec[::interval] = s(f"{prefix}_sf_n")
        fertiliser_data[f"{prefix}_sf_qty"] = qty_vec
        fertiliser_data[f"{prefix}_sf_n"] = n_vec

    # --- Litter ---
    litter_data = {}
    for prefix in ("base", "proj"):
        qty_vec = np.zeros(no_of_years)
        interval = si(f"{prefix}_lit_int")
        if interval > 0:
            qty_vec[::interval] = s(f"{prefix}_lit_qty")
        litter_data[f"{prefix}_lit_qty"] = qty_vec

    # --- Fire ---
    fire_data = {}
    for suffix in ("base", "proj"):
        fire_on = np.zeros(no_of_years)
        fire_off = np.zeros(no_of_years) if si(f"fire_off_{suffix}") == 0 else np.ones(no_of_years)
        interval = si(f"fire_int_{suffix}")
        if interval > 0:
            fire_on[::interval] = s(f"fire_pres_{suffix}")
        fire_data[f"fire_on_{suffix}"] = fire_on
        fire_data[f"fire_off_{suffix}"] = fire_off

    # --- Cover ---
    cover_data = {}
    for prefix in ("base", "proj"):
        cover_year = np.zeros(12)
        cover_year[si(f"{prefix}_cvr_mth_st"):si(f"{prefix}_cvr_mth_en")] = si(f"{prefix}_cvr_pres")
        cover_data[f"{prefix}_cover"] = np.tile(cover_year, no_of_years)

    # --- Tree management ---
    # Build thinning/mortality arrays and copy them to every cohort present in
    # this scenario.  Base always has exactly one cohort in the single-row
    # format; proj can have up to three (species1–species3).
    tree_data = {}
    for mgmt in ("base", "proj"):
        thinning_array = np.zeros(no_of_years + 1)
        for i in range(1, 5):
            yr = si(f"thin_{mgmt}_yr{i}")
            pc = s(f"thin_{mgmt}_pc{i}")
            if yr > 0:
                thinning_array[yr] = pc

        if mgmt == "base":
            cohort_indices = [1]
        else:
            cohort_indices = [i for i in range(1, 4) if f"species{i}" in raw] or [1]

        for c in cohort_indices:
            tree_data[f"thin_{mgmt}_cohort{c}"] = thinning_array
            tree_data[f"mort_{mgmt}_cohort{c}"] = np.full(no_of_years + 1, s(f"{mgmt}_mort"))
            for pool in ("br", "st"):
                for turnover in ("thin", "mort"):
                    tree_data[f"{turnover}_{mgmt}_{pool}_cohort{c}"] = np.full(
                        no_of_years + 1, s(f"{turnover}_{mgmt}_{pool}")
                    )

    mgmt_input_data = crop_data | fertiliser_data | litter_data | fire_data | tree_data

    return scalar_input_data, tree_size_data, mgmt_input_data, cover_data

