#!/usr/bin/python

"""Module for data related functions in the SHAMBA program."""

from marshmallow.validate import Range, OneOf
from marshmallow import Schema, ValidationError, fields
import numpy as np
import re
from typing import Optional


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
    "evap": "float", # TODO: evap OR pet required
    "pet": "float",
    "base_cover": "binary",
    "proj_cover": "binary",
    "fire_on_base": "binary",
    "fire_on_proj": "binary",
    "fire_off_base": "binary", # TODO: current code has fire_off as a bool, then applied every year
    "fire_off_proj": "binary",
}

ANCHOR_HEADER_DATATYPE_PATTERNS = {
    r"^crop_(base|proj)_spp": "scalar integer",
     r"^(base|proj)_species": "scalar integer",  # TODO: this is different from current naming "species_base" "species1" etc
     r"^(base|proj)_sf_qty": "float", # only SF not LIT here, as only SF needs a matching _n proportion
}

CROP_HEADER_DATATYPE_PATTERNS = {
    # Crops (baseline & project), any index
    r"^crop_(base|proj)_spp": "scalar integer",
    r"^crop_(base|proj)_yd": "float",
    r"^crop_(base|proj)_left": "proportion",}

SPECIES_HEADER_DATATYPE_PATTERNS = { # TODO: this needs a specific check: what species numbers are contained in the data under headers {r"^(base|proj)_species"}, and also needs to match the species data in the related file
    # Tree ages/diams: tree1 / sp2 / sp3 generalized # TODO: this needs a new data input file: species index should be embedded in the header, and there may be more than 3 species, so will need to be in a different input file and validated separately
    r"^(age_sp)": "integer",
    r"^(diam_sp)": "float",
}

COHORT_HEADER_DATATYPE_PATTERNS = {
    # Cohort species, planting years & densities by cohort index
    r"^(base|proj)_species": "scalar integer",  # TODO: this is different from current naming "species_base" "species1" etc
    r"^(base|proj)_plant_yr": "scalar integer", # TODO: base doesn't currently have cohort-specific planting years, but may need to be added
    r"^(base|proj)_plant_dens": "scalar integer", # TODO: base doesn't currently have cohort-specific planting densities, but may need to be added

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
                    f"Header matching '{required_regex}' is required " # TODO: this prints the regex so not particularly specific or readable
                    f"because a header matching '{anchor_pattern}{i}$' is present"
                )
    return errors


def validate_all_grouped_headers(data):
    errors = []
    for l in [CROP_HEADER_DATATYPE_PATTERNS,
               SPECIES_HEADER_DATATYPE_PATTERNS,
               COHORT_HEADER_DATATYPE_PATTERNS,
               FERT_HEADER_DATATYPE_PATTERNS,
               LITTER_HEADER_DATATYPE_PATTERNS]:
                patterns = list(l.keys())
                errors.extend(validate_grouped_headers(list(data.keys()), anchor_pattern=patterns[0], required_patterns=patterns))
    return errors

def expand_single_row_data_input(file_path):
        # identify scalar headers through their type in data_handler
        # also treat any header beginning with 'species' as scalar

        row_input_data = read_and_validate_timeseries_by_header(file_path, permitted_vector_lengths=[1], target_vector_length=1)

        no_of_years = "yrs_proj"

        scalar_input_headers = [
            h
            for h in list(row_input_data.keys())
            if (
                (get_header_type(h) and get_header_type(h).startswith("scalar"))
                or re.match(r"^species(?:_base|\d*)$", h)
                or re.match(r"base_plant_(?:yr|dens*)", h) # TODO: remove "or" when headers updated
            )
        ]
        scalar_input_data = {}
        for h in scalar_input_headers:
            # pop (remove) the value from row_input_data; ignore missing keys
            val = row_input_data.pop(h, None)
            if val is not None:
                try:
                    scalar_input_data[h] = np.atleast_1d(np.asarray(val, dtype=float))
                except Exception:
                    # keep original if conversion fails
                    scalar_input_data[h] = val

        # --- tree size ---
        tree_size_data = {}
        age_base = ["age1", "age2", "age3", "age4", "age5", "age6"]
        diam_base = ["diam1", "diam2", "diam3", "diam4", "diam5", "diam6"]

        # determine which species are present in the scalar inputs
        # species headers can be 'species_base' or 'species1', 'species2', etc. # TODO: fix this when strings updated in test files, check if match N_COHORTS
        species_keys = [
            h
            for h in scalar_input_data.keys()
            if re.match(r"^species(?:_base|\d*)$", h)
        ]
        sp_list = []
        for k in species_keys:
            try:
                # scalar_input_data stores arrays, take first element
                arr = np.atleast_1d(scalar_input_data[k])
                sp_list.append(int(arr[0]))
            except Exception:
                # missing or non-numeric
                continue

        for spp_number in set(sp_list):
            if spp_number == 1:
                sp_index = ""
            else:
                sp_index = f"sp{spp_number}_"

            age_input = [f"{sp_index}{key}" for key in age_base]
            age_vals = [float(row_input_data.get(k, 0)) for k in age_input]
            age_arr = np.array(age_vals)
            age_arr = np.array(sorted(age_arr, key=int))

            diam_input = [f"{sp_index}{key}" for key in diam_base]
            diam_vals = [float(row_input_data.get(k, 0)) for k in diam_input]
            diam_arr = np.array(diam_vals)

            tree_size_data[f"age_sp{spp_number}"] = age_arr
            tree_size_data[f"diam_sp{spp_number}"] = diam_arr
        # --- crop data --- #
        crop_data = {}

        for index in set({1,2,3}):
            for prefix in set({"crop_base","crop_proj"}):
                start_year = int(row_input_data[f"{prefix}_start{index}"]) 
                end_year = int(row_input_data[f"{prefix}_end{index}"])
                harvest_yield = harv_frac = np.zeros(no_of_years)
                harvest_yield[start_year:end_year] = float(row_input_data[f"{prefix}_yd{index}"])
                harv_frac[start_year:end_year] = float(row_input_data[f"{prefix}_left{index}"])
                crop_data[f"{prefix}_yd{index}"] = harvest_yield
                crop_data[f"{prefix}_left{index}"] = harv_frac

        #---fertiliser---#
        fertiliser_data = {}

        for prefix in set({"base","proj"}):
            DMinput = np.zeros(no_of_years)
            nitrogen = np.zeros(no_of_years)
            interval = int(row_input_data[f"{prefix}_sf_int"])
            quantity = row_input_data[f"{prefix}_sf_qty"]
            n_content = row_input_data[f"{prefix}_sf_n"] # TODO: check between 0 and 1
            if interval > 0:
                DMinput[::interval] = quantity
                nitrogen[::interval] = n_content
            fertiliser_data[f"{prefix}_sf_qty"] = DMinput
            fertiliser_data[f"{prefix}_sf_n"] = nitrogen

        litter_data = {}

        for prefix in set({"base","proj"}):
            DMinput = np.zeros(no_of_years)
            interval = int(row_input_data[f"{prefix}_lit_int"])
            quantity = row_input_data[f"{prefix}_lit_qty"]
            if interval > 0:
                DMinput[::interval] = quantity
            litter_data[f"{prefix}_lit_qty"] = DMinput

        #--- fire---#
        fire_data = {}

        for suffix in set({"base","proj"}):
            fire_on = np.zeros(no_of_years)
            fire_off = (np.zeros(no_of_years) if int(row_input_data[f"fire_off_{suffix}"])==0 else np.ones(no_of_years) )
            interval = int(row_input_data[f"fire_int_{suffix}"])
            if interval > 0:
                fire_on[::interval] = 1
            fire_data[f"fire_on_{suffix}"] = fire_on
            fire_data[f"fire_off_{suffix}"] = fire_off

        #---cover---#
        cover_data = {}

        for prefix in set({"base","proj"}):
            cover_year = np.zeros(12)
            cover_year[int(row_input_data[f"{prefix}_cvr_mth_st"]) : 
                  int(row_input_data[f"{prefix}_cvr_mth_en"])] = int(row_input_data[f"{prefix}_cvr_pres"])
            cover = np.tile(cover_year, no_of_years)
            cover_data[f"{prefix}_cover"] = cover

        #-- tree mgmt---#
        tree_data = {}

        for mgmt in set({"base", "proj"}):
            thinning_array = np.zeros(no_of_years + 1)
            for i in range(1, 5):
                key_yr = f"thin_{mgmt}_yr{i}"
                key_pc = f"thin_{mgmt}_pc{i}"
                if key_yr in row_input_data and key_pc in row_input_data:
                    year = int(row_input_data[key_yr])
                    percent = float(row_input_data[key_pc]) # TODO: check 0-1
                    thinning_array[year] = percent
            tree_data[f"thin_{mgmt}_cohort"] = thinning_array # TODO: need to copy this to number of cohorts
            tree_data[f"mort_{mgmt}_cohort"] = np.repeat(row_input_data[f"{mgmt}_mort"],no_of_years+1)
            for pool in set({"br","st"}):
                for turnover in set({"thin","mort"}):
                    percent = float(row_input_data[f"{turnover}_{mgmt}_{pool}"])
                    tree_data[f"{turnover}_{mgmt}_{pool}_cohort"] = np.repeat(percent,no_of_years+1) # TODO: no_of_years + 1?
        mgmt_input_data = crop_data | fertiliser_data | litter_data | fire_data | tree_data

        return scalar_input_data, tree_size_data, mgmt_input_data, cover_data

