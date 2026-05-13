from typing import Dict, Any, Tuple, Union

import csv
import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler

# # ----------------------------------
# # Read species data from csv
# # (run when this module is imported)
# # ----------------------------------
SPP_LIST = [1, 2, 3]  # Abridged species list


def read_csv(filename: str, cols: Tuple[int, ...]) -> np.ndarray:
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        data = [[float(row[col]) for col in cols] for row in reader]
    return np.array(data)


def load_tree_species_data(
    filename: str = "tree_params.csv",
) -> Dict[Union[int, str], Dict]:
    data = csv_handler.read_csv(filename, cols=(2, 3, 4, 5, 6, 7, 8, 9))

    nitrogen = data[:, :5]
    carbon = data[:, 5]
    root_to_shoot = data[:, 6]
    wood_density = data[:, 7]

    return {
        spp: {
            "species": spp,
            "wood_dens": wood_density[i],
            "carbon": carbon[i],
            "nitrogen": nitrogen[i],
            "root_to_shoot": root_to_shoot[i],
        }
        for i, spp in enumerate(SPP_LIST)
    }

class TreeParamsData:
    """
    Object holding tree params.

    Instance variables
    ----------------
    species         tree species name
    wood_dens            tree density in g cm^-3
    carbon          tree carbon content as a fraction
    nitrogen        tree nitrogen content as a fraction
    root_to_shoot   tree root-to-shoot ratio
    """

    def __init__(
        self,
        species,
        wood_dens,
        nitrogen,
        carbon,
        root_to_shoot,
    ):
        self.species = species
        self.wood_dens = wood_dens
        self.nitrogen = nitrogen
        self.carbon = carbon
        self.root_to_shoot = root_to_shoot


def validate_species(value):
    # Determining whether the value can be interpreted as a string or an integer
    errors = [f"{value} must be convertible to a string."] * (
        not isinstance(str(value), str)
    ) + [f"{value} must be convertible to an integer."] * (
        not value.isdigit() if isinstance(value, str) else False
    )
    return errors


class TreeParamsSchema(Schema):
    species = fields.Raw(required=True, validate=lambda v: validate_species(v))
    wood_dens = fields.Float(required=True)
    carbon = fields.Float(required=True)
    nitrogen = fields.List(fields.Float, required=True)
    root_to_shoot = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeParamsData(**data)


def create(tree_params) -> TreeParamsData:
    """
    Create a TreeParams object from a dict.

    Args: tree_params: dict with tree params

    Returns: TreeParamsData object
    """
    params = {
        "species": tree_params["species"],
        "wood_dens": tree_params["wood_dens"],
        "carbon": tree_params["carbon"],
        "nitrogen": tree_params["nitrogen"],
        "root_to_shoot": tree_params["root_to_shoot"],
    }

    schema = TreeParamsSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree params: {errors}")

    return schema.load(params)  # type: ignore


def from_species_name(species: str):
    """
    Same as create, but with species name.
    """
    TREE_SPP = load_tree_species_data()
    return create(TREE_SPP[species])


def from_species_index(index: int):
    """
    Same as create, but with species index.
    """
    index = int(index)
    species = SPP_LIST[index - 1]
    TREE_SPP = load_tree_species_data()

    return create(TREE_SPP[species])


def from_csv(species_name: str, filename: str, row=0):
    """
    Construct Tree object using data from a csv which
    is structured like the master csv (tree_defaults.csv)

    Args:
        species_name: name of species (can be anything)
        filename: csv file with the info
        row: row in the csv to be read (0-indexed)
    Returns:
        TreeParamsData
    """

    data = csv_handler.read_csv(filename, cols=(2, 3, 4, 5, 6, 7, 8, 9))
    data = np.atleast_2d(data)  # to account for if there's only one row

    params = {
        "species": species_name,
        "nitrogen": np.array(
            [
                data[row, 0],
                data[row, 1],
                data[row, 2],
                data[row, 3],
                data[row, 4],
            ]
        ),
        "carbon": data[row, 5],
        "root_to_shoot": data[row, 6],
        "wood_dens": data[row, 7],
    }
    return create(params)


def save(tree_params: TreeParamsData, file="tree_params.csv"):
    """Save tree params to a csv.
    Default path is in OUTPUT_DIR with filename 'tree_params.csv'

    Args:
        file: name or path to csv file

    """
    # index is 0 if not in the SPP_LIST
    if tree_params.species in SPP_LIST:
        index = SPP_LIST.index(tree_params.species) + 1
    else:
        index = 0

    data = [
        index,
        tree_params.species,
        tree_params.nitrogen[0],
        tree_params.nitrogen[1],
        tree_params.nitrogen[2],
        tree_params.nitrogen[3],
        tree_params.nitrogen[4],
        tree_params.carbon,
        tree_params.root_to_shoot,
        tree_params.wood_dens,
    ]
    cols = [
        "Sc",
        "Name",
        "N_leaf",
        "N_branch",
        "N_stem",
        "N_croot",
        "N_froot",
        "C",
        "rw",
        "wood_dens",
    ]
    csv_handler.print_csv(file, data, col_names=cols)


def create_tree_params_from_species_index(
    csv_input_data: Dict[str, Any], cohort_count: int
):
    tree_params = []

    for i in range(cohort_count):
        key = f"species{i + 1}"
        try:
            species_index = int(csv_input_data[key].item())
            tree_params.append(from_species_index(species_index))
        except KeyError:
            raise KeyError(f"Warning: Missing key '{key}' in input data.")
    return tree_params
