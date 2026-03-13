#!/usr/bin/python

"""Module for csv related functions in the SHAMBA program."""

import csv
import logging as log
import os

import numpy as np

from model import configuration


class FileOpenError(Exception):
    """
    Exception to be called when a file can't be opened.
    Prints error message and closes program with exit code 2.

    """

    def __init__(self, filename):
        """Initialise FileOpenError.

        Args:
            filename: name of file which couldn't be opened

        """
        super(FileOpenError, self).__init__()
        log.exception("Could not open %s" % filename)


def print_csv(
    file_out,
    array,
    col_names=[],
    print_years=False,
    print_total=False,
    print_column=False,
):
    """Custom method for printing an array or list to a csv file.
    Uses numpy.savetxt.

    Args
        array: array to be printed
        file_out: where to print (put in output unless path specified)
        col_names: list of column names to put at top of csv
        print_years: whether to print the years (index of array) in
                     left-most column
        print_total: whether to print the total of each row in last column
        print_column: whether to print a 1d array to a column instead of row
    """

    def round_(x):
        try:
            x = "%.5f" % x
        except TypeError:
            pass
        return x

    # See if existing path was given, put file in OUTPUT_DIR if not
    if not os.path.isdir(os.path.dirname(file_out)):
        file_out = os.path.join(configuration.OUTPUT_DIR, file_out)

    # See if the array given is actually a list (because it has strings)
    if isinstance(array, list):
        is_list = True

        # WARNIN - broken if data is 3d (list is doubly nested)
        # but not sure how you'd print that to csv anyway
        if not isinstance(array[0], list):
            array = [array, []]  # to make sure it's at least 2d
    else:
        is_list = False

    if print_total and not is_list:
        # Add total as last column
        total = np.sum(array, axis=1)
        col_names.append("total")
        array = np.column_stack((array, total))

    if print_years and not is_list:
        # Add years as first column
        years = np.array(list(range(array.shape[0])))
        col_names.insert(0, "year")
        array = np.column_stack((years, array))

    # manually do header since numpy 1.6 doesn't
    # support header argument to savetxt - FFS
    try:
        with open(file_out, "w") as outcsv:
            writer = csv.writer(outcsv, lineterminator="\n")
            writer.writerow(col_names)
            if is_list:
                for row in array:
                    if row:
                        writer.writerow([round_(x) for x in row])
            else:
                if array.ndim == 1 and print_column:
                    np.savetxt(
                        outcsv, np.atleast_2d(array).T, delimiter=",", fmt="%.5f"
                    )
                else:
                    # 2d
                    np.savetxt(outcsv, np.atleast_2d(array), delimiter=",", fmt="%.5f")

    except IOError:
        log.exception("Cannot print to file %s", file_out)


def read_csv(file_in, cols=None):
    """Read data from a .csv file. Usees numpy.loadtxt.

    Args:
        file_in: name of file to read
        cols: tuple of columns to read (read all if cols==None)
    Returns:
        array: numpy array with data from file_in
    Raises:
        IOError: if file can't be found/opened

    """

    # if full path not specified, search through the files in the
    # project data folder /input, then in the 'defaults' folder
    default_path = os.path.join(configuration.BASE_PATH, "default_input")
    filename, extension = file_in.rsplit(".", maxsplit=1)
    default_file_in = "_".join((filename, "defaults.csv"))

    if not os.path.isfile(file_in):
        if os.path.isfile(os.path.join(configuration.INPUT_DIR, file_in)):
            file_in = os.path.join(configuration.INPUT_DIR, file_in)
        elif os.path.isfile(os.path.join(default_path, default_file_in)):
            file_in = os.path.join(default_path, default_file_in)
        else:
            # not in either folder, and not in full path
            raise FileOpenError(file_in)

    array = np.genfromtxt(
        file_in, skip_header=1, usecols=cols, comments="#", delimiter=","
    )

    return array


def read_mixed_csv(file_in, cols=None, types=None):
    """Read data from a mixed csv (strings and numbers).
    Uses numpy.loadfromtxt

    NOTE: when used to read HWSD stuff, make sure to use usecols=(0,1,2,3..12)
    since genfromtxt seems to think there's 15 cols and gives an error

    Args:
        file_in: name of file to be read
        cols: tuple of columns to read (read all if cols==None)
        types: tuple of the expected types (e.g. int, float, "|S25", etc.)
    Returns:
        array: ndarray of data from file_in
    Raises:
        IOError: if file can't be read/opened or if types don't work

    """

    try:
        if not os.path.isfile(file_in):
            if os.path.isfile(os.path.join(configuration.INPUT_DIR, file_in)):
                file_in = os.path.join(configuration.INPUT_DIR, file_in)
            else:
                # not in either folder, and not in full path
                raise IOError

        array = np.genfromtxt(
            file_in, usecols=cols, dtype=types, delimiter=",", skip_header=1
        )
    except IOError:
        raise FileOpenError(file_in)

    return array


def get_csv_input_data(n, file_path):
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    keys = data[0]
    values = data[n + 1]

    # Create dictionary directly using zip
    return dict(zip(keys, values))
