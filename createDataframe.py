# Purpose: Wrapper function for producing Python dataframe
from fileExists import fileExists

import pandas as pd
import sys


def createDataframe(filename, header=None, colNames=[], delim_whitespace=True):

    # Verify existence of file
    if not(fileExists(filename)):
        print("Error in createDataframe:")
        print("File not found: \"%s\"" % (filename))
        sys.exit()

    # Read data into dataframe
    df = pd.read_csv(filename, header=header,
                     delim_whitespace=delim_whitespace, error_bad_lines=False)

    # Add column names
    if df.shape[1] == len(colNames):
        df.columns = colNames

    # Return dataframe
    return df
