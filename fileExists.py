# Purpose: Wrapper function that determines if a file exists

import os.path as arg


def fileExists(filename):
    if arg.isfile(filename):
        return True
    else:
        return False
