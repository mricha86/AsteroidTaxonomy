# Purpose: Plot all asteroid spectra for a single target

import argparse as ap
import createDataframe as cdf
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import seaborn as sns
import sys
sns.set()


def main(argv):

    # Retrieve arguments
    parser = ap.ArgumentParser()
    parser.add_argument("asteroidSpectraList",
                        help="File containing spectral data")
    parser.add_argument("target",
                        help="Name of asteroid to build merged spectrum")
    args = parser.parse_args()
    asteroidSpectraList = args.asteroidSpectraList
    target = args.target

    # Create dataframe for each file in list that matches target asteroid
    data = []
    targetNames = []
    inputFile = open(asteroidSpectraList)
    for spectrum in inputFile:
        if target in spectrum:
            data.append(cdf.createDataframe(spectrum.rstrip()))
            targetNames.append(op.basename(spectrum))

    # Extract wavelength and reflectance values
    x_values = []
    y_values = []
    for i in range(len(data)):
        x_values.append(
            data[i].loc[:, 0].to_numpy())  # Wavelength Unit: micrometer
        y_values.append(
            data[i].loc[:, 1].to_numpy())  # Reflectance Unit: unitless

    # Remove anomalous points
    for i in range(len(data)):
        pos = np.where(y_values[i] <= 0.8)
        x_values[i] = np.delete(x_values[i], pos, 0)
        y_values[i] = np.delete(y_values[i], pos, 0)

    # Plot spectra
    for i in range(len(data)):
        plt.plot(x_values[i], y_values[i], '.', label=targetNames[i])
    plt.legend()
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength")
    plt.title(target+" Spectra")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
