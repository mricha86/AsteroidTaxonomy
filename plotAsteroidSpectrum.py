# Purpose: Plot single asteroid spectrum

#!/bin/local/python

import argparse as ap
import createDataframe as cdf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sns.set()


def SimpleScatterPlot(x,
                      y,
                      label='',
                      marker='o',
                      title='',
                      xlabel='',
                      ylabel='',
                      xrange=None,
                      yrange=None):
    # Create figure
    fig = plt.figure()

    # Plot data
    if len(x.shape) == 1:
        # Plot data
        plt.scatter(x, y, marker=marker, s=1, label=label)

    if len(x.shape) == 2:
        n = x.shape[0]
        for i in range(n):
            # Plot data
            plt.scatter(x[i], y[i], marker=marker, s=1, label=label[i])

    # X-axis title
    plt.xlabel(xlabel)

    # Y-axis title
    plt.ylabel(ylabel)

    # Title of graph
    plt.title(title)

    # Legend
    # plt.legend()

    # Setting x and y axis range
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])
    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])

    return fig


def main(argv):

    # Retrieve arguments
    parser = ap.ArgumentParser()
    parser.add_argument("asteroidSpectrum",
                        help="File containing spectral data")
    parser.add_argument('--title', default='Astroid Spectrum', 
                        help="Plot title.")
    args = parser.parse_args()
    asteroidSpectrum = args.asteroidSpectrum
    title = args.title

    # Create dataframe for file
    data = cdf.createDataframe(asteroidSpectrum)

    # Extract wavelength and reflectance values
    x = data.loc[:, 0].to_numpy()  # Wavelengths. Unit: micrometer
    y = data.loc[:, 1].to_numpy()  # Reflectance. Unit: unitless

    # Remove points that are less than 2
    pos = np.where(x < 2)
    x = np.delete(x, pos, 0)
    y = np.delete(y, pos, 0)

    # Create plot of spectrum
    SimpleScatterPlot(x,
                      y,
                      title=title,
                      xlabel=r'Wavelength [$\mu$m]',
                      ylabel='Reflectance')

    # Show plot of spectrum
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
