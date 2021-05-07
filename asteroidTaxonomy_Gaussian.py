# Purpose: Algorithms used to calculate Gaussian
# characteristics of bands in asteroid spectra

# Imported modules/functions
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def gaus(x, a, mean, sigma, offset):
    arg = ((x - mean) / sigma)**2
    return a * exp(-arg / 2) + offset


def gaussianFitPlot(x_o, y_o, x_g, params, continuum):
    # Continuum estimate y values
    y_continuum = continuum(x_o)

    # Spectrum with continuum removed
    y_continuum_removed = y_o / y_continuum

    # Gaussian fit to continuum removed data
    gfit_continuum_removed = gaus(x_g, *params)

    # Gaussian fit to original data
    gfit = gaus(x_g, *params) * continuum(x_g)

    # Plot original data and polynomial fit
    plt.figure(figsize=(12, 6))
    # plt.title('Gaussian Fit')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    sns.scatterplot(x=x_o, y=y_o, label='Original Data')
    sns.scatterplot(x=x_o,
                    y=y_continuum_removed,
                    label='Original Data w/o Continuum')
    sns.scatterplot(x=x_g, y=gfit, label='Gaussian Fit /w Continuum')
    sns.scatterplot(x=x_g,
                    y=gfit_continuum_removed,
                    label='Gaussian Fit w/o Continuum')
    sns.scatterplot(x=x_o, y=y_continuum, label='Continuum Estimate')
    plt.legend()
    plt.show()


def gaussianFitResults(x, y, continuum, plot=False):
    # Constants
    xmin = 2.48
    xmax = 3.4

    # Calculate y values for continuum
    y_continuum = continuum(x)

    # Calculate spectrum with continuum removed
    y_continuum_removed = y / y_continuum

    # Extract data on the interval [xmin, xmax]
    indicies = np.where((xmin <= x) & (x <= xmax))
    xg = ar(x[indicies])
    yg = ar(y_continuum_removed[indicies])

    # Gaussian parameter initial estimates
    n = sum(yg)
    mean = sum(xg * yg) / n
    sigma = math.sqrt(sum(yg * (xg - mean)**2) / n)
    params = [1, mean, sigma, 0]  # [Amp, mean, sd, offset]

    # Fit gaussian to modified spectrum on the interval [xmin, xmax]
    good_data = True
    try:
        popt, pcov = curve_fit(gaus, xg, yg, p0=params)
    except RuntimeError:
        # print("Error - curve_fit failed")
        good_data = False
        popt = [0, 0, 0, 0]

    # Calculate band center and corresponding depth
    if good_data and (popt[0] < 0) and (popt[0] > -1.0) and (
            pcov[0][0] < 4.0) and (pcov[1][1] < 4.0):
        band_center = popt[1]
        band_center_err = math.sqrt(pcov[1][1])
        band_depth = 1.0 - (popt[3] + popt[0])
        band_depth_err = math.sqrt(pcov[0][0])
    else:
        band_center = float('inf')
        band_center_err = float('inf')
        band_depth = float('inf')
        band_depth_err = float('inf')

    # If plotting is enabled, do the following (Checking Results)
    if plot and good_data:
        print(band_center, band_depth)
        gaussianFitPlot(x, y, xg, popt, continuum)

    # Store and round results
    results = np.array(
        [band_center, band_center_err, band_depth, band_depth_err])
    #results = np.around(results, decimals=2)

    return results
