# Purpose: Algorithms used to estimate asteroid spectral continuum

######################
# Imported Libraries #
######################
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
import asteroidTaxonomy_Auxilary as ata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def nearest(arr, val):
    return np.argmin(abs(arr - val))


def continuumEstimation_1(x, y, target=None, plot=False):
    # Retrieve points closest to x1 and x2
    x1 = 2.4
    x2 = 3.75
    indicies = np.array([nearest(x, x1), nearest(x, x2)])
    x_selected = x[indicies]
    y_selected = y[indicies] 

    # Estimate coninuum using linear interpolation
    continuum = np.poly1d(np.polyfit(x_selected, y_selected, deg=1))

    # If plotting is enabled, do the following (Checking Results)
    if plot:
        continuumPlot(x, y, continuum=continuum, target=target)

    # Return continuum y data points
    return continuum


def continuumEstimation_2(x, y, target=None, plot=False):
    # Reference points
    x1 = 2.5
    x2 = 3.45
    
    # Retrieve segment of spectrum that is less than the wavelength x1
    index1 = nearest(x, x1)
    indicies = [i for i in range(index1+1)]
    x_seg1 = x[indicies]
    y_seg1 = y[indicies]
        
    # Retrieve the top n maximum spectral values located beyond the wavelength x2
    n = 10
    index2 = nearest(x, x2)
    indicies = [i for i in range(index2, len(x))]
    y_temp = y[indicies]
    y_temp = np.sort(y_temp)
    y_temp = y_temp[-n:]
    
    # Retrieve segment greater than the wavelength x2
    indicies = []
    for val in y_temp:
        index = np.where((x >= x2) & (y == val))[0][0]
        indicies.append(index)
    indicies.sort()
    x_seg2 = x[indicies]
    y_seg2 = y[indicies]
    
    # Append all segments
    x_seg = np.append(x_seg1, x_seg2)
    y_seg = np.append(y_seg1, y_seg2)
    
    # Linearly fit continuum using selected points less than x1 and greater than x2
    solution = np.polyfit(x_seg, y_seg, deg=1, cov=True)
    coefs = solution[0]
    cov = solution[1]
    errs = [cov[i][i] for i in range(len(coefs))]
    error = np.sqrt(np.sum(errs))
    print(solution)
    print(coefs)
    print(errs)
    print(error)
    continuum = np.poly1d(coefs)

    # If plotting is enabled, do the following (Checking Results)
    if plot:
        continuumPlot(x, y, error=error, continuum=continuum, target=target)
    exit()
    
    # Return continuum y data points
    return continuum


def continuumEstimation_3(x, y, ri=None, plot=False):
    # Determine number of restricted intervals
    nri = 0
    if ri is not None:
        nri = len(ri)

    xmin = x.min()
    index = 0
    maxima_x = ()
    maxima_y = ()
    sm = ()
    while index <= nri:
        # Determine maximum allowed x value
        if nri and (index < nri):
            xmax = ri[index][0]
        else:
            xmax = x.max()

        # Extract wavelength and reflectance values (partials)
        indicies = np.where((xmin <= x) & (x <= xmax))
        xp = x[indicies]
        yp = y[indicies]

        # Select smoothing polynomial degree
        sd = 8

        # Select a window length (ad hoc solution)
        wl = int(0.4 * len(xp))
        wl = wl if ((wl % 2) != 0) else wl - 1
        wl = np.amax([wl, sd if ((sd % 2) != 0) else (sd + 1)])

        # Smooth data on the extracted interval
        yp = signal.savgol_filter(yp, wl, sd)

        # Store smoothed function
        sm = np.append(sm, yp)

        # Retrieve average spacing between data points
        wbin = (xp[-1] - xp[0]) / len(xp)

        # Restriction
        thresh = wbin * 5

        # Fit spline to current segment of smoothed spectrum
        spl = InterpolatedUnivariateSpline(xp, yp, k=4)

        # Calculate 1st and 2nd derivative
        spl_prime = spl.derivative()
        spl_2 = spl_prime.derivative()

        # Find extrema
        extrema = spl_prime.roots()

        # Find maxima
        maxima = ()
        minima = ()
        for val in extrema:
            if (spl_2(val) < 0):
                maxima = np.append(maxima, val)
            else:
                minima = np.append(minima, val)
            if (len(minima) == 2):
                arg = maxima[-1]
                min_1 = minima[0]
                min_2 = minima[1]
                if (arg - thresh <= min_1) and (arg + thresh >= min_2):
                    maxima = np.delete(maxima, -1)
                minima = np.delete(minima, 0)

        # Add end points of interval as maxima and sort array
        maxima = np.append(maxima, xp[0])
        maxima = np.append(maxima, xp[-1])
        maxima = np.sort(maxima)

        # Store maxima points x values
        maxima_x = np.append(maxima_x, maxima)

        # Retrieve maxima points y values
        maxima_y = np.append(maxima_y, spl(maxima))

        # Update min x
        if nri and (index < nri):
            xmin = ri[index][1]

        # Update the index
        index += 1

    # Interpolate continuum using maxima points
    continuum = InterpolatedUnivariateSpline(maxima_x, maxima_y, k=1)

    # Interpolate smoothing function using stored smoothing points
    smoothed = InterpolatedUnivariateSpline(x, sm, k=1)

    # If plotting is enabled, do the following (Checking Results)
    if plot:
        continuumPlot(x, y, continuum, sm=smoothed, target=target)

    # Return continuum y data points
    return continuum


def continuumPlot(x, y, continuum, error=None, sm=None, target=None):
    # Target official name (if given)
    target = ata.asteroidOfficialName(target) if target is not None else "Data"
    
    # Plot original data and continuum fit
    plt.figure(figsize=(12, 6))
    plt.xlabel('Wavelength [$\\mu$m]')
    plt.ylabel('Normalized Reflectance')
    sns.scatterplot(x=x, y=y, label=target)
    if error is None:
        sns.scatterplot(x=x, y=continuum(x), label='Continuum Fit', color='blue')
    else:
        plt.errorbar(x=x, y=continuum(x), yerr=error, label='Continuum Fit', color='orange')
    if sm is not None:
        sns.scatterplot(x=x, y=sm(x), label='Smoothed Data')
    plt.legend()
    plt.show()
