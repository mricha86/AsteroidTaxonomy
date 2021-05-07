# Purpose: Algorithms used to calculate polynomial
# characteristics of bands in asteroid spectra

# Imported modules/functions
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
import asteroidTaxonomy_Auxilary as ata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def polynomialConversion(x, y, deg):
    # Fit polynomial of specified degree to data
    pc = np.polyfit(x, y, deg)  # Polynomial coefficients
    p = np.poly1d(pc)

    return p


def polynomialExtrema(x, y, deg, minima=False, maxima=False, y_vals=False):
    # Retrieve minimum and maximum x values
    xmin = x.min()
    xmax = x.max()

    # Fit polynomial of specified degree to data
    p = polynomialConversion(x, y, deg)

    # Calculate first derivative of the polynomial
    p_prime = np.polyder(p)

    # Calculate extrema of polynomial function
    extrema = p_prime.roots
    extrema = extrema.real[abs(extrema.imag) < 1e-5]

    # If minima or maxima is true, return minima or maxima
    # respectively
    mins = np.empty(0)
    maxs = np.empty(0)
    if (minima != maxima):
        # Calculate second derivative of the polynomial
        p_2 = np.polyder(p_prime)

        # Iterate through extrema
        for val in extrema:
            if minima and (p_2(val) > 0):  # If minima is wanted
                mins = np.append(mins, val)
            if maxima and (p_2(val) < 0):  # If maxima is wanted
                maxs = np.append(maxs, val)

        # Check to see if left endpoint is a minimum
        # (Applying special considerations here!)
        if minima:
            a = list(mins < 2.95)
            b = any(a)
            a = list(p(xmin) < p(extrema))
            c = any(a)
            if b or c:
                mins = mins[mins > 2.95]
                mins = np.append(mins, xmin)
        extrema = mins if minima else maxs
        extrema.sort()

    # Remove all extrema values outside of wanted interval
    indicies = np.where((extrema >= xmin) & (extrema <= xmax))
    extrema = extrema[indicies]

    # If extrema y values are wanted, return extrema points
    if y_vals:
        # Calculate corresponding y values for each extrema x value
        extrema_y_vals = p(extrema)

        # Create extrema points
        extrema = (extrema, extrema_y_vals)

    return extrema


def polynomialFitPlot(x_o, y_o, x_p, y_p, deg, continuum=None, target=None):
    # Target official name (if given)
    target = ata.asteroidOfficialName(target) if target is not None else "Data"
    
    # Fit polynomial of specified degree to data
    p = polynomialConversion(x_p, y_p, deg)

    # Plot original data and polynomial fit
    plt.figure(figsize=(12, 6))
    # plt.title('Polynomial Fit')
    plt.xlabel('Wavelength [$\\mu$m]')
    plt.ylabel('Normalized Reflectance')
    sns.scatterplot(x=x_o, y=y_o, label=target)
    sns.scatterplot(x=x_p, y=p(x_p), label='Polynomial Fit')
    if continuum is not None:
        sns.scatterplot(x=x_o, y=continuum(x_o), label='Continuum Estimate')
    plt.legend()
    plt.show()


def polynomialFitResults(x, y, continuum=None, target=None, correction=False, plot=False):
    # Note: Wavelength interval [2.9, 3.4], polynomial degree 6, 
    # rounding results to 2 decimal places, and outlier removal 
    # on aforementioned interval appears to yield ideal results 
    # for region 2
    
    # Constants
    xmin = 2.9
    xmax = 3.4
    deg = 6
    roundtoplace = 2
    lof_rejection = -1.00

    # Extract data on the interval [xmin, xmax]
    indicies = np.where((xmin <= x) & (x <= xmax))
    xp = x[indicies]
    yp = y[indicies]

    # Determine if we are using the original 
    # data on the above extracted interval 
    # or a smoothed version of the data
    if correction:
        # Perform local outlier removal
        y_temp = [np.array([i,j]) for i, j in zip(xp, yp)]
        # y_temp = np.array([yp])
        # y_temp = y_temp.reshape(-1, 1)
        clf = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
        y_pred = clf.fit_predict(y_temp)
        X_scores = clf.negative_outlier_factor_
        indicies = np.where(X_scores >= lof_rejection)[0]
        if indicies[0] != 0:
            indicies = np.insert(indicies, 0, 0)
        if indicies[-1] != len(X_scores)-1:
            indicies = np.append(indicies, len(X_scores)-1)
        x_temp = xp[indicies]
        y_temp = yp[indicies]
        
        # Check results of outlier identification scheme
        # for i, j, k in zip(xp, yp, X_scores):
        #    print("({0},{1}): {2}".format(i, j, k))
        # plt.title("Local Outlier Factor (LOF)")
        # plt.scatter(yd[:, 0], yd[:, 1], color='k', s=3., label='Data points')
        # plt.scatter(xp, yd, color='k', s=3., label='Data points')
        # plot circles with radius proportional to the outlier scores
        # radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        # plt.scatter(yd[:, 0], yd[:, 1], s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')
        # plt.scatter(xp, yd, s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')
        # plt.axis('tight')
        # plt.xlim((2.85, 3.35))
        # plt.ylim((0.75, 1.15))
        # plt.xlabel("prediction errors: %d" % (n_errors))
        # legend = plt.legend(loc='upper left')
        # legend.legendHandles[0]._sizes = [10]
        # legend.legendHandles[1]._sizes = [20]
        # plt.show()
        # exit(0)
        
        # Perform data smoothing
        # wl = 17
        # po = 3
        # y_smooth = savgol_filter(y_temp, window_length=wl, polyorder=po)
        
        # Apply spline to recover missing data points
        # spl = InterpolatedUnivariateSpline(x_temp, y_smooth, k=3)
        spl = InterpolatedUnivariateSpline(x_temp, y_temp, k=1)
        
        # Check results of smoothing
        if False:
            sns.scatterplot(xp, yp, label="All Data")
            sns.scatterplot(x_temp, y_temp, label="Retained Data")
            sns.scatterplot(xp, spl(xp), label="Spline Estimation")
            plt.show() 
        
        # Retrieve corrected data set
        yp = spl(xp)
    
    # Calculate band center(s) and corresponding depths
    band_center_pts = polynomialExtrema(xp, yp, deg, minima=True, y_vals=True)
    band_centers = band_center_pts[0]
    if continuum is None:
        band_depths = 1.0 - band_center_pts[1]
    else:
        if isinstance(type(continuum), float):
            band_depths = continuum - band_center_pts[1]
        else:
            band_depths = np.empty(0)
            for v1, v2 in zip(band_center_pts[0], band_center_pts[1]):
                band_depths = np.append(band_depths,
                                        continuum(v1) - v2)

    # Delete depths where continuum is below or at "minimum"
    indicies = np.where(band_depths <= 0)
    band_centers[indicies] = float('inf')
    band_depths[indicies] = float('inf')

    # If plotting is enabled, do the following (Checking Results)
    if plot:
        print(np.around(band_centers, decimals=roundtoplace), np.around(band_depths, decimals=roundtoplace))
        polynomialFitPlot(x, y, xp, yp, deg, continuum=continuum, target=target)
    
    # Store, sort, and round results
    results = np.array([np.around(band_centers, decimals=roundtoplace), np.around(band_depths, decimals=roundtoplace)])
    indicies = np.argsort(results[0])
    results = [results[0][indicies], results[1][indicies]]

    return results
