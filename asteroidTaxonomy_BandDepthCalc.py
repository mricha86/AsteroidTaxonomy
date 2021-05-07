# Purpose: Algorithm that calculates band depth at a given wavelength

######################
# Imported Libraries #
######################
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
import asteroidTaxonomy_Auxilary as ata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def bandDepthCalc(x, y, ref_x, continuum, target=None, plot=False):
    # Constants
    lof_rejection = -1.0
    
    # Extract data at wavelengths longer than reference wavelength 
    # (include reference wavelength as an endpoint)
    r = 0.3
    indicies = np.where((ref_x <= x) & (x <= ref_x+r))
    x_seg = x[indicies]
    y_seg = y[indicies]
    
    # Perform local outlier removal
    y_temp = [np.array([i,j]) for i, j in zip(x_seg, y_seg)]
    clf = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
    y_pred = clf.fit_predict(y_temp)
    X_scores = clf.negative_outlier_factor_
    indicies = np.where(X_scores >= lof_rejection)[0]
    if indicies[0] != 0:
        indicies = np.insert(indicies, 0, 0)
    x_temp = x_seg[indicies]
    y_temp = y_seg[indicies]
    
    # Check results of outlier identification scheme
    check = False
    if check:
        for i, j, k in zip(x_seg, y_seg, X_scores):
            print("({0},{1}): {2}".format(i, j, k))
       
        # Plot circles with radius proportional to the outlier scores
        plt.title("Local Outlier Factor (LOF)")
        plt.scatter(x_seg, y_seg, color='k', s=3., label='Data points')
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(x_seg, y_seg, s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')
        plt.axis('tight')
        plt.xlim((2.85, 3.35))
        plt.ylim((0.75, 1.15))
        legend = plt.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        plt.show()
        exit(0)
    
    band_depth = -1
    while band_depth < 0:
        # Apply spline to recover missing data points
        spl = InterpolatedUnivariateSpline(x_temp, y_temp, k=1)
        
        # Perform data smoothing
        wl = 9
        po = 3
        y_smooth = savgol_filter(spl(x_seg), window_length=wl, polyorder=po)
    
        # Apply local polynomial fit on data
        p = np.poly1d(np.polyfit(x_seg, y_smooth, deg=6))
        
        # Determine y value at reference x value
        ref_y = p(ref_x)
        
        # Sanity check
        # sns.scatterplot(x_seg, y_seg, label="Original Data")
        # sns.scatterplot(x_seg, spl(x_seg), label="Interpolated Data")
        # sns.scatterplot(x_seg, y_smooth, label="Smoothed Data")
        # sns.scatterplot(x_seg, p(x_seg), label="Polynomial Estimated Data")
        # plt.show()
        
        # Determine corresponding y value at continuum
        con_y = continuum(ref_x)

        # Calculate band depth
        band_depth = np.fabs(con_y - ref_y)/con_y
        
        # Make correction to retained data points
        if (X_scores[0] < lof_rejection) and (indicies[0] == 0):
            indicies = np.delete(indicies, 0)
        
        # Secondary condition to break from loop
        if len(x_seg) == len(x_temp):
            break
        
        # Lessen constraint on LOF rejection value
        lof_rejection = lof_rejection - 0.05
        
        # Recalculate allowed indicies
        indicies = np.where(X_scores >= lof_rejection)[0]
        x_temp = x_seg[indicies]
        y_temp = y_seg[indicies]  
    
    # If plotting is enabled, do the following (Checking Results)
    if plot:
        print(ref_x, band_depth)
        bandDepthPlot(x, y, ref_x, ref_y, continuum, target=target)

    # Return depth at reference x
    return band_depth


def bandDepthPlot(x, y, ref_x, ref_y, continuum, target=None):
    # Target official name (if given)
    target = ata.asteroidOfficialName(target) if target is not None else "Data"
    
    # Plot original data, continuum fit, and location of band
    plt.figure(figsize=(12, 6))
    # plt.title('Band Depth')
    plt.xlabel('Wavelength [$\\mu$m]')
    plt.ylabel('Normalized Reflectance')
    sns.scatterplot(x=x, y=y, label=target)
    sns.scatterplot(x=x, y=continuum(x), label='Continuum Fit')
    bottom, top = plt.ylim()
    range = top - bottom
    ymin = (ref_y - bottom)/range
    ymax = (continuum(ref_x) - bottom)/range
    plt.axvline(x=ref_x,
                ymin=ymin,
                ymax=ymax,
                linestyle='--',
                label='Band Depth')
    plt.legend()
    plt.show()


def bandDepthResults(x, y, ref_x_vals, continuum, target=None, plot=False):
    # Constants 
    roundtoplace = 3
    
    # Iterate through band center values to determine
    # band depth values
    band_depths = np.empty(0)
    for ref_x in ref_x_vals:
        band_depths = np.append(
            band_depths, np.around(bandDepthCalc(x, y, ref_x, continuum, target=target, plot=plot), decimals=roundtoplace))

    return band_depths


def nearest(arr, val):
    return np.argmin(abs(arr - val))
