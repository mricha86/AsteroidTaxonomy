# Purpose: Auxilary function for asteroid taxonomy

# Imported modules/functions
from scipy.interpolate import InterpolatedUnivariateSpline
import asteroidTaxonomy_BandDepthCalc as atb
import asteroidTaxonomy_Continuum as atc
import asteroidTaxonomy_Gaussian as atg
import asteroidTaxonomy_Polynomial as atp
import createDataframe as cdf
import numpy as np
import os.path as op
import pandas as pd
import re


def asteroidName(str):
    # Extract base name
    basename = op.basename(str)
    
    # Extract asteroid name
    if "reduced" in basename:
        targetName = extractField('(\w)*\_', basename)
        targetName = targetName[:-1]
    else:
        targetName = extractField('\_(\w)*', basename)
        targetName = targetName[1:]    
    
    # Remove anomalous characters
    targetName = re.sub('[0-9][a-z]$', '',
                        targetName)  # Remove a digit and a character
    targetName = re.sub('[0-9]', '', targetName)  # Remove all digits
    targetName = re.sub('_all', '', targetName)  # Remove substring '_all'
    targetName = re.sub('comb', '', targetName)  # Remove substring 'comb'
    targetName = re.sub('_interp', '', targetName)  # Remove substring '_interp'
    targetName = re.sub('most', '', targetName)  # Remove substring 'most'
    targetName = re.sub('old$', '', targetName)  # Remove substring 'old'
    targetName = re.sub('part', '', targetName)  # Remove substring 'part'
    targetName = re.sub('_resam', '', targetName)  # Remove substring 'resam'
    
    # Return corrected string
    return targetName


def colorDictionary(clusterlabels, n, largestCluster=False):

    cDict = {}
    fig, ax = plt.subplots()
    
    if largestCluster:
        r = [i for i in range(4)]
        temp = ax.scatter(x=r, y=r, c=r, cmap=plt.get_cmap('cool'));
        lgCluster = {}
        keys = []
        values = []
        clusterlabels = list(clusterlabels)
        for cl in clusterlabels:
            if cl not in keys:
                keys.append(cl)
                values.append(clusterlabels.count(cl))
            if len(keys) == n:
                break
        keys = np.array(keys)
        values = np.array(values)
        indicies = np.argsort(keys)
        keys = keys[indicies]
        values = values[indicies]
        indicies = np.argsort(values)
        keys = keys[indicies]
        values = values[indicies]
        keys = keys[::-1]
        values = values[::-1]
        for i, key in enumerate(keys):
            values[i] = i
            lgCluster[key] = values[i]
        for cl in clusterlabels:
            cDict[cl] = temp.to_rgba(lgCluster[cl])
    else:
        temp = ax.scatter(x=clusterlabels, y=clusterlabels, c=clusterlabels, cmap=plt.get_cmap('cool'));
        for cl in clusterlabels:
            cDict[cl] = temp.to_rgba(cl)
            
    return cDict


def dataProcessing(data, restriction_intervals=None, outlier_removal=False):
    # Constants
    xmin = 2.25
    # xmax = 3.6
    normalizationWavelength = 2.3

    # Sort data in ascending order based on x wavelength
    data = data.sort_values(0)
    
    # Delete rows where wavelength or reflectance values
    # are less than or equal to 0
    indicies = data.loc[(data[0] <= 0) | (data[1] <= 0)].index
    data = data.drop(indicies, axis='index')
    
    # Remove data points on restricted wavelength interval(s)
    if restriction_intervals is not None:
        for p in restriction_intervals:
            indicies = data.loc[(data[0] >= p[0]) & (data[0] <= p[1])].index
            data = data.drop(indicies, axis='index')

    # Remove data points below minimum wavelength
    indicies = data.loc[(data[0] < xmin)].index
    data = data.drop(indicies, axis='index')
    
    # Remove outliers (Sigma rejection)
    z = 3.5
    if outlier_removal:
        mean = data[1].describe()[1]
        std = data[1].describe()[2]
        indicies = data.loc[(data[1] < mean-z*std) | (data[1] > mean+z*std)].index
        data = data.drop(indicies, axis='index')    
    
    # Rescale data to a value of 1 at wavelength of 2.3 (Normalization)
    index = data[0].map(lambda x: np.abs(x - normalizationWavelength)).idxmin()
    normalizationFactor = data[1].loc[index]
    data[1] = data[1] / normalizationFactor

    # Determine if data encompasses the required wavelength domain
    # If not, exclude from analysis
    # if (data[0].min() > xmin) or (data[0].max() < xmax):
    #    data = None

    return data


def extractField(pat, str):
    field = re.search(pat, str)
    return field.group(0) if field else '&'


def observationDate(str):
    # Constants
    monDict = {
        'jan': '01',
        'feb': '02',
        'mar': '03',
        'apr': '04',
        'may': '05',
        'jun': '06',
        'jul': '07',
        'aug': '08',
        'sep': '09',
        'oct': '10',
        'nov': '11',
        'dec': '12'
    }

    # Extract base name
    basename = op.basename(str)

    # Preprocessed data doesn't have date info in filename
    if 'reduced' in basename:
        return '2000-01-01'

    # Extract asteroid observation date
    obsdate = extractField('[0-9]*[a-zA-Z]*[0-9]*', basename)
    year = re.sub('^[0-9]*[a-zA-Z]*', '', obsdate)
    year = "20" + year if (int(year) < 90) else "19" + year
    month = re.sub('[0-9]', '', obsdate)
    month = monDict[month]
    day = re.sub('[a-zA-Z]*[0-9]*$', '', obsdate)
    obsdate = year + '-' + month + '-' + day  # Reformat date string

    # Return string
    return obsdate


def retrieveDatum(filename, plot=False): 
    # Constants
    ri = ((2.5, 2.89), )  # Restricted interval(s)
    region_1 = np.array([2.95, 3.0])
    region_2 = np.array([3.0, 3.25])
    region_3 = np.array([3.25, 3.5])

    # Retrieve asteroid name
    target = asteroidName(filename)

    # Retrieve asteroid observation date
    obsdate = observationDate(filename)

    # If plotting is enabled
    if plot:
        print("Target: {0} - Observation Date: {1}".format(target, obsdate))
        #if target != "patientia":
        #    plot = False
    
    # Retrieve data from file and store in Pandas DataFrame
    # colNames = ["Wavelength", "Reflectance", "Reflectance_Error", "Flag"]
    data = cdf.createDataframe(filename)

    # Data processing
    data = dataProcessing(data, restriction_intervals=ri, outlier_removal=True)

    # Extract wavelength and reflectance values (full)
    x = data.loc[:, 0].to_numpy()  # Wavelengths. Unit: micrometer
    y = data.loc[:, 1].to_numpy()  # Reflectance. Unit: unitless

    # Estimate spectral continuum
    # continuum = atc.continuumEstimation_1(x, y, target=target, plot=plot)
    continuum = atc.continuumEstimation_2(x, y, target=target, plot=plot)
    # continuum = atc.continuumEstimation_3(x, y, ri=ri, plot=plot)

    # Determine band centers and depths via Polynomial Fitting
    polynomial_results = atp.polynomialFitResults(x,
                                                  y,
                                                  correction=True,
                                                  continuum=continuum,
                                                  target=target,
                                                  plot=plot)

    # Determine band centers and depths via Gaussian Fitting
    gaussian_results = atg.gaussianFitResults(x, y, continuum, plot=plot)

    # Determine band depths at specified wavelengths
    ref_x_vals = np.array([2.9, 3.2])
    band_depth_results = atb.bandDepthResults(x,
                                              y,
                                              ref_x_vals,
                                              continuum,
                                              target=target,
                                              plot=plot)

    # Store data in array
    row_data = []
    row_data.append(obsdate)
    row_data.append(target)
    index = np.where((region_1[0] <= polynomial_results[0])
                     & (polynomial_results[0] < region_1[1]))
    if len(index[0]) > 0:
        row_data.append(polynomial_results[0][index][0])
        row_data.append(polynomial_results[1][index][0])
    else:
        row_data.append(np.NaN)
        row_data.append(np.NaN)
    index = np.where((region_2[0] <= polynomial_results[0])
                     & (polynomial_results[0] < region_2[1]))
    if len(index[0]) == 0:
        index = np.where(polynomial_results[0] <= 2.95)
    if len(index[0]) > 0:
        row_data.append(polynomial_results[0][index][0])
        row_data.append(polynomial_results[1][index][0])
    else:
        row_data.append(np.NaN)
        row_data.append(np.NaN)
    index = np.where((region_3[0] <= polynomial_results[0])
                     & (polynomial_results[0] < region_3[1]))
    if len(index[0]) > 0:
        row_data.append(polynomial_results[0][index][0])
        row_data.append(polynomial_results[1][index][0])
    else:
        row_data.append(np.NaN)
        row_data.append(np.NaN)
    index = np.where((region_2[0] <= gaussian_results[0])
                     & (gaussian_results[0] < region_2[1]))
    if len(index[0]) > 0:
        row_data.append(gaussian_results[0])
        row_data.append(gaussian_results[1])
        row_data.append(gaussian_results[2])
        row_data.append(gaussian_results[3])
    else:
        row_data.append(np.NaN)
        row_data.append(np.NaN)
        row_data.append(np.NaN)
        row_data.append(np.NaN)
    if len(band_depth_results) > 0:
        row_data.append(band_depth_results[0])
        row_data.append(band_depth_results[1])
    else:
        row_data.append(np.NaN)
        row_data.append(np.NaN)

    # Return results for spectrum
    return row_data

def retrieveDatum2(filename):

    # Constants
    ri = ((2.5, 2.89), )  # Restricted interval(s)
    region_1 = np.array([2.95, 3.0])
    region_2 = np.array([3.0, 3.25])
    region_3 = np.array([3.25, 3.5])

    # Retrieve asteroid name
    target = asteroidName(filename)

    # Retrieve asteroid observation date
    obsdate = observationDate(filename)

    # Retrieve data from file and store in Pandas DataFrame
    data = cdf.createDataframe(filename)

    # Data processing
    data = dataProcessing(data, restriction_intervals=ri, outlier_removal=True)

    # Extract wavelength and reflectance values (full)
    x = data.loc[:, 0].to_numpy()  # Wavelengths. Unit: micrometer
    y = data.loc[:, 1].to_numpy()  # Reflectance. Unit: unitless

    # Calculate cubic spline
    spl = InterpolatedUnivariateSpline(x, y, k=3)

    # Generate desired x and y values
    nx = [round(i, 2) for i in np.arange(2.30, 4.00, 0.01) if (i <= 2.5) or (2.9 <= i)]
    ny = list(spl(nx))

    # Return y values of spectrum
    return ny


# Asteroid class dictionary based on optical observations
def asteroidClass(asteroidName):
    
    classification = {"aegle": "T",
                      "aemilia": "Ch",
                      "ani": "Ch",
                      "arachne": "Ch",
                      "arsinoe": "Ch",
                      "artemis": "Ch",
                      "bamberga": "Cb",
                      "berbericia": "Ch",
                      "ceres": "C",
                      "circe": "Ch",
                      "daphne": "Ch",
                      "davida": "C",
                      "diana": "Ch",
                      "doris": "Ch",
                      "dynamene": "Ch",
                      "egeria": "Ch",
                      "ekard": "Ch",
                      "elektra": "Ch",
                      "emanuela": "Ch",
                      "erigone": "Ch",
                      "euphrosyne": "Cb",
                      "europa": "C", 
                      "felicitas": "Ch",
                      "fortuna": "Ch",
                      "hedda": "Ch",
                      "hygiea": "C",
                      "iduna": "Ch",
                      "interamnia": "B", 
                      "isolda": "Ch",
                      "johanna": "Ch",
                      "leda": "Ch",
                      "malabar": "Ch",
                      "marianna": "Ch",
                      "maja": "Ch",
                      "mashona": "Ch",
                      "nemausa": "Ch",
                      "pallas": "B",
                      "panopaea": "Ch",
                      "patientia": "Cb",
                      "peraga": "Ch",
                      "sibylla": "Ch",
                      "tercidina": "Ch",
                      "thia": "Ch",
                      "themis": "B",
                      "thisbe": "B",
                      "vibilia": "Ch",
                      "xanthippe": "Ch",
                      "zelinda": "Ch"}
    
    return classification[asteroidName]


# Asteroid official name
def asteroidOfficialName(asteroidName):
    
    officialName = {"aegle": "96 Aegle",
                    "aemilia": "159 Aemilia",
                    "ani": "791 Ani",
                    "arachne": "407 Arachne",
                    "arsinoe": "404 Arsinoe",
                    "artemis": "105 Artemis",
                    "bamberga": "324 Bamberga",
                    "berbericia": "776 Berbericia",
                    "ceres": "1 Ceres",
                    "circe": "34 Circe",
                    "daphne": "41 Daphne",
                    "davida": "511 Davida",
                    "diana": "78 Diana",
                    "doris": "48 Doris",
                    "dynamene": "200 Dynamene",
                    "egeria": "13 Egeria",
                    "ekard": "694 Ekard",
                    "elektra": "130 Elektra",
                    "emanuela": "576 Emanuela",
                    "erigone": "163 Erigone",
                    "euphrosyne": "31 Euphrosyne",
                    "europa": "52 Europa", 
                    "felicitas": "109 Felicitas",
                    "fortuna": "19 Fortuna",
                    "hedda": "207 Hedda",
                    "hygiea": "10 Hygiea",
                    "iduna": "176 Iduna",
                    "interamnia": "704 Interamnia", 
                    "isolda": "211 Isolda",
                    "johanna": "127 Johanna",
                    "leda": "38 Leda",
                    "malabar": "754 Malabar",
                    "marianna": "602 Marianna",
                    "maja": "66 Maja",
                    "mashona": "1467 Mashona",
                    "nemausa": "51 Nemausa",
                    "pallas": "2 Pallas",
                    "panopaea": "70 Panopaea",
                    "patientia": "451 Patientia",
                    "peraga": "554 Peraga",
                    "sibylla": "168 Sibylla",
                    "tercidina": "345 Tercidina",
                    "thia": "405 Thia",
                    "themis": "24 Themis",
                    "thisbe": "88 Thisbe",
                    "vibilia": "144 Vibilia",
                    "xanthippe": "156 Xanthippe",
                    "zelinda": "654 Zelinda"}
    
    return officialName[asteroidName]
