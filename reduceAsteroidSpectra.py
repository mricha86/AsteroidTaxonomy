# Purpose: Simple data reduction

import argparse as ap
import createDataframe as cdf
import numpy as np
import re
import scipy.interpolate as inter
import sys


def extractField(pat, str):
    field = re.search(pat, str)
    return field.group(0) if field else '&'


def removeCharacters(str):
    # Extract asteroid name field
    newstr = extractField('_(\w)*\.', str)

    # Remove first and last characters
    newstr = newstr[1:-1]

    # Remove anomalous characters
    newstr = re.sub('[0-9][a-z]$', '', newstr)  # Remove a digit and a character
    newstr = re.sub('[0-9]', '', newstr)  # Remove all digits
    newstr = re.sub('_all', '', newstr)  # Remove substring '_all'
    newstr = re.sub('comb', '', newstr)  # Remove substring 'comb'
    newstr = re.sub('most', '', newstr)  # Remove substring 'most'
    newstr = re.sub('old$', '', newstr)  # Remove substring 'most'
    newstr = re.sub('part', '', newstr)  # Remove substring 'part'

    # Return corrected string
    return newstr


def main(argv):
    # Constants
    x_axis_min = 2.25
    x_axis_max = 4.00
    stepsize = 0.001
    NormPos = 2.3

    # Retrieve arguments
    parser = ap.ArgumentParser()
    parser.add_argument(
        "asteroidSpectraList",
        help="File containing file names of asteroid spectral data")
    args = parser.parse_args()
    asteroidSpectraList = args.asteroidSpectraList

    # Store filenames in list
    inputFile = open(asteroidSpectraList)
    inputFile_list = []
    for f in inputFile:
        inputFile_list.append(f.rstrip())

    # Sort list of filenames
    inputFile_list.sort(key=lambda l: extractField('_(\w)*\.', l))
    # print(np.asarray(inputFile_list)); exit(0)

    # Combine spectra for the same target
    nspectra = len(inputFile_list)
    targetIndex = 0
    targetName = removeCharacters(inputFile_list[targetIndex])
    for i in range(1, nspectra):
        # Retrieve comparison spectra target name
        compName = removeCharacters(inputFile_list[i])

        # Reduce spectra associated with current targetName if
        # new name is encountered. Otherwise, continue to next
        # spectrum.
        if (targetName == compName):
            continue
        else:
            # Determine number of spectra for current target
            nspectraTarget = i - targetIndex

            # Calculate cubic spline for each spectrum
            spls_x = []
            spls_y = []
            for j in range(targetIndex, targetIndex + nspectraTarget):
                # Retreive spectrum from current target
                data = cdf.createDataframe(inputFile_list[j])

                # Sort the data by wavelength value
                data = data.sort_values(by=0)

                # Extract wavelength and reflectance values
                x = data.loc[:, 0].to_numpy()  # Wavelengths. Unit: micrometer
                y = data.loc[:, 1].to_numpy()  # Reflectance. Unit: unitless

                # Remove anomalous points
                pos = np.where(x <= 0)
                x = np.delete(x, pos, 0)
                y = np.delete(y, pos, 0)

                # Determine if data encompasses the wavelength
                # domain defined at beginning of function. If not,
                # move on to next spectrum
                if (x_axis_min < np.amin(x)) or (np.amax(x) < x_axis_max):
                    continue

                # Calculate Cubic Spline
                spl = inter.InterpolatedUnivariateSpline(x, y)

                # Sample spline function for values excluding
                # intervals [2.5, 2.9]
                xx = np.arange(x_axis_min, x_axis_max + stepsize, stepsize)
                x_spline = np.delete(xx, np.argwhere((2.5 < xx) & (xx < 2.9)))
                y_spline = spl(x_spline)

                # Store spline data
                spls_x.append(x_spline)
                spls_y.append(y_spline)

            # Convert list to array
            spls_x = np.asarray(spls_x)
            spls_y = np.asarray(spls_y)

            # Average along columns
            spls_x_avg = np.mean(spls_x, axis=0)
            spls_y_avg = np.mean(spls_y, axis=0)

            # Normalize spectrum to unity at 2.30 microns
            pos, = np.where(np.isclose(spls_x_avg, NormPos))
            NormFactor = spls_y_avg[pos[0]]
            spls_y_avg = spls_y_avg / NormFactor

            # Save reduced spectrum to file
            outputFilename = targetName + "_reduced.bin"
            outputFile = open(outputFilename, "w")
            for l in range(len(spls_x_avg)):
                outputFile.write('%f\t%f\n' % (spls_x_avg[l], spls_y_avg[l]))
            outputFile.close()

            # Update to new target
            targetName = compName
            targetIndex = i


if __name__ == "__main__":
    main(sys.argv[1:])
