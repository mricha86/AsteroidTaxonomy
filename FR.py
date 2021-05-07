import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as inter
import seaborn as sns
sns.set()

# Read-in data
datafile = "/Users/MRichardson/Desktop/Work_With_Andy/AsteroidThermalModeling/asia.auto_SEGMENT.thcor"
df = pd.read_csv(datafile, header=None, names=["Wavelength", "Reflectance Ratio",
                                               "Reflectance Ratio Error", "Flag"], delim_whitespace=True)

# Parse data
x1 = df.loc[:, 'Wavelength'].values
y1 = df.loc[:, 'Reflectance Ratio'].values
xx = np.arange(np.amin(x1), np.amax(x1), 0.001)

# Plot
spl = inter.InterpolatedUnivariateSpline(x1, y1)
plt.plot(x1, y1, '.', label='Data')
plt.plot (xx, spl(xx), 'k--', label='Spline, correct order')
plt.minorticks_on()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
