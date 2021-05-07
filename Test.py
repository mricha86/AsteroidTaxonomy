import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

# Read-in data
datafile1 = "/Users/MRichardson/Desktop/Work_With_Andy/AsteroidThermalModeling/asia.list.new_FSPEC"
datafile2 = "/Users/MRichardson/Desktop/Work_With_Andy/AsteroidThermalModeling/asia.auto_SEGMENT_STM.thcor"
df1 = pd.read_csv(datafile1,
                  header=None,
                  names=[
                      "Wavelength", "Reflectance Ratio",
                      "Reflectance Ratio Error", "Flag"
                  ],
                  delim_whitespace=True)
df2 = pd.read_csv(datafile2,
                  header=None,
                  names=[
                      "Wavelength", "Reflectance Ratio",
                      "Reflectance Ratio Error", "Flag"
                  ],
                  delim_whitespace=True)

# Parse data
x1 = df1.loc[:, 'Wavelength'].values
y1 = df1.loc[:, 'Reflectance Ratio'].values
x2 = df2.loc[:, 'Wavelength'].values
y2 = df2.loc[:, 'Reflectance Ratio'].values

# Visualization 1
fig = plt.figure()
sns.scatterplot(x1, y1, marker='.', label='Uncorrected Spectrum')
sns.scatterplot(x2, y2, marker='.', label='Corrected Spectrum')
plt.ylim(0.5, 2.5)
plt.xlabel('Wavelength')
plt.ylabel('Reflectance')
plt.title('Asia Reflectance Spectrum')
plt.show()

