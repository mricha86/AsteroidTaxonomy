# Purpose: Algorithms used for plotting

# Imported modules/functions
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def bandCenterHist(series):
    plt.figure(figsize=(12, 6))
    plt.title('Band Center Distribution')
    plt.ylabel('Frequency')
    sns.distplot(series, axlabel='Wavelength ($\\mu$m)', kde=False)
    plt.show()


def bandCenterPlot(df):
    # Retrieve  column names
    columns = df.columns

    # Determine if appropriate column names are present
    # If so, continue with plotting procedure
    if ('R2: PBC' in columns) and ('R2: GBC' in columns) and (
            'R2: GBC Error' in columns):
        x = df['R2: GBC'].values
        y = df['R2: PBC'].values
        xerr = df['R2: GBC Error'].values
        ref = np.arange(x.min(), x.max(), 0.01)
        plt.figure(figsize=(12, 6))
        plt.title(
            'Region 2: Polynomial band center versus Gaussian band center')
        plt.xlabel('Gaussian-derived band center wavelength ($\\mu$m)')
        plt.ylabel('Polynomial-derived band center wavelength ($\\mu$m)')
        plt.errorbar(x, y, xerr=xerr, fmt='.')
        sns.lineplot(x=ref, y=ref)
        plt.show()
    else:
        print("Unable to create plot.")


def bandDepthClusteringPlot(df, clusterlabels):

    # Determine axes titles
    cols = df.columns

    # Extract plotting information
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_title = cols[0]
    y_title = cols[1]

    plt.figure(figsize=(12, 6))
    plt.title('Band Depth Clustering')
    sns.scatterplot(x=x,
                    y=y,
                    hue=clusterlabels,
                    data=df)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(title='Cluster Number')
    plt.show()


def bandDepthClusteringPlot3D(df, clusterlabels, colorDictionary=None, annotate=None):

    # Determine axes titles
    cols = df.columns

    # Determine number of axes
    naxes = len(cols)

    # Abort if number of axes is less than 2 or exceeds 3
    if (naxes < 2) or (naxes > 3):
        return None

    # Use different plotting algorithm for 2 axes
    if naxes == 2:
        bandDepthClusteringPlot(df, clusterlabels)
        return None

    # Extract plotting information
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]
    x_title = cols[0]
    y_title = cols[1]
    z_title = cols[2]
    
    # Initialize 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot axes titles
    #ax.set_title('')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_zlabel(z_title)
    
    # Plot viewing angle
    ax.view_init(40, -225)
    
    # Plot data
    if colorDictionary is None:
        ax.scatter(xs=x, ys=y, zs=z, c=clusterlabels, cmap=plt.get_cmap('cool'))
    else:
        colors = [ colorDictionary[cl] for cl in clusterlabels ]
        ax.scatter(xs=x, ys=y, zs=z, c=colors)
        
    # Annotate data
    if annotate is not None:
        if annotate.upper() == 'ALL':
            for i in range(len(x)):
                ax.text(x[i], y[i], z[i], "{0}".format(df.index[i]), size=6, zorder=1, color='k')
                # ax.annotate(df.index[i], (x[i], y[i], z[i]))
    plt.show()


def bandDepthPlot(df):
    # Retrieve  column names
    columns = df.columns

    # Determine if appropriate column names are present
    # If so, continue with plotting procedure
    if ('R1: PBD' in columns) and ('R2: GBD' in columns) and (
            'R2: GBD Error' in columns):
        x = df['R2: GBD'].values
        y = df['R1: PBD'].values
        xerr = df['R2: GBD Error'].values
        plt.figure(figsize=(12, 6))
        plt.title('Polynomial band depth versus Gaussian band depth')
        plt.xlabel('3.2 $\\mu$m Band Depth')
        plt.ylabel('2.9 $\\mu$m Band Depth')
        plt.errorbar(x, y, xerr=xerr, fmt='.')
        plt.show()
    else:
        print("Unable to create plot.")


def correlationMatrix(df, title="Correlation Matrix",  plot = False):
    # Calculate correlation matrix
    correlationMatrix = None
    if isinstance(df, pd.DataFrame):
        correlationMatrix = df.corr()
    if isinstance(df, np.ndarray):
        data = pd.DataFrame(data=df)
        correlationMatrix = data.corr()

    # Plot correlation matrix
    if(plot):
        sns.heatmap(correlationMatrix, annot=True)
        plt.title(title)
        plt.show()
    
    return correlationMatrix


def plot3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='.')
    # ax.set_xlabel()
    # ax.set_ylabel()
    # ax.set_zlabel()
    plt.show()
