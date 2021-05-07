# Purpose: Wrapper for KNN Clustering ML Algorithm.
#                Plus a few other useful algorithms

# Imported modules/functions
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def KNNClassifierAnimation(data, clusterlabels, colorDictionary=None, save = False):
    ###################
    # Data generation #
    ###################
    
    # Determine axes titles
    cols = data.columns

    # Determine number of data points
    ndatapoints = data.shape[0]
    
    # Extract plotting information
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    x_title = cols[0]
    y_title = cols[1]
    z_title = cols[2]
    
    # Initialize figure and establish 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Setting the axes properties
    ax.set_title('K-Nearest Neighbors')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_zlabel(z_title)
    
    # Initialize scatter plot
    if colorDictionary is None:
        ax.scatter(xs=x, ys=y, zs=z, c=clusterlabels, cmap=plt.get_cmap('cool'))
    else:
        colors = [ colorDictionary[cl] for cl in clusterlabels ]
        ax.scatter(xs=x, ys=y, zs=z, c=colors)
        
    # Provide starting angle for the view.
    # ax.view_init(40, -225)
    
    # Calculate new viewing angle positions
    #pos = [[40, -225]]
    pos = [[40, -180]]
    for side in range(4):
        t_b = (side % 2 == 0)
        while t_b:
            if side == 0:
                if pos[-1][1] != -270:
                    pos.append([pos[-1][0], pos[-1][1]-1])
                else:
                    break
            else:
                if pos[-1][1] != -180:
                    pos.append([pos[-1][0], pos[-1][1]+1])
                else:
                    break
        while not t_b:
            if side == 1:
                if pos[-1][0] != 0:
                    pos.append([pos[-1][0]-1, pos[-1][1]])
                else:
                    break
            else:
                if pos[-1][0] != 40:
                    pos.append([pos[-1][0]+1, pos[-1][1]])
                else:
                    break
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate_scatters, len(pos), fargs=(ax, pos), interval=500, blit=False, repeat=True)
    
    # Save to file
    if save:
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        # ani.save('3D-KNN-animated.mp4', writer=writer)
        ani.save('3D-KNN-animated.gif', writer='imagemagick', fps=30)
   
    # Show plot
    plt.show()


def animate_scatters(iteration, ax, pos):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        pos (list): List of all plot viewing angles
    Returns:
        list: Figure at new viewing angle
    """
 
    # Change viewing angle
    ax.view_init(pos[iteration][0], pos[iteration][1]) 


def myKNNClassifier(data, k):
    # Initialize Agglomerative Clustering algorithm
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')

    # Train Model
    knn = knn.fit(data)

    # Return result of training
    return knn


def optimalNumberNeighbors(data, n=10, plot=False):
    pass


def KNNClassifierPieChart(data, classifications, colorDictionary=None, plot=False, save=False):
    # Store clusters in dictionarysave=False
    myDict = {}
    for key, val in zip(classifications, data.index.values):
        if key in myDict:
            myDict[key].append(val)
        else:
            myDict[key] = [val]
    keys = list(myDict.keys())
    keys.sort()
    labels = ['Cluster '+str(key) for key in keys]
    
    # Determine number of members for each cluster and create table data
    n_members_per_cluster = []
    table_data = []
    for key in keys:
        n_members_per_cluster.append(len(myDict[key]))
        table_data.append(myDict[key])

    # Make table data a rectangular list
    max_entries = max(np.asarray(n_members_per_cluster))
    index = 0
    for n in n_members_per_cluster:
        if n < max_entries:
            diff = max_entries - n
            extension = [''] * diff
            table_data[index].extend(extension)
        index += 1
    table = np.asarray(table_data)
    table = np.transpose(table)

    # Calculate cluster percentages
    n_members = len(classifications)
    percentages = [n / n_members for n in n_members_per_cluster]
    
    # Print clusters
    for key in keys:
        print("{0}: {1}\n".format(key, table_data[key]))
        
    # Retrieve RGBA colors
    colors = []
    if colorDictionary is None:
        temp = plt.scatter(x=keys, y=keys, c=keys, cmap=plt.get_cmap('cool'))
        for key in keys:
            colors.append(temp.to_rgba(key))
    else:
        for key in keys:
            colors.append(colorDictionary[key])
    
    # Cluster pie chart
    if plot:
        fig, ax = plt.subplots()
        ax.pie(
            percentages,
            labels=labels,
            colors=colors,
            autopct='%1.0f%%',
            shadow=False,
            startangle=120,
            pctdistance=1.2,
            labeldistance=1.4)
        ax.axis('equal')
        ax.set_title("K-Nearest Neighbors: Class Percentage for {0} Spectra".format(n_members))
    if save:
        fig.savefig('KNNPieChart.png')
    else:
        plt.show()