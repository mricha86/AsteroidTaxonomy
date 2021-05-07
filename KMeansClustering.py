# Purpose: Wrapper for K-Means Clustering ML Algorithm.
#          Plus a few other useful related algorithms.

# Imported modules/functions
from matplotlib import animation
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def myKMeansClustering(X_Train, **kwargs):
    # Initialize K-Means Clustering algorithm
    kmclustering = KMeans(**kwargs)

    # Train Model
    kmclustering = kmclustering.fit(X_Train)

    # Return result of training
    return kmclustering


def optimalNumberClusters_elbow(data, n=10, plot=False):
    # Calculate sum of squared distances of each data
    # point to their closest cluster center per k clusters
    k_clusters = []
    WCSS = []
    for k in range(1, n+1):
        kmeans = myKMeansClustering(data, n_clusters = k)
        k_clusters.append(k)
        WCSS.append(kmeans.inertia_)
    k_clusters = np.array(k_clusters)
    WCSS = np.array(WCSS)

    # Plot number of clusters versus with-in cluster sum of squares (WCSS)
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=WCSS, label='Data')
        ax.set_title("K-Means Optimal Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_xlabel("Number of Clusters")
        plt.show()

    # Calculate fit and first derivative of WCSS
    f = CubicSpline(k_clusters, WCSS)
    df = f.derivative()
    df_2 = df.derivative()
    DF = df(k_clusters).real

    # Calculate percent difference for consecutive derivatives.
    # If the percent difference drops below threshold, return index
    idx = 0
    for i in range(0, n-1):
        pd = abs((DF[i+1] - DF[i]) / DF[i+1])
        if(pd <= 0.15):
            idx = i
            break

    # Plot fit of WCSS
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=WCSS, label='Data')
        af = sns.scatterplot(x=k_clusters, y=f(k_clusters).real, label='Fit')
        ax.set_title("K-Means Optimal Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_xlabel("Number of Clusters")
        plt.show()
    
    # Plot first derivative of WCSS
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=DF, label='Derivative')
        ax.set_title("K-Means Optimal Number of Clusters")
        ax.set_ylabel("WCSS Derivative")
        ax.set_xlabel("Number of Clusters")
        plt.show()
        
    return k_clusters[idx]

def optimalNumberClusters_silhouette(data, n=11, plot=False, save=False):
    # Calculate silhouette score per k clusters
    k_clusters = []
    SS = []
    for k in range(2, n):
        k_clusters.append(k)
        kmeans = myKMeansClustering(data, n_clusters = k)
        labels = kmeans.labels_
        ss = silhouette_score(data, labels)
        SS.append(ss)

    # Plot number of clusters versus silhouette score
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=SS)
        ax.set_title("K-Means Clustering: Optimal Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("Number of Clusters")
    if save:
        plt.savefig("KMeansClusteringSilhouetteScore.png")
    else:
        plt.show()

    # Return number of clusters that correspond to maximum silhouette score
    idx = np.argmax(np.array(SS))
    return k_clusters[idx]


def animate_scatters(iteration, ax, pos):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        scatters (list): List of all the scatters (One per element)
        colors (list): List of the data point colors at each iteration.
    Returns:
        list: List of scatters (One per element) with new colors
    """
    # Modify color of element at the ith iteration
    #for i in range(len(colors)):
    #    scatters[i]._facecolor3d = [ colors[i] ] if i <= iteration else [ [0, 0, 0, 1] ]
    #return scatters
    
    # Change viewing angle
    ax.view_init(pos[iteration][0], pos[iteration][1])


def KMeansClusteringAnimation(data, clusterlabels, colorDictionary=None, save = False):
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
    ax.set_title('K-Means Clustering')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_zlabel(z_title)
    
    # Initialize scatter plot
    if colorDictionary is None:
        ax.scatter(xs=x, ys=y, zs=z, c=clusterlabels, cmap=plt.get_cmap('cool'))
    else:
        colors = [ colorDictionary[cl] for cl in clusterlabels ]
        ax.scatter(xs=x, ys=y, zs=z, c=colors)
    
    # Calculate new viewing angle positions
    # pos = [[40, -225]]
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
        # ani.save('3D-KMeans-animated.mp4', writer=writer)
        ani.save('3D-KMeans-animated.gif', writer='imagemagick', fps=30)
    
    # Show plot
    plt.show()


def KMeansClusteringPieChart(df, kmeans, colorDictionary=None, plot=False, save=False):
    # Store clusters in dictionary
    myDict = {}
    clusters = kmeans.labels_.astype(int)
    for key, val in zip(clusters, df.index.values):
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
    n_members = len(clusters)
    percentages = [n / n_members for n in n_members_per_cluster]

    # Print clusters
    for key in keys:
        print("{0}: {1}\n".format(key, table_data[key]))
    
    # Cluster relationship bar chart
    if plot:
        fig = go.Figure(data=[go.Table(
            columnwidth = [40, 40],
            header=dict(
                values=["Cluster", 'Asteroid'], 
                align='left',
                fill_color='white',
                font=dict(color='black', size=16),
                height=30,
            ),
            cells=dict(
                values=[[0, 1, 2, 3], ["24 Themis", "1 Ceres", "51 Nemausa", "2 Pallas"]], 
                align='left',
                fill_color=[['rgb(0, 255, 255)', 'rgb(85, 170, 255)', 'rgb(170, 85, 255)', 'rgb(255, 0.0, 255)']*2],
                font=dict(color='black', size=16),
                height=30
            )
        )])
        fig.show()
    
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
            startangle=45,
            pctdistance=1.2,
            labeldistance=1.4)
        ax.axis('equal')
        ax.set_title("K-Means Clustering: Cluster Proportions for {0} Spectra".format(n_members))
        # ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))
        # fig, ax = plt.subplots()
        # ax.axis('tight')
        # ax.axis('off')
        # ax.table(cellText=table, colLabels=labels, colLoc='center', loc='center')
    if save:
        fig.savefig('KMeansClusteringPieChart.png')
    else:
        plt.show()
