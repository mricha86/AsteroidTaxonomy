# Purpose: Wrapper for Agglomerative Clustering ML Algorithm.
#                Plus a few other useful algorithms

# Imported modules/functions
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def AgglomerativeClusteringAnimation(data, clusterlabels, colorDictionary=None, save = False):
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
    ax.set_title('Agglomerative Clustering')
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
    
    # Create animationß
    ani = animation.FuncAnimation(fig, animate_scatters, len(pos), fargs=(ax, pos), interval=500, blit=False, repeat=True)
    
    # Save to file
    if save:
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        # ani.save('3D-Agglomerative-animated.mp4', writer=writer)
        ani.save('3D-Agglomerative-animated.gif', writer='imagemagick', fps=30)

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

    
def calculateWSS(nclusters, data, linkage='ward'):
    WSS = []
    for i in range(nclusters):
        cluster = AgglomerativeClustering(n_clusters=i+1, affinity='euclidean', linkage=linkage)  
        cluster.fit(data)
        
        # Retrieve cluster indicies
        label = cluster.labels_

        # Calculate WSS associated with current number of clusters
        wss = []
        for j in range(i+1):
            # Extract each cluster member according to its cluster's index
            idx = [t for t, e in enumerate(label) if e == j]
            cluster = data.iloc[idx,].values
            
            # Calculate the WSS associated with i+1 clusters:
            cluster_mean = cluster.mean(axis=0)
            distance = np.sum(np.abs(cluster - cluster_mean)**2, axis=-1)
            wss.append(sum(distance))
        WSS.append(sum(wss))

    return WSS


def myAgglomerativeClustering(X_Train, **kwargs):
    # Initialize Agglomerative Clustering algorithm
    agclustering = AgglomerativeClustering(**kwargs)

    # Train Model
    agclustering = agclustering.fit(X_Train)

    # Return result of training
    return agclustering


def optimalNumberClusters_elbow(data, n=10, linkage='ward', plot=False, save=False):
    # Calculate sum of squared distances of each data
    # point to their closest cluster center per k clusters
    k_clusters = [k+1 for k in range(n)]
    WCSS = calculateWSS(n, data, linkage=linkage)

    # Plot WSS versus number of clusters
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=WCSS)
        ax.set_title("Agglomerative Clustering: Optimal Number of Clusters via Elbow Method")
        ax.set_ylabel("Total WCSS")
        ax.set_xlabel("Number of Clusters")
    if save:
        plt.savefig("AgglomerativeClusteringElbowMethod.png")
        plt.show()
    else:
        plt.show()


def optimalNumberClusters_silhouette(data, n=11, plot=False, save=False):
    # Calculate silhouette score per k clusters
    k_clusters = []
    SS = []
    for k in range(2, n):
        k_clusters.append(k)
        agglomerative = myAgglomerativeClustering(data, n_clusters = k)
        labels = agglomerative.labels_
        ss = silhouette_score(data, labels)
        SS.append(ss)

    # Plot number of clusters versus silhouette score
    if plot:
        ax = sns.scatterplot(x=k_clusters, y=SS)
        ax.set_title("Agglomerative Clustering: Optimal Number of Clusters via Silhouette Method")
        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("Number of Clusters")
    if save:
        plt.savefig("AgglomerativeClusteringSilhouetteScore.png")
        plt.show()
    else:
        plt.show()

    # Return number of clusters that correspond to maximum silhouette score
    idx = np.argmax(np.array(SS))
    return k_clusters[idx]


def AgglomerativeClusteringPieChart(df, agclustering, colorDictionary=None, plot=False, save=False):
    # Store clusters in dictionary
    myDict = {}
    clusters = agclustering.labels_.astype(int)
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
        
    # Create modified color dictionary
    mod_color_dict = {}
    for key in keys:
        mod_color_dict[key] = colorDictionary[key]
        vals = tuple(np.array(mod_color_dict[key])*255)
        mod_color_dict[key] = vals
    
    # Cluster relationship bar chart
    asteroid_official_names = ["1 Ceres", "51 Nemausa", "2 Pallas", "24 Themis"]
    asteroids = ["ceres", "nemausa", "pallas", "themis"]
    asteroid_classes = []
    asteroid_class_color = []
    for ast in asteroids:
        for key in keys:
            if ast in myDict[key]:
                asteroid_classes.append(key)
                asteroid_class_color.append("rgb{0}".format(mod_color_dict[key][0:3]))

    # Retrieve RGBA colors
    colors = []
    if colorDictionary is None:
        temp = plt.scatter(x=keys, y=keys, c=keys, cmap=plt.get_cmap('cool'))
        for key in keys:
            colors.append(temp.to_rgba(key))
    else:
        for key in keys:
            colors.append(colorDictionary[key])
    
    # Quick chart of spectral types
    if plot:
        table = go.Figure(data=[go.Table(
            columnwidth = [40, 40],
            header=dict(
                values=["Cluster", 'Asteroid'], 
                align='left',
                fill_color='white',
                font=dict(color='black', size=16),
                height=30,
            ),
            cells=dict(
                values=[asteroid_classes, asteroid_official_names], 
                align='left',
                fill_color=[asteroid_class_color*2], 
                font=dict(color='black', size=16),
                height=30
            )
        )])
        table.show()
    
    # Cluster pie chart and table
    if plot:
        fig, ax = plt.subplots()
        ax.pie(
            percentages,
            labels=labels,
            colors=colors,
            autopct='%1.0f%%',
            shadow=False,
            startangle=-45,
            pctdistance=1.2,
            labeldistance=1.4)
        ax.axis('equal')
        ax.set_title("Agglomerative Clustering: Cluster Proportions for {0} Spectra".format(n_members))
        # ax.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))
        # fig, ax = plt.subplots()
        # ax.axis('tight')
        # ax.axis('off')
        # ax.table(cellText=table, colLabels=labels, colLoc='center', loc='center')
    if save:
        fig.savefig('AgglomerativeClusteringPieChart.png')
        plt.show()
    else:
        plt.show()
        


def Dendrogram(df, distance_threshold=None, method='ward', cluster_labels=None, colorDictionary=None, plot=False, save=True):
    # Create linkage matrix
    linkage_matrix = linkage(df, method=method)
    
    if cluster_labels is not None:
        # Cluster indicies
        index = [i for i in range(len(cluster_labels))]
        
        # Retrieve RGBA colors
        colors = []
        if colorDictionary is None:
            temp = plt.scatter(x=cluster_labels, y=cluster_labels, c=cluster_labels, cmap=plt.get_cmap('cool'))
            for cl in cluster_labels:
                colors.append(temp.to_rgba(cl))
        else:
            for cl in cluster_labels:
                colors.append(colorDictionary[cl])
        
        # Convert RGBA colors to Hex colors
        colors = [ mpl.colors.to_hex(colors[i], keep_alpha=True) for i in range(len(colors)) ]
        
        # Cluster color mapping
        D_leaf_colors = {}
        for i in range(len(cluster_labels)):
            D_leaf_colors[index[i]] = colors[i]
        
        # notes:
        # * rows in Z correspond to "inverted U" links that connect clusters
        # * rows are ordered by increasing distance
        # * if the colors of the connected clusters match, use that color for link
        default_col = "#000000ff"
        link_cols = {}
        for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linkage_matrix) else D_leaf_colors[x] for x in i12)
            link_cols[i+1+len(linkage_matrix)] = c1 if c1 == c2 else default_col
    
    # Plot the corresponding dendrogram
    if(plot):
        plt.figure(figsize=(12, 6))
        #dendrogram(linkage_matrix)
        dendrogram(linkage_matrix, link_color_func=lambda x: link_cols[x])
        if distance_threshold:
            plt.axhline(distance_threshold, color='m', linestyle='dashdot')
        plt.title('Agglomerative Clustering: Dendrogram')
        plt.xlabel('Observation')
        plt.ylabel('Distance ('+method+')')
    if save:
        plt.savefig('AgglomerativeClusteringDendrogram.png')
        plt.show()
    else:
        plt.show()

    return linkage_matrix
