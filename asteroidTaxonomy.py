# Purpose: Construction of asteroid taxonomy using 3-micron data

######################
# Imported Libraries #
######################
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import AgglomerativeClustering as ac
import argparse as ap
import asteroidTaxonomy_Auxilary as ata
import asteroidTaxonomy_Plotting as atplot
import KMeansClustering as kmc
import KNNClassifier as knn
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sys
sns.set()

#################
# Main Function #
#################
def main(argv):
    # Retrieve arguments
    parser = ap.ArgumentParser()
    parser.add_argument("spectraList1",
                        help="List containing spectra file names")
    parser.add_argument("spectraList2",
                        help="List containing spectra file names")
    args = parser.parse_args()
    spectraList1 = args.spectraList1
    spectraList2 = args.spectraList2

    # Create empty Pandas dataframe
    columns = [
        'Date', 'Asteroid', 'R1: PBC', 'R1: PBD', 'R2: Polynomial-Derived Band Center', 'R2: PBD',
        'R3: PBC', 'R3: PBD', 'R2: GBC', 'R2: GBC Error', 'R2: GBD',
        'R2: GBD Error', '2.9 \u03BCm Band Depth', '3.2 \u03BCm Band Depth'
    ]
    df1 = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame(columns=columns)

    # Retrieve data for classification
    index = 0
    files = open(spectraList1)
    for f in files:
        row = ata.retrieveDatum(f.rstrip(), plot=True)
        if len(row) == len(columns):
            df1.loc[index] = row  # Append row to data array
        index += 1  # Update index
    files.close()

    exit()
    
    index = 0
    files = open(spectraList2)
    for f in files:
        row = ata.retrieveDatum(f.rstrip(), plot=True)
        if len(row) == len(columns):
            df2.loc[index] = row  # Append row to data array
        index += 1  # Update index
    files.close()

    exit()
    
    # Convert string date into real date
    df1.Date = pd.to_datetime(df1.Date).dt.date
    df2.Date = pd.to_datetime(df2.Date).dt.date

    # Extract data subsets
    cols = ['Asteroid', '3.2 \u03BCm Band Depth', '2.9 \u03BCm Band Depth', 'R2: Polynomial-Derived Band Center']
    df_1 = df1[cols]
    df_1 = df_1.dropna()
    df_1 = df_1.loc[df_1['2.9 \u03BCm Band Depth'] >= 0]
    df_1 = df_1.loc[df_1['3.2 \u03BCm Band Depth'] >= 0]
    df_1 = df_1.set_index('Asteroid')
    df_2 = df2[cols]
    df_2 = df_2.dropna()
    df_2 = df_2.loc[df_2['2.9 \u03BCm Band Depth'] >= 0]
    df_2 = df_2.loc[df_2['3.2 \u03BCm Band Depth'] >= 0]
    df_2 = df_2.set_index('Asteroid')
    
    # Create pretty tables for first data set
    cols = ['Date', 'Asteroid', '3.2 \u03BCm Band Depth', '2.9 \u03BCm Band Depth', 'R2: Polynomial-Derived Band Center']
    df_3 = df1[cols]
    df_3 = df_3.dropna()
    df_3 = df_3.loc[df_3['2.9 \u03BCm Band Depth'] >= 0]
    df_3 = df_3.loc[df_3['3.2 \u03BCm Band Depth'] >= 0]
    df_3['Asteroid'] = [ata.asteroidOfficialName(name) for name in df_3['Asteroid'].values]    
    df_3['Sample Index'] = [i for i in range(len(df_3))]
    cols = ['Sample Index', 'Date', 'Asteroid', '3.2 \u03BCm Band Depth', '2.9 \u03BCm Band Depth', 'R2: Polynomial-Derived Band Center']
    df_3 = df_3[cols]
    table1 = go.Figure(
        data = [go.Table(
            header = dict(values = list(df_3.columns),
                          fill_color = 'paleturquoise',
                          align = 'left'), 
            cells = dict(values = [df_3['Sample Index'], 
                                   df_3['Date'], 
                                   df_3['Asteroid'],
                                   df_3['3.2 \u03BCm Band Depth'], 
                                   df_3['2.9 \u03BCm Band Depth'],  
                                   df_3['R2: Polynomial-Derived Band Center']], 
                         fill_color='lavender',
                         align='left')
        )]
    )
    
    # table1.show()
    #table1.update_layout(width = 1200, height = 1200, font = dict(size=15))
    #table1.write_image("Table1.png")
    
    # Create pretty tables for second data set
    cols = ['Date', 'Asteroid', '3.2 \u03BCm Band Depth', '2.9 \u03BCm Band Depth', 'R2: Polynomial-Derived Band Center']
    df_4 = df2[cols]
    df_4 = df_4.dropna()
    df_4 = df_4.loc[df_4['2.9 \u03BCm Band Depth'] >= 0]
    df_4 = df_4.loc[df_4['3.2 \u03BCm Band Depth'] >= 0]
    df_4['Asteroid'] = [ata.asteroidOfficialName(name) for name in df_4['Asteroid'].values]
    df_4['Sample Index'] = [i for i in range(len(df_4))]
    cols = ['Sample Index', 'Date', 'Asteroid', '3.2 \u03BCm Band Depth', '2.9 \u03BCm Band Depth', 'R2: Polynomial-Derived Band Center']
    df_4 = df_4[cols]
    table2 = go.Figure(data=[go.Table(
    header=dict(values=list(df_4.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_4['Sample Index'], df_4['Date'], df_4['Asteroid'], df_4['3.2 \u03BCm Band Depth'], df_4['2.9 \u03BCm Band Depth'], df_4['R2: Polynomial-Derived Band Center']],
               fill_color='lavender',
               align='left'))
    ])
    # table2.show()
    #table2.update_layout(width = 1200, height = 1200, font = dict(size=15))
    #table2.write_image("Table2.png")
    #exit()
    
    # Polynomial band center histogram
    # series = df['R2: PBC'].dropna()
    #atplot.bandCenterHist(series)

    # Gaussian band center histogram
    #series = df['R2: GBC'].dropna()
    #atplot.bandCenterHist(series)

    # Polynomial band center versus Gaussian band center
    # atplot.bandDepthPlot(df_2)

    # Data selection
    data1 = df_1
    data2 = df_2
    
    # Normalization/Scale data and record to DataFrame
    data_scaled1 = preprocessing.minmax_scale(data1)
    data_scaled1 = pd.DataFrame(data_scaled1)
    data_scaled1.columns = cols[1:4]
    data_scaled1 = data_scaled1.set_index(data1.index)
    
    # Correlation Matrix
    # atplot.correlationMatrix(data, plot=True)
    # atplot.correlationMatrix(data_scaled, plot=True)
    # exit()
    
    #########################
    # Clustering algorithms #
    #########################    
    
    # Agglomerative Clustering Optimization
    linkage = 'ward'
    distance_threshold = None
    ac.optimalNumberClusters_elbow(data_scaled1, n=len(data_scaled1), linkage=linkage, plot=False, save=False)
    oncAgglomerativeClustering = ac.optimalNumberClusters_silhouette(data_scaled1, n=39, plot=False, save=False)
    
    # Agglomerative Clustering Model Training
    n_clusters = oncAgglomerativeClustering
    agclustering = ac.myAgglomerativeClustering(data_scaled1, 
                                                n_clusters=n_clusters, 
                                                distance_threshold=distance_threshold, 
                                                linkage=linkage)
    agglomerative_clustering_labels = agclustering.labels_
    
    # Determine color dictionary
    cDict1 = ata.colorDictionary(agglomerative_clustering_labels, n_clusters, largestCluster=True)
    
    # Agglomerative Clustering Visualizations
    atplot.bandDepthClusteringPlot3D(data1, agglomerative_clustering_labels, colorDictionary=cDict1, annotate='All')
    # ac.AgglomerativeClusteringPieChart(data1, agclustering, colorDictionary=cDict1, plot=True, save=True)
    # ac.Dendrogram(data_scaled1, method=linkage, cluster_labels=agglomerative_clustering_labels, colorDictionary=cDict1, plot=True, save=True)
    # ac.AgglomerativeClusteringAnimation(data1, agglomerative_clustering_labels, colorDictionary=cDict1, save=True)
    exit(0)
    
    # K-Means Clustering Optimization
    #onc = kmc.optimalNumberClusters_elbow(data_scaled1, n=), plot=True)
    oncKMeans = kmc.optimalNumberClusters_silhouette(data_scaled1)

    # K-Means Clustering Model Training
    n_clusters = oncKMeans
    kmeans = kmc.myKMeansClustering(data_scaled1, n_clusters=n_clusters, random_state = 0)
    kmeans_labels = kmeans.labels_
    
    # Determine color dictionary
    cDict2 = ata.colorDictionary(kmeans_labels, n_clusters, largestCluster=True)
    
    # K-Means Clustering Visualizations
    # kmc.KMeansClusteringPlot(data1, kmeans)
    # atplot.bandDepthClusteringPlot3D(data1, kmeans_labels, colorDictionary=cDict2, annotate='All')
    # kmc.KMeansClusteringPieChart(data1, kmeans, colorDictionary=cDict2, plot=True, save=False)
    # kmc.KMeansClusteringAnimation(data1, kmeans_labels, colorDictionary=cDict2, save=False)
    # exit(0)

    ##########################################
    # Splitting data into training/test sets #
    ##########################################
    # trainingData, testingData, trainingLabels, testingLabels = train_test_split(data_scaled, kmeans_labels, train_size = 0.8, random_state = 1)
    # print("Total number of data points:", len(data_scaled), "| Training data set size:", len(trainingData), "| Testing data set size:", len(testingData))
    # exit()    
    
    #############################
    # Classification algorithms #
    #############################
    # k = int(math.sqrt(data_scaled1.shape[0]))
    # k = k - 1 if (k % 2 == 0) else k
    k = 3
    knnclassifications = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    knnclassifications.fit(data1, agglomerative_clustering_labels)
    # knnclassifications.fit(data_scaled1, kmeans_labels)
    
    # Make predictions
    predictions = knnclassifications.predict(data2)
    # atplot.bandDepthClusteringPlot3D(data2, predictions, colorDictionary=cDict1, annotate='All')
    knn.KNNClassifierAnimation(data2, predictions, colorDictionary=cDict1, save=True)
    knn.KNNClassifierPieChart(data2, predictions, colorDictionary=cDict1, plot=True, save=True)
    # print(predictions)

    # Determine accuracy of model
    # score = mod.score(testingData, testingLabels)
    # print(score)

    # Data Visualization
    # cm = metrics.confusion_matrix(kmeans_labels, predictions)
    # sns.heatmap(cm, annot=True)
    # plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
