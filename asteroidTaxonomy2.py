# Purpose:

# Imported modules/functions
import AgglomerativeClustering as ac
import argparse as ap
import asteroidTaxonomy_Auxilary as ata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys


# Main Function
def main(argv):

    #######################
    # Retrieve arguements #
    #######################
    parser = ap.ArgumentParser()
    parser.add_argument("spectralFeaturesTable",
                        help="Table containing spectral features")
    args = parser.parse_args()
    spectralFeaturesTable = args.spectralFeaturesTable

    ###################################
    # Read data into Pandas DataFrame #
    ###################################
    df = pd.read_csv(spectralFeaturesTable, sep=',', header=0)
    print(df.head())
    
    #########################
    # Extract required data #
    #########################
    #cols = ["Asteroid", ""]
    #data = df[cols]
    
    ######################################
    # Evaluate with clustering algorithm #
    ######################################

    # Agglomerative Clustering Optimization
    #linkage = 'ward'
    #distance_threshold = None
    #ac.optimalNumberClusters_elbow(data_scaled, n=len(data_scaled), linkage=linkage, plot=True, save=False)
    #oncAgglomerativeClustering = ac.optimalNumberClusters_silhouette(data_scaled, n=39, plot=False, save=False)
    #print(oncAgglomerativeClustering)
    
    

# Execute main function
if __name__ == "__main__":
   main(sys.argv[1:])
