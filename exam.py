########################################
# 
# Unsupervised Learning - Final Project
# 
# Author: Stefano Tumino
# Date: 18/04/2025
# 
########################################
from datetime import datetime as dt
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform

from func import (nan_deletion_approach, nan_substitution_approach, MultiColumnLabelEncoder, MultiColumnOneHotEncoder, kmeans_clustering,
                hierarchical_clustering, PCA_tSNE_visualization)

# Parameters
DATASET_NAME = 'mehra-complete-1000.csv'
PAL = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'magenta', 'cyan', 'brown']
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

TITLE_COLOR = '\033[0;34m'
RESULTS_COLOR = '\033[1;32m'
STANDARD_COLOR = '\033[0m'

CLEANING_APPROACHES = ['NaN deletion', 'NaN substitution']
CATEGORICAL_TRANSFORMATION = ['Label Encoder', 'One Hot Encoder']
SCALING_METHODS = ['MinMax Scaler', 'Robust Scaler']
CLUSTER_METHODS = ['K-Means++', 'Hierarchical']


########################################
# DATA READING AND INITIALIZATION
########################################
# Check for the directories
current_dir = os.getcwd()
print(current_dir)

output_dir = f'{current_dir}/output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Output directory did not esist! It has just been generated")

# Data reading
data = pd.read_csv(DATASET_NAME)

# Data cleaning from negative values | Step 1: identify the columns
problematic_features = []
for feat in list( data.columns ):
  if (data[feat].dtype == 'float64') and (data[feat].min() < 0):
    problematic_features.append(feat)

print('Features with negative values in some row are: \n', problematic_features)
problematic_features = problematic_features[2:]     # First two are normal
print('\nFeatures with negative and problematic values in some row are: \n', problematic_features)

# Data cleaning from negative values | Step 2: replace with nan
nan_data = data.copy()
for feat in problematic_features:
   nan_data.loc[ nan_data[feat] < 0, feat ] = np.nan

# Separate discrete from contrinuous features
continuous_features = []
discrete_features = []
for feat in list( data.columns ):
    if (data[feat].dtype == 'float64') or (data[feat].dtype == 'int64'):
        continuous_features.append(feat)
    else:
        discrete_features.append(feat)

print('\nContinuous features are:\n', continuous_features)
print('Discrete features are:\n', discrete_features)


########################################
# MAIN BODY
########################################
# Configurations
isToBePlotted = True


# Iterate over different cleaning data methods
for i, cleaning_name in enumerate(CLEANING_APPROACHES):

    if i == 0:
        clean_data = nan_deletion_approach(nan_data)
    elif i == 1:
        clean_data = nan_substitution_approach(nan_data, problematic_features)
    
    continuous_features = []
    discrete_features = []
    for feat in list( data.columns ):
        if (data[feat].dtype == 'float64') or (data[feat].dtype == 'int64'):
            continuous_features.append(feat)
        else:
            discrete_features.append(feat)

    # Iterate over different categorical data transformations
    for j, transf_name in enumerate(CATEGORICAL_TRANSFORMATION):

        if j == 0:
            categ_transformation = MultiColumnLabelEncoder(columns=discrete_features)
        elif j == 1:
            categ_transformation = MultiColumnOneHotEncoder(columns=discrete_features)

        categ_data = categ_transformation.fit(clean_data).transform(clean_data)

        # Iterate over different clustering methods
        for h,scaling_name in enumerate(SCALING_METHODS):

            if h == 0:
                scaler = ColumnTransformer(
                    transformers=[('minmax', MinMaxScaler(), continuous_features)], remainder='passthrough'
                )
            elif h == 1:
                scaler = ColumnTransformer(
                    transformers=[('robust', RobustScaler(), continuous_features)], remainder='passthrough'
                )
            
            scaled_data = scaler.fit_transform(categ_data)
            scaled_feature_names = scaler.get_feature_names_out()
            scaled_data = pd.DataFrame(scaled_data, columns=scaled_feature_names, index=categ_data.index)
            scaled_data.columns = [col.split('__')[-1] for col in scaled_data.columns]

            # Plot PCA and tSNE
            if (isToBePlotted):
                filename = output_dir + f'/{i}_{j}_{h}_PCA_tSNE_visualization.png'
                PCA_tSNE_visualization(scaled_data, 5, np.ones( scaled_data.shape[0] ), PAL, filename)

            predicted_labels = []
            max_nclusters = 0

            # Iterate over different clustering techniques
            for ii, cluster_name in enumerate(CLUSTER_METHODS):
                print('--------------------------------------------------')
                starting_time = dt.now()

                print(f'\n>>>>>>{TITLE_COLOR} {cleaning_name:<17} {STANDARD_COLOR}' +
                        f'-->{TITLE_COLOR} {transf_name:<16} {STANDARD_COLOR}' +
                        f'-->{TITLE_COLOR} {scaling_name:<15} {STANDARD_COLOR}' +
                        f'-->{TITLE_COLOR} {cluster_name:<15} {STANDARD_COLOR}')
                print('Pre-processed data has %d observations and %d features' % (scaled_data.shape[0], scaled_data.shape[1]))
                
                if ii == 0:
                    labels, nclusters = kmeans_clustering(scaled_data)
                
                elif ii == 1:
                    labels, nclusters = hierarchical_clustering(scaled_data, False)

                S = silhouette_score(scaled_data, labels, metric='euclidean')
                print(f'\nSilhouette score = [{RESULTS_COLOR}{S:.3f}{STANDARD_COLOR}]')

                # Time needed to computations
                duration = (dt.now() - starting_time).seconds
                print(f'\nComputational Time = [{RESULTS_COLOR}{duration} s{STANDARD_COLOR}]\n')

                # Add clustering method and predicted labels 
                predicted_labels.append(
                    {'name': cluster_name, 'labels': labels}
                )
                if nclusters > max_nclusters:
                    max_nclusters = nclusters
            
            # Comparison between clusters methods
            print('------------------------- Comparison -------------------------')
            if isToBePlotted:
                filename = output_dir + f'/{i}_{j}_{h}_Comparison.png'
                fig = plt.figure(figsize=(10, 5))
                for jj,prediction in enumerate(predicted_labels):
                    plt.plot(np.sort( prediction['labels'] ), color=PAL[jj], label=prediction['name'])
                sns.set_theme(style='dark')
                plt.xlabel('Object id')
                plt.yticks( np.arange(0, max_nclusters+1) )
                plt.ylabel('Predicted labels')
                plt.title('Comparison of predicted labels')
                plt.legend()
                plt.grid()
                plt.savefig(filename, dpi=150)
                plt.close(fig)
                print('Comparison figure saved in output directory')

            # Compute the adjusted Rand score (R) for each pair
            c = list( itertools.combinations( list(range(0, len(predicted_labels))),2 ) )
            c = np.array(c)
            for el in c:
                R = adjusted_rand_score(predicted_labels[el[0]]['labels'], predicted_labels[el[1]]['labels'])
                print(f'\nR between {el[0]} and {el[1]} = {R:.2f}\n')
            
            # Similarity matrix
            PM = pdist(scaled_data, metric='euclidean')
            PM = squareform(PM).round(2)
            for prediction in predicted_labels:
                SM = np.eye(scaled_data.shape[0])
                y = prediction['labels']
                for jj in range(y.size):
                    for hh in range(y.size):
                        if y[jj] == y[hh]:
                            SM[jj, hh] = 1
                            SM[hh, jj] = 1
                
                PM_values = PM[np.triu_indices_from(PM, k=1)]
                SM_values = SM[np.triu_indices_from(SM, k=1)]
                rho = np.abs( np.corrcoef(PM_values, SM_values)[0,1].round(2) )

                print(f'For {prediction['name']} the cross-correlation is {rho:.2f}')
