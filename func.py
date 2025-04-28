import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#######################################
# NaN deletion
########################################
def nan_deletion_approach(data):
    clean_data = data.copy()
    return clean_data.dropna()


########################################
# NaN substitution
########################################
def nan_substitution_approach(data, problematic_features):
    clean_data = data.copy()
    for feat in problematic_features:
        mu = clean_data[feat].mean()
        sigma = clean_data[feat].std()

        nan_indices = clean_data[feat].isna()

        new_values = []
        while len(new_values) < nan_indices.sum():
            value = np.random.normal(mu, sigma)
            if value >= 0:
                new_values.append(value)

        clean_data.loc[ nan_indices, feat ] = new_values

    return clean_data


########################################
# Label Encoder
########################################
class MultiColumnLabelEncoder:

  def __init__(self, columns, dtype=np.int32):
    self.columns = columns     # Array of columns name to encode
    self.encoder = {}     # save the label encoder

  def fit(self, data):
    new_data = data.copy()
    for col in self.columns:
        labelEncoder = LabelEncoder()
        labelEncoder.fit( new_data[col] )
        self.encoder[col] = labelEncoder
    return self

  def transform(self, data):
    transformed_data = data.copy()
    for col, labelEncoder in self.encoder.items():
        transformed_data[col] = labelEncoder.transform( transformed_data[col] )
    return transformed_data

  def inverse_transform(self, data):
    transformed_data = data.copy()
    for col, labelEncoder in self.encoder.items():
        transformed_data[col] = labelEncoder.inverse_transform( transformed_data[col] )
    return transformed_data


########################################
# One Hot Encoder
########################################
class MultiColumnOneHotEncoder:
    def __init__(self, columns):
        self.columns = columns     # Categorical columns
        self.encoder = None
        self.col_names = None     # Name of expanded columns

    def fit(self, data):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(data[self.columns])
        self.col_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, data):
        new_data = data.copy()
        ohe_cols = pd.DataFrame(
            self.encoder.transform( new_data[self.columns] ),
            columns=self.col_names,
            index=new_data.index
        ).astype(int)
        new_data = new_data.drop(columns=self.columns)
        return pd.concat([ohe_cols, new_data], axis=1)

    def inverse_transform(self, data):
        new_data = data.copy()
        num_part = new_data.drop(columns=self.col_names)
        categ_part = pd.DataFrame(
            self.encoder.inverse_transform( new_data[self.col_names] ),
            columns=self.columns,
            index=new_data.index
        )
        return pd.concat([num_part, categ_part], axis=1)


########################################
# K-Means Clustering
########################################
def kmeans_clustering(data):
    '''
    INPUT:
    data - data set to be clustered
    '''
    n_clusters = random.randint(5, 11)
    print('\nThe number of clusters is %d' % n_clusters)

    classifier = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5, random_state=71, max_iter=300, tol=0.001, verbose=False)
    classifier.fit(data)
    print('The final SSE is %.2f' % classifier.inertia_)

    return classifier.labels_, n_clusters


########################################
# Hierarchical Clustering
########################################
def hierarchical_clustering(data, isToBePlotted:bool):
    '''
    INPUT:
    data          - data set to be clustered
    isToBePlotted - True if there is the will to make plots
    '''
    Z = linkage(data, 'ward', metric='euclidean', optimal_ordering='true')
    max_d = find_max_d(Z)
    print('\nThe y value used for cutting the tree is %.2f' % max_d)

    # Graphical representation of the dendrogram
    if isToBePlotted:
        plt.figure(figsize=(20,15))
        dn = dendrogram(Z, no_plot=0)
        plt.tick_params('y', which='major', labelsize=15)
        plt.tick_params('x', which='major', labelsize=8)
        plt.xlabel('Distance')
        plt.axhline(max_d, color='black', ls='--')
        plt.grid()
        plt.savefig('/output/Clustering - Hierarchical dendrogram', dpi=150)
    
    labels = fcluster(Z, max_d, criterion='distance') - 1
    print('We got %d clusters' % (labels.max()+1))

    #centers = find_cluster_centers(data.to_numpy(), labels.max(), labels)
    #PCA_tSNE_visualization(data, 5, labels, PAL)

    return labels, labels.max()+1


########################################
# Find Max D
########################################
def find_max_d(Z, min_clusters=5, max_clusters=15, attempts=500):
    values = np.linspace(Z[:,2].min(), Z[:,2].max(), attempts)
    for value in values[::-1]:
        labels = fcluster(Z, value, criterion='distance')
        n_clusters = len(np.unique(labels))
        if min_clusters <= n_clusters <= max_clusters:
            return value
    
    return Z[:,2].max()


########################################
# Find Cluster Centers
########################################
def find_cluster_centers(data, K, labels):
    '''
    INPUT
    data    - data matrix for which to compute the proximity matrix
    K       - the expected number of clusters
    labels  - predicted labels from the clustering solution applied to data
    '''

    '''
    OUTPUT
    cluster_centers   - cluster centres from the clustering solution applied to data
    '''
    # Initialize the output
    cluster_centers = np.zeros((K, np.shape(data)[1]))   # np.shape(data)[1] = no. of attributes
    #print("%d centroids are being computed, as we have %d clusters." % (K, K) )

    for k in range(0, K):
        ind = np.array( np.where( labels == k ) )
        cluster_points = data[ind, :][0]
        cluster_centers[k,:] = np.mean(cluster_points, axis=0) # cluster_points.mean(axis=0)
        #print("The centroid of cluster %d has coordinates: " % (k), *cluster_centers[k,:].round(2))

    return cluster_centers


########################################
# PCA and tSNE 
########################################
def PCA_tSNE_visualization(data2visualize, NCOMP, LABELS, PAL, filename):
    '''
    INPUT
    data2visualize    - data matrix to visualize
    NCOMP             - no. of components to decompose the dataset during PCA
    LABELS            - labels given by the clustering solution
    PAL               - palette of colours to distinguish between clusters
    filename          - filename for saving the figure + output directory
    '''

    '''
    OUTPUT
    Two figures: one using PCA and one using tSNE
    '''

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=NCOMP)
    pca_result = pca.fit_transform(data2visualize)
    print('\nPCA: explained variation per principal component: {}'.format(pca.explained_variance_ratio_.round(2)))

    # tSNE
    from sklearn.manifold import TSNE
    #print('\nApplying tSNE...')
    np.random.seed(100)
    tsne = TSNE(n_components=2, verbose=0, perplexity=20, max_iter=300)
    tsne_results = tsne.fit_transform(data2visualize)

    # Plots
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Dimensionality reduction of the dataset', fontsize=16)

    # Plot 1: 2D image of the entire dataset
    ax1 = fig.add_subplot(121)
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], ax=ax1, hue=LABELS, palette=PAL)
    ax1.set_xlabel('Dimension 1', fontsize=10)
    ax1.set_ylabel('Dimension 2', fontsize=10)
    ax1.title.set_text('PCA')
    plt.grid()

    ax2= fig.add_subplot(122)
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], ax=ax2, hue=LABELS, palette=PAL)
    ax2.set_xlabel('Dimension 1', fontsize=10)
    ax2.set_ylabel('Dimension 2', fontsize=10)
    ax2.title.set_text('tSNE')
    plt.grid()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print('Figure of PCA and tSNE saved in output directory')