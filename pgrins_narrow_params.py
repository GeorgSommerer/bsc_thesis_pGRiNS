from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
def get_dist_matrix(grins_data):
    grins_matrix = grins_data.layers["missing"].todense().T
    dist_matrix = pairwise_distances(grins_matrix)
    return dist_matrix

def calc_clusters(dist_matrix):
    # How to deal with multiple control datasets? Maybe calc clusters for all of them and find the best mean?
    # Probably also need to subset the genes in GRN for each dataset
    # Use average linkage because done so in Racipe paper
    # Use Silhouette Score to find distinct clusters above a certain size first, then compare them to dataset (optional)
    n_clusters = dist_matrix.shape[0]-1
    cluster_labels = []
    silhouette_scores = []
    while n_clusters > 0:
        break#dist_matrix = 
    pass

def reduce_param_ranges():
    # Get param ranges of best cluster as .csv
    # 
    pass

def set_param_ranges():
    pass

if __name__ == "__main__":
    main()