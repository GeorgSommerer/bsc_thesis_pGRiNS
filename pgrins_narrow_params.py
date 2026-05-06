from scipy.cluster.hierarchy import single, complete, average, ward, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy import sparse

from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

import scanpy as sc
import hdf5plugin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
import argparse
import pickle
import timeit

# Global variables necessary for memory-efficient multiprocessing: by setting the global variables in the parent function and then reading (but not changing!) them in the helper functions, they do not need to be copied many times

g_cluster_res = None

g_dist_matrix = None

g_adata_mean = None
g_grins_matrix = None
g_cluster_size = None
g_metric = None

def calc_fcluster(height):
    return fcluster(g_cluster_res,t=height,criterion="distance")

def calc_silhouette_score(clustering):
    return silhouette_score(g_dist_matrix,clustering)

def calc_metric_on_clustering(clustering):
    if g_metric == "MSE":
        best_score_clustering = np.inf
    elif g_metric == "cosine":
        best_score_clustering = 0
    best_cluster_in_clustering = None

    # Get all clusters in clustering with more than cluster_size cells
    clustering_dict = pd.DataFrame({"Cluster":clustering}).groupby(by="Cluster").indices
    metric_col = np.zeros((g_grins_matrix.shape[0]))

    for cluster in clustering_dict:
        cluster_indices = clustering_dict[cluster]
        if len(cluster_indices) < g_cluster_size:
            continue

        # Calculate score between cluster_mean and adata_mean
        cluster_mean = np.mean(g_grins_matrix[cluster_indices],axis=0)
        if g_metric == "MSE":
            score = mean_squared_error(np.asarray(cluster_mean),np.asarray(g_adata_mean))
            if score < best_score_clustering:
                best_score_clustering = score
                best_cluster_in_clustering = cluster
        elif g_metric == "cosine":
            score = cosine_similarity(np.asarray(cluster_mean),np.asarray(g_adata_mean)).T[0][0]
            if score > best_cluster_score:
                best_score_clustering = score
                best_cluster_in_clustering = cluster
        metric_col[cluster] = score

    return metric_col, best_cluster_in_clustering, best_score_clustering



def calc_clusters(grins_matrix : sparse.csc_matrix, num_processes : int = 10, clustering_method : str = "average", sample_clusterings : bool = False) -> [sparse.csc_matrix, np.array(float), np.matrix(int)]:
    """
    Calcuates a distance matrix between the rows of the log1p grins matrix and then performs UPGMA (average linkage clustering) on it, since the authors of the original Racipe paper employ a similar method.
    Then, it finds all distinct clusterings (ways to assign each cell to a cluster) from the resulting dendrogram (cluster_res).

    Parameters:
    -----------
    grins_matrix : sparse.csc_matrix
        A matrix containing the log1p normalized values from the simulated GRiNS data.
    num_processes : int, optional
        The number of processes used for multiprocessing. Defaults to 10. Set to 1 for no multiprocessing.
    clustering_method : str, optional
        The method used for clustering the cells. Options are "single", "complete", "average", and "ward".
    sample_clusterings : bool, optional
        Whether or not the clusterings at all possible heights of the dendrogram should be evaluated. Defaults to False. If True, only log2 n many heights are sampled and complexity of find_all_params is reduced from O(n^3) to O(n^2log n).

    Returns:
    --------
    dist_matrix : sparse.csc_matrix
        A sparse distance matrix between all cells.
    heights : np.array(float)
        A list of the lowest heights at which distinct clusters occur
    clusterings : np.matrix(int)
        For each height, this matrix assigns each cell to a certain cluster.
    """

    # Cluster:
    print("Calculating distance matrix")
    dist = pdist(grins_matrix)
    print("Clustering")
    # Column 1 and 2 of cluster_res contain the indices of nodes clustered together; column 3 contains the height of the newly formed cluster; column 4 contains its size
    if clustering_method == "single":
        cluster_res = single(dist) 
    elif clustering_method == "complete":
        cluster_res = complete(dist) 
    elif clustering_method == "average":
        cluster_res = average(dist) 
    elif clustering_method == "ward":
        cluster_res = ward(dist) 

    # Get all heights at which unique clusterings occur:
    dist_matrix = sparse.csc_matrix(squareform(dist)) 
    #heights = np.unique(np.round(cluster_res[:,2]-1e-5,decimals=5)) # Without subtracting 1e-5, some heights would be rounded to the same number
    heights = np.unique(cluster_res[:,2])
    if sample_clusterings:
        heights = heights[np.linspace(0,len(heights),int(np.log2(len(heights))))]
    print("Getting clusters")
    # Forms flat clusterings from the structure within cluster_res
    if num_processes > 1:
        global g_cluster_res
        g_cluster_res = cluster_res
        with Pool(processes=num_processes) as pool:
            clusterings = pool.map(calc_fcluster,heights)
    else:
        clusterings = np.empty((0,dist_matrix.shape[0]),int)
        for height in heights:
            clustering = fcluster(cluster_res,t=height,criterion="distance") 
            clusterings = np.append(clusterings,[clustering],axis=0)


    # Remove single cluster or n clusters (no silhouette score possible to calculate)
    if len(np.unique(clusterings[0])) == dist_matrix.shape[0]:
        heights = heights[1:]
        clusterings = clusterings[1:]
    if len(np.unique(clusterings[-1])) == 1:
        heights = heights[:-1]
        clusterings = clusterings[:-1]
    
    return dist_matrix, heights, clusterings


    

def find_best_params(grins_matrix : sparse.csc_matrix, adata_matrix : sparse.csc_matrix, clusterings : np.matrix(int), dist_matrix : sparse.csc_matrix, cluster_size : int = 5, metric : str = "MSE", num_processes : int = 10):
    """
    This function first calculates the silhouette score of each clustering.
    Then, clusterings with silhouette scores that are the highest within their vicinity (i.e., they are prominent and the highest within +- 100 other clusterings) are kept.
    This means that the best clusterings, which are analyzed afterwards, contain only clusterings where the clusters are well-separated relative to other clusters with a similar height.

    If experimental data is provided, each cluster in each clustering among these best clusterings has its mean calculated, which is then compared to the mean of the experimental control data.
        Among all these clusters, the one with the highest cosine similarity relative to the experimental data is kept and used to narrow down the parameter ranges.
    If no experimental data is provided, the biggest cluster from the clustering with the highest silhouette score is returned.

    Parameters:
    -----------
    grins_matrix : sparse.csc_matrix
        A matrix containing the log1p normalized values from the simulated GRiNS data.
    adata_matrix : sparse.csc_matrix | None
        If experimental data is provided, this is the log1p layer of the adata object of one of the experimental datasets.
    clusterings : np.matrix(int)
        For each height, this matrix assigns each cell to a certain cluster.
    dist_matrix : sparse.csc_matrix
        A sparse distance matrix between all cells. Only required for perturbed runs.
    cluster_size : int, optional
        How many cells a cluster must include in order to have a metric computed. Defaults to 5.
    metric : str, optional
        What metric should be used. Options include cosine similarity ("cosine") or MSE.
    num_processes : int, optional
        The number of processes for the multiprocessing of the calculations.

    Returns:
    --------
    sil_scores : np.array[float]
        The silhouette scores for each height.
    cluster_metrics : list[list[float]]
        For ctrl: A list of lists where cluster_metrics[i][j] contains the cosine similarity of the mean expression of all the cells labeled j in the clustering i to the adata unperturbed mean.
            Also, len(cluster_metrics[i]) is equivalent to np.unique(clusterings[i]).
        For pert: A list where cluster_metrics[i] contains the silhouette score of the clustering i.
    best_cluster_total : tuple[int,int]
        A tuple where the first element contains the index of the clustering with the best overall cluster in clusterings, and the second element is the label this cluster has.
        This means that clusterings[best_cluster_total[0]]==best_cluster_total[1] gives a mask that if applied on the obs_names of grins_data, returns the param/IC combinations for cells in this best cluster.
    """


    # Get the cluster that is the closest to the control mean in some way:
    if adata_matrix is not None:
        adata_mean = np.mean(adata_matrix,axis=0)

        if num_processes > 1:
            global g_metric
            g_metric = metric

            global g_adata_mean
            g_adata_mean = adata_mean

            global g_cluster_size
            g_cluster_size = cluster_size

            global g_grins_matrix
            g_grins_matrix = grins_matrix
        else:
            metric_matrix = np.zeros((clusterings.shape[0],grins_matrix.shape[0])) # Matrix of shape clustering x cells (max no. of clusters)
            best_cluster_total = None # Will have shape [index in clusterings with highest score, cluster label within that clustering with the highest score]
            
            if metric == "MSE":
                best_cluster_score = np.inf
            elif metric == "cosine":
                best_cluster_score = 0
        
        print(f"Going through {len(clusterings)} clusterings")
        t1 = timeit.default_timer()
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                metric_cols, best_cluster_in_clusterings, best_score_clusterings = zip(*pool.map(calc_metric_on_clustering,clusterings))
            metric_matrix = np.vstack(metric_cols)
            if metric == "MSE":
                best_score_total = np.min(best_score_clusterings) # Get best score overall
                best_cluster_total = [np.argmin(best_score_clusterings),best_cluster_in_clusterings[np.argmin(best_score_clusterings)]] # Get the clustering with the highest score, and the cluster inside it with that score
            elif metric == "cosine":
                best_score_total = np.max(best_score_clusterings)
                best_cluster_total = [np.argmax(best_score_clusterings),best_cluster_in_clusterings[np.argmax(best_score_clusterings)]]
        else:
            for i, clustering in zip(range(len(clusterings)),clusterings):
                # Get all clusters in clustering with more than cluster_size cells
                clusters = sorted([c for c in np.unique(clustering) if sum(clustering==c)>=cluster_size])
                for cluster in clusters:
                    # Calculate score between cluster_mean and adata_mean
                    cluster_mean = np.mean(grins_matrix[clustering == cluster],axis=0)
                    if metric == "MSE":
                        score = mean_squared_error(np.asarray(cluster_mean),np.asarray(adata_mean))
                        if score < best_cluster_score:
                            best_cluster_score = score
                            best_cluster_total = [i,cluster]
                    elif metric == "cosine":
                        score = cosine_similarity(np.asarray(cluster_mean),np.asarray(adata_mean)).T[0][0]
                        if score > best_cluster_score:
                            best_cluster_score = score
                            best_cluster_total = [i,cluster]
                    metric_matrix[i,cluster] = score
        t2 = timeit.default_timer()
        print(f"Time required for calculating scores for each cluster (not parallelized): {t2-t1}")
        metric_matrix = sparse.csr_matrix(metric_matrix)

    else:
        # Calculate Silhouette scores for each clustering
        print("Calculating silhouette scores")
        global g_dist_matrix # Add non-mp option (together with silhouette_samples)
        g_dist_matrix = dist_matrix
        with Pool(processes=num_processes) as pool:
            sil_scores = pool.map(calc_silhouette_score,clusterings)
        pool.close()

        # Choose the biggest cluster from the clustering with the highest silhouette score (i.e. the most distinct large cluster) as the best one
        best_clustering = clusterings[np.argmax(sil_scores)]
        largest_cluster = np.argmax(np.bincount(best_clustering))

        metric_matrix = sil_scores
        best_cluster_total = [best_clustering, largest_cluster]

    return metric_matrix, best_cluster_total



def set_param_ranges(grins_data,clusters_and_metrics,best_cluster_results):
    # From the best cluster, get the indices of the init/params corresponding to this and from this information, build up a way to reduce param ranges
    # e.g. a distribution
    
    best_cluster_rows = list(clusters_and_metrics.keys())[best_cluster_results["Bestest_Clustering"]]==best_cluster_results["Best_Cluster_Within"]
    best_cluster_indices = grins_data.obs_names[best_cluster_rows]
    pass



def main(grn_file,grins_data = None, adata_file = None,pert_file = None, clustering_method = "average",sample_clusterings : bool = False, metric : str = "MSE", cluster_size : int = 5, num_processes : int = 10):
    assert metric in ["MSE","cosine"]
    assert clustering_method in ["single","complete","average","ward"]

    if pert_file is not None:
        pert_list = pgrins_prepare_input.extract_pert_info(grn_file,pert_file)
    else:
        pert_list = [{"ctrl":0}]

    if grins_data is None: # grins_data can either be provided in the pipeline, or loaded if the file is run standalone
        grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/perturb_norm.h5ad")
        grins_data = grins_data[:,grins_data.var["pct_dropout_by_counts"]<90]

    if isinstance(adata_file,str): # adata can either be provided in the pipeline, or loaded if the file is run standalone, or be None
        adata = sc.read_h5ad(f"Data/Experimental/{adata_file}/perturb_subset_{grn_file}.h5ad")
        adata = adata[:,adata.var["pct_dropout_by_counts"]<90]
    else:
        adata = adata_file # either the adata object, or None

    with open(f"Data/Projects/{grn_file}/cluster_res_{clustering_method}.pickle","wb") as handle:
        for pert in pert_list:
            pert = "_".join(list(pert.keys()))
            if adata is not None: # if not none: find common genes again
                print("Subsetting datasets")
                common_genes = list(set(adata.var_names) & set(grins_data.var_names))
                grins_matrix = grins_data[grins_data.obs["perturbation"]==pert][:,common_genes].layers["log1p"].todense()
                adata_matrix = adata[adata.obs["perturbation"]==pert][:,common_genes].layers["log1p"].todense()
            else:
                grins_matrix = grins_data.layers["log1p"].todense()
                adata_matrix = None

            dist_matrix, heights, clusterings = calc_clusters(grins_matrix,num_processes=num_processes,clustering_method=clustering_method,sample_clusterings=sample_clusterings)
            metric_matrix, best_cluster_total = find_best_params(grins_matrix,adata_matrix,clusterings,dist_matrix,cluster_size=cluster_size,metric=metric,num_processes=num_processes)

            # Save first results:
            pickle.dump({"Pert":pert,"Heights":heights,"Clusterings":clusterings,"Cluster_Metrics":metric_matrix,"Best_Cluster":best_cluster_total},handle,protocol=pickle.HIGHEST_PROTOCOL)
            

if __name__ == "__main__":
    # Example: python3 -u pgrins_narrow_params.py projectname -e pseq_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e", "--experimental", help="The name of the experimental control data used for comparison")
    parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
    parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")
    parser.add_argument("--clustering_method",help="Which method should be used to cluster the distance matrix. Options are single, complete, average (default) and ward.")
    parser.add_argument("--sample_clusterings",help="Whether or not all distinct clusterings at all heights should be calculated. Defaults to False. If True, runtime is reduced at risk of not discovering the best cluster.")
    parser.add_argument("--metric",help="Which metric should be used to calculate the score for each cluster during the unperturbed run. Options are mean squared error (MSE, default) and cosine similarity (cosine).")
    parser.add_argument("--cluster_size",type=int,help="The minimum size of the clusters in order to be compared to the control data. Defaults to 5.")
    parser.add_argument("--num_processes", type=int,help="The number of processes used for silhouette score computation; defaults to 10.")
    
    args = parser.parse_args()

    grn_file = args.grn
    if args.experimental:
        adata_file = args.experimental
    else:
        adata_file = None

    if args.use_perts:
        if args.pert_file:
            pert_file = args.pert_file
        else:
            pert_file = f"{grn_file}_perts"
    else:
        pert_file = None

    kwargs = {}
    if args.clustering_method:
        kwargs["clustering_method"] = args.clustering_method
    if args.sample_clusterings:
        kwargs["sample_clusterings"] = args.sample_clusterings
    if args.metric:
        kwargs["metric"] = args.metric
    if args.cluster_size:
        kwargs["cluster_size"] = args.cluster_size
    if args.num_processes:
        kwargs["num_processes"] = args.num_processes

    main(grn_file, adata_file=adata_file, pert_file=pert_file, **kwargs)