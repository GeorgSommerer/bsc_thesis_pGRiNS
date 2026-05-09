from scipy.cluster.hierarchy import single, complete, average, ward, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy import sparse

from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
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

from tqdm import tqdm

# Global variables necessary for memory-efficient multiprocessing: by setting the global variables in the parent function and then reading (but not changing!) them in the helper functions, they do not need to be copied many times

g_cluster_res = None
g_dist_matrix = None
g_adata_mean = None

def calc_mse_to_adata_mean(data_cell):
    return mean_squared_error(np.asarray(data_cell),g_adata_mean)

def calc_cosine_to_adata_mean(data_cell):
    return cosine_similarity(np.asarray(data_cell),g_adata_mean).T[0][0]

def calc_fcluster(height):
    return fcluster(g_cluster_res,t=height,criterion="distance")

def calc_silhouette_score(clustering):
    return silhouette_score(g_dist_matrix,np.asarray(clustering).squeeze())

def calc_silhouette_samples(clustering):
    return silhouette_samples(g_dist_matrix,np.asarray(clustering).squeeze())


def get_candidate_cells(grins_matrix : sparse.csc_matrix, adata_matrix : sparse.csc_matrix, metric : str = "MSE", use_mad : bool = False, times_dev : float = 10, candidate_limit : float = 0.1, num_processes : int = 10):
    """
    If experimental control data is available for the unperturbed run, makes a first selection to remove all cells that are too different from the control data mean.
    This is done by removing all GRiNS cells whose score (cosine similarity or MSE) is not within times_dev many MAD (or SD) of the mean of means of each gene in the control dataset.

    Parameters:
    -----------
    grins_matrix : sparse.csc_matrix
        A matrix containing the log1p normalized values from the simulated GRiNS data.
    adata_matrix : sparse.csc_matrix | None
        If experimental data is provided, this is the log1p layer of the adata object of one of the experimental datasets.
    metric : str, optional
        What metric should be used. Options include cosine similarity ("cosine") or MSE.
    use_mad : bool, optional
        If true, MAD is used as a deviation measure. If False (default), SD is used.
    times_dev : float, optional
        How many deviations the GRiNS scores can be away from the adata mean of means in order for the cell to be included in candidate_matrix.
    candidate_limit : float, optional
        What percentage of cells should be kept at most. Combinable with times_dev.
    num_processes : int, optional
        The number of processes for the multiprocessing of the calculations.

    Returns:
    --------
    candidate_matrix : sparse.csc_matrix
        A matrix where all cells that are too dissimilar from the control data are removed.
    """
    adata_mean = np.asarray(np.mean(adata_matrix,axis=0))
    if num_processes > 1:
        global g_adata_mean
        g_adata_mean = adata_mean
    
    # General function to calculate the metric between each cell and the adata mean
    def calc_metric_for_cells(data_matrix):
        if metric == "MSE":
            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    data_score_list = np.array(list(tqdm(pool.imap(calc_mse_to_adata_mean,data_matrix),total=data_matrix.shape[0])))
            else:
                data_score_list = np.array([mean_squared_error(np.asarray(data_matrix[i,:]),adata_mean) for i in tqdm(range(data_matrix.shape[0]))])

        elif metric == "cosine":
            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    data_score_list = np.array(list(tqdm(pool.imap(calc_cosine_to_adata_mean,data_matrix),total=data_matrix.shape[0])))
            else:
                data_score_list = np.array([cosine_similarity(np.asarray(data_matrix[i,:]),adata_mean).T[0][0] for i in tqdm(range(data_matrix.shape[0]))])
        return data_score_list

    print(f"Getting {metric} scores for adata...")
    adata_scores = calc_metric_for_cells(adata_matrix)
    adata_score_mean = np.mean(adata_scores)
    if use_mad:
        adata_score_dev = np.median(np.abs(adata_scores-adata_score_mean))
    else:
        adata_score_dev = np.sqrt(np.var(adata_scores))

    print(f"Getting {metric} scores for GRiNS...")
    grins_scores = calc_metric_for_cells(grins_matrix)

    print("Getting candidate cells...")
    candidate_cell_mask = np.asarray((grins_scores>=adata_score_mean-times_dev*adata_score_dev) & (grins_scores<=adata_score_mean+times_dev*adata_score_dev)) # Get all grins cells with values within the range of [mean-x*sd, mean+x*sd]

    if sum(candidate_cell_mask)<10:
        print("Warning: <10 of cells were kept. Try using a higher times_dev or turning use_mad off.")
    if candidate_limit is not None and sum(candidate_cell_mask)>candidate_limit*grins_matrix.shape[0]:
        idx = np.random.choice(np.where(candidate_cell_mask == True)[0],size=int(sum(candidate_cell_mask)-candidate_limit*grins_matrix.shape[0]),replace=False) # Set enough cells to False so that only 10% remain True
        candidate_cell_mask[idx] = False 

    print(f"{100*sum(candidate_cell_mask)/grins_matrix.shape[0]}% ({sum(candidate_cell_mask)}) of cells were kept.")
            
    return candidate_cell_mask



def calc_clusters(candidate_matrix : sparse.csc_matrix, num_processes : int = 10, clustering_method : str = "average", sample_clusterings : bool = False, cluster_size : int = 5) -> [sparse.csc_matrix, np.array(float), np.matrix(int)]:
    """
    Calcuates a distance matrix between the rows of the log1p grins matrix and then performs hierarchical on it.
    Then, it finds all distinct clusterings (ways to assign each cell to a cluster) from the resulting dendrogram (cluster_res).

    Parameters:
    -----------
    candidate_matrix : sparse.csc_matrix
        A matrix containing the log1p normalized values from the simulated GRiNS data. Possibly subset so that all dissimilar cells are removed.
    num_processes : int, optional
        The number of processes used for multiprocessing. Defaults to 10. Set to 1 for no multiprocessing.
    clustering_method : str, optional
        The method used for clustering the cells. Options are "single", "complete", "average", and "ward".
    sample_clusterings : bool, optional
        Whether or not the clusterings at all possible heights of the dendrogram should be evaluated. Defaults to False. If True, only log2 n many heights are sampled and complexity of find_all_params is reduced from O(n^3) to O(n^2log n).
    cluster_size : int, optional
        How many cells a cluster must include in order to have a metric computed. Defaults to 5.

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
    dist = pdist(candidate_matrix)
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

    dendrogram(cluster_res)
    plt.savefig("Dendrogram.png")

    # Get all heights at which unique clusterings occur:
    dist_matrix = sparse.csc_matrix(squareform(dist)) 
    heights = np.unique(np.round(cluster_res[:,2]-1e-5,decimals=5)) # Without subtracting 1e-5, some heights would be rounded to the same number
    #heights = np.unique(cluster_res[:,2])
    if sample_clusterings:
        heights = heights[np.linspace(0,len(heights),int(np.log2(len(heights))))]
    print("Getting clusters")
    # Forms flat clusterings from the structure within cluster_res
    if num_processes > 1:
        global g_cluster_res
        g_cluster_res = cluster_res
        with Pool(processes=num_processes) as pool:
            clusterings = np.matrix(list(tqdm(pool.imap(calc_fcluster,heights),total=len(heights))))#,total=len(heights))
    else:
        clusterings = np.empty((0,dist_matrix.shape[0]),int)
        for height in tqdm(heights):
            clustering = fcluster(cluster_res,t=height,criterion="distance") 
            clusterings = np.append(clusterings,[clustering],axis=0)

    # Remove single cluster or n clusters (no silhouette score possible to calculate)
    if len(np.unique(clusterings[0])) == dist_matrix.shape[0]:
        heights = heights[1:]
        clusterings = clusterings[1:]
    if len(np.unique(clusterings[-1])) == 1:
        heights = heights[:-1]
        clusterings = clusterings[:-1]

    print("Calculating silhouette scores")
    if num_processes < 1:
        global g_dist_matrix
        g_dist_matrix = dist_matrix
        with Pool(processes=num_processes) as pool:
            sil_scores = np.array(list(tqdm(pool.imap(calc_silhouette_score,clusterings),total=clusterings.shape[0])))
        with Pool(processes=num_processes) as pool:
            sil_samples = np.matrix(list(tqdm(pool.imap(calc_silhouette_samples,clusterings),total=clusterings.shape[0])))
    else:
        sil_scores = np.array([silhouette_score(dist_matrix,np.asarray(clustering).squeeze()) for clustering in tqdm(clusterings)])
        sil_samples = np.matrix([silhouette_samples(dist_matrix,np.asarray(clustering).squeeze()) for clustering in tqdm(clusterings)])

    plt.scatter(heights,sil_scores,s=1)
    plt.savefig("Sil.png")

    high_sil_indices = find_peaks(sil_scores) # Get all significant silhouette score indices <- maybe add prominence?

    # Go through all clusterings with significant silhouette scores
    best_cluster_total = None
    best_sil_cluster = -1
    for i in range(high_sil_indices):
        clustering_dict = pd.DataFrame({"Cluster":clusterings[i]}).groupby(by="Cluster").indices # Get the indices of clusters for each cluster label in the clustering
        for cluster in clustering_dict:
            if len(clustering_dict[cluster])<cluster_size:
                continue
            sil_mean_cluster = np.mean(sil_samples[clustering_dict[cluster]]) # Calculate the mean silhouette score within each cluster if it is big enough
            if sil_mean_cluster > best_sil_cluster:
                best_cluster_total = [i,int(cluster)]
                best_sil_cluster = sil_mean_cluster
    
    return clusterings, best_cluster_total



def set_param_ranges(grins_data,clusters_and_metrics,best_cluster_results):
    # From the best cluster, get the indices of the init/params corresponding to this and from this information, build up a way to reduce param ranges
    # e.g. a distribution
    
    best_cluster_rows = list(clusters_and_metrics.keys())[best_cluster_results["Bestest_Clustering"]]==best_cluster_results["Best_Cluster_Within"]
    best_cluster_indices = grins_data.obs_names[best_cluster_rows]
    pass



def main(grn_file,grins_data = None, adata_file = None,pert_file = None, clustering_method = "average",sample_clusterings : bool = False, metric : str = "MSE", use_mad : bool = False, time_dev : float = 10, candidate_limit : float = 0.1, cluster_size : int = 5, num_processes : int = 10):
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

            if adata_matrix is None:
                idx = np.random.choice(list(range(grins_matrix.shape[0])),size=int(candidate_limit*grins_matrix.shape[0]),replace=False) # Get candidate_limit % of cells
                candidate_cell_mask = np.asarray(pd.Series(np.arange(grins_matrix.shape[0])).isin(idx)) 
            else:
                candidate_cell_mask = get_candidate_cells(grins_matrix,adata_matrix,metric=metric,use_mad=use_mad,times_dev=times_dev,candidate_limit=candidate_limit,num_processes=num_processes)


            clusterings, best_cluster_total = calc_clusters(grins_matrix,num_processes=num_processes,clustering_method=clustering_method,sample_clusterings=sample_clusterings, cluster_size=cluster_size)

            # Save first results:
            pickle.dump({"Pert":pert,"Clusterings":clusterings,"Best_Cluster":best_cluster_total},handle,protocol=pickle.HIGHEST_PROTOCOL)
            

if __name__ == "__main__":
    # Example: python3 -u pgrins_narrow_params.py projectname -e pseq_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e", "--experimental", help="The name of the experimental control data used for comparison")
    parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
    parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")
    parser.add_argument("--clustering_method",help="Which method should be used to cluster the distance matrix. Options are single, complete, average (default) and ward.")
    parser.add_argument("--sample_clusterings",action="store_true",help="Whether or not all distinct clusterings at all heights should be calculated. If turned on, runtime is reduced at risk of not discovering the best cluster.")
    parser.add_argument("--metric",help="Which metric should be used to calculate the score for each cluster during the unperturbed run. Options are mean squared error (MSE, default) and cosine similarity (cosine).")
    parser.add_argument("--cluster_size",type=int,help="The minimum size of the clusters in order to be compared to the control data. Defaults to 5.")
    parser.add_argument("--num_processes", type=int,help="The number of processes used for silhouette score computation; defaults to 10.")
    parser.add_argument("--use_mad",action="store_true",help="Turn this option on if the MAD should be used as a deviation measure instead of SD for get_candidate_cells.")
    parser.add_argument("--times_dev",type=float,help="How many deviations away the score of the GRiNS data of a cell can be from the mean of means of control data in order to not be discarded. Defaults to 10.")
    parser.add_argument("--candidate_limit",type=float,help="What percentage of candidate cells should be kept at most. Defaults to 10%. Combinable with times_dev.")

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
    if args.use_mad:
        kwargs["use_mad"] = args.use_mad
    if args.times_dev:
        kwargs["times_dev"] = args.times_dev
    if args.candidate_limit:
        kwargs["candidate_limit"] = args.candidate_limit

    main(grn_file, adata_file=adata_file, pert_file=pert_file, **kwargs)