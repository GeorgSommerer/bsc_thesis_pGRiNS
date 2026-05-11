from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import scanpy as sc
import anndata as ad
import hdf5plugin
import igraph

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
import argparse
import timeit
from tqdm import tqdm

from Prep_Data import pgrins_prepare_output

# Global variables necessary for memory-efficient multiprocessing: by setting the global variables in the parent function and then reading (but not changing!) them in the helper functions, they do not need to be copied many times

g_adata_mean = None

def calc_mse_to_adata_mean(data_cell):
    return mean_squared_error(np.asarray(data_cell),g_adata_mean)

def calc_cosine_to_adata_mean(data_cell):
    return cosine_similarity(np.asarray(data_cell),g_adata_mean).T[0][0]



def get_candidate_cells(grins_data : sparse.csc_matrix, adata : sparse.csc_matrix, layer="log1p",metric : str = "MSE", use_mad : bool = False, times_dev : float = 10, candidate_limit : float = 1.0, num_processes : int = 10):
    """
    If experimental control data is available for the unperturbed run, makes a first selection to remove all cells that are too different from the control data mean.
    This is done by removing all GRiNS cells whose score (cosine similarity or MSE) is not within times_dev many MAD (or SD) of the mean of means of each gene in the control dataset.

    Parameters:
    -----------
    grins_data : ad.AnnData
        The simulated GRiNS data.
    adata : ad.AnnData
        If experimental data is provided, this is the adata object of one of the experimental datasets.
    layer : str, optional
        The layer of both grins_data and adata to perform the calculation on. If none, done on raw data.
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
    if layer is None:
        adata_mean = np.asarray(np.mean(adata.X,axis=0))
    else:
        adata_mean = np.asarray(np.mean(adata.layers["log1p"],axis=0))
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
    if layer is None:
        adata_scores = calc_metric_for_cells(adata.X.todense())
    else:
        adata_scores = calc_metric_for_cells(adata.layers["log1p"].todense())
    adata_score_mean = np.mean(adata_scores)

    if use_mad:
        adata_score_dev = np.median(np.abs(adata_scores-adata_score_mean))
    else:
        adata_score_dev = np.sqrt(np.var(adata_scores))

    print(f"Getting {metric} scores for GRiNS...")
    if layer is None:
        grins_scores = calc_metric_for_cells(grins_data.X.todense())
    else:
        grins_scores = calc_metric_for_cells(grins_data.layers[layer].todense())

    print("Getting candidate cells...")
    candidate_cell_mask = np.asarray((grins_scores>=adata_score_mean-times_dev*adata_score_dev) & (grins_scores<=adata_score_mean+times_dev*adata_score_dev)) # Get all grins cells with values within the range of [mean-x*sd, mean+x*sd]

    if sum(candidate_cell_mask)<10:
        print("Warning: <10 of cells were kept. Try using a higher times_dev or turning use_mad off.")
    if candidate_limit is not None and sum(candidate_cell_mask)>candidate_limit*grins_data.shape[0]:
        idx = np.random.choice(np.where(candidate_cell_mask == True)[0],size=int(sum(candidate_cell_mask)-candidate_limit*grins_data.shape[0]),replace=False) # Set enough cells to False so that only 10% remain True
        candidate_cell_mask[idx] = False 

    print(f"{100*sum(candidate_cell_mask)/grins_data.shape[0]}% ({sum(candidate_cell_mask)}) of cells were kept.")
            
    return candidate_cell_mask



def get_weights(grins_data : ad.AnnData) -> dict[str,list[float]]:
    """
    Calculates the weights mostly as proposed by Meija et al.
    First, the t-scores are calculated for all genes, but against the GRiNS control cells.
    Then, an absolute value transformation and a min-max transformation to [0,1] are applied.
    These weights are then squared and normalized to add up to 1.

    Parameters:
    -----------
    grins_data : ad.AnnData
        The simulated GRiNS data.

    Returns:
    --------
    grins_data : ad.AnnData
        Same as the input data, but adata.uns contains a dictionary with a list of weights for each perturbation set.
    """

    sc.tl.rank_genes_groups(
        grins_data,
        groupby = 'perturbation',
        method = 't-test_overestim_var',
        reference = 'ctrl',
        layer=layer)
    
    weights_dict = {}
    for pert in gt_index_dict.keys():
        # norm_data.uns["rank_genes_groups"] contains the p values for the DEGs of each perturbation
        pert_weights = this_norm_data.uns["rank_genes_groups"]["scores"][pert]
        pert_weights = np.abs(pert_weights) # Absolute value transformation
        pert_weights = (pert_weights-min(pert_weights))/(max(pert_weights)-min(pert_weights)+1e-8) # Min-max transformation to [0,1]
        pert_weights = pert_weights**2 # Squaring
        pert_weights = pert_weights/sum(pert_weights) # Normalization

        weights_dict[pert] = pert_weights
    
    return weights_dict



def calc_clusters(grins_data : ad.AnnData, adata : ad.AnnData = None, grins_ctrl_data : ad.AnnData = None, layer = "missing_log1p", min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10, deg_imp : float = 0.5) -> ad.AnnData:
    """
    Clusters the data by performing PCA -> UMAP -> Leiden, then getting large clusters with high silhouette scores and (if adata is provided) finding the one closest to adata.
    For the control cells, this is done by calculating the MSE between the mean expression of each clusters and the experimental control mean (if provided, otherwise biggest cluster)
    For the perturbations, an MSE based metric is calculated, inspired by a paper from Meija et al.:
        First, weights w are calculated based on the t scores of the DEGs between the whole perturbation and the synthetic control cells.
        Then in order to maximize distance for the DEGs and minimize for the non-DEGs, (1-a)(1-w)MSE(nonDEG)-awMSE(DEG) is calculated, which is to be minimized.
        a acts as a balancing hyperparameter between the two terms (is usually 1/2).

    Parameters:
    -----------
    grins_data : ad.AnnData
        The generated GRiNS data.
    adata : ad.AnnData, optional
        The experimental control data.
    grins_ctrl_data : ad.AnnData, optional
        Already clustered control data for the perturbed run.
    layer : str, optional
        The layer of grins_data on which to perform the operations. Defaults to missing_log1p (log1p with missingness layer applied).
    min_num_pcs : int, optional
        The minimal number of principal components to use for UMAP. Defaults to 10.
    min_cluster_size_pct : float, optional
        The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 1%.
    max_num_clusters : int, optional
        The maximal number of clusters considered as best cluster. Defaults to 10.
    deg_imp : float, optional
        How much importance the MSE of the DEGs should have for the pert score relative to the nonDEG MSE. Defaults to 0.5 (same importance).

    Returns:
    --------
    grins_data : ad.AnnData
        The original grins_data, but with additional columns and data indicating which cluster is the best.
    
    """
    
    print("Running PCA")
    sc.pp.pca(grins_data, svd_solver="arpack",layer=layer)

    # Number of PCs to keep:
    cum_pca = np.cumsum(grins_data.uns["pca"]["variance_ratio"])
    redundant_pcs = np.where(np.array([cum_pca[i+1]/cum_pca[i] for i in range(len(cum_pca)-1)])<1.01)[0] # Find the PC where the increase in the cumulative elbow plot is less than 1%
    if len(redundant_pcs)==0:
        n_pcs = len(cum_pca)
    else:
        n_pcs = max(min_num_pcs,redundant_pcs[0]) # Have at least min_num_pcs many PCs

    print(f"Running UMAP using {n_pcs} PCs")
    sc.pp.neighbors(grins_data,use_rep="X_pca",n_pcs=n_pcs)
    sc.tl.umap(grins_data)
    print("Clustering")
    sc.tl.leiden(grins_data, flavor="igraph")

    cluster_labels = grins_data.obs["leiden"]
    cluster_counts = cluster_labels.value_counts()
    clusters = np.array(list(cluster_counts.keys()))

    print(f"Calculating Silhouette score for {len(clusters)} clusters")
    sample_silhouette_values = silhouette_samples(grins_data.obsm["X_umap"], cluster_labels)
    grins_data.obs["silhouette_samples"] = sample_silhouette_values

    # Get large, homogenous clusters
    best_clusters = np.array(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys())) # At least min_cluster_size_pct of total cells in cluster
    best_clusters = clusters[np.argsort([np.mean(sample_silhouette_values[cluster_labels==c]) for c in best_clusters])[::-1][:max_num_clusters]] # max_num_clusters clusters with highest silhouette score

    print(f"Calculating MSES for {len(best_clusters)} clusters")
    if grins_ctrl_data is None:
        # Get the MSEs between the adata control mean and each cluster mean:
        adata_mean = np.asarray(np.mean(adata.layers["log1p"],axis=0))
        mse_results = [
            mean_squared_error(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers[layer],axis=0)),adata_mean)
            for cluster in best_clusters
        ]
    else:
        degs = grins_data[grins_data.var["is_DEG"]]
        ndegs = grins_data[~grins_data.var["is_DEG"]]
        ctrl_mean = np.asarray(np.mean(grins_ctrl_data.layers[layer],axis=0))
        mse_results = [
            (1-deg_imp)*np.dot(np.asarray(1-ndegs.var["weights"]),mean_squared_error(np.asarray(np.mean(ndegs[cluster_labels==cluster].layers[layer],axis=0)),adata_mean))-deg_imp*np.dot(np.asarray(degs.var["weights"]),mean_squared_error(np.asarray(np.mean(degs[cluster_labels==cluster].layers[layer],axis=0)),adata_mean))
            for cluster in best_clusters
        ]

    # Get the cells in GRiNS in the cluster with the lowest MSE:
    best_cluster_overall = best_clusters[np.argmin(mse_results)]
    grins_data.obs["best_cells"] = cluster_labels==best_cluster_overall

    print(f"{sum(grins_data.obs["best_cells"])}/{grins_data.shape[0]} cells retained")
    grins_data.uns["clustering_results"] = {}
    grins_data.uns["clustering_results"]["best_clusters"]=best_clusters
    grins_data.uns["clustering_results"]["scores"]=mse_results
    
    # Optional saving for plotting
    
    print("Saving file")
    ad.settings.allow_write_nullable_strings = True
    grins_data.write_h5ad(
        f"Data/Projects/{grn_file}/perturb_norm_{grins_data.obs["perturbation"][0]}_clustered.h5ad",
        compression=hdf5plugin.FILTERS["zstd"]
    )    
    
    
    return np.asarray(grins_data.obs["best_cells"])



def main(grn_file,grins_data = None, adata_file = None,pert_file = None, metric : str = "MSE", use_mad : bool = False, time_dev : float = 10, candidate_limit : float = 1.0, num_processes : int = 10,layer = "missing_log1p", min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10, deg_imp : float = 0.5):
    assert metric in ["MSE","cosine"]
    if pert_file is None:
        suffix = "ctrl"
    else:
        suffix = "pert"

    if grins_data is None: # grins_data can either be provided in the pipeline, or loaded if the file is run standalone
        grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/perturb_norm_{suffix}.h5ad")
        grins_data = grins_data[:,grins_data.var["pct_dropout_by_counts"]<90]
        if layer == "missing_log1p":
            try:
                grins_data.layers[layer]
            except:
                grins_data = pgrins_prepare_output.apply_missingness(grins_data)

    if adata_file is None:
        adata = None
    else:
        if isinstance(adata_file,str): # adata can either be provided in the pipeline, or loaded if the file is run standalone, or be None
            adata = sc.read_h5ad(f"Data/Experimental/{adata_file}/perturb_subset_{grn_file}.h5ad")
            adata = adata[:,adata.var["pct_dropout_by_counts"]<90]
        else:
            adata = adata_file 
        print("Subsetting datasets")
        common_genes = list(set(adata.var_names) & set(grins_data.var_names))
        grins_data = grins_data[:,common_genes]
        adata = adata[:,common_genes]       
        

    if pert_file is None:
        pert_list = [{"ctrl":0}]
        grins_ctrl_data = None
    else:
        pert_list = pgrins_prepare_input.extract_pert_info(grn_file,pert_file)

        # Add control cells to unclustered pert cells to calculate weights
        grins_ctrl_data = sc.read_h5ad(f"Data/Projects/{grn_file}/perturb_norm_ctrl_reduced.h5ad")
        grins_ctrl_data = grins_ctrl_data[grins_ctrl_data.obs["best_cells"]]
        grins_data = ad.concat([grins_ctrl_data,grins_data])
        weights_dict = get_weights(grins_data,layer)

    # Collect the chosen cells for each perturbation, then subset the whole adata object by them
    best_cells = np.array([],dtype=np.bool)
    for pert_dict in pert_list:
        pert = "_".join(list(pert_dict.keys()))
        if adata is not None:
            grins_data_subset = grins_data[grins_data.obs["perturbation"]==pert]
            adata_subset = adata[adata.obs["perturbation"]==pert]
        else:
            grins_data_subset = grins_data[grins_data.obs["perturbation"]==pert]
            adata_subset = None

        if pert_file is not None:
            grins_data_subset.var["weights"]=weights_dict[pert]
            grins_data_subset.var["is_DEG"]=grins_data.uns["rank_genes_groups"]["pval_adj"][pert]<0.05
        

        # Candidate sampling:
        if candidate_limit < 1.0:
            if adata_subset is not None:
                candidate_cell_mask = get_candidate_cells(grins_data,adata_subset,metric=metric,use_mad=use_mad,times_dev=times_dev,candidate_limit=candidate_limit,num_processes=num_processes, layer=layer)
                
            else:
                idx = np.random.choice(list(range(grins_data_subset.shape[0])),size=int(candidate_limit*grins_data_subset.shape[0]),replace=False) # Get candidate_limit % of cells randomly
                candidate_cell_mask = np.asarray(pd.Series(np.arange(grins_data_subset.shape[0])).isin(idx)) 
            grins_data_subset = grins_data_subset[candidate_cell_mask]

        # Clustering:
        best_cells_cluster = calc_clusters(grins_data_subset,adata=adata_subset,grins_ctrl_data=grins_ctrl_data,layer=layer,min_num_pcs=min_num_pcs,min_cluster_size_pct=min_cluster_size_pct, max_num_clusters=max_num_clusters,deg_imp=deg_imp)     
        best_cells = np.append(best_cells, best_cells_cluster)

    grins_data = grins_data[best_cells]

    print("Saving final file")
    ad.settings.allow_write_nullable_strings = True
    grins_data.write_h5ad(
        f"Data/Projects/{grn_file}/perturb_norm_{suffix}_reduced.h5ad",
        compression=hdf5plugin.FILTERS["zstd"]
    )    
   



if __name__ == "__main__":
    # Example: python3 -u pgrins_narrow_params.py projectname -e pseq_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e", "--experimental", help="The name of the experimental control data used for comparison")
    parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
    parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")
    parser.add_argument("--metric",help="Which metric should be used to calculate the score for each cluster during the unperturbed run. Options are mean squared error (MSE, default) and cosine similarity (cosine).")
    parser.add_argument("--num_processes", type=int,help="The number of processes used for silhouette score computation; defaults to 10.")
    parser.add_argument("--use_mad",action="store_true",help="Turn this option on if the MAD should be used as a deviation measure instead of SD for get_candidate_cells.")
    parser.add_argument("--times_dev",type=float,help="How many deviations away the score of the GRiNS data of a cell can be from the mean of means of control data in order to not be discarded. Defaults to 10.")
    parser.add_argument("--candidate_limit",type=float,help="What percentage of candidate cells should be kept at most. Defaults to 100%. If 100%, no candidate sampling is done at all. Otherwise combinable with times_dev.")
    parser.add_argument("--layer", help="The layer of grins_data on which to perform the operations. Defaults to missing_log1p (log1p with missingness layer applied).")
    parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
    parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 1%.")
    parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")
    parser.add_argument("--deg_imp",type=float,help="How much importance the MSE of the DEGs should have for the pert score relative to the nonDEG MSE. Defaults to 0.5 (same importance).")
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
    if args.metric:
        kwargs["metric"] = args.metric
    if args.num_processes:
        kwargs["num_processes"] = args.num_processes
    if args.use_mad:
        kwargs["use_mad"] = args.use_mad
    if args.times_dev:
        kwargs["times_dev"] = args.times_dev
    if args.candidate_limit:
        kwargs["candidate_limit"] = args.candidate_limit
    if args.layer:
        kwargs["layer"] = args.layer
    if args.min_num_pcs:
        kwargs["min_num_pcs"] = args.min_num_pcs
    if args.min_cluster_size_pct:
        kwargs["min_cluster_size_pct"] = args.min_cluster_size_pct
    if args.max_num_clusters:
        kwargs["max_num_clusters"] = args.max_num_clusters
    if args.deg_imp:
        kwargs["deg_imp"] = args.deg_imp

    main(grn_file, adata_file=adata_file, pert_file=pert_file, **kwargs)