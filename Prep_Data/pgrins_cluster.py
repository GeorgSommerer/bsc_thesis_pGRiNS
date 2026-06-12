import numpy as np
import pandas as pd
from itertools import combinations

import scanpy as sc
import anndata as ad
import igraph

from sklearn.metrics import silhouette_samples, mean_squared_error
from scipy import sparse
from scipy.stats import spearmanr

import os
from glob import glob
import pickle

from Prep_Data import pgrins_prepare_output
from Analysis import plotting_funs



def get_adata_ctrl_mean(adata_list : list[ad.AnnData]) -> tuple[list[str], list[np.float]]:
    """
    For each gene, calculates the mean across ctrl cells of all datasets that contain the gene.
    For example, for datasets A,B,C, some genes will be in A&B&C, some only in A&B, A&C, or B&C, but some only in A,B, or C.

    Parameters:
    -----------
    adata_list : list[ad.AnnData]
        A list of all experimental dataset objects.

    Returns:
    --------
    mean_genenames : list[str]
        A list of the gene names across all datasets in alphabetical order.
    mean_vals : list[np.float]
        A list of the mean values across the datasets, where the gene belonging to mean_vals[i] is mean_genenames[i] (the i-th gene in alphabetical order).
    """
    done_genes = set() # Stores all genes already evaluated
    mean_genenames = []
    mean_vals = np.empty(0)

    for comb_size in range(len(adata_list),0,-1):
        # Get all subsets of adata_list with comb_size elements (start with intersection of all of them, end with individual datasets)
        for comb_perm in combinations(range(len(adata_list)),comb_size): 
            # Get the adata objects from the current combination
            adata_comb = [adata_list[i] for i in comb_perm] 
            # Get the names of their genes
            comb_genenames = [set(adata.var_names) for adata in adata_comb]
            # Get the intersection of the names and remove all genes for which a mean was already calculated
            # -> Keep all genes ONLY in the intersection of the current combination in the order in which they'll be added to mean_genenames
            comb_genenames = list(set().union(*comb_genenames).intersection(*comb_genenames) - done_genes)
            # Stack the expression data of these genes and calculate the mean
            comb_mean = np.mean(sparse.vstack([adata[:,comb_genenames].layers["log1p"] for adata in adata_comb]),axis=0)

            # Update the genes already iterated over
            done_genes = done_genes | set(comb_genenames)
            mean_genenames += comb_genenames
            mean_vals = np.append(mean_vals, comb_mean)

    # Sort values so that the corresponding genes are in alphabetical order again:
    mean_df = pd.DataFrame({"Gene":mean_genenames,"Val":mean_vals}).sort_values(by="Gene")
    mean_df = mean_df[~mean_df["Gene"].str.contains(r"\-|\.",regex=True)]

    return mean_df["Gene"].values, mean_df["Val"].values



def calc_clusters(grins_data : ad.AnnData, adata_mean : np.array = None, min_num_pcs : int = 10, max_num_pcs : int = 25, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10, eval_metric : str = "MSE") -> tuple[list[str],ad.AnnData]:
    """
    Clusters the data by performing PCA -> UMAP -> Leiden.
    Then, large clusters with high silhouette scores are calculated and (if experimental data is provided) the one with the closest mean expression is kept.
    
    Parameters:
    -----------
    grins_data : ad.AnnData
        The unperturbed GRiNS data.
    adata_mean : np.array, optional
        The mean of the experimental control data from get_adata_ctrl_mean.
    min_num_pcs : int, optional
        The minimal number of principal components to use for UMAP. Defaults to 10.
    max_num_pcs : int, optional
        The maximal number of principal components to use for UMAP. Defaults to 25.
    min_cluster_size_pct : float, optional
        The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.
    max_num_clusters : int, optional
        The maximal number of clusters considered as best cluster. Defaults to 10.
    eval_metric : str, optional
        If experimental data is used, whether or not the clusters should be evaluated using MSE or Spearman correlation. Defaults to MSE
        
    Returns:
    --------
    best_cells : list[str]
        The poitions of cells belonging to the best cluster (NOT a subset of grins_data.obs_names).
    grins_data : ad.AnnData
        The original grins_data, but with additional columns and data indicating which cluster is the best.
    """

    print(f"Running PCA, keeping at least {min_num_pcs} PCs")
    sc.pp.pca(grins_data, svd_solver="arpack",layer="log1p")

    # Find the PCs where the increase in the cumulative elbow plot is less than 1% (if they exist)
    cum_pca = np.cumsum(grins_data.uns["pca"]["variance_ratio"])
    redundant_pcs = np.where(np.array([cum_pca[i+1]/cum_pca[i] for i in range(len(cum_pca)-1)])<1.01)[0]
    if len(redundant_pcs)==0:
        n_pcs = len(cum_pca)
    else:
        n_pcs = redundant_pcs[0]

    # n_pcs is in [min_num_pcs, max_num_pcs]
    if n_pcs > max_num_pcs:
        n_pcs = max_num_pcs
    elif n_pcs < min_num_pcs:
        n_pcs = min_num_pcs

    print(f"Running UMAP using {n_pcs} PCs")
    sc.pp.neighbors(grins_data,use_rep="X_pca",n_pcs=n_pcs)
    sc.tl.umap(grins_data)

    print("Clustering")
    sc.tl.leiden(grins_data, flavor="igraph",resolution=1)
    cluster_labels = grins_data.obs["leiden"] # Cluster of each cell
    cluster_counts = cluster_labels.value_counts() # Number of cells per cluster
    clusters = np.array(list(cluster_counts.keys())) # Unique cluster labels

    # Get all clusters that contain at least min_cluster_size_pct% of total cells
    best_clusters = np.array(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys())) 

    # If no maximal number of clusters is specified: all clusters with at least min_cluster_size_pct% of total cells are kept
    if max_num_clusters == 0:
        max_num_clusters = len(best_clusters)

    print(f"{len(best_clusters)} clusters definable with >{100*min_cluster_size_pct}% of cells (at most, {max_num_clusters} are needed).")

    # Get large, homogenous clusters (i.E. clusters with a high silhouette score)
    print(f"Calculating Silhouette score for {len(clusters)} clusters")
    sample_silhouette_values = silhouette_samples(grins_data.obsm["X_umap"], cluster_labels)
    grins_data.obs["silhouette_samples"] = sample_silhouette_values

    # Keep the max_num_clusters clusters with the highest silhouette score
    best_clusters = clusters[np.argsort([np.mean(sample_silhouette_values[cluster_labels==c]) for c in best_clusters])[::-1][:max_num_clusters]]
    
    # If experimental data is available, keep the large, homogenous cluster with the lowest MSE/highest spearman correlation compared to experimental mean
    if adata_mean is not None:
        if eval_metric.lower() == "mse":
            # Get the MSEs between the adata control mean and each cluster mean:
            print(f"Calculating MSE for {len(best_clusters)} clusters")
            metric_results = [
                mean_squared_error(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(adata_mean).squeeze())
                for cluster in best_clusters
            ]
            # Keep the cluster with the lowest MSE
            best_cluster_overall = best_clusters[np.argmin(metric_results)]

        elif eval_metric.lower() == "spearman":
            # Get the spearman correlation between the adata control mean and each cluster mean:
            print(f"Calculating Spearman correlation for {len(best_clusters)} clusters")
            metric_results = [
                spearmanr(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(adata_mean).squeeze()).correlation
                for cluster in best_clusters
            ]
            # Keep the cluster with the highest correlation
            best_cluster_overall = best_clusters[np.argmax(metric_results)]

        else:
            raise ValueError("eval_metric must be MSE or Spearman.")

        for i in range(len(best_clusters)):
            print(f"Cluster {best_clusters[i]} ({grins_data[cluster_labels==best_clusters[i]].shape[0]} cells): {eval_metric} = {metric_results[i]}")

    # If no experimental data provided, Keep the cluster with the highest silhouette score
    else: 
        best_cluster_overall = best_clusters[0]

    # Get the positions of the cells of the best cluster
    grins_data.obs["best_cells"] = cluster_labels == best_cluster_overall
    best_cells = list(grins_data.obs["best_cells"])
    print(f"{sum(best_cells)}/{grins_data.shape[0]} cells retained")

    # Copy clustering information to grins_data
    grins_data.uns["clustering_results"] = {}
    grins_data.uns["clustering_results"]["best_clusters"]=best_clusters
    if adata_mean is not None:
        grins_data.uns["clustering_results"]["metric"]=eval_metric
        grins_data.uns["clustering_results"]["scores"]=metric_results

    return best_cells, grins_data



def main(grn_file : str, experimental : bool = False, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10, max_num_pcs : int = 25, num_replicates:int=1, eval_metric : str = "MSE"):
    """
    Loads the unperturbed synthetic control data, calculates the mean of the experimental control data if provided.
    Then, clusters the synthetic data so that a large, homogenous cluster (with expression levels close to the experimental data) is kept and used as the final control data.

    Parameters:
    -----------
    grn_file : str
        The name of the project.
    experimental : bool, optional
        Whether or not experimental data is provided.
    min_num_pcs : int, optional
        The minimal number of principal components to use for UMAP. Defaults to 10.
    max_num_pcs : int, optional
        The maximal number of principal components to use for UMAP. Defaults to 25.
    min_cluster_size_pct : float, optional
        The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.
    max_num_clusters : int, optional
        The maximal number of clusters considered as best cluster. Defaults to 10.
    eval_metric : str, optional
        If experimental data is used, whether or not the clusters should be evaluated using MSE or Spearman correlation. Defaults to MSE

    Returns:
    --------
    None
    Results are saved in Projects/{grn_file}/ctrl_best_cells.pickle
    Projects/{grn_file}/pert_norm_ctrl.h5ad is overwritten and now contains information on the clustering as well.
    """

    for replicate in range(1,num_replicates+1):
        if not os.path.exists(f"Data/Projects/{grn_file}/{replicate:03}/ctrl_best_cells.pickle"):
            grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl.h5ad")

            if experimental:
                print("Calculating adata mean")
                adata_list = [sc.read_h5ad(f"{pseq_path}/perturb_norm_subset_{grn_file}.h5ad") for pseq_path in glob(f"Data/Experimental/*")]
                adata_list = [adata[adata.obs["perturbation"]=="ctrl"] for adata in adata_list]

                adata_genes, adata_mean = get_adata_ctrl_mean(adata_list)
                # Remove source genes not in adata from grins_data
                grins_data = grins_data[:,sorted(list(set(grins_data.var_names) & set(adata_genes)))] 
                del adata_list
            else:
                adata_mean = None      

            # Get the cells of the best cluster, as well as the augmented grins_data
            best_cells, grins_data_clustered = calc_clusters(grins_data,adata_mean=adata_mean,min_num_pcs=min_num_pcs,min_cluster_size_pct=min_cluster_size_pct, max_num_clusters=max_num_clusters,eval_metric=eval_metric)     
            #Turn the positions of the best cells into indices (subset of grins_data.obs_names).
            grins_data = grins_data[best_cells]
            best_cell_index = list(grins_data.obs.index)

            print("Saving final file")
            # Plot results
            plotting_funs.plot_pca_results(grn_file,grins_data_clustered,replicate)
            plotting_funs.plot_umap_results(grn_file,grins_data_clustered,replicate)
            
            # Overwrite perturb_norm_ctrl.h5ad
            grins_data_clustered.write_h5ad(f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl.h5ad") 
            with open(f'Data/Projects/{grn_file}/{replicate:03}/ctrl_best_cells.pickle', 'wb') as f:
                pickle.dump(best_cell_index, f, pickle.HIGHEST_PROTOCOL)
        else:
            print("Done!")