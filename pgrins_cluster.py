from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import scanpy as sc
import anndata as ad
import igraph

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from multiprocessing import Pool
import argparse
import timeit
from tqdm import tqdm
import os
from glob import glob
import pickle

from Prep_Data import pgrins_prepare_output


def get_adata_ctrl_mean(adata_list : ad.AnnData) -> np.array:
    """
    Goes through the control cells of all experimental datasets and calculates for each gene the mean across the ctrl cells of all datasets that contain the gene.
    For example, for datasets A,B,C, some genes will be in A&B&C, some only in A&B, A&C, or B&C, but some only in A,B, or C.

    Parameters:
    -----------
    adata_list : list[ad:AnnData]
        A list of all experimental dataset objects.

    Returns:
    --------
    mean_genenames : list[str]
        A list of the gene names across all datasets.
    mean_vals : list[np.float]
        A list of the mean values across the corresponding datasets.
    """
    done_genes = set()
    mean_genenames = []
    mean_vals = np.empty(0)
    for comb_size in range(len(adata_list),0,-1):
        for comb_perm in combinations(range(len(adata_list)),comb_size): # Get all comb_size combinations of adata sets
            # Get the adata objects from the current combination
            adata_comb = [adata_list[i] for i in comb_perm] 
            # Get the names of their genes
            comb_genenames = [set(adata.var_names) for adata in adata_comb]
            # Get the intersection of the names and remove all genes for which a mean was already calculated -> all genes ONLY in the intersection of the current combination
            comb_genenames = list(set().union(*comb_genenames).intersection(*comb_genenames) - done_genes)
            # Stack the expression data of these genes and calculate the mean
            comb_mean = np.mean(sparse.vstack([adata[:,comb_genenames].layers["log1p"] for adata in adata_comb]),axis=0)
            # Update the genes already iterated over
            done_genes = done_genes | set(comb_genenames)
            mean_genenames += comb_genenames
            mean_vals = np.append(mean_vals, comb_mean)
    # Sort values so that the corresponding genes are in alphabetical order again:
    mean_df = pd.DataFrame({"Gene":mean_genenames,"Val":mean_vals}).sort_values(by="Gene")
    return mean_df["Gene"].values, mean_df["Val"].values



def calc_clusters(grins_data : ad.AnnData, adata_mean : np.array = None, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10) -> ad.AnnData:
    """
    Clusters the data by performing PCA -> UMAP -> Leiden, then getting large clusters with high silhouette scores and (if adata is provided) finding the one closest to adata.
    For the control cells, this is done by calculating the MSE between the mean expression of each clusters and the experimental control mean (if provided, otherwise biggest cluster)
    
    Parameters:
    -----------
    grins_data : ad.AnnData
        The generated GRiNS data.
    adata_mean : np.array, optional
        The mean of the experimental control data.
    min_num_pcs : int, optional
        The minimal number of principal components to use for UMAP. Defaults to 10.
    min_cluster_size_pct : float, optional
        The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 1%.
    max_num_clusters : int, optional
        The maximal number of clusters considered as best cluster. Defaults to 10.
        
    Returns:
    --------
    grins_data : ad.AnnData
        The original grins_data, but with additional columns and data indicating which cluster is the best.
    
    """
    
    print(f"Running PCA, keeping at least {min_num_pcs} PCs")
    sc.pp.pca(grins_data, svd_solver="arpack",layer="log1p")

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
    sc.tl.leiden(grins_data, flavor="igraph",resolution=1)
    cluster_labels = grins_data.obs["leiden"]
    cluster_counts = cluster_labels.value_counts()
    large_clusters = len(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys()))
    print(f"{large_clusters} clusters definable with >{100*min_cluster_size_pct}% of cells (at most, {max_num_clusters} are needed).")

    clusters = np.array(list(cluster_counts.keys()))
    print(f"Calculating Silhouette score for {len(clusters)} clusters")
    sample_silhouette_values = silhouette_samples(grins_data.obsm["X_umap"], cluster_labels)
    grins_data.obs["silhouette_samples"] = sample_silhouette_values

    # Get large, homogenous clusters
    if large_clusters > 0:
        best_clusters = np.array(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys())) # At least min_cluster_size_pct of total cells in cluster
    else:
        best_clusters = np.array(list(cluster_counts.keys()))

    best_clusters = clusters[np.argsort([np.mean(sample_silhouette_values[cluster_labels==c]) for c in best_clusters])[::-1][:max_num_clusters]] # max_num_clusters clusters with highest silhouette score

    if adata_mean is not None:
        print(f"Calculating MSES for {len(best_clusters)} clusters")
        # Get the MSEs between the adata control mean and each cluster mean:
        mse_results = [
            mean_squared_error(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(adata_mean).squeeze())
            for cluster in best_clusters
        ]
        # Get the cells in GRiNS in the cluster with the lowest MSE:
        best_cluster_overall = best_clusters[np.argmin(mse_results)]
    else: # ctrl case without experimental data: take the cluster with the highest silhouette score
        best_cluster_overall = best_clusters[0]

    grins_data.obs["best_cells"] = cluster_labels==best_cluster_overall

    print(f"{sum(grins_data.obs["best_cells"])}/{grins_data.shape[0]} cells retained")

    # Optional saving for plotting
    
    grins_data.uns["clustering_results"] = {}
    grins_data.uns["clustering_results"]["best_clusters"]=best_clusters
    if adata_mean is not None:
        grins_data.uns["clustering_results"]["scores"]=mse_results

    
    return list(grins_data.obs["best_cells"]), grins_data



def main(grn_file,experimental, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10,num_replicates:int=1):
    for replicate in range(1,num_replicates+1):
        if not os.path.exists(f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl_clustered.h5ad"):

            grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl.h5ad")

            if experimental:
                print("Calculating adata mean")
                adata_list = [sc.read_h5ad(f"{pseq_path}/perturb_norm_subset_{grn_file}.h5ad") for pseq_path in glob(f"Data/Experimental/*")]
                adata_list = [adata[adata.obs["perturbation"]=="ctrl"] for adata in adata_list]
                adata_genes, adata_mean = get_adata_ctrl_mean(adata_list)
                grins_data = grins_data[:,adata_genes] # Remove source genes not in adata from grins_data
                del adata_list
            else:
                adata_mean = None      

            # Collect the chosen cells, then subset the whole adata object by them
            best_cells, grins_data_clustered = calc_clusters(grins_data,adata_mean=adata_mean,min_num_pcs=min_num_pcs,min_cluster_size_pct=min_cluster_size_pct, max_num_clusters=max_num_clusters)     
            grins_data = grins_data[best_cells]
            best_cell_index = list(grins_data.obs.index)

            print("Saving final file")
            # File containing all cells and information about the clustering process (for plotting):
            grins_data_clustered.write_h5ad(
                f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl_clustered.h5ad"
            ) 
            with open(f'Data/Projects/{grn_file}/{replicate:03}/ctrl_best_cells.pickle', 'wb') as f:
                pickle.dump(best_cell_index, f, pickle.HIGHEST_PROTOCOL)
            



if __name__ == "__main__":
    # Example: python3 -u pgrins_narrow_params.py projectname -e pseq_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e", "--experimental", action="store_true",help="Whether or not experimental data is used for comparison.")
    parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
    parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
    parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")
    args = parser.parse_args()

    grn_file = args.grn
    if args.experimental:
        experimental = True
    else:
        experimental = False


    kwargs = {}
    if args.min_num_pcs:
        kwargs["min_num_pcs"] = args.min_num_pcs
    if args.min_cluster_size_pct:
        kwargs["min_cluster_size_pct"] = args.min_cluster_size_pct
    if args.max_num_clusters:
        kwargs["max_num_clusters"] = args.max_num_clusters
    if args.num_replicates:
        kwargs["num_replicates"]=args.num_replicates

    main(grn_file, experimental, **kwargs)