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
from itertools import combinations

from multiprocessing import Pool
import argparse
import timeit
from tqdm import tqdm
import os
from glob import glob

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



def calc_clusters(grins_data : ad.AnnData, adata_mean : np.array = None, grins_ctrl_data : ad.AnnData = None, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10) -> ad.AnnData:
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
    adata_mean : np.array, optional
        The mean of the experimental control data.
    grins_ctrl_data : ad.AnnData, optional
        Already clustered control data for the perturbed run.
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
    
    print("Running PCA")
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
    # Lower cluster resolution until cluster exists that fulfills min_cluster_size_pct condition
    #res = 1.0
    #large_clusters = 0
    #while large_clusters < max_num_clusters and res > 0.01:
    sc.tl.leiden(grins_data, flavor="igraph",resolution=1)
    cluster_labels = grins_data.obs["leiden"]
    cluster_counts = cluster_labels.value_counts()
    large_clusters = len(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys()))
    #    res /= 2
    print(f"{large_clusters} clusters definable with >{100*min_cluster_size_pct}% of cells.")

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

    if adata_mean is not None or grins_ctrl_data is not None:
        print(f"Calculating MSES for {len(best_clusters)} clusters")
        if grins_ctrl_data is None:
            # Get the MSEs between the adata control mean and each cluster mean:
            mean_compare = adata_mean
        else:
            mean_p = np.asarray(np.mean(grins_data.layers["log1p"],axis=0))
            mean_c = np.asarray(np.mean(grins_ctrl_data.layers["log1p"],axis=0))
            padj = grins_data.uns["rank_genes_groups"]["pval_adj"][pert][np.argsort(grins_data.uns["rank_genes_groups"]["names"][pert])] # Get pvals in the correct order (so that the corresponding genes are alphabetical)
            mean_compare = np.dot(1-padj,mean_p)+np.dot(padj,mean_c)
        mse_results = [
            mean_squared_error(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(mean_compare).squeeze())
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
    grins_data.uns["clustering_results"]["scores"]=mse_results
    print("Saving file")
    ad.settings.allow_write_nullable_strings = True
    grins_data.write_h5ad(
        f"Data/Projects/Keggoro/perturb_norm_{grins_data.obs["perturbation"][0]}_clustered.h5ad",
        compression=hdf5plugin.FILTERS["zstd"]
    ) 
    

    return np.asarray(grins_data.obs["best_cells"])



def main(grn_file,experimental, grins_data = None,pert_file = None, pval_treshold : float = 0.1, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10):
    if pert_file is None:
        suffix = "ctrl"
    else:
        suffix = "pert"

    if not os.path.exists(f"Data/Projects/{grn_file}/perturb_norm_{suffix}_reduced.h5ad"):

        if grins_data is None: # grins_data can either be provided in the pipeline, or loaded if the file is run standalone
            grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/perturb_norm_{suffix}.h5ad")

        if experimental:
            print("Calculating adata mean")
            adata_list = [sc.read_h5ad(f"{pseq_path}/perturb_norm_subset_{grn_file}.h5ad") for pseq_path in glob(f"Data/Experimental/*")]
            adata_list = [adata[adata.obs["perturbation"]=="ctrl"] for adata in adata_list]
            adata_genes, adata_mean = get_adata_ctrl_mean(adata_list)
            del adata_list
        else:
            adata_mean = None
            
        if pert_file is None:
            pert_list = [{"ctrl":0}]
            grins_ctrl_data = None
        else:
            pert_list = pgrins_prepare_input.extract_pert_info(grn_file,pert_file)

            # Add control cells to unclustered pert cells to calculate weights
            grins_ctrl_data = sc.read_h5ad(f"Data/Projects/{grn_file}/perturb_norm_ctrl_reduced.h5ad")
            grins_ctrl_data = grins_ctrl_data[grins_ctrl_data.obs["best_cells"]]
            grins_data = ad.concat([grins_ctrl_data,grins_data])
            sc.tl.rank_genes_groups(
            grins_data,
            groupby = 'perturbation',
            method = 't-test_overestim_var',
            reference = 'ctrl',
            layer="log1p")

        # Collect the chosen cells for each perturbation, then subset the whole adata object by them
        best_cells = np.array([],dtype=np.bool)
        for pert_dict in pert_list:
            pert = "_".join(list(pert_dict.keys()))
            grins_data_pert = grins_data[grins_data.obs["perturbation"]==pert]            

            # Clustering:
            best_cells_cluster = calc_clusters(grins_data_pert,adata_mean=adata_mean,grins_ctrl_data=grins_ctrl_data,min_num_pcs=min_num_pcs,min_cluster_size_pct=min_cluster_size_pct, max_num_clusters=max_num_clusters)     
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
    parser.add_argument("-e", "--experimental", action="store_true",help="Whether or not experimental data is used for comparison.")
    parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
    parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")
    parser.add_argument("--pval_treshold", type=float,help="The critical treshold level where genes with p values below that value are assigned DEGs. Defaults to 0.1 (0.05 is not chosen because of the high variance between perturbed cells).")
    parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
    parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
    parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")
    args = parser.parse_args()

    grn_file = args.grn
    if args.experimental:
        experimental = True
    else:
        experimental = False

    if args.use_perts:
        if args.pert_file:
            pert_file = args.pert_file
        else:
            pert_file = f"{grn_file}_perts"
    else:
        pert_file = None

    kwargs = {}
    if args.min_num_pcs:
        kwargs["min_num_pcs"] = args.min_num_pcs
    if args.pval_treshold:
        kwargs["pval_treshold"] = args.pval_treshold
    if args.min_cluster_size_pct:
        kwargs["min_cluster_size_pct"] = args.min_cluster_size_pct
    if args.max_num_clusters:
        kwargs["max_num_clusters"] = args.max_num_clusters

    main(grn_file, experimental, pert_file=pert_file, **kwargs)