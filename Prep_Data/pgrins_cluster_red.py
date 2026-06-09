from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.stats import spearmanr

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
from pathlib import Path

try:
    from Prep_Data import pgrins_prepare_output
    from Analysis import plotting_funs
except:
    sys.path.append("..")
    sys.path.append("/".join(str(Path.cwd()).split("/")[:-1]))
    from Prep_Data import pgrins_prepare_output
    from Analysis import plotting_funs


def get_nonzero_stats(layer):
    layer = layer.tocsc()
    mean_nz = []
    sd_nz = []
    non_zero_entries = np.split(layer.indices,layer.indptr[1:-1])
    for i in range(len(non_zero_entries)):
        non_zero_exprs = (layer[non_zero_entries[i],i]).todense().T
        if non_zero_exprs.shape[1] < 1:
            mean_nz.append(0)
            continue
        mean_nz.append(np.mean(non_zero_exprs))
        sd_nz.append(np.sqrt(np.var(non_zero_exprs)))
    mean_nz = np.asarray(mean_nz)
    sd_nz = np.asarray(sd_nz)
    return mean_nz, sd_nz


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
    sd_vals = np.empty(0)
    for comb_size in range(len(adata_list),0,-1):
        for comb_perm in combinations(range(len(adata_list)),comb_size): # Get all comb_size combinations of adata sets
            # Get the adata objects from the current combination
            adata_comb = [adata_list[i] for i in comb_perm] 
            # Get the names of their genes
            comb_genenames = [set(adata.var_names) for adata in adata_comb]
            # Get the intersection of the names and remove all genes for which a mean was already calculated -> all genes ONLY in the intersection of the current combination
            comb_genenames = list(set().union(*comb_genenames).intersection(*comb_genenames) - done_genes)
            # Stack the expression data of these genes and calculate the mean
            comb_mean_nz, comb_sd_nz = get_nonzero_stats(sparse.vstack([adata[:,comb_genenames].layers["log1p"] for adata in adata_comb]))
            # Update the genes already iterated over
            done_genes = done_genes | set(comb_genenames)
            mean_genenames += comb_genenames
            mean_vals = np.append(mean_vals, comb_mean_nz)
            sd_vals = np.append(sd_vals, comb_sd_nz)
    # Sort values so that the corresponding genes are in alphabetical order again:
    mean_df = pd.DataFrame({"Gene":mean_genenames,"Mean":mean_vals,"Sd":sd_vals}).sort_values(by="Gene")
    mean_df = mean_df[~mean_df["Gene"].str.contains(r"\-|\.",regex=True)]
    
    return mean_df["Gene"].values, mean_df["Mean"].values, mean_df["Sd"].values



def calc_clusters(grins_data : ad.AnnData, adata_mean : np.array = None, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10,eval_metric : str = "MSE") -> ad.AnnData:
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
    clusters = np.array(list(cluster_counts.keys()))

    best_clusters = np.array(list(cluster_counts[cluster_counts>grins_data.shape[0]*min_cluster_size_pct].keys())) # At least min_cluster_size_pct of total cells in cluster
    if max_num_clusters == 0:
        max_num_clusters = len(best_clusters)
    print(f"{len(best_clusters)} clusters definable with >{100*min_cluster_size_pct}% of cells (at most, {max_num_clusters} are needed).")

    # Get large, homogenous clusters
    print(f"Calculating Silhouette score for {len(clusters)} clusters")
    sample_silhouette_values = silhouette_samples(grins_data.obsm["X_umap"], cluster_labels)
    grins_data.obs["silhouette_samples"] = sample_silhouette_values

    best_clusters = clusters[np.argsort([np.mean(sample_silhouette_values[cluster_labels==c]) for c in best_clusters])[::-1][:max_num_clusters]] # max_num_clusters clusters with highest silhouette score
    
    if adata_mean is not None:
        if eval_metric.lower() == "mse":
            print(f"Calculating MSE for {len(best_clusters)} clusters")
            # Get the MSEs between the adata control mean and each cluster mean:
            metric_results = [
                mean_squared_error(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(adata_mean).squeeze())
                for cluster in best_clusters
            ]
            best_cluster_overall = best_clusters[np.argmin(metric_results)]
        elif eval_metric.lower() == "spearman":
            print(f"Calculating Spearman correlation for {len(best_clusters)} clusters")
            # Get the values between the adata control mean and each cluster mean:
            print(grins_data.var_names[:10])
            metric_results = [
                spearmanr(np.asarray(np.mean(grins_data[cluster_labels==cluster].layers["log1p"],axis=0)).squeeze(),np.asarray(adata_mean).squeeze()).correlation
                for cluster in best_clusters
            ]
            best_cluster_overall = best_clusters[np.argmax(metric_results)]
        else:
            raise ValueError("eval_metric must be MSE or Spearman.")
        for i in range(len(best_clusters)):
            print(f"Cluster {best_clusters[i]} ({grins_data[cluster_labels==best_clusters[i]].shape[0]} cells): {eval_metric} = {metric_results[i]}")
    else: # ctrl case without experimental data: take the cluster with the highest silhouette score
        best_cluster_overall = best_clusters[0]

    grins_data.obs["best_cells"] = cluster_labels==best_cluster_overall

    print(f"{sum(grins_data.obs["best_cells"])}/{grins_data.shape[0]} cells retained")

    # Optional saving for plotting
    
    grins_data.uns["clustering_results"] = {}
    grins_data.uns["clustering_results"]["best_clusters"]=best_clusters
    if adata_mean is not None:
        grins_data.uns["clustering_results"]["metric"]=eval_metric
        grins_data.uns["clustering_results"]["scores"]=metric_results

    
    return list(grins_data.obs["best_cells"]), grins_data



def main(grn_file, sim_it : int, experimental, min_num_pcs : int = 10, min_cluster_size_pct : float = 0.01, max_num_clusters : int = 10,num_replicates:int=1, eval_metric : str = "MSE"):
    for replicate in range(1,num_replicates+1):
        if not os.path.exists(f"Data/SimulResults_Racipe_{sim_it+1}"):

            grins_data = sc.read_h5ad(f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl_{sim_it}.h5ad")

            if experimental:
                print("Calculating adata mean")
                adata_list = [sc.read_h5ad(f"{pseq_path}/perturb_norm_subset_{grn_file}.h5ad") for pseq_path in glob(f"Data/Experimental/*")]
                adata_list = [adata[adata.obs["perturbation"]=="ctrl"] for adata in adata_list]
                adata_genes, adata_mean, adata_sd = get_adata_ctrl_mean(adata_list)
                grins_data = grins_data[:,sorted(list(set(grins_data.var_names) & set(adata_genes)))] # Remove source genes not in adata from grins_data
                del adata_list
            else:
                adata_mean = None 
            print("Getting entries close to experimental...") 
            init_conds = pd.read_parquet(f"Data/SimulResults_Racipe_{sim_it}/{grn_file}/{replicate:03}/{grn_file}_init_conds_{replicate:03}.parquet")    
            proxim_matrix = np.abs(grins_data.layers["log1p"]-adata_mean)<adata_sd
            source_idx = np.argwhere(grins_data.var_names.isin(init_conds.columns)).squeeze()

            print("Finding initial condition ranges...")
            genes = []
            mins = []
            maxs = []
            for i in tqdm(source_idx):
                gene = grins_data.var_names[i]
                gene_merged = pd.merge(grins_data[np.asarray(proxim_matrix[:,i]).squeeze()].obs["InitCondNum"],init_conds.loc[:,[gene,"InitCondNum"]])
                bin_size=20
                ic_hist = np.histogram(gene_merged[gene],bins=bin_size,range=(0,100))
                peak_starts = ic_hist[1][:-1][ic_hist[0]>gene_merged.shape[0]/bin_size].astype(int)
                if len(peak_starts)==0:
                    greatest_cluster = [0,100]
                else:
                    peak_clusters=[[peak_starts[0],peak_starts[0]+int(100/bin_size)]]
                    for i in range(1,len(peak_starts)):
                        if peak_clusters[-1][1]==peak_starts[i]:
                            peak_clusters[-1][1]+=int(100/bin_size)
                        else:
                            peak_clusters.append([peak_starts[i],peak_starts[i]+int(100/bin_size)])
                    greatest_cluster = peak_clusters[np.argmax([cluster[1]-cluster[0] for cluster in peak_clusters])]
                genes.append(gene)
                mins.append(greatest_cluster[0])
                maxs.append(greatest_cluster[1])
            
            irange_df = pd.DataFrame({"Gene":genes,"Minimum":mins,"Maximum":maxs})
            os.makedirs(f"Data/SimulResults_Racipe_{sim_it+1}/{grn_file}/{replicate:03}",exist_ok=True)
            irange_df.to_csv(f"Data/SimulResults_Racipe_{sim_it+1}/{grn_file}/{replicate:03}/{grn_file}_init_conds_range_{replicate:03}.csv",index=False,sep="\t")
            """
            # Collect the chosen cells, then subset the whole adata object by them
            best_cells, grins_data_clustered = calc_clusters(grins_data,adata_mean=adata_mean,min_num_pcs=min_num_pcs,min_cluster_size_pct=min_cluster_size_pct, max_num_clusters=max_num_clusters,eval_metric=eval_metric)     
            grins_data = grins_data[best_cells]
            best_cell_index = list(grins_data.obs.index)

            print("Saving final file")
            plotting_funs.plot_pca_results(grn_file,grins_data_clustered,replicate)
            plotting_funs.plot_umap_results(grn_file,grins_data_clustered,replicate)
            
            # File containing all cells and information about the clustering process (for plotting):
            grins_data_clustered.write_h5ad(
                f"Data/Projects/{grn_file}/{replicate:03}/perturb_norm_ctrl_clustered.h5ad"
            ) 
            
            with open(f'Data/Projects/{grn_file}/{replicate:03}/ctrl_best_cells.pickle', 'wb') as f:
                pickle.dump(best_cell_index, f, pickle.HIGHEST_PROTOCOL)
            """
        else:
            print("Done!")
            



if __name__ == "__main__":
    # Example: python3 -u pgrins_narrow_params.py projectname -e pseq_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e", "--experimental", action="store_true",help="Whether or not experimental data is used for comparison.")
    parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
    parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
    parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")
    parser.add_argument("--eval_metric",help="If experimental data is used, whether or not the clusters should be evaluated using MSE or Spearman correlation. Defaults to MSE.")
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
    if args.eval_metric:
        kwargs["eval_metric"]=args.eval_metric

    main(grn_file, experimental, **kwargs)