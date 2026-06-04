# BASELINES

import anndata as ad
import numpy as np
import scanpy as sc
import random
from anndata import AnnData
import scipy

# 1. control baseline: mean of all unperturbed cells (i.e. cells with perturbation == "*")

def control_baseline(norm_data: AnnData, ctrl_i_vec: list[str]) -> np.ndarray:
    """
    Calculate mean expression per gene across unperturbed control cells.
    
    Parameters
    ----------
    norm_data : anndata.AnnData
        The annotated data matrix
    ctrl_i_vec : list[str]
        A list of indices for control cells
    Returns
    -------
    mean_unperturbed : np.ndarray
        Array of length n_genes with mean expression in control cells
    """
    # Todo: rewrite function so that ctrl_i_vec are used as indices
    
    # create boolean mask for unperturbed cells
    unperturbed_mask = norm_data.obs["perturbation"] == "ctrl"

    if unperturbed_mask.sum() == 0:
        raise ValueError("No control cells found with label 'ctrl'")

    # calculate mean expression per gene across control cells
    mean_unperturbed = np.asarray(np.mean(norm_data[unperturbed_mask].layers["log1p"], axis=0)).squeeze()

    return mean_unperturbed


# 2. Mean Baseline function 

def mean_baseline(norm_data: AnnData, pert_i_dict: dict[str,list[str]], train_perts: list[str]) -> np.ndarray[np.float64]:
    """
    Calculate the average gene expression counts for all genes for each perturbation. 
    Then compute the mean expressions of each gene across all perturbations

    Parameters 
    ----------
    norm_data : anndata object 
        The annotated data matrix
    pert_i_dict : dict[str, list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values
    train_perts : list[str]
        A list of all perturbations used for testing (i.E. used for calculating the baseline)

    Returns
    -------
    mu_all : np.ndarray[np.float64]
        The mean baseline of norm_data for a specified layer
    """

    non_existent_perturbations = []
    skipped_perturbations = []

    pert_means = []
    for pert in train_perts:

        pert_indices = pert_i_dict[pert]

        # Perturbations with too few cells are skipped:
        if len(pert_indices) == 0:
            non_existent_perturbations.append(pert)
            continue
        elif len(pert_indices) < 10:
            skipped_perturbations.append(pert)
            continue

        # Computing avg gene expression for each perturbation
        mu_pert_i = np.asarray(np.mean(norm_data[pert_indices].layers["log1p"], axis=0)).ravel()
        pert_means.append(mu_pert_i)

    mu_all = np.asarray(np.mean(np.stack(pert_means, axis=0), axis=0)).ravel()

    return mu_all



def get_linear_GWb(norm_data : AnnData, train_perts : list[str], K : int = 10, l : float = 0.1) -> tuple[np.matrix,np.matrix,np.matrix]:
    """
    Calculate a linear baseline as described by Ahlmann-Eltze et al. used for single perturbations
    For a matrix Y_train of expression values with 1 column per trained perturbation and 1 row per gene (taken from Norman or Replogle), calculate
    the model W from G (the top K principal components of Y), P (the rows of G that correspond to the perturbed genes), b (the row means of the training data) and l (a ridge regression penalty of 0.1).
    The equation used is W=(G^TG+lI)^(-1)*G^T(Y-b)P(P^TP+lI)^(-1)

    Parameters:
    -----------
    K : int
        The number of PCs to keep.
    l : float
        The ridge penalty.
    """
    train_single_perts = [t for t in train_perts if "_" not in t]
    
    Y = np.vstack([np.mean(norm_data[norm_data.obs["perturbation"]==p].layers["log1p"],axis=0) for p in train_single_perts]).T # Shape g x p
    b = np.mean(Y,axis=1) # Shape g x 1
    G = sc.pp.pca(Y,n_comps=K,return_info=False) # Shape g x K
    P = G[np.argwhere(norm_data.var_names.isin(train_single_perts)).squeeze()] # Shape p x K

    W = np.linalg.inv(G.T @ G+l*np.eye(G.shape[1])) @ G.T @ (Y-b) @ P @ np.linalg.inv(P.T @ P + l*np.eye(P.shape[1])) # Shape K x K

    #print("Shape of data:",norm_data.shape,"Shape of Y:",Y.shape,"Shape of b:",b.shape,"Shape of G:",G.shape,"Shape of P:",P.shape,"Shape of W:",W.shape)

    return(G,W,b)



def informed_model(norm_data : ad.AnnData, train_perts : list[str], test_perts : list[str], pert_i_dict : dict[str,list[str]], control_baseline : np.ndarray[np.float64], K : int = 10, l : float = 0.1) -> np.ndarray[np.float64]:
    """
    A mix of the linear baseline for single perturbations, and an additive baseline for double perturbations.
    Calculate the linear baseline as Y=GWP^T+b, where P is now subset to the genes from the test perturbations.
    The additive baseline calculates the mean vector for each single gene in the double perturbation (using the TD cells only), sums them up, and subtracts the control mean.
        Also possible for n-perturbations by subtracting the control baseline n-1 times.
        ! This assumes that for every double perturbation, the single perturbations are also part of the dataset. !

    """
    test_single_perts = [t for t in test_perts if "_" not in t]
    test_double_perts = [t for t in test_perts if "_" in t]

    informed_dict = {}
    
    # Linear baseline:
    #single_test_perts_ordered = list(norm_data.var_names[norm_data.var_names.isin(single_test_perts)])
    #P = G[np.argwhere(norm_data.var_names.isin(single_test_perts_ordered))]
    G,W,b = get_linear_GWb(norm_data,train_perts,K,l)
    for pert in test_single_perts:
        P_tilde = G[np.argwhere(norm_data.var_names.isin(test_single_perts)).squeeze()]

        Y_pred = (G @ W @ P_tilde.T + b).T
        informed_dict[pert] = np.asarray(np.mean(Y_pred,axis=0)).squeeze() # Mean over all created cells

    # Additive baseline:
    for pert in test_double_perts:   
        pert_genes = pert.split("_")
        pert_gene_means = [np.asarray(np.mean(norm_data[pert_i_dict[single_gene]].layers["log1p"],axis=0)).squeeze() for single_gene in pert_genes] # What if single pert is not in dataset?? -> For KeggoRo_0206 not a problem
        informed_dict[pert] = np.sum(pert_gene_means) - (len(pert_genes)-1)*control_baseline
    
    return informed_dict



def td_gt_split(pert_i_dict : dict[str,list[str]],test_perts : list[str], td_ratio : float = 0.5) -> tuple[dict[str,list[str]],dict[str,list[str]]]:
    """
    For all test perturbations, assign td_ratio% of the cells to a technical duplicate (TD), which corresponds to the mean per gene across these cells.
    The remaining cells are assigned to the ground truth (GT).

    Parameters:
    -----------
    test_perts : list[str]
        A list of all perturbations used for testing (i.E. used for calculating the baseline)
    pert_i_dict : dict[str,list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values
    td_ratio : float, optional
        The percentage of cells used for the technical duplicate (rounded up) (Default: 0.5).

    Returns:
    --------
    td_index_dict : dict[str,list[str]]
        For each perturbed genes as the key, the values are a list of the cells of this perturbation assigned to the technical duplicate.
    gt_index_dict : dict[str, list[str]]
        Same as td_indices, but for the ground truth.
    """    
    td_index_dict = {}
    gt_index_dict = {}

    non_existent_perturbations = []
    skipped_perturbations = []
    
    for pert in test_perts:
        pert_indices = pert_i_dict[pert]
        
        # Perturbations with too few cells are skipped:
        if len(pert_indices) == 0:
            non_existent_perturbations.append(pert)
            continue
        elif len(pert_indices) < 10:
            skipped_perturbations.append(pert)
            continue
        
        # Split the cells into technical duplicate and ground truth
        random.shuffle(pert_indices)
        td_size = np.ceil(td_ratio*len(pert_indices)).astype(int)
        td_indices = list(pert_indices[:td_size])
        gt_indices = list(pert_indices[td_size:])

        td_index_dict[pert] = td_indices
        gt_index_dict[pert] = gt_indices

    # Print which entries in test_perts were skipped:
    if len(non_existent_perturbations)>0:
        print("The following perturbations do not exist: ", end = "")
        for p in non_existent_perturbations:
            print(p,end="\t")
        print("\n",end="")
    
    if len(skipped_perturbations)>0:
        print("The following perturbations were skipped due to a too low number of cells (<10): ", end = "")
        for p in skipped_perturbations:
            print(p,end="\t")
        print("\n",end="")

    return td_index_dict, gt_index_dict
        
        

# Generate a technical duplicate (positive control) for a certain perturbation:
def technical_duplicate(norm_data: ad.AnnData, td_index_dict : dict[str,list[str]], gt_index_dict : dict[str,list[str]])  -> tuple[dict, dict]:
    """
    Generate the mean expression levels of each gene in the technical duplicate and the ground truth.

    Parameters
    ----------
    norm_data : anndata
        The normalized anndata object.    
    td_index_dict : dict[str,list[str]]
        For each perturbed genes as the key, the values are a list of the cells of this perturbation assigned to the technical duplicate.
    gt_index_dict : dict[str, list[str]]
        Same as td_indices, but for the ground truth.

    Returns
    -------
    mu_td_dict : dict[str, np.ndarray[np.float64]]
        A dictionary with the perturbations in test_perts as keys and an array containing the mean of each gene in the technical duplicate cells of the perturbation as values.
    mu_gt_dict : dict[str, np.ndarray[np.float64]]
        Same as mu_td_dict, buth with the mean of each gene in the ground truth cells.
    """
    mu_td_dict = {}
    mu_gt_dict = {}

    for pert in td_index_dict.keys():
        # Calculate the means
        mu_td = np.asarray(np.mean(norm_data[td_index_dict[pert]].layers["log1p"],axis=0)).squeeze()
        mu_gt = np.asarray(np.mean(norm_data[gt_index_dict[pert]].layers["log1p"],axis=0)).squeeze()
        
        # Add the mean counts to the dict
        mu_td_dict[pert] = mu_td
        mu_gt_dict[pert] = mu_gt
    
    return mu_td_dict, mu_gt_dict



def interpolated_duplicate(norm_data : AnnData, mu_td_dict : dict[str, np.ndarray[np.float64]], mu_all : np.ndarray[np.float64], td_index_dict : dict[str,list[str]]) -> dict[str, np.ndarray[np.float64]]:
    """
    Calculates the interpolated duplicate for each perturbation i through the equation
        alpha_i * mu_td_i + (1 - alpha_i) * mu_all,
    where alpha_i = 1 - p(DEGs, i) is derived for each gene from the probability of it being a DEG when comparing the control cells and the cells of perturbation i

    Parameters:
    -----------
    norm_data : anndata
        The normalized anndata object.    
    mu_td_dict : dict[str, np.ndarray[np.float64]]
        An dictionary with the perturbations in test_perts as keys and an array containing the mean of each gene in the technical duplicate cells of the perturbation as values.
    mu_all : np.ndarray[np.float64]
        The mean baseline of norm_data for a specified layer
    td_index_dict : dict[str, list[str]]
        A dictionary with the perturbations in test_perts as keys and an array containing the mean of each gene in the technical duplicate cells of the perturbation as values.

    Returns
    -------
    mu_id_dict : dict[str, np.ndarray[np.float64]]
        A dictionary of the same shape as mu_td_dict, but with the interpolated duplicate instead of the technical duplicate.
    """

    td_cells = np.concatenate(list(td_index_dict.values())).tolist()
    this_norm_data = norm_data[td_cells]

    sc.tl.rank_genes_groups(
        this_norm_data,
        groupby = 'perturbation',
        method = 't-test_overestim_var',
        reference = 'rest',
        layers="log1p")

    mu_id_dict = {}
    for keys, values in mu_td_dict.items():
        # this_norm_data.uns["rank_genes_groups"] contains the p values for the DEGs of each perturbation
        pvals = this_norm_data.uns["rank_genes_groups"]["pvals_adj"][keys][np.argsort(this_norm_data.uns["rank_genes_groups"]["names"][keys])]
        mu_id_dict[keys] = (1-pvals)*values + pvals*mu_all
    return mu_id_dict
    
"""
# Small Test
input = ad.read_h5ad("../Data/AdamsonWeissman2016_GSM2406681_10X010.h5ad")
input.layers["raw_layer"] = input.X
norm = sc.pp.normalize_total(input, target_sum=None, copy=True)
input.layers["norm_layer"] = norm.X

print(mean_baseline(norm_data=input))
print(mean_baseline(norm_data=input))


mu_td, mu_gt = technical_duplicate(adata_norm_median,"OST4_pDS353")
"""








