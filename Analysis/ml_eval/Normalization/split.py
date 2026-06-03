import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import random
from typing import Dict, Generator
import pickle
import hdf5plugin
import os

# Returns the indices of the control cells, and a dictionary that contains the name of each perturbation together with the corresponding indices
def ctrl_pert_split_dataset(norm_data: ad.AnnData, bad_perts : list[str] = None) -> tuple[list[str], dict[str,list[str]]]:
    """
    Split the norm_data.obs dataframe into control cells and each perturbation.

    Parameters
    ----------
    norm_data : anndata
        The anndata object containing the data.
    bad_perts : list
        A list of perturbations at least one of the models cannot process.

    Returns
    -------
    ctrl_i_vec : list[str]
        A vector of all indices of control cells in norm_data.obs (corresponds to the rows in norm_data.X).
    pert_i_dict : dict[str, list[str]]
        A dictionary with the perturbation name and the indices in norm_data.obs where this perturbation is found as key value pairs.
    """
    # Split df into perturbed and control cells
    
    by_pert = [pert for _, pert in norm_data.obs.groupby(norm_data.obs["perturbation"])]
    # Find indices for each perturbation
    pert_i_dict = {pert.iloc[0]["perturbation"]:pert.index.to_list() for pert in by_pert}
    # Extract the control cells separately
    ctrl_i_vec = pert_i_dict.pop("ctrl")

    # Optional: remove all perturbations that do not appear in a dataset
    if bad_perts is not None:
        for bp in bad_perts:
            pert_i_dict.pop(bp[:-5]) # the last 5 letters are +ctrl and need to be ignored
    
    return (ctrl_i_vec, pert_i_dict)

# Perform cross-validation by randomly splitting the perturbations each time.
def cross_validate_split_random(norm_data: ad.AnnData, pert_i_dict: Dict[str, list[str]], no_iterations: int = 5, need_val: bool = False, train_ratio: float = 0.8) -> Generator[tuple[list[str], list[str], list[str]], None, None]:
    """
    --- Deprecated ---
    Assign each perturbation randomly to the train dataset, the validation dataset, and the test dataset.
    train_ratio% of perturbations will be assigned to the train dataset.
    If no validation dataset is needed, the remaining perturbations will yield the test dataset (standard: 80/20 split).
    Otherwise, half of the remaining perturbations will be the validation, and half the test dataset (standard: 80/10/10 split).
    For cross validation, this split will be repeated no_iterations times, using the yield keyword.
    
    Parameters
    ----------
    norm_data : anndata
        The anndata object containing the normalized data. 
    pert_i_dict : Dict[str, list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values.
    no_iterations : int
        The number of times the perturbations will be randomly split (for cross validation).
    need_val : bool
        A bool indicating whether or not validation data is needed.
    train_ratio : float
        The percentage of perturbations assigned to the train dataset.
    
    Yields
    ------
    train_perts
        The names of perturbations in the train dataset.
    val_perts
        The names of perturbations in the validation dataset.
    train_perts
        The names of perturbations in the train dataset.
    """
    # Set the perturbation names and size of the train dataset
    perts = list(pert_i_dict.keys())
    train_size = np.ceil(train_ratio*len(perts)).astype(int)
    
    for it in range(no_iterations):
        # Shuffle the perturbations and get the train dataset
        random.shuffle(perts)
        train_perts = perts[:train_size]
        test_val_perts = perts[train_size:]
        
        # If a validation dataset is needed, the remaining perturbations are split 50/50
        if need_val:
            val_perts = test_val_perts[len(test_val_perts)//2:]
            test_perts = test_val_perts[:len(test_val_perts)//2]
        # Otherwise, the validation dataset is empty
        else:
            val_perts = []
            test_perts = test_val_perts
        
        yield train_perts, val_perts, test_perts
        
        
# Perform cross-validation by splitting the perturbations into fix blocks.
def cross_validate_split_blockwise(norm_data: ad.AnnData, pert_i_dict: Dict[str, list[str]], filename : str, modelname : str,no_iterations: int = 5, need_val: bool = False) -> list[tuple[list[str],list[str],list[str]]]:
    """
    Split all perturbations into no_iterations blocks.
    For cross validation, all but one block will consolidate the train dataset.
    If no validation dataset is needed, the remaining block will be the test dataset.
    Otherwise, the first half of the remaining block will be the validation dataset, and the other half the test dataset.
    Each block will form the validation/test datasets only once, meaning that no_iterations is equivalent to the number of blocks.
    All datasets are returned via a list of dictionaries.
    
    Parameters
    ----------
    norm_data : anndata
        The anndata object containing the normalized data. 
    pert_i_dict : Dict[str, list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values.
    filename : str
        The name of the dataset used
    no_iterations : int
        The number of iterations for cross-validation. Corresponds the number of blocks the perturbations will be split into.
    need_val : bool
        A bool indicating whether or not validation data is needed.
    
    Returns
    -------
    splits : list
        A list of tuples with the perturbation names for train, val, and test for each split.
    """
    # Set the perturbation names and find the indices of the list where each block will begin/end (standard: [0, n/5, 2n/5, 3n/5, 4n/5, n])
    perts = list(pert_i_dict.keys())
    block_borders = [np.ceil(i/no_iterations*len(perts)).astype(int) for i in range(no_iterations)]
    block_borders.append(len(perts))
    random.shuffle(perts)
    splits = []
    
    for it in range(no_iterations):
        # Each block is once assigned to the validation/test datasets,
        train_perts = perts[:block_borders[it]] + perts[block_borders[it+1]:]
        val_test_perts = perts[block_borders[it]:block_borders[it+1]]
        
        # If a validation dataset is needed, the remaining perturbations are split 50/50
        if need_val:
            val_perts = val_test_perts[len(val_test_perts)//2:]
            test_perts = val_test_perts[:len(val_test_perts)//2]
        # Otherwise, the validation dataset is empty
        else:
            val_perts = train_perts
            test_perts = val_test_perts

        splits.append((train_perts, val_perts, test_perts))

        # Create and pickle the splits in the format used for GEARS
        os.makedirs(f"Models/Splits/{filename}/{modelname}",exist_ok=True)
        with open(f'Models/Splits/{filename}/{modelname}/split_{it}.pickle','wb') as handle:
            train_perts = [p.rename("_","+") if "_" in p or "+" in p else p+"+ctrl" for p in train_perts]
            val_perts = [p.rename("_","+") if "_" in p or "+" in p else p+"+ctrl" for p in val_perts]
            test_perts = [p.rename("_","+") if "_" in p or "+" in p else p+"+ctrl" for p in test_perts]
            
            pickle.dump({'train':train_perts,'val':val_perts,'test':test_perts}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return splits

def hvg_subsets(norm_data : ad.AnnData, filename : str, no_iterations : int):
    print("Generating HVGs for the perts of each training set...")
    genes_per_it = []
    for i in range(no_iterations):
        print(f"{i+1}/{no_iterations}")

        splits = pickle.load(open(f"Models/Splits{filename}/split_{i}.pickle",'rb'))
        train_perts = list([p[:-5] for p in splits["train"]])
        dataset = norm_data[norm_data.obs["perturbation"].isin(train_perts)]
        sc.pp.highly_variable_genes(dataset,n_top_genes=5000, subset=True,layer="log1p")
        genes_per_it.append(set(dataset.var_names))
        
        ad.settings.allow_write_nullable_strings = True

        norm_data.write_h5ad(
            f"Data/{filename}/perturb_norm_hvg_{i}.h5ad",
            compression=hdf5plugin.FILTERS["zstd"]
        )
    print(f"The union of the HVGs contains {len(set().union(*genes_per_it))} genes.")
