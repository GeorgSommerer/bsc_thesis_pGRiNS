import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

import scanpy as sc
import anndata as ad
import hdf5plugin

import os
from glob import glob
import argparse

import normalize


def prepare_datasets(grn_file, topo_files : list[str], adata_dict : dict = None, subset : str = "Union"):
    """
    Takes a GRN and normalized anndata objects as inputs.
    After updating the gene names, all genes in the anndata objects not in the GRN are removed.
    From the GRN, genes not present in any | all of the adata sets are removed.
    
    Parameters:
    -----------
    grn_df : str
        The name of the combined GRN.
    topo_files : list[str]
        A list of names of .topo files to be combined. They must have the column names "Source", "Target", and "Type", where Type=1 for an activating and Type=2 for an inhibiting regulation.
    adata_dict : dict[str,anndata.AnnData]
        A dictionary of all the dataset names as keys and the normalized adata files as values.
    subset : str
        How the genes are subset. Choices are ["union", "intersection"].

    Returns:
    --------
    grn_df : DataFrame
        A pandas dataframe containing all the unique rows
    adada_dict : dict[str,anndata.AnnData]
        The input adata_dict (if not None), but with subset genes and filtered by the top 5000 HVGs.
    """
    assert subset.lower() in ["union","intersection"]

    topo_dfs = [pd.read_csv(f"Data/Topos/{grn_file}.topo",sep=" ") for grn_file in topo_files]
    grn_df = pd.concat(topo_dfs,axis=0).drop_duplicates(subset=["Source","Target"]).sort_values(by=["Source","Target"])

    if adata_dict is None:
        grn_df.to_csv(f"Data/{grn_file}/{grn_file}.topo",sep=" ",index=False)

        return grn_df

    else: 
        grn_genes = set(list(grn_df["Source"])+list(grn_df["Target"]))

        # Remove genes not in GRN from adata sets
        adata_genes_list = []
        for pseq, adata in adata_dict.items():
            adata_genes = set(adata.var_names)
            adata = adata[:,sorted(list(grn_genes & adata_genes))]
            adata_genes_list.append(adata_genes)
            
            # Find highly variable genes and save new datasets
            sc.pp.highly_variable_genes(adata,n_top_genes=5000, subset=True,layer="log1p")
            adata_dict[pseq] = adata
            if not os.path.isfile(f"Data/Experimental/{pseq}/perturb_subset_{grn_file}.h5ad"):
                ad.settings.allow_write_nullable_strings = True
                adata.write_h5ad(
                    f"Data/Experimental/{pseq}/perturb_subset_{grn_file}.h5ad",
                    compression=hdf5plugin.FILTERS["zstd"]
                )

        # Remove genes not in datasets from GRN:
        if subset.lower() == "union":
            all_adata_genes = set().union(*adata_genes_list)
        elif subset.lower() == "intersection":
            all_adata_genes = set().union(*adata_genes_list).intersection(*adata_genes_list)
        
        # Only remove a gene from the GRN if it is not in the datasets AND if it has no outgoing edges, otherwise important relationships in the graph could be lost
        grn_remaining_genes = set(grn_genes & all_adata_genes)
        grn_source_genes = set(grn_df["Source"].values)

        grn_df = grn_df[grn_df["Source"].isin(list(grn_remaining_genes | grn_source_genes))]
        grn_df = grn_df[grn_df["Target"].isin(list(grn_remaining_genes | grn_source_genes))]

        grn_df.to_csv(f"Data/{grn_file}/{grn_file}.topo",sep=" ",index=False)

        return grn_df, adata_dict


def extract_pert_info(grn_file : str) -> list:
    """
    Taking an input file called Data/pert_list.pert as the input, the perturbations in the file are turned into a list of perturbation set, where
    each set is represented as a dict with the gene as the key and the perturbation type as the value.
    Perturbation sets where the perturbed genes are not in the GRN are removed.

    Parameters:
    -----------
    grn_file : str
        The name of the project.
    
    Returns:
    --------
    pert_list : list
        A list of dicts where each dict contains the genes and perturbation types of a perturbation set.
    """

    perts_df = pd.read_csv("Data/pert_list.pert",sep=" ")
    pert_list = [{} for i in range(1+max(perts_df["Index"]))]

    grn = pd.read_csv(f"Data/{grn_file}/{grn_file}.topo",sep=" ")
    grn_genes = sorted(list(set(grn["Source"])|set(grn["Target"])))
    
    for index, row in perts_df.iterrows():
        if row["Gene"] not in grn_genes:
            raise Exception(f"Gene {row["Gene"]} at index {row["Index"]} not in GRN.")
        pert_list[row["Index"]][row["Gene"]]=row["Type"]
    
    return pert_list


def perts_from_adata():
    """
    Turns certain perturbations from an adata object into a list that is inputtable into extract_pert_info.
    """
    pass

def main(project_name, experimental):
    if experimental:
        pseqs = sorted([pseq_path.split("/")[-1] for pseq_path in glob(f"Data/Experimental/*")])
        print(pseqs)

    topo_files = sorted([pseq_path.split("/")[-1].split(".")[0] for pseq_path in glob(f"Data/Topos/*.topo")])
    print(topo_files)

    # Normalize perturb seq adata:
    if experimental:
        norm_data_dict = {}
        for pseq_file in pseqs:
            norm_data_dict[pseq_file] = normalize.normalize_adata_main(pseq_file)
    else:
        norm_data_dict = None

    grn_df, norm_data_dict = prepare_datasets(project_name, topo_files, norm_data_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e","--experimental", help="Whether or not perturb-seq control data is supplied",action="store_true")
    args = parser.parse_args()

    project_name = args.grn
    experimental = args.experimental
    print(extract_pert_info(project_name))
    #main(project_name, experimental)