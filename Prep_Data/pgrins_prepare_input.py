import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

import scanpy as sc
import anndata as ad

import os
import sys
from glob import glob
import argparse

try:
    import normalize
except:
    from Prep_Data import normalize


def subset_from_adata(grn_df : pd.DataFrame, adata_dict : dict[str,ad.AnnData], project_name : str, max_missingness : int = 90) -> tuple[pd.DataFrame, dict[str,ad.AnnData]]:
    """
    From the experimental datasets, removes all genes with a high amount of missing entries, as well as genes not present in grn_df.
    Then, perturbations where perturbed genes have been removed are removed as well.
    Finally, all nodes from the GRN that are not in the experimental datasets are removed, except for nodes in the Source column: if they are removed, important interactions in the GRN might get lost.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    adata_dict : dict[str,ad.AnnData]
        A dictionary with the experimental dataset names as keys and the adata objects as values.
    project_name : str
        The name of the project.
    max_missingness : int, optional
        The maximal percentage of cells per gene that can have missing entries.

    Returns:
    --------
    grn_df : pd.DataFrame
        The subset GRN.
    adata_dict : dict[str,ad.AnnData]
        The subset experimental datasets.
    """

    print("Created subset adata files...")

    # Remove genes that are missing:
    for name, adata in adata_dict.items():
        adata_dict[name] = adata[:,adata.var["pct_dropout_by_counts"]<max_missingness]

    adata_gnames = set().union(*[set(adata.var_names) for adata in adata_dict.values()])
    grn_df_adata = grn_df[(grn_df["Source"].isin(adata_gnames)) & (grn_df["Target"].isin(adata_gnames))] # Keep all rows where both genes are in the adata datasets
    grn_df_plus_sources = grn_df[grn_df["Target"].isin(set(list(grn_df["Source"])))] # Also keep all rows that impact outgoing edges, otherwise important relationships in the graph could be lost
    
    grins_gnames = set(list(grn_df_adata["Source"])) | set(list(grn_df_adata["Target"]))

    for pseq, adata in adata_dict.items():
        # Remove genes in adata not present in grn
        adata_genes = set(adata.var_names)
        adata = adata[:,sorted(list(grins_gnames & adata_genes))]

        # Remove perturbations for which genes are not expressed
        kept_perts = ["ctrl"]
        for pert in adata.obs["perturbation"].unique():
            if False not in [(p in list(grn_df_adata["Source"])) & (p in adata_genes) for p in pert.split("_")]: # All perturbed genes of a perturbation set must be both among the source nodes of the GRN, and among the genes in adata
                kept_perts.append(pert)
        adata = adata[adata.obs["perturbation"].isin(kept_perts)]
        adata_dict[pseq] = adata

        print(f"Save subset of {pseq}...")
        adata_dict[pseq].write_h5ad(
            f"Data/Experimental/{pseq}/perturb_norm_subset_{project_name}.h5ad"
        )     

    grn_df = pd.concat([grn_df_adata,grn_df_plus_sources]).drop_duplicates()

    return grn_df, adata_dict



def extract_pert_info(project_name : str, pert_file : str) -> list[dict[str,int]]:
    """
    Taking a .pert file as the input, the perturbations in the file are turned into a list of perturbation sets, where
    each set is represented as a dict with the perturbed gene as the key and the perturbation type as the value.

    Parameters:
    -----------
    project_name : str
        The name of the project.
    pert_file : str, optional
        The name of the .pert file.
    
    Returns:
    --------
    pert_list : list
        A list of dicts where each dict contains the genes and perturbation types of a perturbation set.
    """

    perts_df = pd.read_csv(f"Data/Perts/{pert_file}.pert",sep=" ")
    pert_list = [{} for i in range(1+max(perts_df["Index"]))]
    for index, row in perts_df.iterrows():
        pert_list[row["Index"]][row["Gene"]]=(row["Type"])
    
    return pert_list



def get_perts(grn_df : pd.DataFrame, project_name : str, adata_dict : dict[str,ad.AnnData]):
    """
    If no .pert file was provided, all perturbation sets remaining in the experimental datasets after subsetting are compiled into a .pert file.

    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    project_name : str
        The name of the project.
    adata_dict : dict[str,ad.AnnData]
        A dict with the experimental dataset names as keys and the adata objects as values.

    Returns:
    --------
    None
    """

    if not os.path.exists(f"Data/Perts/{project_name}_perts.pert"): # Generate perturbation .pert file if not specified
        print("Creating new .pert file...")
        index = []
        current_index = 0
        gene = []
        types = []
        for pseq, adata in adata_dict.items():
            pert_sets = sorted([pert.split("_") for pert in set(list(adata.obs["perturbation"]))])
            pert_sets.remove(["ctrl"])

            if "Norman" in pseq:
                typ = 1
            elif "Replogle" in pseq or "Adamson" in pseq:
                typ = 2
            else:
                typ = int(input(f"Input the type of perturbation in the dataset {pseq} (1: CRISPRa, 2: CRISPRi, 3: CRISPR-KO)"))
                assert typ in [1,2,3]
            for pert_set in pert_sets:
                index += [current_index]*len(pert_set)
                current_index += 1
                gene += pert_set
                types += [typ]*len(pert_set)
        pert_df = pd.DataFrame({"Index":index,"Gene":gene,"Type":types}).sort_values(by=["Index","Gene"])
        pert_df.to_csv(f"Data/Perts/{project_name}_perts.pert",index=False,sep=" ")# Takes the first part of the dataset name (before the _) as the identifier



def split_sinks_from_grn(grn_df : pd.DataFrame, project_name : str) -> pd.DataFrame:
    """
    Removes all the sink nodes from the Dataframe and saves them in a separate file in Data/Projects/{project_name}/{project_name}_sinks.topo.
    The full GRN will be in Data/Projects/{project_name}/{project_name}_full.topo.
    Data/Projects/{project_name}/{project_name}.topo will contain all nonsinks edges, plus additional edges for all source genes (that have no incoming edge) that only point towards sink nodes, as they would otherwise not be simulated.
    These source-to-sink nodes are included with edges pointing towards a dummy node which is removed befeore simulation.
    Since sink nodes are not necessary to simulate the steady state of non-sink nodes and their own steady state can be calculated algebraically as long as parameters are sampled, this saves computational cost.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    project_name : str
        The name of the project.

    Returns:
    --------
    grn_df : pd.DataFrame
        The input dataframe without edges to sink nodes.
    """

    grn_df.to_csv(f"Data/Projects/{project_name}/{project_name}_full.topo",sep=" ",index=False)
    print("Splitting sinks from GRN...")
    sinks = set(list(grn_df["Target"])) - set(list(grn_df["Source"])) # All genes that have an outdeg of 0
    sink_df = grn_df[grn_df["Target"].isin(sinks)]
    sink_df.to_csv(f"Data/Projects/{project_name}/{project_name}_sinks.topo",sep=" ",index=False)
    
    grn_df = grn_df[~grn_df["Target"].isin(sinks)] # Remove these sinks from the df

    sources = set(list(sink_df["Source"])) - (set(list(grn_df["Source"]))|set(list(grn_df["Target"]))) # All genes from sink_df that do not appear in grn_df (lone sources)
    source_df = pd.DataFrame({"Source":list(sources),"Target":"dummy","Type":1}) # Sink nodes must also be included in grn_df
    grn_df = pd.concat([grn_df,source_df]).sort_values(by=["Source","Target"])
    return grn_df



def main(project_name : str, experimental : bool, make_pert_file : bool = False, split_sinks : bool = False, max_missingness : int = 90):
    """
    Creates a new project folder in Data/{project_name}, in which the input topos from Data/Topos are concatenated.
    If experimental datasets are provided in Data/Experimental, they are normalized and used for various subsetting procedures designed to remove unnecessary edges and nodes from the GRN.
    If requested, a .pert file of all perturbed genes in the subset datasets is created in Data/Perts/{project_name}_perts.pert.
    If split_sinks is True, edges pointing towards sink nodes are put into a separate GRN.

    The subset experimental datasets are saved in Data/Experimental/{pseq}/{pseq}_subset_{project_name}.h5ad.
    The subset and filtered GRN is saved in Data/{project_name}/{project_name}.topo.

    Parameters:
    -----------
    project_name : str
        The name of the project. From now on equivalent to grn_file, which refers to the storage location of grn_df.
    experimental : bool
        Whether or not experimental data is provided.
    pert_file : str
        Name of the newly created .pert file from experimental datasets.
    split_sinks : bool, optional
        Whether or not the sink nodes should be treated separately and not simulated in Racipe.
    max_missingness : int, optional
        The maximal percentage of cells per gene that can have missing entries.
    
    Returns:
    --------
    None
    """

    if not os.path.exists(f"Data/Projects/{project_name}/{project_name}.topo"):
        os.makedirs(f"Data/Projects/{project_name}",exist_ok=True)

        topo_files = sorted([pseq_path.split("/")[-1].split(".")[0] for pseq_path in glob(f"Data/Topos/*.topo")])
        print(f".topo files {topo_files} are turned into {project_name}")

        # Concatenate the input .topo files into grn_df
        topo_dfs = [pd.read_csv(f"Data/Topos/{topo_file}.topo",sep=" ") for topo_file in topo_files]
        grn_df = pd.concat(topo_dfs,axis=0).drop_duplicates(subset=["Source","Target"]).sort_values(by=["Source","Target"])

        # Gene names that contain symbols that cannot be in python variable names must be removed (because GRiNS turns the gene names into python variable names)
        grn_df = grn_df[(~grn_df["Source"].str.contains(r"\-|\.",regex=True)) & (~grn_df["Target"].str.contains(r"\-|\.",regex=True))]

        # -e: Get experimental perturb seq datasets and normalize them
        if experimental:
            pseqs = sorted([pseq_path.split("/")[-1] for pseq_path in glob(f"Data/Experimental/*")])
            print(f"Experimental perturb seq datasets used: {pseqs}")
            adata_dict = {pseq_file : normalize.normalize_adata_main(pseq_file) for pseq_file in pseqs}
        else:
            adata_dict = None
        # Remove genes from GRN not in datasets
        if adata_dict is not None:
            grn_df, adata_dict = subset_from_adata(grn_df, adata_dict, project_name, max_missingness=max_missingness)

        # -p: Get a list of perturbations of genes of interest that are in the GRN
        if make_pert_file:
            get_perts(grn_df, project_name, adata_dict)
            
        # -s: Split sink nodes into separate .topo file
        if split_sinks:
            grn_df = split_sinks_from_grn(grn_df, project_name)

        # Save the output data structures to files:
        grn_df.to_csv(f"Data/Projects/{project_name}/{project_name}.topo",sep=" ",index=False)
    else:
        print("Done!")