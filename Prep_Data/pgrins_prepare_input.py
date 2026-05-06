import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

import scanpy as sc
import anndata as ad
import hdf5plugin

import os
import sys
from glob import glob
import argparse

try:
    import normalize
except:
    from Prep_Data import normalize

def extract_pert_info(project_name : str, pert_file : str) -> list:
    """
    Taking a .pert file as the input, the perturbations in the file are turned into a list of perturbation sets, where
    each set is represented as a dict with the gene as the key and the perturbation type as the value.
    Perturbation sets where the perturbed genes are not in the GRN are removed.

    Parameters:
    -----------
    project_name : str
        The name of the project.
    pert_file : str, optional
        If None and experimental is True, a new .pert file is created from the perturbed genes of interestin the experimental datasets.
        Otherwise, Data/Perts/pert_file.pert is loaded and potentially used for downstream sampling.
    
    Returns:
    --------
    pert_list : list
        A list of dicts where each dict contains the genes and perturbation types of a perturbation set.
    """

    perts_df = pd.read_csv(f"Data/Perts/{pert_file}.pert",sep=" ")
    pert_list = [{} for i in range(1+max(perts_df["Index"]))]

    grn = pd.read_csv(f"Data/Projects/{project_name}/{project_name}.topo",sep=" ")
    grn_genes = sorted(list(set(grn["Source"])|set(grn["Target"])))
    
    for index, row in perts_df.iterrows():
        if row["Gene"] not in grn_genes:
            raise Exception(f"Gene {row["Gene"]} at index {row["Index"]} not in GRN.")
        pert_list[row["Index"]][row["Gene"]]=row["Type"]
    
    return pert_list


def get_perts(grn_df : pd.DataFrame, project_name : str, pert_file : str, adata_dict : dict[str,ad.AnnData]) -> tuple[dict[str,ad.AnnData], list[str]]:
    """
    Either reads a list of perturbations from a given .pert file, or generates it from the intersection of all non-sink genes fron GRN and the perturbed genes fron the experimental datasets and saves it in Data/Perts/project_name.pert.
    
    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    project_name : str
        The name of the project.
    adata_dict : dict[str,ad.AnnData]
        A dict with the experimental dataset names as keys and the adata objects as values.
    pert_file : str, optional
        If None and experimental is True, a new .pert file is created from the perturbed genes of interestin the experimental datasets.
        Otherwise, Data/Perts/pert_file.pert is loaded and potentially used for downstream sampling.


    Returns:
    --------
    adata_dict : dict[str, ad.AnnData]
        The input dict without the perturbations not in the GRN.
    all_perts : list[str]
        A list of all perturbed genes that will be used for pGRiNS.
    """
    if pert_file is None: # Generate perturbation .pert file if not specified
        print("Creating new .pert file...")
        index = []
        current_index = 0
        gene = []
        types = []
        for pseq, adata in adata_dict.items():
            pert_sets = [pert.split("_") for pert in set(list(adata.obs["perturbation"]))]
            pert_sets_in_grn = [pert_set for pert_set in pert_sets if sum([pert not in list(grn_df["Source"]) for pert in pert_set])==0 and sum([pert not in list(adata.var_names) for pert in pert_set])==0]
                # Requirements: All perts from the pert set must be among the sources nodes and actually expressed in the dataset they come from (looking at you, Replogle)

            # Remove unused perts from adata subset
            adata_dict[pseq] = adata[adata.obs["perturbation"].isin([*["_".join(perts) for perts in pert_sets],"ctrl"])]

            if "Norman" in pseq:
                typ = 1
            elif "Replogle" in pseq or "Adamson" in pseq:
                typ = 2
            else:
                typ = int(input(f"Input the type of perturbation in the dataset {pseq} (1: CRISPRa, 2: CRISPRi, 3: CRISPR-KO)"))
                assert typ in [1,2,3]
            for pert_set in pert_sets_in_grn:
                index += [current_index]*len(pert_set)
                current_index += 1
                gene += pert_set
                types += [typ]*len(pert_set)
        pert_df = pd.DataFrame({"Index":index,"Gene":gene,"Type":types}).sort_values(by=["Index","Gene"])
        pert_df.to_csv(f"Data/Perts/{project_name}_perts.pert",index=False,sep=" ")# Takes the first part of the dataset name (before the _) as the identifier
        all_perts = gene
    else: # Otherwise read existing .pert file
        print("Reading .pert file...")
        all_perts = list(set(list(pd.read_csv(f"Data/Perts/{pert_file}.pert,",sep=" ")["Gene"])))
    return adata_dict, all_perts



def downstream_filtering(grn_df : pd.DataFrame, upstream_genes : list[str], downstream_depth : int) -> pd.DataFrame:
    """
    Given a list of perturbed genes, this function returns all the edges along paths from these perturbed genes to (other) genes with a length of <= downstream_depth edges.
    In other words, only edges closely downstream from the perturbed genes of interest are kept in order to keep the ODE small.

    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    upstream_genes : list[str]
        A list of genes whose adjacent nodes downstream should be found. Starts with the pGOIs.
    downstream_depth : int, optional
        How long the paths from pGOIs that are kept are. If None, all are kept.

    Returns:
    --------
    grn_df : pd.DataFrame
        A subset of the edges (rows) from the input dataframe.
    """

    print(f"Performing downstream filtering at depth {downstream_depth}...")
    edges_kept = []
    for i in range(downstream_depth):
        new_edges = grn_df[grn_df["Source"].isin(upstream_genes)]
        edges_kept.append(new_edges)
        upstream_genes = list(set(list(new_edges["Target"])))
    grn_df = pd.concat(edges_kept).drop_duplicates().sort_values(by=["Source","Target"])
    return grn_df



def subset_from_adata(grn_df : pd.DataFrame, adata_dict : dict[str,ad.AnnData], subset_method : str) -> tuple[pd.DataFrame,dict[str,ad.AnnData]]:
    """
    Given a set of experimental datasets, this function removes all genes from them not in the GRN, and from the GRN it removes all genes not in any (union) or all (intersection) datasets.

    Parameters:
    -----------
    grn_df : pd.DataFrame
        The GRN.
    adata_dict : dict[str,ad.AnnData]
        A dict with the experimental dataset names as keys and the adata objects as values.
    subset_method : str, optional
        Whether or not genes from the GRN should be kept only if they are present in all experimental datasets (intersection), or any of them (union).

    Returns:
    --------
    grn_df : pd.DataFrame
        The input GRN, but without the genes not in the datasets.
    adata_dict : dict[str,ad.AnnData]]
        The input datasets, but without genes not in grn_df.
    """

    print("Created subset adata files...")
    grn_genes = set(list(grn_df["Source"])+list(grn_df["Target"]))
    # Remove genes not in GRN from adata sets
    adata_genes_list = []
    for pseq, adata in adata_dict.items():
        adata_genes = set(adata.var_names)
        adata = adata[:,sorted(list(grn_genes & adata_genes))]
        adata_genes_list.append(adata_genes)
        
        # Find highly variable genes and save new datasets
        #sc.pp.highly_variable_genes(adata,n_top_genes=5000, subset=True,layer="log1p")
        adata_dict[pseq] = adata

    # Remove genes not in datasets from GRN:
    if subset_method.lower() == "union":
        all_adata_genes = set().union(*adata_genes_list)
    elif subset_method.lower() == "intersection":
        all_adata_genes = set().union(*adata_genes_list).intersection(*adata_genes_list)
    
    # Only remove a gene from the GRN if it is not in the datasets AND if it has no outgoing edges, otherwise important relationships in the graph could be lost
    grn_remaining_genes = set(grn_genes & all_adata_genes)
    grn_source_genes = set(grn_df["Source"].values)

    grn_df = grn_df[grn_df["Source"].isin(list(grn_remaining_genes | grn_source_genes))]
    grn_df = grn_df[grn_df["Target"].isin(list(grn_remaining_genes | grn_source_genes))]

    return grn_df, adata_dict



def split_sinks_from_grn(grn_df : pd.DataFrame, project_name : str) -> pd.DataFrame:
    """
    Removes all the sink nodes from the Dataframe and saves them in a separate file in Data/Projects/project_name/project_name_sinks.topo.
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

    print("Splitting sinks from GRN...")
    sinks = set(list(grn_df["Target"])) - set(list(grn_df["Source"])) # All genes that have an outdeg of 0
    sink_df = grn_df[grn_df["Target"].isin(sinks)]
    sink_df.to_csv(f"Data/Projects/{project_name}/{project_name}_sinks.topo",sep=" ",index=False)
    
    grn_df = grn_df[~grn_df["Target"].isin(sinks)] # Remove these sinks from the df

    sources = set(list(sink_df["Source"])) - (set(list(grn_df["Source"]))|set(list(grn_df["Target"]))) # All genes from sink_df that do not appear in grn_df (lone sources)
    source_df = pd.DataFrame({"Source":list(sources),"Target":"dummy","Type":1}) # Sink nodes must also be included in grn_df
    grn_df = pd.concat([grn_df,source_df]).sort_values(by=["Source","Target"])
    return grn_df


def main(project_name : str, experimental : bool, pert_file : str = None, downstream_depth : int = None, subset_method : str = "Union", split_sinks : bool = False):
    """
    Creates a new project folder in Data/project_name, in which the input topos from Data/Topos are concatenated.
    If experimental data is provided in Data/Experimental, it is normalized and used for various subsetting procedures designed to remove unnecessary edges and nodes from the GRN.
    If required, a .pert file of all perturbed genes of interest (pGOIs) is created in Data/Perts/project_name_perts.pert.

    The subset experimental datasets are saved in Data/Experimental/pseq/pseq_subset_project_name.h5ad.
    The subset and filtered GRN is saved in Data/project_name/project_name.topo.

    Parameters:
    -----------
    project_name : str
        The name of the project. From now on equivalent to grn_file, which refers to the storage location of grn_df.
    experimental : bool
        Whether or not experimental data is provided.
    pert_file : str, optional
        If None and experimental is True, a new .pert file is created from the perturbed genes of interestin the experimental datasets.
        Otherwise, Data/Perts/pert_file.pert is loaded and potentially used for downstream sampling.
    downstream_depth : int, optional
        How long the paths from pGOIs that are kept are. If None, all are kept.
    subset_method : str, optional
        Whether or not genes from the GRN should be kept only if they are present in all experimental datasets (intersection), or any of them (union).
    split_sinks : bool, optional
        Whether or not the sink nodes should be treated separately and not simulated in Racipe.
    
    Returns:
    --------
    None
    """
    
    assert subset_method.lower() in ["union","intersection"]
    os.makedirs(f"Data/Projects/{project_name}",exist_ok=True)

    topo_files = sorted([pseq_path.split("/")[-1].split(".")[0] for pseq_path in glob(f"Data/Topos/*.topo")])
    print(f".topo files {topo_files} are turned into {project_name}")

    # Concatenate the input .topo files into grn_df
    topo_dfs = [pd.read_csv(f"Data/Topos/{topo_file}.topo",sep=" ") for topo_file in topo_files]
    grn_df = pd.concat(topo_dfs,axis=0).drop_duplicates(subset=["Source","Target"]).sort_values(by=["Source","Target"])

    # -e: Get experimental perturb seq datasets and normalize them
    if experimental:
        pseqs = sorted([pseq_path.split("/")[-1] for pseq_path in glob(f"Data/Experimental/*")])
        print(f"Experimental perturb seq datasets used: {pseqs}")
        adata_dict = {pseq_file : normalize.normalize_adata_main(pseq_file) for pseq_file in pseqs}
    else:
        adata_dict = None
    
    # -p: Get a list of perturbations of genes of interest that are in the GRN
    if adata_dict is not None or pert_file is not None:
        adata_dict, all_perts = get_perts(grn_df, project_name, pert_file, adata_dict)

        # -d: Remove nodes from the GRN not closely downstream of those GOIs
        if downstream_depth is not None:
            grn_df = downstream_filtering(grn_df, all_perts, downstream_depth)

    else:
        print("Neither a .pert file, nor experimental data to draw perturbations from are specified. Downstream depth sampling is not skipped.")

    # -e: Remove genes from GRN not in datasets, and genes from datasets not in GRN
    if adata_dict is not None:
        grn_df, adata_dict = subset_from_adata(grn_df, adata_dict,subset_method)

    # -s: Split sink nodes into separate .topo file
    if split_sinks:
        grn_df = split_sinks_from_grn(grn_df, project_name)

    # Save the output data structures to files:
    grn_df.to_csv(f"Data/Projects/{project_name}/{project_name}.topo",sep=" ",index=False)
    for pseq, adata in adata_dict.items():
        ad.settings.allow_write_nullable_strings = True
        adata.write_h5ad(
            f"Data/Experimental/{pseq}/perturb_subset_{project_name}.h5ad",
            compression=hdf5plugin.FILTERS["zstd"]
        )



if __name__ == "__main__":
    """
    Example:
    python3 Prep_Data/pgrins_prepare_input.py project_name -es -d 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("-e","--experimental", help="Whether or not perturb-seq control data is supplied",action="store_true")
    parser.add_argument("-p","--pert_file",help="The name of the .pert file in Data/Perts to use for genes of interest.")
    parser.add_argument("-d","--downstream_depth",type=int,help="Given a list of perturbed genes, keep only the genes with a minimal distance of d (and the edges along these paths)")
    parser.add_argument("--subset_method", help="Whether or not the GRN should only contains genes in all experimental datasets, or any of them (default: Union)")
    parser.add_argument("-s","--split_sinks", help="Whether or not sinks should be removed from the GRN",action="store_true")
    args = parser.parse_args()

    kwargs = {}
    project_name = args.grn
    if args.experimental:
        experimental = True
    else:
        experimental = False
    if args.split_sinks:
        kwargs["split_sinks"] = args.split_sinks
    if args.subset_method:
        kwargs["subset_method"] = args.subset_method
    if args.downstream_depth:
        kwargs["downstream_depth"] = args.downstream_depth
    if args.pert_file:
        kwargs["pert_file"] = args.pert_file
    main(project_name, experimental, **kwargs)