import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from scipy import sparse
import os
import hdf5plugin


# Normalizes the raw input dataset
def filter_base(
    rawData: ad.AnnData,
    verbose = False
    ) -> ad.AnnData:
    """
    Filter invalid perturbation annotations and normalize control labels.

    This function removes cells with missing, empty, or invalid perturbation
    entries and standardizes control labels across datasets. Any perturbation
    containing "(mod)" (Adamson dataset) or "control" (Norman dataset) is
    relabeled as "ctrl".

    Parameters
    ----------
    rawData : anndata.AnnData
        AnnData object containing the dataset. Must include a column
        "perturbation" in `rawData.obs`.

    verbose : bool, optional (default: False)
        If True, prints the number of cells and genes after filtering.

    Returns
    -------
    data_filtered : anndata.AnnData
        Filtered AnnData object where:
        - Invalid perturbation entries are removed
        - Control cells are labeled as "ctrl"
    """
    # Delete cells with invalid pertubations:
    data_filtered = rawData[
    rawData.obs["perturbation"].notna() &
    (rawData.obs["perturbation"] != "*") &
    (rawData.obs["perturbation"].str.strip() != "")
    ].copy()

    if verbose:
        print("Total measured cells:", data_filtered.n_obs)
        print("Total measured genes:", data_filtered.n_vars)

    # Marks all perturbation entries with control [in Norman dataset] or (mod) [in Adamson dataset] as 'ctrl':

    data_filtered.obs["perturbation"] = data_filtered.obs["perturbation"].astype(str)

    mask_ctrl = (data_filtered.obs["perturbation"].str.contains("(mod)", regex=False, na=False) |
    data_filtered.obs["perturbation"].str.contains("control", regex=False, na=False))

    data_filtered.obs.loc[mask_ctrl, "perturbation"] = "ctrl"

    return data_filtered

# Print the cell statistics after filtering:
def print_cell_stats(data_filtered, verbose):
    """
    Print summary statistics of perturbation annotations.

    Function computes and prints basic statistics about the
    perturbation distribution in a filtered AnnData object. If `verbose`
    is False, the function returns without producing any output.

    The following statistics are reported:
    - Percentage of control ("ctrl") cells
    - Number of unique perturbations (excluding control)

    Parameters
    ----------
    data_filtered : anndata.AnnData
        Filtered AnnData object containing a "perturbation" column in `.obs`.

    verbose : bool
        If True, prints the computed statistics. If False, no output is produced.

    Returns
    -------
    None
        This function does not return any value; it only prints statistics.
    """
    if not verbose:
        return
    
    sumCtrlCells = (data_filtered.obs["perturbation"] == "ctrl").sum()
    total_cells = data_filtered.n_obs
    percent_ctrl = (sumCtrlCells / total_cells) * 100 if total_cells > 0 else 0
    n_perturbations = (data_filtered.obs["perturbation"][data_filtered.obs["perturbation"] != "ctrl"].nunique())

    print("Percent of control cells:", percent_ctrl,"%")
    print("Unique perturbations:", n_perturbations )  # -1 for the'ctrl' cells
    
# Normalizes the raw Adamson dataset:
def filter_Adamson_data(rawData, verbose = False):
    """
    Process and standardize perturbation annotations for the AdamsonWeissman2016 dataset.

    This function applies base filtering and extracts the CRISPRi component
    from perturbation labels of the form "gene_CRISPRiRNA". The CRISPRi
    component is stored in a new column "CRISPRi". Cells without a second
    component are labeled as "ctrl".
    Parameters
    ----------
    rawData : anndata.AnnData
        AnnData object containing the dataset. 
        Must include a column "perturbation" in `rawData.obs`.

    verbose : bool, optional (default: False)
        If True, prints summary statistics including:
        - Number of unique genes
        - Percentage of control cells
        - Number of unique perturbations

    Returns
    -------
    data_filtered : anndata.AnnData
        Filtered AnnData object where:
        - Invalid perturbations are removed
        - Control cells are labeled as "ctrl"
        - A new column "CRISPRi" contains the perturbation subtype
        - The "perturbation" column is converted to categorical
    """

    data_filtered = filter_base(rawData, verbose=verbose)

    # Split the entrys (gen_CRISPRiRNA) of the pertubation column. Create a CRISPRi column for the second entry (= CRISPRiRNA):
    split = data_filtered.obs["perturbation"].str.split("_", n=1, expand=True)
    data_filtered.obs["perturbation"] = split[0]
    data_filtered.obs["CRISPRi"] = split[1].fillna("no_CRISPRi")

    # Data preparation for GEARS:
    process_GEARS(data_filtered)
   
    if verbose:
        genesUnique = data_filtered.obs.loc[data_filtered.obs["perturbation"] != "ctrl"].nunique()
        print("Unique genes in the df (w/o ctrl):",genesUnique)

    print_cell_stats(data_filtered, verbose)
    
    # Converts the pertubation column in category:
    data_filtered.obs["perturbation"] = data_filtered.obs["perturbation"].astype("category")

    return data_filtered

# Normalizes the raw input Norman dataset:
def filter_Norman_data(rawData, verbose = False):
    """
    Process and standardize perturbation annotations for the Norman dataset.

    This function applies base filtering and identifies double perturbations
    of the form "gene_gene". These are stored in a new column
    "double_perturbation", while single perturbations remain unchanged.

    Parameters
    ----------
    rawData : anndata.AnnData
        AnnData object containing the dataset. Must include a column
        "perturbation" in `rawData.obs`.

    verbose : bool, optional (default: False)
        If True, prints summary statistics including:
        - Number of double perturbations
        - Percentage of control cells
        - Number of unique perturbations

    Returns
    -------
    data_filtered : anndata.AnnData
        Filtered AnnData object where:
        - Invalid perturbations are removed
        - Control cells are labeled as "ctrl"
        - The as "double_perturbation" masked cells are removed from the "perturbation" column in the dataset
        - The "perturbation" column is converted to categorical
    """
    
    data_filtered = filter_base(rawData, verbose=verbose)
    """
    # Find double perturbations "gen_gen" : 
    mask_double_pert = data_filtered.obs["perturbation"].str.contains("_", na=False)

    # Delete all double pertubations from the perturbation column:
    data_filtered = data_filtered[~mask_double_pert].copy()

    if verbose:
        # Unique single perturbations (without ctrl or NA)
        mask_single = (
            data_filtered.obs["perturbation"].notna() &
            (data_filtered.obs["perturbation"] != "ctrl")
        )
        genesUnique_single = data_filtered.obs.loc[mask_single, "perturbation"].nunique()

        # Total counts
        n_single = mask_single.sum()
        n_double = mask_double_pert.sum()

        print(f"{n_single} total single perturbations found.")
        print(f"{n_double} total double perturbations found.")
        print(f"{genesUnique_single} unique single genes.")
        
    """
    # Data preparation for GEARS:
    process_GEARS(data_filtered)
    
    #data_filtered.obs["condition"]  = data_filtered.obs["perturbation"]+"+ctrl"
    #data_filtered.obs["condition_name"]  = data_filtered.obs["perturbation"]    # Check if we can delete this 1
    #data_filtered.obs["cell_type"]  = data_filtered.obs["cell_line"]
    

    print_cell_stats(data_filtered, verbose)

    # Converts the pertubation column in category:
    data_filtered.obs["perturbation"] = data_filtered.obs["perturbation"].astype("category")
    

    return data_filtered


# Normalizes the raw input Norman dataset:
def filter_Replogle_data(rawData, verbose = False):
    """
    Process and standardize perturbation annotations for the Replogle K562 dataset.

    This function applies base filtering and identifies double perturbations
    of the form "gene_gene". These are stored in a new column
    "double_perturbation", while single perturbations remain unchanged.

    Parameters
    ----------
    rawData : anndata.AnnData
        AnnData object containing the dataset. Must include a column
        "perturbation" in `rawData.obs`.

    verbose : bool, optional (default: False)
        If True, prints summary statistics including:
        - Number of double perturbations
        - Percentage of control cells
        - Number of unique perturbations

    Returns
    -------
    data_filtered : anndata.AnnData
        Filtered AnnData object where:
        - Invalid perturbations are removed
        - Control cells are labeled as "ctrl"
        - The as "double_perturbation" masked cells are removed from the "perturbation" column in the dataset
        - The "perturbation" column is converted to categorical
    """
    
    data_filtered = filter_base(rawData, verbose=verbose)

    # Data preparation for GEARS:
    process_GEARS(data_filtered)
   
    if verbose:
        genesUnique = data_filtered.obs.loc[data_filtered.obs["perturbation"] != "ctrl"].nunique()
        print("Unique genes in the df (w/o ctrl):",genesUnique)

    print_cell_stats(data_filtered, verbose)
    
    # Converts the pertubation column in category:
    data_filtered.obs["perturbation"] = data_filtered.obs["perturbation"].astype("category")

    return data_filtered

def process_GEARS(norm_data):
    """
    Processes the data in such a way that it can be input into a GEARS PertData object (turns it into perturb_processed.h5ad).
   
    Parameters:
    -----------
    norm_data : anndata
        An anndata object that is the output of normalize.py in the pipeline.
    """
    # Rename cell_line to cell_type:
    norm_data.obs["cell_type"]  = norm_data.obs["cell_line"]
    # Turn gene names into own column:
    norm_data.var["gene_name"] = norm_data.var.index

    # Turn pertGene in the perturbation column into pertGene+ctrl in the condition column
    is_not_ctrl = norm_data.obs["perturbation"] != "ctrl"
    norm_data.obs["condition"] = np.asarray([pert.split("_")[0] for pert in norm_data.obs["perturbation"]])
    norm_data.obs["condition_name"] = norm_data.obs["condition"] # Unsure what condition_name is supposed to be, but apparently necessary (set to same as condition)
    norm_data.obs.loc[is_not_ctrl,"condition"] += "+ctrl"

# Helper function for QC
def is_outlier(x: ad.AnnData, metric: str, nmads: int) -> pd.Series:
    """
    Detect outlier cells for a specified QC metric using a MAD-based threshold.

    This function identifies cells whose values for a given observation-level
    quality control (QC) metric deviate strongly from the dataset median.
    Outliers are defined as observations lying outside the interval

        median ± nmads × MAD

    where MAD is the median absolute deviation scaled to be comparable to the
    standard deviation. This robust approach is less sensitive to extreme values
    than standard deviation–based methods.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData object containing single-cell data with QC metrics stored in
        `adata.obs`.

    metric : str
        Name of the QC metric column in `adata.obs` to evaluate
        (e.g., 'log1p_total_counts', 'log1p_n_genes_by_counts',
        'pct_counts_mt').

    nmads : int
        Number of median absolute deviations from the median used as the cutoff
        for defining outliers. Larger values yield more permissive thresholds.

    Returns
    -------
    pandas.Series
        Boolean series indexed like `adata.obs`, where True indicates that the
        corresponding cell is classified as an outlier for the specified metric.
    """
    M = x.obs[metric]
    med = np.median(M)
    mad = median_abs_deviation(M, scale="normal")
    return (M < med - nmads * mad) | (M > med + nmads * mad)

def data_qc (data_filtered: ad.AnnData, top_genes: int, outlier_mad_threshold: int, mt_mad_threshold: int, mt_cutoff_percent: float, verbose: bool = True) -> ad.AnnData:
    """
    Perform quality control filtering of single cells using MAD-based outlier detection.

    This function computes standard single-cell RNA-seq QC metrics and removes
    low-quality cells based on robust thresholds. Specifically, it:

    1. Annotates mitochondrial, ribosomal, and hemoglobin genes
    2. Computes per-cell QC metrics (library size, gene complexity, etc.)
    3. Optionally visualizes QC distributions
    4. Identifies outliers using median absolute deviation (MAD) thresholds
    5. Removes cells with abnormal library composition or high mitochondrial content

    Cells are flagged as outliers if they strongly deviate in:

    - Total counts (library size)
    - Number of detected genes
    - Expression dominance of top genes
    - Mitochondrial transcript percentage

    Parameters
    ----------
    data_filtered : anndata.AnnData
        AnnData object containing raw count data for filtered cells
        (e.g., after perturbation filtering).

    verbose : bool, optional (default: True)
        If True, displays QC plots prior to filtering, including distributions
        of total counts and mitochondrial content.

    Returns
    -------
    qc_data : anndata.AnnData
        A new AnnData object containing only cells that passed the QC filtering.
        The object includes computed QC metrics and outlier annotations in
        `qc_data.obs`.
    """
    qc_data = data_filtered.copy()

    # Define QC-gen classes 
    qc_data.var["mt"] = qc_data.var_names.str.startswith(("MT-", "mt-"))
    qc_data.var["ribo"] = qc_data.var_names.str.startswith(("RPS", "RPL"))
    qc_data.var["hb"] = qc_data.var_names.str.contains("^HB[^(P)]", regex=True)

    if verbose:
        print("Done defining qc classes")
        print("Mitochondrial genes:", qc_data.var["mt"].sum())
        print("Ribosomal genes:", qc_data.var["ribo"].sum())
        print("Hemoglobin genes:", qc_data.var["hb"].sum())

    # Calculate QC-metrics
    sc.pp.calculate_qc_metrics(
        qc_data,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[top_genes],
        log1p=True,
    )
    
    # QC-Plots before filtering (optional)
    if verbose:
        print("Done calculating qc metrics")
        sns.displot(qc_data.obs["total_counts"], bins=100, kde=False)
        plt.show()
        sc.pl.violin(qc_data, "pct_counts_mt")
        sc.pl.scatter(qc_data, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

    # MAD-based outlier filtering
    qc_data.obs["outlier"] = (
        is_outlier(qc_data, "log1p_total_counts", outlier_mad_threshold)
        | is_outlier(qc_data, "log1p_n_genes_by_counts", outlier_mad_threshold)
        | is_outlier(qc_data, "pct_counts_in_top_20_genes", outlier_mad_threshold)
    )

    qc_data.obs["mt_outlier"] = (
        is_outlier(qc_data, "pct_counts_mt", mt_mad_threshold)
        | (qc_data.obs["pct_counts_mt"] > mt_cutoff_percent)
    )
    if verbose:
        print(f"Total number of cells: {qc_data.n_obs}")
        print(qc_data.obs["outlier"].value_counts())
        print(qc_data.obs["mt_outlier"].value_counts())
        print(f"Number of dropout genes: {sum(qc_data.var["pct_dropout_by_counts"]<90)}")

    # filter cells
    qc_data = qc_data[(~qc_data.obs["outlier"]) & (~qc_data.obs["mt_outlier"])].copy()

    print(f"Total number of cells after filtering of low quality cells: {qc_data.n_obs}")
    
    return qc_data

def data_normalization(qc_data: ad.AnnData, n_top_genes : int = 5000, verbose : bool = False) -> ad.AnnData:
    """
    Perform a log1p-normalization with the median read count per cell as the target sum.

    Parameters
    ----------
    qc_data : anndata
        The anndata object containing the data after filtering and outlier removal.
    n_top_genes : int
        The number of highly variable genes to extract.
    verbose : bool
        A bool that indicates whether or not certain information about the dataset (no. of cells, no. of perturbations, etc.) should be printed.

    Returns
    -------
    norm_data
        A dataset after normalization.
    """
    # Median normalization of the data
    if verbose:
        cell_sums = np.array((qc_data).X.sum(axis=1)).flatten()
        median_counts = np.median(cell_sums)
        print("Median counts:", median_counts)
        
    norm_data = qc_data.copy()
    sc.pp.normalize_total(norm_data, target_sum=None) # sc.pp.normalize_total sums to the median by standard

    # Log1p transformation: 
    norm_data.layers["normalized"] = norm_data.X.copy()
    norm_data.layers["log1p"] = sparse.csc_matrix(norm_data.layers["normalized"].copy())
    sc.pp.log1p(norm_data, layer="log1p")
    #if n_top_genes > 0:
    #    print("Finding highly variable genes...")
    #    sc.pp.highly_variable_genes(norm_data,n_top_genes=n_top_genes, subset=True,layer="log1p")
    #print("Generating p values for DEGs, might take a long time...")
        # Remove perts with too few cells
    frequent_perts = [pert for pert in norm_data.obs["perturbation"].unique() if norm_data.obs["perturbation"].value_counts()[pert]>10]
    norm_data = norm_data[norm_data.obs["perturbation"].isin(frequent_perts)]
    #sc.tl.rank_genes_groups(norm_data, groupby="perturbation",method='t-test_overestim_var',layer="log1p",copy=False)

    print("Done!")
    return norm_data

def normalize_adata_main(pseq_file : str):
    """
    Normalizes a perturb seq dataset in mostly the same way as done in the normalization pipeline in the software project, with the exception of not calling the HVGs (done at a later time) and normalizing to 1e6 instead of the median.

    Parameters:
    -----------
    pseq_file: str:
        The name of the file to normalize.

    Returns:
    --------
    norm_data : anndata.AnnData:
        The normalized adata object.
    """
    out_path = f"Data/Experimental/{pseq_file}/perturb_norm.h5ad"

    if not os.path.isfile(out_path):
        print("Normalizing file: ", pseq_file)
        # Read raw data:
        raw_data = sc.read_h5ad(f"Data/Experimental/{pseq_file}/perturb.h5ad")
        
        if "Adamson" in pseq_file:
            data_filtered = filter_Adamson_data(raw_data, verbose = True)
        elif "Norman" in pseq_file:
            data_filtered = filter_Norman_data(raw_data, verbose = True)
        elif "Replogle" in pseq_file:
            data_filtered = filter_Replogle_data(raw_data,verbose=True)
        else:
            raise ValueError(f"No matching dataset found for filename: {pseq_file}")

        qc_data = data_qc(data_filtered, top_genes = 20, outlier_mad_threshold = 5,  mt_mad_threshold = 3, mt_cutoff_percent= 8, verbose=True)
        norm_data = data_normalization(qc_data, n_top_genes = 0, verbose = True)
        norm_data = norm_data[:,norm_data.var_names.sort_values()]
        print("Saving file")
        ad.settings.allow_write_nullable_strings = True

        norm_data.write_h5ad(
            out_path,
            compression=hdf5plugin.FILTERS["zstd"]
        )

    else:
        print("Reading file: ", out_path)
        norm_data = sc.read_h5ad(out_path)
    return norm_data