import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

import os

from Prep_Data import normalize, prep_data_pgrins
import grins_on_cluster

filenames = ["AdamsonWeissman2016_GSM2406681_10X010"]#,"NormanWeissman2019_filtered","ReplogleWeissman2022_K562_essential"]
topo_files = ["dorothea_abcd","kegg"]
project = "Keggoro_abcd"

# Normalize perturb seq adata:
norm_data_dict = {}
for filename in filenames:
    norm_data_dict[filename] = normalize.normalize_adata_main(filename)

# Get GRN and subset it:
grn_df, norm_data_dict = subset_datasets(project, topo_files, norm_data_dict)

# Train control GRiNS:
grins_on_cluster.racipe_control(project)
grins_on_cluster.ising_control(project)

# Evaluate control GRiNS:
is_racipe = True # Whether or not Racipe or Ising data should be loaded
is_discrete = False # If Ising: whether or not sync data should be loaded
zero_cutoff = 0.1

grins_data = prep_data_pgrins.grins_to_adata(project, replicate, is_racipe, is_discrete, zero_cutoff)
# Specify file path:

