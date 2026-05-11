import argparse
import os

from Prep_Data import pgrins_prepare_input, pgrins_prepare_output
import pgrins_run_grins, pgrins_cluster

########################################################
parser = argparse.ArgumentParser()

# General:
parser.add_argument("grn", help="Name of the GRN used in the project")
parser.add_argument("-m", "--method", help="Racipe or Ising. Defaults to Racipe.")
parser.add_argument("-e","--experimental", help="If control data is supplied, perturbations are taken from all directories in Data/Experimental. Specify the directory name of the main dataset used for clustering.")
parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")

# A:
parser.add_argument("--subset_method", help="Whether or not the GRN should only contains genes in all experimental datasets, or any of them (default: Union)")

# A/B:
parser.add_argument("-s","--split_sinks", help="Whether or not sinks should be removed from the GRN",action="store_true")

# B1 only:
parser.add_argument("--num_init_conds",type=int,help="Number of initial conditions (Default: 100 for Racipe, 2**14 for Ising)")
parser.add_argument("--num_params",type=int,help="Number of params (Default: 1000 for Racipe)")
parser.add_argument("--batch_size", type=int,help="Number of params/conditions to be calculated at the same time (Default: 10 for Racipe, 3 -> 2**3 for Ising)")
parser.add_argument("--sampling_method",help="Way of sampling parameters in Racipe (Default: Uniform)")
parser.add_argument("--max_steps",type=int,help="Number of steps until the Racipe simulation is finished (Default: 2048)")

# B1/B2:
parser.add_argument("--pert_ratio",type=float,help="How many parameter and initial condition sets should be generated for each perturbation compared to the unperturbed run. Defaults to 10%.")
parser.add_argument("--pert_factor",type=int,help="By what order of magnitude the Prod_pert_gene parameters should be scaled. Defaults to 6 (*1 million for CRISPRa, /1 million for CRISPRi).")
parser.add_argument("--pert_batch_size",type=int,help="The number of perturbations for which cells are simulated in one run, as too many at once might not fit into memory parameter/IC wise. Defaults to 1/pert_ratio (10) since this results in the same number of params/ICs as for the unperturbed run. Set to 0 if all perts should be simulated at once.")
    
# B/C:
parser.add_argument("--mode", help="If Ising, sync or async (Default: async)")
parser.add_argument("--num_replicates", type=int,help="Number of replicates to be simulated (Default: 1)")

# C:
parser.add_argument("--max_missingness", type=float,help="The maximal percentage of missing data (Default: 90 percent).")
parser.add_argument("--expon_scale",type=float,help="Given an experimental dataset, this is the mean of pct_dropout_by_counts from adata.var among the genes with pct_dropout_by_counts<max_missingness (Default: 36.36 for Replogle22)")
parser.add_argument("--full_dropouts",type=float,help="How many genes should have 100 percent of their entries missing (Default: 0 percent)")        

# D:
parser.add_argument("--metric",help="Which metric should be used to calculate the score for each cluster for the candidates during the unperturbed run. Options are mean squared error (MSE, default) and cosine similarity (cosine).")
parser.add_argument("--num_processes", type=int,help="The number of processes used for silhouette score computation; defaults to 10.")
parser.add_argument("--use_mad",action="store_true",help="Turn this option on if the MAD should be used as a deviation measure instead of SD for get_candidate_cells.")
parser.add_argument("--times_dev",type=float,help="How many deviations away the score of the GRiNS data of a cell can be from the mean of means of control data in order to not be discarded. Defaults to 10.")
parser.add_argument("--candidate_limit",type=float,help="What percentage of candidate cells should be kept at most. Defaults to 1 (100%). If 100%, no candidate sampling is done at all. Otherwise combinable with times_dev.")
parser.add_argument("--layer", help="The layer of grins_data on which to perform the operations. Defaults to missing_log1p (log1p with missingness layer applied).")
parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")
parser.add_argument("--deg_imp",type=float,help="How much importance the MSE of the DEGs should have for the pert score relative to the nonDEG MSE. Defaults to 0.5 (same importance).")

args = parser.parse_args()

kwargs_a = {}
kwargs_b1 = {}
kwargs_b2 = {}
kwargs_c = {}
kwargs_d = {}

project_name = args.grn

if args.experimental:
    experimental = True
    main_dataset = args.experimental
else:
    experimental = False

if args.method and args.method.lower() == "ising":
    is_racipe = False
else:
    is_racipe = True

if args.use_perts:
    if args.pert_file:
        pert_file = args.pert_file
    else:
        pert_file = f"{project_name}_perts"
else:
    pert_file = None

if args.split_sinks:
    kwargs_a["split_sinks"] = args.split_sinks
    kwargs_b1["split_sinks"] = args.split_sinks
if args.subset_method:
    kwargs_a["subset_method"] = args.subset_method

if args.mode:
    kwargs_b1["mode"]=args.mode
    kwargs_c["mode"]=args.mode
if args.num_replicates:
    kwargs_b1["num_replicates"]=args.num_replicates
    kwargs_c["num_replicates"]=args.num_replicates
if args.num_init_conds:
    kwargs_b1["num_init_conds"]=args.num_init_conds
if args.num_params:
    kwargs_b1["num_params"]=args.num_params
if args.batch_size:
    kwargs_b1["batch_size"]=args.batch_size
if args.sampling_method:
    kwargs_b1["sampling_method"]=args.sampling_method
if args.max_steps:
    kwargs_b1["max_steps"]=args.max_steps
if args.pert_ratio:
    kwargs_b1["pert_ratio"]=args.pert_ratio
    kwargs_b2["pert_ratio"]=args.pert_ratio
if args.pert_factor:
    kwargs_b1["pert_factor"]=args.pert_factor
    kwargs_b2["pert_factor"]=args.pert_factor
if args.pert_batch_size:
    kwargs_b1["pert_batch_size"]=args.pert_batch_size
    kwargs_b2["pert_batch_size"]=args.pert_batch_size

if args.max_missingness:
    kwargs_c["max_missingness"] = args.max_missingness
if args.expon_scale:
    kwargs_c["expon_scale"] = args.expon_scale
if args.full_dropouts:
    kwargs_c["full_dropouts"] = args.full_dropouts

if args.metric:
    kwargs_d["metric"] = args.metric
if args.num_processes:
    kwargs_d["num_processes"] = args.num_processes
if args.use_mad:
    kwargs_d["use_mad"] = args.use_mad
if args.times_dev:
    kwargs_d["times_dev"] = args.times_dev
if args.candidate_limit:
    kwargs_d["candidate_limit"] = args.candidate_limit
if args.layer:
    kwargs_d["layer"] = args.layer
if args.min_num_pcs:
    kwargs_d["min_num_pcs"] = args.min_num_pcs
if args.min_cluster_size_pct:
    kwargs_d["min_cluster_size_pct"] = args.min_cluster_size_pct
if args.max_num_clusters:
    kwargs_d["max_num_clusters"] = args.max_num_clusters
if args.deg_imp:
    kwargs_d["deg_imp"] = args.deg_imp

########################################################
# A: Prepare data
pgrins_prepare_input.main(project_name=project_name, experimental=experimental, pert_file=pert_file, **kwargs_a)

########################################################
# B1: Run unperturbed GRiNS
pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=None,**kwargs_b1)

########################################################
# C1: Create unperturbed adata object
pgrins_prepare_output.main(grn_file=project_name, is_racipe=is_racipe, pert_file=None, **kwargs_c)

########################################################
# D1: Get best control cells
pgrins_cluster.main(grn_file=project_name, adata_file=main_dataset, pert_file=None, **kwargs_d)

########################################################
if pert_file is not None:
    # B2: Run perturbed GRiNS
    pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=pert_file,**kwargs_b2)

    ########################################################
    # C2: Create perturbed adata object
    pgrins_prepare_output.main(grn_file=project_name, is_racipe=is_racipe, pert_file=pert_file, **kwargs_c)

    ########################################################
    # D2: Get best cells for each perturbation and full output
    pgrins_cluster.main(grn_file=project_name, adata_file=main_dataset, pert_file=pert_file, **kwargs_d)
