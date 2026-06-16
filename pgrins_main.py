import argparse

from Prep_Data import pgrins_prepare_input, pgrins_prepare_output, pgrins_cluster
from grins import pgrins_run_grins

########################################################
parser = argparse.ArgumentParser(add_help=True)

# A:
parser.add_argument("project_name", help="Name of the project.")
parser.add_argument("-m", "--method", help="Whether GRiNS simulations are performed using Racipe or Ising. Defaults to Racipe. Ising is not recommended as it is not adapted to handle the prediction of perturbed data.")
parser.add_argument("-e","--experimental", action="store_true",help="If experimental data is supplied, the GRN is subset on the genes of these datasets, and their control data is used to find the best cluster of synthetic unperturbed data. Defaults to False.")
parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not perturbed data should be generated. Defaults to False. If False, the pipeline stops after the clustering step. If True, Data/Perts/{project_name}_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
parser.add_argument("--pert_file", help="A list of perturbations to process other than {project_name}_perts.pert.")
parser.add_argument("-s","--split_sinks", help="Sink nodes in the GRN do not need to be simulated, since their values does not impact the ODEs of any other nodes; therefore, the steady state values can be computed algebraically, reducing memory usage and increasing simulation speed. If this is desired, turn this option on to remove edges pointing towards sink genes from the main GRN. Defaults to False.",action="store_true")

# B:
parser.add_argument("--num_init_conds",type=int,help="Number of initial conditions sampled for each gene during the unperturbed run. Defaults to 100 for Racipe, 2**14 for Ising.")
parser.add_argument("--num_params",type=int,help="Number of values sampled for each kinetic parameter during the unperturbed run. Defaults to 1000 for Racipe (not needed for Ising).")
parser.add_argument("--batch_size", type=int,help="Number of cells (parameter-initial condition pairs) to be simulated at the same time. Defaults to 1000 for Racipe.")
parser.add_argument("--sampling_method",help="Way of sampling parameters in Racipe. Defaults to Uniform. Sobol is recommended if the number of parameters is small enough.")
parser.add_argument("--max_steps",type=int,help="Number of steps until the Racipe simulation is finished. Defaults to 2048")
parser.add_argument("--tmax", type=int, help="Maximum time for the simulation. Defaults to 200.")
parser.add_argument("--pert_ratio",type=float,help="How many cells should be generated per perturbation compared to the unperturbed run. Defaults to 0.01. Accepts values from [0.01,1]. Should be smaller than the percentage of cells kept during clustering, otherwise technical duplicate cells are created.")
parser.add_argument("--pert_factor",type=int,help="By what factor the Prod_{pert_gene} parameters should be scaled. Defaults to 50 (*50 for CRISPRa, /50 for CRISPRi).")
parser.add_argument("--mode", help="If Ising, whether the updates were done sync or async. Defaults to async.")
parser.add_argument("--num_replicates", type=int,help="The number of replicates to be simulated. Defaults to 1. Pipeline has not been tested on more than 1 replicate.")
parser.add_argument("--no_G_scaling",action=store_true,help="Whether or not G_max should be scaled in add_thr_rows. If true, then G_max will scale exponentially with the number of incoming edges, which can lead to numeric problems. Defaults to False.")

# C:
parser.add_argument("--max_missingness", type=float,help="The maximal percentage of cells per gene that can have missing entries. Defaults to 90. Accepts values from [0,100]")
parser.add_argument("--expon_scale",type=float,help="Given an experimental dataset, this value can be calculated as the mean of pct_dropout_by_counts from exp_adata.var among the genes with pct_dropout_by_counts<max_missingness. Defaults to 42.2 (taken from Replogle22).")
parser.add_argument("--full_dropouts",type=float,help="What percentage of genes should have 100 percent of their entries missing. Defaults to 0. Accepts values from [0,100]")        
parser.add_argument("--outlier_cutoff_min_pct", type=float,help="What percentage of cells of a gene should be negative or missing in order for them to be removed. Defaults to 1 percent. Accepts values from [0,100]")
parser.add_argument("--outlier_cutoff_max",type=float,help="How high the mean raw counts should be in order to be treated as an outlier. Defaults to 10000. For reference, Norman19 and Replogle22 have few genes with >100 raw counts.")
parser.add_argument("--remove_outliers",action="store_true",help="If True, outlier genes are removed. Otherwise, negative and NA values are set to zero, and very large entries are kept unchanged. Defaults to False.")

# D:
parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
parser.add_argument("--max_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 25.")
parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10. If 0, all clusters with >min_cluster_size_pct of cells are considered.")
parser.add_argument("--eval_metric",help="If experimental data is used, whether or not the clusters should be evaluated using MSE or Spearman correlation. Defaults to MSE.")
    
args = parser.parse_args()

kwargs_a = {}
kwargs_b = {}
kwargs_c = {}
kwargs_d = {}

project_name = args.project_name

if args.experimental:
    experimental = True
else:
    experimental = False

if args.method and args.method.lower() == "ising":
    is_racipe = False
else:
    is_racipe = True

if args.use_perts:
    if args.pert_file:
        pert_file = args.pert_file
        kwargs_a["make_pert_file"] = False
    else:
        pert_file = f"{project_name}_perts"
        kwargs_a["make_pert_file"] = True
else:
    pert_file = None
    kwargs_a["make_pert_file"] = False

if args.split_sinks:
    kwargs_a["split_sinks"] = args.split_sinks
    kwargs_b["split_sinks"] = args.split_sinks
    kwargs_c["split_sinks"] = args.split_sinks

if args.mode:
    kwargs_b["mode"]=args.mode
    kwargs_c["mode"]=args.mode
if args.num_replicates:
    kwargs_b["num_replicates"]=args.num_replicates
    kwargs_c["num_replicates"]=args.num_replicates
    kwargs_d["num_replicates"]=args.num_replicates
if args.num_init_conds:
    kwargs_b["num_init_conds"]=args.num_init_conds
if args.num_params:
    kwargs_b["num_params"]=args.num_params
if args.batch_size:
    kwargs_b["batch_size"]=args.batch_size
if args.sampling_method:
    kwargs_b["sampling_method"]=args.sampling_method
if args.max_steps:
    kwargs_b["max_steps"]=args.max_steps
if args.tmax:
    kwargs_b["tmax"]=args.tmax
if args.no_G_scaling:
    kwargs_b["no_G_scaling"]=True
else:
    kwargs_b["no_G_scaling"]=False
if args.pert_ratio:
    kwargs_b["pert_ratio"]=args.pert_ratio
    kwargs_b["pert_ratio"]=args.pert_ratio
if args.pert_factor:
    kwargs_b["pert_factor"]=args.pert_factor
    kwargs_b["pert_factor"]=args.pert_factor

if args.max_missingness:
    kwargs_a["max_missingness"] = args.max_missingness
    kwargs_c["max_missingness"] = args.max_missingness
if args.expon_scale:
    kwargs_c["expon_scale"] = args.expon_scale
if args.full_dropouts:
    kwargs_c["full_dropouts"] = args.full_dropouts
if args.outlier_cutoff_min_pct:
    kwargs_c["outlier_cutoff_min_pct"] = args.outlier_cutoff_min_pct
if args.outlier_cutoff_max:
    kwargs_c["outlier_cutoff_max"] = args.outlier_cutoff_max
if args.remove_outliers:
    kwargs_c["remove_outliers"] = True
else:
    kwargs_c["remove_outliers"] = False

if args.min_num_pcs:
    kwargs_d["min_num_pcs"] = args.min_num_pcs
if args.max_num_pcs:
    kwargs_d["max_num_pcs"] = args.max_num_pcs
if args.min_cluster_size_pct:
    kwargs_d["min_cluster_size_pct"] = args.min_cluster_size_pct
if args.max_num_clusters:
    kwargs_d["max_num_clusters"] = args.max_num_clusters
if args.eval_metric:
    kwargs_d["eval_metric"]=args.eval_metric

########################################################
# A: Prepare data
print("*"*10,"A: Prepare data:","*"*10)
pgrins_prepare_input.main(project_name=project_name, experimental=experimental, **kwargs_a)

########################################################
# B1: Run unperturbed GRiNS
print("*"*10,"B1: Run unperturbed GRiNS:","*"*10)
pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=None,**kwargs_b)

########################################################
# C1: Create unperturbed adata object
print("*"*10,"C1: Create unperturbed adata object:","*"*10)
pgrins_prepare_output.main(grn_file=project_name,experimental=experimental,is_racipe=is_racipe, pert_file=None, **kwargs_c)

########################################################
# D: Get best control cells
print("*"*10,"D: Get best control cells:","*"*10)
pgrins_cluster.main(grn_file=project_name, experimental=experimental, **kwargs_d)

########################################################
if pert_file is not None:
    # B2: Run perturbed GRiNS
    print("*"*10,"B2: Run perturbed GRiNS:","*"*10)
    pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=pert_file,**kwargs_b)

    ########################################################
    # C2: Create perturbed adata object
    print("*"*10,"C2: Create perturbed adata object:","*"*10)
    pgrins_prepare_output.main(grn_file=project_name,experimental=experimental,is_racipe=is_racipe, pert_file=pert_file, **kwargs_c)
    
print("All done! Exiting...")