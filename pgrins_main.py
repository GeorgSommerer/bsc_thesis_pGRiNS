import argparse
import os

from Prep_Data import pgrins_prepare_input, pgrins_prepare_output
import pgrins_run_grins, pgrins_cluster

########################################################
parser = argparse.ArgumentParser()

# General:
parser.add_argument("grn", help="Name of the GRN used in the project")
parser.add_argument("-m", "--method", help="Racipe or Ising. Defaults to Racipe.")
parser.add_argument("-e","--experimental", action="store_true",help="If control data is supplied, perturbations are taken from all directories in Data/Experimental.")
parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")

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
parser.add_argument("--pert_factor",type=int,help="By what factor the Prod_pert_gene parameters should be scaled. Defaults to 50 (*50 for CRISPRa, /50 for CRISPRi).")
parser.add_argument("--pert_batch_size",type=int,help="The number of perturbations for which cells are simulated in one run, as too many at once might not fit into memory parameter/IC wise. Defaults to 1/pert_ratio (10) since this results in the same number of params/ICs as for the unperturbed run. Set to 0 if all perts should be simulated at once.")
    
# B/C:
parser.add_argument("--mode", help="If Ising, sync or async (Default: async)")
parser.add_argument("--num_replicates", type=int,help="Number of replicates to be simulated (Default: 1)")

# C:
parser.add_argument("--max_missingness", type=float,help="The maximal percentage of missing data (Default: 90 percent).")
parser.add_argument("--expon_scale",type=float,help="Given an experimental dataset, this is the mean of pct_dropout_by_counts from adata.var among the genes with pct_dropout_by_counts<max_missingness (Default: 36.36 for Replogle22)")
parser.add_argument("--full_dropouts",type=float,help="How many genes should have 100 percent of their entries missing (Default: 0 percent)")        

# D:
parser.add_argument("--pval_treshold", type=float,help="The critical treshold level where genes with p values below that value are assigned DEGs. Defaults to 0.1 (0.05 is not chosen because of the high variance between perturbed cells).")
parser.add_argument("--min_num_pcs", type=int,help="The minimal number of principal components to use for UMAP. Defaults to 10.")
parser.add_argument("--min_cluster_size_pct", type=float,help="The minimal size a cluster must have relative to all cells in grins_data to be considered for the best cluster. Defaults to 0.01.")
parser.add_argument("--max_num_clusters", type=int,help="The maximal number of clusters considered as best cluster. Defaults to 10.")

args = parser.parse_args()

kwargs_a = {}
kwargs_b1 = {}
kwargs_b2 = {}
kwargs_c = {}
kwargs_d = {}

project_name = args.grn

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
    else:
        pert_file = f"{project_name}_perts"
        kwargs_a["pert_file"] = f"{project_name}_perts"
else:
    pert_file = None

if args.split_sinks:
    kwargs_a["split_sinks"] = args.split_sinks
    kwargs_b1["split_sinks"] = args.split_sinks

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

if args.min_num_pcs:
    kwargs_d["min_num_pcs"] = args.min_num_pcs
if args.pval_treshold:
    kwargs_d["pval_treshold"] = args.pval_treshold
if args.min_cluster_size_pct:
    kwargs_d["min_cluster_size_pct"] = args.min_cluster_size_pct
if args.max_num_clusters:
    kwargs_d["max_num_clusters"] = args.max_num_clusters

########################################################
# A: Prepare data
print("*"*10,"A: Prepare data:","*"*10)
pgrins_prepare_input.main(project_name=project_name, experimental=experimental, **kwargs_a)

########################################################
# B1: Run unperturbed GRiNS
print("*"*10,"B1: Run unperturbed GRiNS:","*"*10)
pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=None,**kwargs_b1)

########################################################
# C1: Create unperturbed adata object
print("*"*10,"C1: Create unperturbed adata object:","*"*10)
ugrins_data = pgrins_prepare_output.main(grn_file=project_name, experimental=experimental,is_racipe=is_racipe, pert_file=None, **kwargs_c)

########################################################
# D1: Get best control cells
print("*"*10,"D1: Get best control cells:","*"*10)
pgrins_cluster.main(grn_file=project_name, experimental=experimental,grins_data=ugrins_data, pert_file=None, **kwargs_d)
del ugrins_data
"""
########################################################
if pert_file is not None:
    # B2: Run perturbed GRiNS
    print("*"*10,"B2: Run perturbed GRiNS:","*"*10)
    pgrins_run_grins.main(grn_file=project_name,is_racipe=is_racipe,pert_file=pert_file,**kwargs_b2)

    ########################################################
    # C2: Create perturbed adata object
    print("*"*10,"C2: Create perturbed adata object:","*"*10)
    pgrins_data = pgrins_prepare_output.main(grn_file=project_name, experimental=experimental,is_racipe=is_racipe, pert_file=pert_file, **kwargs_c)

    ########################################################
    # D2: Get best cells for each perturbation and full output
    print("*"*10,"D2: Get best cells for each perturbation and full output:","*"*10)
    pgrins_cluster.main(grn_file=project_name, experimental=experimental,grins_data = pgrins_data, pert_file=pert_file, **kwargs_d)
"""