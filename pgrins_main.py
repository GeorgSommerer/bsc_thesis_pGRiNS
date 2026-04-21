import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

import os

from Prep_Data import pgrins_prepare_input, pgrins_prepare_output
import pgrins_run_grins

########################################################

parser = argparse.ArgumentParser()
parser.add_argument("grn", help="Name of the GRN used in the project")
parser.add_argument("method", help="Racipe or Ising")
parser.add_argument("-e","--experimental", help="Whether or not perturb-seq control data is supplied",action="store_true")
parser.add_argument("-m","--mode", help="If Ising, sync or async (Default: async)")
parser.add_argument("-f","--fragment", type=int,help="If the GRN is too large, the param and init_cond parquet files of Racipe might not fit into memory. If fragment > 1, the parquet files will be split into this many files that are treated separately (for a total of fragment**2 runs)")
parser.add_argument("-r","--num_replicates", type=int,help="Number of replicates to be simulated (Default: 1)")
parser.add_argument("-i","--num_init_conds",type=int,help="Number of initial conditions (Default: 100 for Racipe, 2**14 for Ising)")
parser.add_argument("-p","--num_params",type=int,help="Number of params (Default: 10000 for Racipe)")
parser.add_argument("--batch_size", type=int,help="Number of params/conditions to be calculated at the same time (Default: 10 for Racipe, 8 for Ising)")
parser.add_argument("--sampling_method",help="Way of sampling parameters in Racipe (Default: Uniform)")
parser.add_argument("--max_steps",type=int,help="Number of steps until the Racipe simulation is finished (Default: 2048)")
args = parser.parse_args()

project_name = args.grn # "Keggoro.abcd"
experimental = args.experimental
if args.method.lower() == "racipe":
    is_racipe = True
elif args.method.lower() == "ising":
    is_racipe = False
else:
    raise Exception("method is wrong")

kwargs_ctrl_grins = {}
if args.mode:
    kwargs_ctrl_grins["mode"]=args.mode
if args.fragment:
    kwargs_ctrl_grins["fragment"]=args.fragment
if args.num_replicates:
    kwargs_ctrl_grins["num_replicates"]=args.num_replicates
if args.num_init_conds:
    kwargs_ctrl_grins["num_init_conds"]=args.num_init_conds
if args.num_params:
    kwargs_ctrl_grins["num_params"]=args.num_params
if args.batch_size:
    kwargs_ctrl_grins["batch_size"]=args.batch_size
if args.sampling_method:
    kwargs_ctrl_grins["sampling_method"]=args.sampling_method
if args.max_steps:
    kwargs_ctrl_grins["max_steps"]=args.max_steps

kwargs_output = {}
if args.mode:
    kwargs_output["mode"]=args.mode

########################################################

pgrins_prepare_data.main(experimental,project_name)

########################################################

is_control = True
pgrins_run_grins.main(project_name,is_control,is_racipe,kwargs_ctrl_grins)

########################################################

pgrins_prepare_output.main(project_name, is_racipe, kwargs_output)