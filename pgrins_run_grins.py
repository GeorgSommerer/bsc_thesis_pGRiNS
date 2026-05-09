import jax
import jax.numpy as jnp
from jax import config

import grins.racipe_run as racipe
from grins import ising_bool, gen_params, reg_funcs
from Prep_Data import pgrins_prepare_input
    

import pandas as pd
import numpy as np

import sys
import os
import argparse
from tqdm import tqdm
import gc

from itertools import product, groupby
from operator import itemgetter


def racipe_simulate_sinks(grn_file,sol_df,replicate,num_init_conds,suffix):
    """
    After the steady states of the non-sink nodes of the GRN have been simulated, all that is left is to determine the steady state concentrations of the sink nodes.
    This can be done algebraically: dC/dt=0=G*Prod(H)-kC <-> C=G*Prod(H)/k
    Most of the code in this function deals with efficiently retrieving the columns used in each equation by grouping them by the sink gene C, and then for each term in H by the upstream gene.
    The resulting DataFrames are then added at the correct places to sol_df and saved.

    Parameters:
    -----------
    grn_file : str
        The project name.
    sol_df : pd.DataFrame
        The dataframe containing the nonsink solutions.
    replicate : int
        The current replicate.
    num_init_conds : int
        The number of initial condition files to generate. Necessary to know how many times each parameter must be repeated.
    suffix : str
        ctrl or pert.

    Returns:
    --------
    None
    """
    
    path_to_data = f"Data/SimulResults_Racipe/{grn_file}/{replicate:03}/{grn_file}"
    
    sink_params = pd.read_parquet(f"{path_to_data}_params_{replicate:03}_sinks_{suffix}.parquet").iloc[:,:-1] # Remove ParamNum column
    sink_genes = [col.split("_")[1] for col in list(sink_params.columns) if "Prod_" in col]

    sink_dict = {}
    #sink_gk_dict = {}
    
    split_cols = [col.split("_") for col in list(sink_params.columns)]
    # Sort the parameter columns by the gene whose equation they belong to (last part of the colname) and group the colnames by them:
    split_cols.sort(key=itemgetter(-1)) 
    node_groups = groupby(split_cols,itemgetter(-1))
    with tqdm(total=len(sink_genes)) as progress_bar:
        print("Getting node_groups...")
        for sink_gene, node_colnames in tqdm(node_groups):
            # Get the G and k parameters, whose colnames have the shape varname_downstream_gene
            # Since all parameters are tried for each initial condition (the resulting steady state df is sorted first by the InitCondNum, then by ParamNum), they need to be tiled
            node_colnames = list(node_colnames)
            G = np.tile(sink_params.loc[:,f"Prod_{sink_gene}"].to_numpy(),num_init_conds)
            node_colnames.remove(["Prod",sink_gene])
            k = np.tile(sink_params.loc[:,f"Deg_{sink_gene}"].to_numpy(),num_init_conds)
            node_colnames.remove(["Deg",sink_gene])

            # All remaining colnames relate to the shifted Hill terms and have 3 parts: varname_upstream_gene_downstream_gene
            # Now, sort and group by the upstream_gene to get all 3 columns relating to the same Hill term
            node_colnames.sort(key=itemgetter(1))
            edge_groups = groupby(node_colnames,itemgetter(1))
            H = 1

            for upstream_gene, edge_colnames in edge_groups:
                edge_colnames = list(edge_colnames)
                Node = sol_df.loc[:,upstream_gene].to_numpy()
                """
                try:
                    
                except KeyError as e:
                    # Nodes that are sources with edges only going into sinks will not appear in the non-sink df, but appear as upstream nodes in the sink df despite not having been simulated
                    # However, since they are sources, their equation looks like dC/dt=G-kC, meaning that it decays exponentially and is 0 when the steady state is reached
                    
                    Node = np.repeat(0.0,sol_df.shape[0])
                """
                # Get the Hill and half-max threshold parameters:
                Hill = np.tile(sink_params.loc[:,f"Hill_{upstream_gene}_{sink_gene}"].to_numpy(),num_init_conds)
                edge_colnames.remove(["Hill",upstream_gene,sink_gene])
                Thr = np.tile(sink_params.loc[:,f"Thr_{upstream_gene}_{sink_gene}"].to_numpy(),num_init_conds)
                edge_colnames.remove(["Thr",upstream_gene,sink_gene])

                # The remaining column has the fold change, which also indicates whether the term is activating or inhibiting
                if edge_colnames[0][0] == "ActFld":
                    Fold = np.tile(sink_params.loc[:,f"ActFld_{upstream_gene}_{sink_gene}"].to_numpy(),num_init_conds)

                    thisH = []
                    for i in range(len(Node)):
                        thisH.append(reg_funcs.psH(Node[i],Fold[i],Hill[i],Thr[i]))

                elif edge_colnames[0][0] == "InhFld":
                    Fold = np.tile(sink_params.loc[:,f"InhFld_{upstream_gene}_{sink_gene}"].to_numpy(),num_init_conds)

                    thisH = []
                    for i in range(len(Node)):
                        thisH.append(reg_funcs.nsH(Node[i],Fold[i],Hill[i],Thr[i]))

                H*=np.array(thisH)
            # Calculate algebraic solution to dC/dt=0=GH-kC
            sink_dict[sink_gene] = G*H/k
            #sink_gk_dict[f"gk_{sink_gene}"] = H # gk normalized means diving C=GH/k by G/k, so only H remains
            progress_bar.update(1)
    print("Done!")

    sink_df = pd.DataFrame(sink_dict,dtype=np.float32)
    #sink_gk_df = pd.DataFrame(sink_gk_dict,dtype=np.float32)

    # Save data: 
    #nonsink_df = sol_df.loc[:,list([col.replace("gk_", "") for col in sol_df.columns if "gk_" in col])] # Contains expression values
    info_df = sol_df.loc[:,["PertNum","InitCondNum","ParamNum"]] # Contains combination of ICs and params and perts
    nonsink_df = sol_df.drop(["PertNum","InitCondNum","ParamNum"],axis=1) # Contains combination of ICs and params and perts
    info_df.to_parquet(f"{path_to_data}_steadystate_solutions_{replicate:03}_info_{suffix}.parquet",index=False)
    del sol_df,info_df

    expr_df = pd.concat([nonsink_df,sink_df],axis=1)
    del nonsink_df, sink_df
    expr_df.reindex(sorted(expr_df.columns),axis=1).to_parquet(f"{path_to_data}_steadystate_solutions_{replicate:03}_expr_{suffix}.parquet",index=False)
    # sol_df is now unnecessary


def run_racipe(grn_file, pert_list, split_sinks = False, num_replicates = 1, num_params = 100, num_init_conds = 100, sampling_method = "Uniform",max_steps = 2048, batch_size = 1000, pert_ratio : float = 0.1,pert_batch_size : int = None):
    """
    Generate parameters and run Racipe simulations for the specified GRN.
    Sobol is recommended as the sampling method, but does not work for larger datasets (>20k parameters) due to constraints within the sampler.
    batch_size should be adjusted depending on the available memory.

    Parameters:
    -----------
    grn_file : str
        The project name.
    split_sinks : bool, optional
        Whether or not sinks were removed during pgrins_prepare_input; if that is the case, it is necessary to generate parameters for them here.
    max_steps : int, optional
        Maximum number of steps for the simulation. Defaults to 2048.
    batch_size : int, optional
        Batch size for the simulation. Defaults to 4000.
    num_replicates : int, optional
        The number of replicates to run the simulation for. Defaults to 1.
    num_params : int, optional
        The number of parameter files to generate. Defaults to 1000.
    num_init_conds : int, optional
        The number of initial condition files to generate. Defaults to 100.
    sampling_method : Union[str, dict], optional
        The method to use for sampling the parameter space. Defaults to 'Uniform'. For a finer control over the parameter generation look at the documentation of the gen_param_range_df function and gen_param_df function.
    pert_ratio : float, optional
        How many parameter and initial condition sets should be generated for each perturbation compared to the unperturbed run. Defaults to 10%.
    pert_batch_size : bool, optional
        If turned on, instead of looking at all param/IC combinations, 1 set of parameters and ICs is generated for each cell. This takes longer to generate (and might not fit into memory!), but increases variability.

    Returns:
    --------
    None
    Results are saved in save_dir/grn_file/replicate.
    """

    grn_path = f"Data/Projects/{grn_file}/{grn_file}"
    save_dir = f"Data/SimulResults_Racipe"

    if pert_list is None:
        suffix = "ctrl"
    else:
        suffix = "pert"
        if pert_batch_size is None:
            pert_batch_size = int(1/pert_ratio)
        elif pert_batch_size == 0:
            pert_batch_size = len(pert_list)
        
        num_params = int(num_params*len(pert_list)*pert_ratio)
        num_init_conds = int(num_init_conds*len(pert_list)*pert_ratio)

    print(f"Running {suffix} with {num_replicates} replicates, {num_params} parameters, {num_init_conds} initial conditions, a batch size of {batch_size} and {max_steps} steps.")

    # Generate parameters and initial conditions for the GRN:
    if not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates:03}/{grn_file}_params_{num_replicates:03}_{suffix}.parquet"):
        print("Generating parameters...")
        racipe.gen_topo_param_files(
            f"{grn_path}.topo",
            save_dir,
            num_replicates,
            num_params,
            num_init_conds,
            sampling_method=sampling_method,
            pert_list = pert_list,
            pert_batch_size=pert_batch_size
        )
    
    # If sink nodes were removed, it is necessary to generate parameters for them here (analogous to code from gen_topo_param_files):
    if split_sinks and not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates:03}/{grn_file}_params_{num_replicates:03}_sinks_{suffix}.parquet"):
        print("Generating parameters for sinks...")
        sink_df = pd.read_csv(f"{grn_path}_sinks.topo",sep=" ")
        main_rng = gen_params._get_rng(None)
        for replicate in range(1, num_replicates + 1):
            rep_seed = main_rng.integers(0, 2**32)
            rep_rng = np.random.default_rng(rep_seed)

            sink_param_range_df = gen_params.gen_param_range_df(
                sink_df, num_params, sampling_method=sampling_method, rng=rep_rng
            )
            # Remove all rows corresponding to parameters from upstream nodes that don't have to be simulated twice:
            sink_pr_genes = sink_param_range_df["Parameter"].apply(lambda x: x.split("_")[-1])
            sink_param_range_df = sink_param_range_df[~sink_pr_genes.isin(list(sink_df["Source"]))]
            
            sink_param_range_df.to_csv(
                f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_param_range_{replicate:03}_sinks_{suffix}.csv",
                index=False,
                sep="\t",
            )
            sink_param_df = gen_params.gen_param_df(sink_param_range_df, num_params, rng=rep_rng)
            sink_param_df.to_parquet(
                f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_params_{replicate:03}_sinks_{suffix}.parquet", index=False
            )

    # Run Racipe for all replicates:
    if not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates:03}/{grn_file}_steadystate_solutions_{num_replicates:03}_{suffix}.parquet"):
        if pert_list is None:
            racipe.run_all_replicates(
                    f"{grn_path}.topo",
                    save_dir,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    discretize=False
                )
        else:
            for i in range(0,len(pert_list),pert_batch_size):
                print(f"Simulating pert sets {i} to {min(len(pert_list),i+pert_batch_size-1)}:")
                racipe.run_all_replicates(
                    f"{grn_path}.topo",
                    save_dir,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    pert_list = pert_list[i:i+pert_batch_size],
                    pert_fragment = i,
                    discretize=False
                )

    # If split_sinks is true, calculate non-sinks steady states; in any case, save as filename_expr.parquet for consistency
    if not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates:03}/{grn_file}_steadystate_solutions_{num_replicates:03}_expr_{suffix}.parquet"):
        for replicate in range(1,num_replicates+1):
            if pert_list is None:
                sol_df = pd.read_parquet(f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_steadystate_solutions_{replicate:03}_{suffix}.parquet")
            else:
                sol_df = pd.concat([pd.read_parquet(f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_steadystate_solutions_{replicate:03}_{suffix}_{i}.parquet") for i in range(0,len(pert_list),pert_batch_size)])
            sol_df.columns = [col.replace("_","-") for col in sol_df.columns]  # GRiNS internally replaces "-" with "_" so that it can name its parameters after the genes

            if split_sinks:
                print(f"Simulating sinks for replicate {replicate}...")
                racipe_simulate_sinks(grn_file, sol_df, replicate, num_init_conds,suffix)
            else:
                expr_df = sol_df.loc[:,list([col.replace("gk_", "") for col in sol_df.columns if "gk_" in col])] # Contains expression values
                rest_df = sol_df.loc[:,["PertNum","InitCondNum","ParamNum"]] # Contains combination of ICs and params
                rest_df.to_parquet(f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_steadystate_solutions_{replicate:03}_info_{suffix}.parquet",index=False)
                expr_df.to_parquet(f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_steadystate_solutions_{replicate:03}_expr_{suffix}.parquet",index=False)



def ising_cleanup(grn_file : str,replicate : int,mode : str,no_fragments : int, fragment_size : int):
    """
    If the number of initial conditions was too high to be processed in one go, the resulting parquet files need to be concatenated.
    The resulting concatenated dataframe is then turned into three .csv files, which contain the binary steady states, the binary initial conditions, and the state counts.
    
    Parameters:
    -----------
    grn_file : str
        The project name.
    replicate : int
        The number of replicates.
    mode : str
        Sync or async.
    no_fragments : int
        The number of fragments the initial conditions were split into.
    fragment_size : int
        The number of initial conditions input into each fragmented run.

    Returns:
    --------
    None
    The results are three .csv files stored in the same folder as the .parquet files.
    After the .csv files have been created, it is save to delete the .parquet files.
    """
    
    path_to_data = f"Data/SimulResults_Ising/{grn_file}/{replicate}/{grn_file}_{mode}"

    count_df = pd.read_parquet(f"{path_to_data}_ising_results_0.parquet")
    for f in range(1,no_fragments):
        new_df = pd.read_parquet(f"{path_to_data}_ising_results_{f}.parquet")

        # Since the Initnum column will be the same for every parquet file, it is necessary to make them different
        new_df = new_df.astype({"Initnum":"int32"})
        new_df["Initnum"] = new_df["Initnum"]+f*2**fragment_size 
        count_df = pd.concat([count_df,new_df])

    # Since packbits causes 8 genes to be combined in each column, their numeric representation (0~256) is first turned into 8-length bools.
    print("Creating .csv files...")
    for eight_genes in count_df.columns[2:]:
        count_df[eight_genes] = [format(int(ic),"08b")[:eight_genes.count("|")+1] for ic in list(count_df[eight_genes])] # The genes in the column are separated with a "|", so "[:eight_genes.count("|")+1]" returns as many bits as there are genes
    
    # These bit representations are then combined into one state string
    count_df["State"] = count_df[count_df.columns[2:]].astype(str).agg("".join, axis=1) 
    steady_df = count_df[["Initnum","Step","State"]]
    steady_df.loc[:,"Replicate"] = replicate
    steady_df["Mode"] = mode.capitalize()

    ic_df = steady_df[steady_df["Step"]==0] # The rows with Step==0 are the initial conditions
    steady_df = steady_df[steady_df["Step"]!=0] # The remaining rows are the final states

    state_counts = steady_df["State"].value_counts(normalize=True).reset_index()
    state_counts.columns = ["State", "Fraction"]
    state_counts.loc[:,"Replicate"] = replicate
    state_counts["Mode"] = mode.capitalize()
    print("Saving .csv files...")

    ic_df.drop("Step",axis=1).to_csv(f"{path_to_data}_SteadyStates_ICs.csv", index=False) # A df of the initial conditions
    steady_df.drop("Step",axis=1).to_csv(f"{path_to_data}_SteadyStates.csv", index=False) # A df of the steady states
    state_counts.to_csv(f"{path_to_data}_StateCounts_Main.csv", index=False) # How often each steady state appears

    #for f in range(no_fragments):
    #    os.remove(f"{path_to_data}_ising_results_{f}.parquet")



def run_ising(grn_file : str, mode : str = "async", num_replicates : int = 1, num_init_conds : int = 13, batch_size : int = 1, fragment_size : int = 13):
    """
    Runs the Boolean Ising model for the given grn_file.
    If the number of initial conditions is too large for the memory to handle, it is run in multiple fragments/iterations, each of which fits into memory.

    This approach was abandoned during the creation of pGRiNS, and cannot be used anymore for the generation of perturbed data.

    Parameters:
    -----------
    grn_file : str
        The project name.
    num_init_conds : int, optional
        Number of initial conditions to sample. Defaults to 2**10 if not provided.
    num_replicates : int, optional
        The number of replicates to run the simulation for. Defaults to 1.
    batch_size : int, optional
        Number of samples per batch. Defaults to 2**10.
    mode : str, optional
        The simulation mode, either "sync" or "async". The default is "sync".
    fragment_size : int, optional
        2**fragment_size is the maximal number of fragments the memory can handle. The FU cluster can handle at most 2**13=8192.
    
    Returns:
    --------
    None
    The resulting parquet files are saved in save_dir/grn_file/replicate/grn_file_mode_ising_results_fragment.parquet
    They are then combined in ising_cleanup into .csv files.
    """

    no_fragments = max(1,2**(num_init_conds-fragment_size)) # Gives the number of fragments to split into in order to achieve desired number of initial conditions

    num_init_conds = 2**num_init_conds
    batch_size = 2**batch_size
    grn_path = f"Data/Projects/{grn_file}/{grn_file}.topo"
    save_dir = "Data/SimulResults_Ising"

    if not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates}/{grn_file}_{mode}_ising_results_{no_fragments-1}.parquet"):
        print(f"Running Ising in {no_fragments} iterations...")
        for f in range(no_fragments):
            # Has been modified so that only initial conditions and final states are stored in the .parquet file.
            ising_bool.run_all_replicates_ising( 
                grn_path,
                num_initial_conditions=2**fragment_size,
                batch_size=batch_size,
                save_dir=save_dir,
                mode=mode,
                packbits=True,
                num_replicates=num_replicates,
                fragment = f
            )

    # Concatenate .parquet files and create output .csv files:
    for replicate in range(1,num_replicates+1):
        ising_cleanup(grn_file, replicate, mode, no_fragments,fragment_size)



def main(grn_file : str, is_racipe : bool,pert_file : str, **kwargs):
    """
    The main file. Assures that computations are performed on the GPU and whether Racipe or Ising is used.
    """
    if pert_file is not None:
        pert_list = pgrins_prepare_input.extract_pert_info(grn_file,pert_file)
    else:
        pert_list = None
        
    if is_racipe:
        print("RACIPE:") 
        run_racipe(grn_file, pert_list, **kwargs)
    else:
        print("Boolean Ising:")
        run_ising(grn_file,**kwargs)



if __name__ == "__main__":
    """
    Example:
        python3 pgrins_run_grins.py project Racipe -s -p example_pertfile
        python3 pgrins_run_grins.py project Ising -m sync --batch_size 16
    """
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["XLA_FLAGS"] = "--xla_dump_to=./racipe_xla_dump.log"
    kwargs = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("method", help="Racipe or Ising")
    parser.add_argument("-p", "--use_perts", action="store_true",help="Whether or not pertubations should be analyzed. If true, Data/Perts/grn_perts.pert is loaded. Specify a different filename in Data/Perts/filename.pert with --pert_file in addition to -p.")
    parser.add_argument("--pert_file", help="A list of perturbations to process other than grn_perts.pert.")
    parser.add_argument("-s","--split_sinks", help="Whether or not sinks were removed from the GRN",action="store_true")
    parser.add_argument("-m","--mode", help="If Ising, sync or async (Default: async)")
    parser.add_argument("--num_replicates", type=int,help="Number of replicates to be simulated (Default: 1)")
    parser.add_argument("--num_init_conds",type=int,help="Number of initial conditions (Default: 100 for Racipe, 2**14 for Ising)")
    parser.add_argument("--num_params",type=int,help="Number of params (Default: 10000 for Racipe)")
    parser.add_argument("--batch_size", type=int,help="Number of params/conditions to be calculated at the same time (Default: 10 for Racipe, 3 -> 2**3 for Ising)")
    parser.add_argument("--sampling_method",help="Way of sampling parameters in Racipe (Default: Uniform)")
    parser.add_argument("--max_steps",type=int,help="Number of steps until the Racipe simulation is finished (Default: 2048)")
    parser.add_argument("--pert_ratio",type=float,help="How many parameter and initial condition sets should be generated for each perturbation compared to the unperturbed run. Defaults to 10%.")
    parser.add_argument("--pert_batch_size",type=int,help="The number of perturbations for which cells are simulated in one run, as too many at once might not fit into memory parameter/IC wise. Defaults to 1/pert_ratio (10) since this results in the same number of params/ICs as for the unperturbed run. Set to 0 if all perts should be simulated at once.")
    
    args = parser.parse_args()

    grn_file = args.grn
    if args.method.lower() == "racipe":
        is_racipe = True
    elif args.method.lower() == "ising":
        is_racipe = False
    else:
        raise Exception("method is wrong")

    if args.use_perts:
        if args.pert_file:
            pert_file = args.pert_file
        else:
            pert_file = f"{grn_file}_perts"
    else:
        pert_file = None

    if args.split_sinks:
        kwargs["split_sinks"]=args.split_sinks
    if args.mode:
        kwargs["mode"]=args.mode
    if args.num_replicates:
        kwargs["num_replicates"]=args.num_replicates
    if args.num_init_conds:
        kwargs["num_init_conds"]=args.num_init_conds
    if args.num_params:
        kwargs["num_params"]=args.num_params
    if args.batch_size:
        kwargs["batch_size"]=args.batch_size
    if args.sampling_method:
        kwargs["sampling_method"]=args.sampling_method
    if args.max_steps:
        kwargs["max_steps"]=args.max_steps
    if args.pert_ratio:
        kwargs["pert_ratio"]=args.pert_ratio
    if args.pert_batch_size:
        kwargs["pert_ratio"]=args.pert_batch_size

    main(grn_file,is_racipe,pert_file,**kwargs)