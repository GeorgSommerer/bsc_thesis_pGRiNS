import jax
import jax.numpy as jnp
import grins.racipe_run as racipe
from grins import ising_bool
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from itertools import product
import argparse

def racipe_cleanup(grn_file : str,num_replicates : int,frag : list):
    # !!! Untested !!!
    """
    If the params and IC parquet files were fragmented, the fragmented result parquet and csv files are joined and the fragments removed.

    Parameters:
    -----------
    grn_file : str
        The project name.
    num_replicates : int
        The number of replicates.
    frag : list
        A list of all param and IC fragment combinations.
    """
    # Untested!!!
    for replicate in range(1,num_replicates+1):
        # Remove the fragmented init condition and parameter parquet files:
        for which in ["init_conds","params"]:
            path_to_data = f"Data/SimulResults_Racipe/{grn_file}/00{replicate}/{grn_file}_{which}_00{replicate}"
            for i in range(fragment):
                os.remove(f"{path_to_data}_{i}.parquet")

    for replicate in range(1,num_replicates+1):
        # Concatenate steadystate solution parquet files:
        path_to_data = f"Data/SimulResults_Racipe/{grn_file}/00{replicate}/{grn_file}_steadystate_state_solutions_00{replicate}"
        concat_df = pd.read_parquet(f"{path_to_data}_{frag[0][0]}_{frag[0][1]}.parquet")
        for ij in range(1,len(frag)):
            concat_df = pd.concat([concat_df,pd.read_parquet(f"{path_to_data}_{frag[ij][0]}_{frag[ij][1]}.parquet")])
        concat_df.to_parquet(f"{path_to_data}.parquet")
        # Remove fragmented files:
        if os.path.exists(f"{path_to_data}.parquet"):
            for ij in range(1,len(frag)):
                os.remove(f"{path_to_data}_{frag[ij][0]}_{frag[ij][1]}.parquet")
    
    for replicate in range(1,num_replicates+1):
        # Concatenate steady state count csv files:
        path_to_data = f"Data/SimulResults_Racipe/{grn_file}/00{replicate}/{grn_file}_steadystate_state_solutions_00{replicate}"
        concat_df = pd.read_csv(f"{path_to_data}_{frag[0][0]}_{frag[0][1]}.csv")
        for ij in range(1,len(frag)):
            concat_df = pd.concat([concat_df,pd.read_csv(f"{path_to_data}_{frag[ij][0]}_{frag[ij][1]}.csv")])
        concat_df.to_csv(f"{path_to_data}.csv",index=False,sep="\t")
        # Remove fragmented files:
        if os.path.exists(f"{path_to_data}.csv"):
            for ij in range(1,len(frag)):
                os.remove(f"{path_to_data}_{frag[ij][0]}_{frag[ij][1]}.csv")

def racipe_control(grn_file, num_replicates = 1, num_params = 100, num_init_conds = 10,sampling_method = "Uniform",max_steps = 2048, batch_size = 10, fragment : int = 10):
    """
    Generate parameters and run Racipe simulations for the specified GRN.

    Parameters
    ----------
    grn_file : str
        The project name.
    max_steps : int, optional
        Maximum number of steps for the simulation. Defaults to 2048.
    batch_size : int, optional
        Batch size for the simulation. Defaults to 1000.
    fragment : list(int,int), optional
        The fragment from the params and init conds parquet file analyzed in this run.
    num_replicates : int, optional
        The number of replicates to run the simulation for. Defaults to 1.
    num_params : int, optional
        The number of parameter files to generate. Defaults to 2**10.
    num_init_conds : int, optional
        The number of initial condition files to generate. Defaults to 2**7.
    sampling_method : Union[str, dict], optional
        The method to use for sampling the parameter space. Defaults to 'Sobol'. For a finer control over the parameter generation look at the documentation of the gen_param_range_df function and gen_param_df function.
    """
    grn_path = f"Data/{grn_file}/{grn_file}.topo"
    save_dir = f"Data/SimulResults_Racipe"

    # Generate parameters and initial conditions:
    if not os.path.exists(f"Data/SimulResults_Racipe/{grn_file}/00{num_replicates}/{grn_file}_params_00{num_replicates}.parquet"):
        print("Generating param files...")
        racipe.gen_topo_param_files(
            grn_path,
            save_dir,
            num_replicates,
            num_params,
            num_init_conds,
            sampling_method=sampling_method,
        )

    # If the whole parquet files of ICs and params don't fit into memory, split them into fragment many blocks, leading to fragment**2 many runs
    for replicate in range(num_replicates):
        path_to_data = f"Data/SimulResults_Racipe/{grn_file}/00{replicate}/{grn_file}"
        if fragment > 1 and not os.path.exists(f"{path_to_data}_init_conds_00{replicate}_{fragment-1}.parquet"):
            params = pd.read_parquet(f"{path_to_data}_params_00{replicate}.parquet")
            inits = pd.read_parquet(f"{path_to_data}__init_conds_00{replicate}.parquet")

            # Determine where to make the cut:
            params_bounds = np.linspace(0,len(params),fragment+1,dtype=int)
            inits_bounds = np.linspace(0,len(inits),fragment+1,dtype=int)
            print(f"Splitting params at positions {params_bounds}")
            for i in range(fragment):
                params.iloc[params_bounds[i]:params_bounds[i+1]].to_parquet(f"{path_to_data}__params_00{replicate}_{i}.parquet")
            print(f"Splitting params at positions {inits_bounds}")
            for j in range(fragment):
                inits.iloc[inits_bounds[j]:inits_bounds[j+1]].to_parquet(f"{path_to_data}__init_conds_00{replicate}_{j}.parquet")
            del params
            del inits

    # Speed computation up via multiprocessing of fragments:
    available_devices = jax.devices()
    frag = list(product(fragment,fragment)) # Iterate through all combinations of IC and param fragments
    for ij in range(0,len(frag),len(available_devices)):
        for d in range(min(len(frag)-ij,len(available_devices))): # d indicates the device num; us as many devices as are available and still necessary
            print(f"Running Racipe for params {frag[ij][0]} and init_conds {frag[ij][1]}:")
            kwargs = {
                "max_steps":max_steps,
                "batch_size":batch_size,
                "fragment":frag[ij],
                "device":available_devices[d]
            }
            p = mp.process(target=racipe.run_all_replicates,args=(grn_path,save_dir),kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()               
        """
        racipe.run_all_replicates(
            grn_path,
            save_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            fragment = frag[ij],
            #device = available_devices[d]
        )    
        """ 

    if fragment > 1:
        racipe_cleanup(grn_file,num_replicates,frag)

def ising_cleanup(grn_file : str,num_replicates : int,mode : str,no_fragments : int):
    # !!! Untested !!!
    """
    If the number of initial conditions was too high to be processed in one go, the resulting parquet files are concatenated and deleted.
    Also, the resulting concatenated parquet file is then turned into three .csv files, which contain the discrete Steady States, the discrete Initial Conditions, and the State Counts.

    Parameters:
    -----------
    grn_file : str
        The project name.
    num_replicates : int
        The number of replicates.
    mode : str
        Sync or async.
    no_fragments : int
        The number of fragments the no. of initial conditions was split into
    """
        # 8 genes are combined in each column, meaning that in order to get the full state count string, these need to be joined.
    # Also, the result as is written mixes initial conditions and steady states
    for replicate in range(1,num_replicates+1):
        path_to_data = f"Data/SimulResults_Ising/{grn_file}/{replicate}/{grn_file}_{mode}"
        print(f"{path_to_data}_ising_results.parquet")        
        count_df = pd.read_parquet(f"{path_to_data}_ising_results_0.parquet")
        for f in range(1,no_fragments):
            new_df = pd.read_parquet(f"Read {path_to_data}_ising_results_{f}.parquet...")
            new_df["Initnum"] = new_df["Initnum"]+f*2**fragment_size # Since the Initnum column will be the same for every parquet file, make them different
            count_df = pd.concat([count_df,new_df])
        # Since 8 genes are combined in each column, their numeric representation (0~256) is first turned into 8-length bools.
        print("Creating .csv files...")
        for eight_genes in count_df.columns[2:]:
            count_df[eight_genes] = [format(int(ic),"08b")[:eight_genes.count("|")+1] for ic in list(count_df[eight_genes])] # The genes in the column are separated with a "|", so "[:eight_genes.count("|")+1]" returns as many bits as there are genes
        count_df["State"] = count_df[count_df.columns[2:]].astype(str).agg("".join, axis=1) # These bit representations are then combined
        steady_df = count_df[["Initnum","Step","State"]]
        steady_df.loc[:,"Replicate"] = replicate
        steady_df["Mode"] = mode.capitalize()
        ic_df = steady_df[steady_df["Step"]==0] # The rows with Step==0 are the initial conditions
        steady_df = steady_df[steady_df["Step"]!=0] # The remaining rows are the final states


        state_counts = count_df["State"].value_counts(normalize=True).reset_index()
        state_counts.columns = ["State", "Fraction"]
        state_counts.loc[:,"Replicate"] = replicate
        state_counts["Mode"] = mode.capitalize()
        print("Saving .csv files...")

        ic_df.drop("Step",axis=1).to_csv(f"{path_to_data}_SteadyStates_ICs.csv", index=False) # A df of the initial conditions
        steady_df.drop("Step",axis=1).to_csv(f"{path_to_data}_SteadyStates.csv", index=False) # A df of the steady states
        state_counts.to_csv(f"{path_to_data}_StateCounts_Main.csv", index=False) # How often each steady state appears

        #for f in range(no_fragments):
        #    os.remove(f"{path_to_data}_ising_results_{f}.parquet")

def ising_control(grn_file, mode : str = "async", num_replicates = 1, num_init_conds = 13, batch_size = 1):
    """
    Run multiple replicate of ising model simulations for a given topology and save results.

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
    """
    fragment_size = 13 # 2**fragment_size is the maximum number of initial conditions the cluster can handle, change if used on better hardware
    no_fragments = max(1,2**(num_init_conds-fragment_size))

    num_init_conds = 2**num_init_conds
    batch_size = 2**batch_size
    grn_path = f"Data/{grn_file}/{grn_file}.topo"
    save_dir = "Data/SimulResults_Ising"

    if not os.path.exists(f"{save_dir}/{grn_file}/{num_replicates}/{grn_file}_{mode}_ising_results_{no_fragments-1}.parquet"):
        print(f"Running Ising in {no_fragments} iterations...")
        # Has been modified so that only initial conditions and final states are written.
        for f in range(no_fragments):
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
            
    ising_cleanup(grn_file, num_replicates, mode, no_fragments)

def main(grn_file : str, is_control : bool, is_racipe : bool,**kwargs):
    if is_control:
        print("Running control run for ", end="")
        if is_racipe:
            print("RACIPE:") 
            racipe_control(grn_file, **kwargs)
        else:
            print("Boolean Ising:")
            ising_control(grn_file, **kwargs)


if __name__ == "__main__":
    """
    Example:
        python3 pgrins_run_grins.py project Racipe -f 10 --max_steps 1024
        python3 pgrins_run_grins.py project Ising -m sync --batch_size 16
    """
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    kwargs = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("grn", help="Name of the GRN used in the project")
    parser.add_argument("method", help="Racipe or Ising")
    parser.add_argument("-m","--mode", help="If Ising, sync or async (Default: async)")
    parser.add_argument("-f","--fragment", type=int,help="If the GRN is too large, the param and init_cond parquet files of Racipe might not fit into memory. If fragment > 1, the parquet files will be split into this many files that are treated separately (for a total of fragment**2 runs)")
    parser.add_argument("-r","--num_replicates", type=int,help="Number of replicates to be simulated (Default: 1)")
    parser.add_argument("-i","--num_init_conds",type=int,help="Number of initial conditions (Default: 100 for Racipe, 2**14 for Ising)")
    parser.add_argument("-p","--num_params",type=int,help="Number of params (Default: 10000 for Racipe)")
    parser.add_argument("--batch_size", type=int,help="Number of params/conditions to be calculated at the same time (Default: 10 for Racipe, 3 -> 2**3 for Ising)")
    parser.add_argument("--sampling_method",help="Way of sampling parameters in Racipe (Default: Uniform)")
    parser.add_argument("--max_steps",type=int,help="Number of steps until the Racipe simulation is finished (Default: 2048)")
    args = parser.parse_args()

    grn_file = args.grn
    if args.method.lower() == "racipe":
        is_racipe = True
    elif args.method.lower() == "ising":
        is_racipe = False
    else:
        raise Exception("method is wrong")
    if args.mode:
        kwargs["mode"]=args.mode
    if args.fragment:
        kwargs["fragment"]=args.fragment
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

    is_control = True
    main(grn_file,is_control,is_racipe,**kwargs)
