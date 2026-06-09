import jax
import jax.numpy as jnp
from jax import config
import pathlib

    

import pandas as pd
import numpy as np

import sys
import os
import argparse
from tqdm import tqdm

from itertools import product, groupby
from operator import itemgetter

try:
    import racipe_run as racipe
    import ising_bool, gen_params, reg_funcs
except:
    import grins.racipe_run as racipe
    from grins import ising_bool, gen_params, reg_funcs
try:
    from Prep_Data import pgrins_prepare_input
except:
    sys.path.append("..")
    sys.path.append("/".join(str(Path.cwd()).split("/")[:-1]))
    from Prep_Data import pgrins_prepare_input

    
def racipe_simulate_sinks(grn_file : str, save_dir : str, sol_df_full : pd.DataFrame, replicate : int, num_init_conds : int, suffix : str, batch_size : int):
    """
    After the steady states of the non-sink nodes of the GRN have been simulated, all that is left is to determine the steady state concentrations of the sink nodes.
    This can be done algebraically: dC/dt=0=G*Prod(H)-kC <-> C=G*Prod(H)/k
    Most of the code in this function deals with efficiently retrieving the columns used in each equation by grouping them by the sink gene C, and then for each term in H by the upstream gene.
    The resulting DataFrames are then added at the correct places to sol_df and saved.

    Parameters:
    -----------
    grn_file : str
        The project name.
    save_dir : str
        The path where the results will be saved.
    sol_df_full : pd.DataFrame
        The dataframe containing the nonsink solutions.
    replicate : int
        The current replicate.
    num_init_conds : int
        The number of initial condition files to generate. Necessary to know how many times each parameter must be repeated.
    suffix : str
        ctrl or pert.
    batch_size : int
        The batch size used for the simulations. Necessary in order to to make sure that everything fits into memory.

    Returns:
    --------
    None
    """
    
    path_to_data = f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}"
    
    # Get sink parameters and genes:
    sink_params_full = pd.read_parquet(f"{path_to_data}_params_{replicate:03}_sinks.parquet")
    sink_genes = [col.split("_")[1] for col in list(sink_params_full.columns) if "Prod_" in col]

    sink_matrix = np.zeros((sol_df_full.shape[0],len(sink_genes))) # empty matrix
    sink_dict = {}
    sink_gene_to_id = {sink_genes[i]:i for i in range(len(sink_genes))}
    
    for batch in tqdm(range(0,sol_df_full.shape[0],batch_size)):
        # Get cells for current batch, and arrange the sink params so that the param numbers match up
        sol_df = sol_df_full.iloc[batch:batch+batch_size]
        sink_params = sol_df.merge(sink_params_full,on="ParamNum")[sink_params_full.columns]

        # Get the individual parts (ParamType, (SourceGene), TargetGene) of each parameter
        split_cols = [col.split("_") for col in list(sink_params.columns)]
        split_cols.remove(["ParamNum"])

        # Remove all gene names that have for some reason forbidden symbols in them, meaning that they contain more "_" than allowed
        split_cols_filtered = []
        for i in range(len(split_cols)):
            if split_cols[i][0] in ["Prod","Deg"] and len(split_cols[i]) == 2:
                split_cols_filtered.append(split_cols[i])
            elif split_cols[i][0] in ["Hill","Thr","ActFld","InhFld"] and len(split_cols[i]) == 3:
                split_cols_filtered.append(split_cols[i])
        split_cols = split_cols_filtered

        # Sort the parameter columns by the gene whose equation they belong to (last part of the colname) and group the colnames by them:
        split_cols.sort(key=itemgetter(-1)) 
        node_groups = groupby(split_cols,itemgetter(-1))
        for sink_gene, node_colnames in node_groups:
            # Get the G and k parameters, whose colnames have the shape varname_downstreamgene
            node_colnames = list(node_colnames)
            G = sink_params.loc[:,f"Prod_{sink_gene}"].to_numpy()
            node_colnames.remove(["Prod",sink_gene])
            k = sink_params.loc[:,f"Deg_{sink_gene}"].to_numpy()
            node_colnames.remove(["Deg",sink_gene])

            # All remaining colnames relate to the shifted Hill terms and have 3 parts: varname_upstreamgene_downstreamgene
            # Now, sort and group by the upstream_gene to get all 3 columns relating to the same Hill term
            node_colnames.sort(key=itemgetter(1))
            edge_groups = groupby(node_colnames,itemgetter(1))
            H = 1

            for upstream_gene, edge_colnames in edge_groups:
                edge_colnames = list(edge_colnames)
                Node = sol_df.loc[:,upstream_gene].to_numpy()
                # Get the Hill and half-max threshold parameters:
                Hill = sink_params.loc[:,f"Hill_{upstream_gene}_{sink_gene}"].to_numpy()
                edge_colnames.remove(["Hill",upstream_gene,sink_gene])
                Thr = sink_params.loc[:,f"Thr_{upstream_gene}_{sink_gene}"].to_numpy()
                edge_colnames.remove(["Thr",upstream_gene,sink_gene])

                # The remaining column has the fold change, which also indicates whether the term is activating or inhibiting
                if edge_colnames[0][0] == "ActFld":
                    Fold = sink_params.loc[:,f"ActFld_{upstream_gene}_{sink_gene}"].to_numpy()

                    thisH = []
                    for i in range(len(Node)):
                        thisH.append(reg_funcs.psH(Node[i],Fold[i],Hill[i],Thr[i]))

                elif edge_colnames[0][0] == "InhFld":
                    Fold = sink_params.loc[:,f"InhFld_{upstream_gene}_{sink_gene}"].to_numpy()

                    thisH = []
                    for i in range(len(Node)):
                        thisH.append(reg_funcs.nsH(Node[i],Fold[i],Hill[i],Thr[i]))

                # Multiply all Hill terms together
                H*=np.array(thisH)
            # Calculate algebraic solution to dC/dt=0=GH-kC and insert it in the right column and the rows for the current batch
            sink_matrix[batch:batch+sol_df.shape[0],sink_gene_to_id[sink_gene]] = G*H/k
    print("Done!")

    sink_df = pd.DataFrame(sink_matrix,columns=sink_genes,dtype=np.float32)
    sink_df.to_parquet(f"{path_to_data}_steadystate_solutions_{replicate:03}_sinks_{suffix}.parquet",index=False)



def run_racipe(grn_file : str, sim_it : int, pert_list : list[dict[str,int]], split_sinks : bool = False, num_replicates : int = 1, num_params : int = 1000, num_init_conds : int = 100, sampling_method : str = "Uniform",max_steps : int = 2048, tmax : int = 200,batch_size : int = 1000, pert_ratio : float = 0.01, pert_factor : int = 50):
    """
    Generate parameters and run Racipe simulations for the specified GRN.
    Sobol is recommended as the sampling method, but does not work for larger datasets (>20k parameters) due to constraints within the sampler.
    batch_size should be adjusted depending on the available memory.

    Changes to normal GRiNS:
        - If split_sinks is True, then the parameter_range.csv file is generated for all parameters at once, but parameters of equations of sink_parameters are split off.
            Then, the ode.py file is regenerated using only equations of nonsink genes, and the initial conditions of all nonsinks genes are sampled.
        - For the perturbed run, a new file pert_genes.parquet with the shape no_perturbations x no_pert_genes is generated, where each entry is 1 if the gene is unperturbed, and pert_factor (CRISPRa) or 1/pert_factor (CRISPRi) or 0 (CRISPR KO) otherwise
            - In _gen_combinations, initial condition/parameter combinations from the unperturbed run that were stored in best_cells_ctrl.pickle are sampled independently for each perturbation and the parameters/initial conditions from the corresponding rows in the .parquet files reused.
            - Therefore, 3 dataframe are loaded into the JAX ODE solver: parameters (pert_factor*len(pert_list)*num_params x no_kinetic_parameters), initial conditions (pert_factor*len(pert_list)*num_init_conds x no_nonsink_genes), and pert factors (len(pert_list) x no_pert_genes)

    Parameters:
    -----------
    grn_file : str
        The project name.
    sim_it : int
        The current iteration of the IC narrowing process.
    pert_list : list[dict[str,int]]
        A list as generated by extract_pert_info. Empty for the unperturbed run.
    split_sinks : bool, optional
        Whether or not sinks were removed during pgrins_prepare_input; if that is the case, it is necessary to generate parameters for them here.
    max_steps : int, optional
        Maximum number of steps for the simulation. Defaults to 2048.
    batch_size : int, optional
        Batch size for the simulation. Defaults to 1000. Should evenly divide the number of cells generated.
    tmax : int, optional
        The maximal number of time steps. Defaults to 200.
    num_replicates : int, optional
        The number of replicates to run the simulation for. Defaults to 1.
    num_params : int, optional
        The number of parameter files to generate. Defaults to 1000.
    num_init_conds : int, optional
        The number of initial condition files to generate. Defaults to 100.
    sampling_method : Union[str, dict], optional
        The method to use for sampling the parameter space. Defaults to 'Uniform'. For a finer control over the parameter generation look at the documentation of the gen_param_range_df function and gen_param_df function.
    pert_ratio : float, optional
        How many parameter and initial condition sets should be generated for each perturbation compared to the unperturbed run. Defaults to 0.01 (1%).
    pert_factor : int, optional
        By what factor the Prod_pert_gene parameters should be scaled up or down. Defaults to 50 (*50 for CRISPRa, /50 for CRISPRi).

    Returns:
    --------
    None
    Results are saved in Data/SimulResults_Racipe_{sim_it}/{grn_file}/{replicate_number}.
    """

    grn_path = f"Data/Projects/{grn_file}/{grn_file}"
    save_dir = f"Data/SimulResults_Racipe_{sim_it}"

    if pert_list == []:
        suffix = "ctrl"
        print(f"Running {suffix} with {num_replicates} replicates, {num_params} parameters, {num_init_conds} initial conditions, a batch size of {batch_size} and {max_steps} steps in the time inveral [0,{tmax}] using {sampling_method} sampling.")

    else:
        suffix = "pert"
        print(f"Running {suffix} with {int(num_params*num_init_conds*pert_ratio)} cells for {len(pert_list)} perturbation sets each (scaling factor of {pert_factor}), a batch size of {batch_size} and {max_steps} steps in the time inveral [0,{tmax}] using {sampling_method} sampling.")


    # Check if files have already been generated:
    if pert_list == []:
        gen_paths = [f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_params_{rep:03}.parquet" for rep in range(1,num_replicates+1)]+[f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_init_conds_{rep:03}.parquet" for rep in range(1,num_replicates+1)]
        if split_sinks:
            gen_paths += [f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_params_{rep:03}_sinks.parquet" for rep in range(1,num_replicates+1)]
    else:
        gen_paths = [f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_pert_genes_{rep:03}.parquet" for rep in range(1,num_replicates+1)]
    # Generate parameters and initial conditions for the GRN:
    if False in [os.path.exists(path) for path in gen_paths]:
        print("Generating parameters...")
        racipe.gen_topo_param_files(
            f"{grn_path}.topo",
            save_dir,
            num_replicates,
            num_params,
            num_init_conds,
            sampling_method=sampling_method,
            pert_list = pert_list,
            pert_factor = pert_factor,
            split_sinks=split_sinks
        )

    # Run Racipe for all replicates:
    if False in [os.path.exists(f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_steadystate_solutions_{rep:03}_{suffix}.parquet") for rep in range(1,num_replicates+1)]:
        #if pert_list is None:
        racipe.run_all_replicates(
                f"{grn_path}.topo",
                save_dir,
                max_steps=max_steps,
                tmax=tmax,
                batch_size=batch_size,
                discretize=False,
                pert_list = pert_list,
                pert_ratio=pert_ratio
            )
        
    # If split_sinks is true, calculate non-sinks steady states; in any case, save as filename_expr.parquet for consistency
    if split_sinks and False in [os.path.exists(f"{save_dir}/{grn_file}/{rep:03}/{grn_file}_steadystate_solutions_{rep:03}_sinks_{suffix}.parquet") for rep in range(1,num_replicates+1)]:
        for replicate in range(1,num_replicates+1):
            sol_df = pd.read_parquet(f"{save_dir}/{grn_file}/{replicate:03}/{grn_file}_steadystate_solutions_{replicate:03}_{suffix}.parquet")
            sol_df.columns = [col.replace("_","-") for col in sol_df.columns]  # GRiNS internally replaces "-" with "_" so that it can name its parameters after the genes
            print(f"Simulating sinks for replicate {replicate:03}...")
            racipe_simulate_sinks(grn_file, save_dir, sol_df, replicate, num_init_conds,suffix,batch_size)
    print("Done!")



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



def main(grn_file : str, sim_it : int, is_racipe : bool = True,pert_file : str = None, **kwargs):
    """
    The main file. Assures that computations are performed on the GPU and whether Racipe or Ising is used.
    """

    if pert_file is not None:
        pert_list = pgrins_prepare_input.extract_pert_info(grn_file,pert_file)
    else:
        pert_list = []
        
    if is_racipe:
        print("RACIPE:") 
        run_racipe(grn_file, sim_it, pert_list, **kwargs)
    else:
        print("Boolean Ising:")
        run_ising(grn_file,**kwargs)