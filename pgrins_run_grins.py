import jax.numpy as jnp
import grins.racipe_run as racipe
from grins import ising_bool

def racipe_control(grn_file, num_replicates = 1, num_params = 10000, num_init_conds = 100,sampling_method = "Uniform",max_steps = 2048, batch_size = 10000):
    grn_path = f"Data/{grn_file}/{grn_file}.topo"
    save_dir = f"Data/SimulResults_Racipe"

    racipe.gen_topo_param_files(
        grn_path,
        save_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method=sampling_method,
    )

    racipe.run_all_replicates(
        grn_path,
        save_dir,
        max_steps=max_steps,
        batch_size=batch_size,
    )

def ising_control(grn_file, mode : str, num_replicates = 1, num_init_conds = 2**14, batch_size = 2**10):
    """
    Copy paste from GRiNS
    """
    grn_path = f"Data/{grn_file}/{grn_file}.topo"
    save_dir = f"Data/SimulResults_Ising"
    replacement_values = jnp.array([-1, 1])

    ising_bool.run_all_replicates_ising(
        grn_path,
        num_initial_conditions=num_init_conds,
        batch_size=batch_size,
        save_dir=save_dir,
        mode=mode,
        packbits=True,
        num_replicates=num_replicates
    )

def main(grn_file : str, is_control : bool, racipe : bool,mode : str=None):
    if is_control:
        print("Running control run for", end="")
        if racipe:
            print("RACIPE:")
            racipe_control(grn_file)
        else:
            print("Boolean Ising:")
            ising_control(grn_file, mode)


if __name__ == "__main__":
    is_control = True
    is_racipe = True
    grn_file = "Keggoro_abcd"
    mode = "async"
    
    main(grn_file,is_control,is_racipe,mode)