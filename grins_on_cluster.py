import jax.numpy as jnp
import grins.racipe_run as racipe

def racipe_control(grn_file, num_replicates = 3, num_params = 10000, num_init_conds = 100,sampling_method = "Uniform",max_steps = 2048, batch_size = 6000):
    save_dir = f"Data/{grn_file}/SimulResults_Racipe"

    racipe.gen_topo_param_files(
        grn_file,
        save_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method=sampling_method,
    )

    racipe.run_all_replicates(
        topo_file,
        save_dir,
        max_steps=max_steps,
        batch_size=batch_size,
    )

def ising_control(grn_file, num_replicates = 3, num_init_conds = 2**14, batch_size = 2**10, mode : str = "sync"):
    save_dir = f"Data/{grn_file}/SimulResults_Ising"
    replacement_values = jnp.array([-1, 1])

    ising_bool.run_all_replicates_ising(
        topo_file,
        num_initial_conditions=num_initial_conds,
        batch_size=batch_size,
        save_dir=save_dir,
        mode=mode,
        packbits=True,
        num_replicates=num_replicates
    )