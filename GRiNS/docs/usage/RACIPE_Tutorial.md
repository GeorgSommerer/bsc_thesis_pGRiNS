# RACIPE Simulation Tutorial

## Overview

This tutorial provides a comprehensive pipeline that integrates various functions for RACIPE simulations from the 'grins' library. It guides you through generating the ODE system file, parameter sets, and initial conditions, ultimately running simulations for all topology files in a specified directory containing gene regulatory network topologies.

If more customization is needed, a similar pipeline can be built using the custom functions provided in the package—this tutorial serves as a guide for structuring such workflows.

For the default implementation of this pipeline, refer to [racipe](RACIPE_Script.md).

## Step 1: Define the number if parallel cores to use

We first specify the number of CPU cores to use for parallel execution. While this step is not mandatory, it significantly speeds up the parameter and initial condition generation process, making the pipeline more efficient if one is simulating a large number of networks.

```python
numCores = 15
print(f"Number of cores: {numCores}")
```

## Step 2: Define Directories

We define the input and output directories to organize the topology files and simulation results efficiently. The topology files, which describe the structure of gene regulatory networks, are stored in a dedicated directory (`TOPOS`). The simulation results, including generated parameter sets, initial conditions, and computed outputs, are saved in a separate directory (`SimulResults`).

Before proceeding, we ensure that the output directory exists by using `os.makedirs(sim_save_dir, exist_ok=True)`. This command creates the directory if it does not already exist.

```python
import os
from glob import glob

# Topology file directory
topo_dir = "TOPOS"
# Directory to store simulation results
sim_save_dir = "SimulResults"
# Create output directory if it does not exist
os.makedirs(sim_save_dir, exist_ok=True)
```

## Step 3: Load Topology Files

We retrieve all topology files that need to be simulated.

The `glob` function is used to search for all files with a `.topo` extension in the specified directory (`topo_dir`). The `sorted()` function ensures that the files are processed in a the alphbetical oder, but this is not necessary.

```python
# Get the list of all topology files which need to be simulated
topo_files = sorted(glob(f"{topo_dir}/*.topo"))
print(f"Number of topology files: {len(topo_files)}")
```

## Step 4: Define Simulation Parameters

Specify the number of replicates per topology file, the number of parameter sets to generate, and the number of initial conditions to be generated. These values determine the scale of the simulation:

- Replicates per topology file: The number of times each topology file will be simulated to account for variability.
- Parameter sets: The number of different sets of parameters to generate for each topology file.
- Initial conditions: The number of different initial conditions to be considered for each parameter set.

The simulations will be run for all combinations of initial conditions and parameter sets. The standard deviation of the measured metric across all replicates is a useful indicator of how well the chosen number of parameters and initial conditions capture the dynamics of the gene regulatory network (GRN). If the standard deviation is high, increasing the number of parameter sets or initial conditions may improve the robustness of the results.

```python
num_replicates = 3
num_params = 10000
num_init_conds = 100
print(f"Number of replicates: {num_replicates}")
print(f"Number of parameters: {num_params}")
print(f"Number of initial conditions: {num_init_conds}\n")
```

## Step 5: Parallelized Parameter and Initial Condition Generation

We use multiprocessing to generate parameter and initial condition files in parallel. This setp is not necessary, but will significantly speed up the paramter sets and intial conditions generation time when simulating large number of networks.

```python
from multiprocessing import Pool

# Start multiprocessing pool
pool = Pool(numCores)
print("Generating Parameter and Initial Condition files...")

# Parallel execution of file generation
pool.starmap(
    gen_topo_param_files,
    [
        (
            topo_file,
            sim_save_dir,
            num_replicates,
            num_params,
            num_init_conds,
        )
        for topo_file in topo_files
    ],
)
print("Parameter and Initial Condition files generated.\n")

# Close multiprocessing pool
pool.close()
```

## Step 6: Running Simulations

Once parameter files are generated, we run simulations for each topology file.

Fine-tune the `batch_size` in `run_all_replicates` based on network size to balance performance and memory use. Large batches may cause out-of-memory errors, while small ones slow down simulations. Group similarly sized GRNs and monitor GPU VRAM and utilization to optimize batch size.

```python
import jax.numpy as jnp

for topo_file in topo_files:
    # Generate parameters using Sobol sampling (optional - if the paramters are not already generated in parallel)
    gen_topo_param_files(
        topo_file,
        sim_save_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method="Sobol",
    )
    
    # Run time-series simulations
    run_all_replicates(
        topo_file,
        sim_save_dir,
        tsteps=jnp.array([25.0, 75.0, 100.0]),
        max_steps=2048,
        batch_size=4000,
    )
    
    # Run steady-state simulations
    run_all_replicates(
        topo_file,
        sim_save_dir,
        batch_size=4000,
    )
```

The results of the simulations will be stored in the sim_save_dir, with each topology file having its own dedicated folder named after the topology file. Within these folders, the following structure will be maintained:

### Topology File and ODE System

- The original topology (.topo) file.
- The generated ODE system function using diffrax, which is created by parsing the topology file.

### Replicate Subdirectories

Each topology folder will contain multiple replicate subdirectories (one per replicate). These will store:

-   Initial Conditions: Saved as a Parquet file (`<topo_name>_init_conds_<replicate_number>.parquet`).
-   Parameter Sets: Stored in another Parquet file (`<topo_name>_params_<replicate_number>.parquet`).
-   Parameter Range: A CSV file (`parameter_range.csv`) defining the parameter bounds before the simulation is run.

### Simulation Output Files

Once the simulations are completed, the solution files will be stored in the respective replicate folders. Depending on the simulation type, the output files will follow these naming conventions:

-   Steady-state solutions: `<topo_file_name>_steady_state_solutions_<replicate_number>.parquet`
-   Time-series solutions: `<topo_file_name>_time_series_solutions_<replicate_number>.parquet`
-   Discretized State Data (if applicable): If the user has opted to discretize the states, an additional file will be present containing the unique states and their occurrence counts.
