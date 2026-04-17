import numpy as np
import pandas as pd
import os
import glob
import jax.numpy as jnp
from jax import vmap, lax, jit, debug
from scipy.stats import qmc
import time


# Function to convert the dataframe into a numpy matrix
def parse_topo_to_matrix(topofile_path):
    """
    Parses a topology file into an adjacency matrix.

    This function reads a topology file, and converts it into an adjacency matrix.
    The adjacency matrix is then converted to a JAX array.

    Parameters
    ----------
    topofile_path : str
        The path to the topology file. The file should be in a format readable by pandas ```read_csv``` with whitespace as the delimiter.

    Returns
    -------
    tuple
        A tuple containing:
        topo_adj : jax.numpy.ndarray
            The adjacency matrix as a JAX array.
        node_names : list
            A list of node names in the order they appear in the adjacency matrix.
    """
    # Read the topo file as a pandas dataframe
    topo_df = pd.read_csv(topofile_path, sep=r"\s+")
    # Get the node names
    node_names = set(
        list(topo_df["Source"].unique()) + list(topo_df["Target"].unique())
    )
    # Adding self loops with type 0 for all the nodes if self loop of that node is not present
    # This is done to ensure that the adjacency matrix is square i.e all the only target or source nodes are also included
    # Subsetting the dataframe to get the self loops
    self_loops = topo_df[topo_df["Source"] == topo_df["Target"]]
    # Getting the list of nodes without self loops
    nodes_without_self_loops = list(node_names - set(self_loops["Source"]))
    # Creating a dataframe for the nodes without self loops
    self_loops_df = pd.DataFrame(
        [
            nodes_without_self_loops,
            nodes_without_self_loops,
            [0] * len(nodes_without_self_loops),
        ]
    ).T
    # Renaming the columns
    self_loops_df.columns = ["Source", "Target", "Type"]
    # Adding the self loops to the topo dataframe
    topo_df = pd.concat([topo_df, self_loops_df], ignore_index=True)
    # Convert the type column to float
    topo_df["Type"] = topo_df["Type"].astype(float)
    # Replacing the 2s with -1s
    topo_df["Type"] = topo_df["Type"].replace({2: -1})
    # Pivot the dataframe to get the adjacencey matrix
    topo_df = topo_df.pivot(index="Source", columns="Target", values="Type")
    # Make the node names set into a list
    node_names = sorted(list(node_names))
    # Reorder the columns and rows
    topo_df = topo_df.loc[node_names, node_names]
    # Replace the NaN values with 0s
    topo_df.fillna(0, inplace=True)
    # Convert the dataframe to a numpy adjacency matrix
    topo_adj = topo_df.to_numpy()
    # Convert the adjacency matrix to a jax array
    topo_adj = jnp.array(topo_adj, dtype=jnp.int16)
    return topo_adj, node_names


# Function to sample and scale the distributions
def generate_intial_conditions(num_nodes, num_samples):
    """
    Generate initial conditions for a given number of nodes and samples.

    This function generates initial conditions using Sobol sequences and scales the generated samples to [0, 1].

    Parameters
    ----------
    num_nodes : int
        The number of nodes for which to generate initial conditions.
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    jax.numpy.ndarray
        The generated initial conditions as a JAX array of integers.
    """

    # Internal function to generate sobol sequences
    def _gen_sobol_seq(dimensions, num_samples, optimise=False):
        """
        Generate Sobol sequence samples.

        Parameters
        ----------
        dimensions : int
            The number of dimensions for the Sobol sequence.
        num_samples : int
            The number of samples to generate.
        optimise : bool, optional
            Whether to use optimization for generation. Defaults to False.

        Returns
        -------
        numpy.ndarray
            The generated Sobol sequence samples.
        """
        if not optimise:
            samples = qmc.Sobol(d=dimensions, scramble=True).random(num_samples)
        else:
            # Optimisation leads to a significant slowdown in the generation
            # Use only if needed
            samples = qmc.Sobol(
                d=dimensions, scramble=True, optimization="lloyd"
            ).random(num_samples)
        return samples

    # Function which scales a given distribution to the required ranges
    def _scale_distribution(
        sample,
        minmax_vals,
        round_int=False,
    ):
        """
        Scale the given distribution to the required ranges.

        Parameters
        ----------
        sample : numpy.ndarray
            The distribution to be scaled.
        minmax_vals : tuple
            The minimum and maximum values for scaling.
        round_int : bool, optional
            Whether to round the values to the nearest integer. Defaults to False.

        Returns
        -------
        numpy.ndarray
            The scaled distribution.
        """
        min_vals = minmax_vals[:, 0]
        max_vals = minmax_vals[:, 1]
        if round_int:
            min_vals = min_vals - 1
        # Scaling the values for the log distributions
        sample = min_vals + (max_vals - min_vals) * (
            sample - np.min(sample, axis=0)
        ) / (np.max(sample, axis=0) - np.min(sample, axis=0))
        # If round is required, return the scaled and rounded values
        if round_int:
            # Rounding the values to the next integer
            sample = np.ceil(sample)
            # # If values are present below the minimum value, shift them to the minimum value
            # If values are present above the maximum value, shift them to the maximum value
            sample = np.clip(sample, min_vals + 1, max_vals)
            # # Convert the values to integers
            sample = sample.astype(int)
            return sample
        else:
            return sample

    # Generate the MinMax values
    minmax_vals = np.array([[0, 1]] * num_nodes)
    # Generate the Sobol sequences
    unscaled_samples = _gen_sobol_seq(len(minmax_vals), num_samples)
    # Scaling the samples
    scaled_samples = _scale_distribution(
        unscaled_samples, np.array(minmax_vals), round_int=True
    )
    # COnvert the jax array
    scaled_samples = jnp.array(scaled_samples, dtype=jnp.int16)
    return scaled_samples


@jit
def sync_eval_next_state(
    prev_state,
    topo_adj,
    replacement_values,  # Vector of three values
):
    """
    Evaluate the next state of a system synchronously based on the previous state,
    topology adjacency matrix, and replacement values.

    Parameters
    ----------
    prev_state : jnp.ndarray
        The previous state of the system.
    topo_adj : jnp.ndarray
        The topology adjacency matrix representing the connections between nodes in the system.
    replacement_values : jnp.ndarray
        A vector of two values used for replacement based on the computed state conditions. The values are: [value_if_negative, value_if_positive].
        Value of 0 is not included as it is assumed that the state will remain the same if the node evaluates to 0 in that step.
        prev_state (jnp.ndarray): The previous state of the system.

    Returns
    -------
    jnp.ndarray
        The new state of the system as an array of int16.
    """
    # Compute the state of the incoming links for the state
    new_state = jnp.dot(prev_state, topo_adj)
    # debug.print("Scaled state: {}", new_state)
    # Apply replacement values based on conditions
    new_state = jnp.where(
        new_state < 0,
        replacement_values[0],
        jnp.where(new_state > 0, replacement_values[1], prev_state),
    )
    # Convert the new state to int16
    new_state = new_state.astype(jnp.int16)
    return new_state


@jit
def simulate_sync_trajectory(
    initial_condition,
    ic_index,
    topo_adj,
    replacement_values,
    max_steps,
):
    """
    Simulates a synchronous trajectory of a system based on the given initial condition, topology adjacency matrix, and replacement values.

    Parameters
    ----------
    initial_condition : jnp.ndarray
        The initial state of the system.
    ic_index : int
        The index of the current initial condition.
    topo_adj : jnp.ndarray
        The topology adjacency matrix representing the connections between nodes in the system.
    replacement_values : jnp.ndarray
        The values used to replace the states during the simulation. It is a vector of two values in the form [value_if_negative, value_if_positive].
    max_steps : jnp.arange
        The range of steps to simulate. The simulation will run for each step in the range.

    Returns
    -------
    jnp.ndarray
        A JAX array containing the states of the system at certain steps, with the initial condition included at the beginning. The array also includes a column for the step indices. All -1 values in the states are replaced with 0 if the replacement values are [-1, 1].
    """

    # Initialize states array
    def step_fn(carry, _):
        current_state = carry
        next_state = sync_eval_next_state(
            current_state,
            topo_adj,
            replacement_values,
        )
        return next_state, next_state
    # Run the simulation loop
    _, states = lax.scan(step_fn, initial_condition, xs=max_steps)
    # Add the initial condition to the states at the beginning
    states = jnp.concatenate((jnp.expand_dims(initial_condition, axis=0), states))
    # Add column for IC vals
    states = jnp.hstack((jnp.repeat(ic_index,states.shape[0]).reshape(-1,1),states))
    # Add a column for the steps
    states = jnp.hstack((jnp.arange(states.shape[0]).reshape(-1, 1), states))
    # Make sure that states are a jax array
    states = jnp.array(states, dtype=jnp.int16)
    # For the results where the replacement values are -1 and 1, some nodes will have -1 values, but for all purposes those should be considered as 0 values, so turning all -1 values to 0
    states = jnp.where(states == -1, 0, states)

    returned_states = jnp.vstack((states[:1,:],states[-1:,:])) # Determine which states should be returned; here, the initial conditions & end states are returned

    return returned_states


@jit
def async_eval_next_state(
    prev_state,
    topo_adj,
    replacement_values,  # Vector of two values
    update_index,  # Index of the node to update
):
    """
    Asynchronously evaluates the next state of a node in an Ising model.

    Parameters
    ----------
    prev_state : jnp.ndarray
        The previous state vector of the system.
    topo_adj : jnp.ndarray
        The adjacency matrix representing the topology of the system.
    replacement_values : jnp.ndarray
        A vector of two values used for state replacement based on conditions. The values of the vector should be [value_if_negative, value_if_positive]. The value for 0 is not included as it is assumed that the state will remain the same if the node evaluates to 0 in that step.
    update_index : int
        The index of the node to update.

    Returns
    -------
    jnp.ndarray
        The new state vector after updating the specified node.
    """
    # debug.print("Update index: {}", update_index)
    # debug.print("Prev state: {}", prev_state)
    # Compute the state update only for the selected index
    new_value = jnp.dot(prev_state, topo_adj[update_index])  # Only update one row
    # debug.print("New value0: {}", new_value)
    # Apply replacement values based on conditions
    new_value = jnp.where(
        new_value < 0,
        replacement_values[0],
        jnp.where(new_value > 0, replacement_values[1], prev_state[update_index]),
    ).astype(jnp.int16)
    # debug.print("New value1: {}", new_value)
    # Scatter the updated value into the state vector
    new_state = prev_state.at[update_index].set(new_value)
    # debug.print("New state: {}", new_state)
    return new_state


@jit
def simulate_async_trajectory(
    initial_condition,
    ic_index,
    topo_adj,
    replacement_values,
    update_indices,  # Vector of indices specifying which node to update at each step
):
    """
    Simulates an asynchronous trajectory of a system given an initial condition and update indices.

    Parameters
    ----------
    initial_condition : jnp.ndarray
        The initial condition of the system.
    ic_index : int
        The index of the current initial condition.
    topo_adj : jnp.ndarray
        The adjacency matrix representing the topology of the system.
    replacement_values : jnp.ndarray
        The values used to replace the states during the simulation. It is a vector of two values in the form [value_if_negative, value_if_positive].
    update_indices : jnp.ndarray
        A vector of indices specifying which node to update at each step. The length of the vector should be equal to the number of steps. If not, the simulation will only run until the length of the update_indices.

    Returns
    -------
    jnp.ndarray
        A JAX array containing the states of the system at each step, with the initial condition included at the beginning. The array also includes a column for the step indices. All -1 values in the states are replaced with 0 if the replacement values are [-1, 1].
    """

    def step_fn(carry, update_index):
        next_state = async_eval_next_state(
            carry,
            topo_adj,
            replacement_values,
            update_index,
        )
        return next_state, next_state

    # Run the simulation loop, passing update_indices as xs
    _, states = lax.scan(step_fn, initial_condition, xs=update_indices)
    # Add the initial condition to the states at the beginning
    states = jnp.concatenate((jnp.expand_dims(initial_condition, axis=0), states))
    # Add column for IC vals
    states = jnp.hstack((jnp.repeat(ic_index,states.shape[0]).reshape(-1,1),states))
    # Add a column for the steps
    states = jnp.hstack((jnp.arange(states.shape[0]).reshape(-1, 1), states))
    # Make sure that states are the right dtype
    states = jnp.array(states, dtype=jnp.int16)
    # For the results where the replacement values are -1 and 1, some nodes will have -1 values, but for all purposes those should be considered as 0 values, so turning all -1 values to 0
    states = jnp.where(states == -1, 0, states)

    returned_states = jnp.vstack((states[:1,:],states[-1:,:])) # Determine which states should be returned; here, the initial conditions & end states are returned

    return returned_states


# Function to pack the 0/1 states into bits
@jit
def packbit_states(states):
    states = jnp.concatenate(
        [states[:, 0:2], jnp.packbits(states[:, 1:], axis=1)], axis=1
    )
    return states


def run_ising(
    topo_file,
    num_initial_conditions=None,
    inital_conditions=None,
    max_steps=None,
    batch_size=None,
    replacement_values=jnp.array([-1, 1]),
    mode="sync",
    packbits=False,
):
    """
    Run synchronous or asynchronous simulations for a given topology.

    Parameters
    ----------
    topo_file : str
        The path to the topology file.
    num_initial_conditions : int, optional
        The number of initial conditions to sample. If not provided, the default is 2**10.
    inital_conditions : jax.numpy.ndarray, optional
        The initial conditions matrix with the individual initial conditions as rows of the matrix. If provided, num_initial_conditions is ignored.
    max_steps : int, optional
        The maximum number of steps to simulate. If not provided, it is calculated to be 10 times the number of nodes.
    batch_size : int, optional
        The number of samples per batch. If not provided, the default is 2**10.
    replacement_values : jax.numpy.ndarray, optional
        These values are used for replacement after each evaluation in the simulation. The first value specifies what all elements less than 0 in the evaluated state will be converted to, and the second value specifies the replacement for all elements greater than 0. The default replacement values are [-1, 1]. When saving to file, all -1 and 0 values are converted to 0, and 1 remains 1. This ensures compatibility with packbits.
    mode : str, optional
        The simulation mode, either "sync" or "async". The default is "sync".
    packbits : bool, optional
        Whether to pack the 0/1 states into bits to reduce memory usage. The default is False.

    Returns
    -------
    pd.DataFrame
        Simulation results are returned as a pandas DataFrame.
        The dataframe as a `Step` column indicating the simulation step. Other columns represent node values at each step. If `packbits=True`, node names are concatenated with `"|"` in the column names. During unpacking, values can be assigned to individual nodes in the same order.

    Example
    -------
    Run the synchronous simulation for a topology file:

        >>> run_ising(
        ...     topo_file="TOPOS/TS.topo",
        ...     num_initial_conditions=2**10,
        ...     max_steps=100,
        ...     batch_size=2**10,
        ...     replacement_values=jnp.array([0, 1]),
        ...     mode="sync",
        ...     packbits=True,
        ... )


    Similary, the asynchronous simulation can be run by setting mode="async".

    If the initial conditions matrix is provided, the num_initial_conditions parameter is ignored. In this case, the initial_conditions matrix should have the individual initial conditions as rows.
    If only specfic inital conditions are to be used, the initial conditions matrix can be provided with the individual initial conditions as rows of the matrix. This provides control over simulating specific pre-defined initial conditions.

        >>> initial_conditions = jnp.array([[0, 1], [1, 0], [0, 0]])
        >>> run_ising(
        ...     topo_file="TOPOS/TS.topo",
        ...     initial_conditions=initial_conditions,
        ...     max_steps=100,
        ...     batch_size=2**10,
        ...     replacement_values=jnp.array([0, 1]),
        ...     mode="sync",
        ...     packbits=True,
        ... )

    For cases where the replacement values are not [-1, 1], the replacement values should be provided as a jax array of length 2 with the first value less than the second.

        >>> replacement_values = jnp.array([0, 1]) # Replacement values are 0 for negetive and 1 for positive
        >>> run_ising(
        ...     topo_file="TOPOS/TS.topo",
        ...     num_initial_conditions=2**10,
        ...     max_steps=100,
        ...     batch_size=2**10,
        ...     replacement_values=replacement_values,
        ...     mode="sync",
        ...     packbits=True,
        ... )

    The results for [-1, 1] replacment values will also be converted to 0 for all the -1 or 0 values in the states and 1s will remain as 1s when saving to the file. This is important as otherwise the packbits  would not work.

    The packbits function used is jnp.packbits which packs the 0/1 states into bits to reduce memory usage. This is useful when the number of nodes is large and the number of steps is also large. The memory usase can be reduced by a factor of 8 by packing the states into bits. If packbits is not set to True, the states are saved as is.

        >>> run_ising(
        ...     topo_file="TOPOS/TS.topo",
        ...     num_initial_conditions=2**10,
        ...     max_steps=100,
        ...     batch_size=2**10,
        ...     replacement_values=jnp.array([0, 1]),
        ...     mode="sync",
        ...     packbits=False,
        ... )

    The final dataframe which is written to the parquet file has the following columns for the packbits=False case:

    - Step: The step number for the simulation.
    - Node names: The names of the nodes in the network.

    If the packbits=True, the final dataframe has the following columns:

    - Step: The step number for the simulation.
    - Byte_i: The ith byte of the packed states for the simulation. The actual column name is the node names conactenated by the string `|` in order.

    In both these cases a step value of 0 in the dataframes will signify the initial condition of the simulation and the subsequent steps will be the states of the network at each step until the max_steps value is reached after which a new initial condition will be present (with step value 0).

    The Byte_i columns are created based on the number of nodes in the network. For example, if there are 100 nodes, there will be 13 columns for the packed states. They can be unpacked using jnp.unpackbits to get the unpacked state values. And the unpacked column can be named by splitting the corresponding column name by the string `|`.
    """
    # Get the adjacency matrix and node names
    topo_adj, node_names = parse_topo_to_matrix(topo_file)
    print(f"Running {mode} simulations for the network: {topo_file}")
    # Default batch size if not specified
    if batch_size is None:
        batch_size = 2**10
    # Default max steps if not specified
    if max_steps is None:
        # Determine max steps based on number of nodes
        max_steps = 10 * len(node_names)
    # Check if Inital conditions matrix is provided
    if inital_conditions is None:
        # Default number of inital conditions if not specified
        if num_initial_conditions is None:
            num_initial_conditions = 2**10
        # Generate random initial conditions
        initial_conditions = generate_intial_conditions(
            len(node_names), num_initial_conditions
        )
    else:
        # Check if the initial conditions matrix has the correct number of nodes
        if inital_conditions.shape[1] != len(node_names):
            raise ValueError(
                f"Initial conditions matrix should have the same number of columns as the number of nodes in the network. Expected {len(node_names)} columns, but got {inital_conditions.shape[1]}."
            )
        num_initial_conditions = inital_conditions.shape[0]
    # Check if the replacement values provided have length 2 and that the first value is less than the second
    # Sort the values
    replacement_values = jnp.sort(replacement_values)
    if len(replacement_values) != 2 or replacement_values[0] >= replacement_values[1]:
        raise ValueError(
            "Replacement values must be a jax array of length 2 with the first value less than the second."
        )
    # Initialize an empty numpy array to store the results
    if not packbits:
        results_array = np.empty((0, len(node_names) + 2), dtype=np.int16)
        # Create the column names for the dataframe
        df_cols = ["Step"] + node_names
    else:
        results_array = np.empty(
            (0, int(np.ceil(len(node_names) / 8)) + 2), dtype=np.int16
        )
        #### OLD Behaviour ####
        ## # Create the column names for the dataframe
        ## df_cols = ["Step"] + [
        ##     f"Byte_{i}" for i in range(int(np.ceil(len(node_names) / 8)))
        ## ]
        ## Create the column names by joining the nodes in batches of 8 seperated by "|"
        ## # Create a text file with the node names seperated by a comma for easy reference
        ## with open(f"{save_dir}/{topo_name}_{mode}_node_names_order.csv", "w") as f:
        ##     f.write(",".join(node_names))
        ######
        df_cols = ["Step"] + ["Initnum"] + [
            "|".join(node_names[i : i + 8]) for i in range(0, len(node_names), 8)
        ]
    # Start the simulation timer
    start_time = time.time()
    # Run synchronous or asynchronous simulations
    if mode == "sync":
        # Run synchronous simulations in batches
        for batch in range(0, num_initial_conditions, batch_size):
            batch_initial_conditions = initial_conditions[batch : batch + batch_size]
            batch_ic_indices = jnp.arange(start=batch,stop=min([batch+batch_size,len(initial_conditions)]))
            batch_results = vmap(
                lambda x,z: simulate_sync_trajectory(
                    x,
                    z,
                    topo_adj,
                    replacement_values,
                    jnp.arange(max_steps),
                ),
                in_axes=0,
            )(batch_initial_conditions,batch_ic_indices)
            # Stack the batch results into a single array
            batch_results = jnp.vstack(batch_results)
            # If packbits is True, pack the 0/1 states into bits to reduce memory usage
            if packbits:
                batch_results = packbit_states(batch_results)
            # Append the batch results to the main results array
            results_array = np.vstack((results_array, batch_results))

    elif mode == "async":
        # Generate random update indices for asynchronous updates
        update_indices_matrix = np.random.randint(
            0, len(node_names), (max_steps, num_initial_conditions)
        )
        update_indices_matrix = jnp.array(update_indices_matrix, dtype=jnp.int16)
        # Run asynchronous simulations in batches
        for batch in range(0, num_initial_conditions, batch_size):
            batch_initial_conditions = initial_conditions[batch : batch + batch_size]
            batch_update_indices = update_indices_matrix[:, batch : batch + batch_size]
            batch_ic_indices = jnp.arange(start=batch,stop=min([batch+batch_size,len(initial_conditions)]))
            batch_states = vmap(
                lambda x, y, z: simulate_async_trajectory(
                    x,
                    z,
                    topo_adj,
                    replacement_values,
                    y,
                ),
                in_axes=(0, 1, 0),
            )(batch_initial_conditions, batch_update_indices, batch_ic_indices)
            # Stack the batch results into a single array
            batch_states = jnp.vstack(batch_states)
            # If packbits is True, pack the 0/1 states into bits to reduce memory usage
            if packbits:
                batch_states = packbit_states(batch_states)
            # Append the batch results to the main results array
            results_array = np.vstack((results_array, batch_states))
    # Create a dataframe from the results array
    results_df = pd.DataFrame(results_array, columns=df_cols)
    # End the simulation timer
    print(f"Simulation time for {mode} mode: {time.time() - start_time:.2f} seconds")
    return results_df


def run_all_replicates_ising(
    topo_file,
    num_initial_conditions=None,
    inital_conditions=None,
    max_steps=None,
    batch_size=None,
    save_dir="IsingSimulResults",
    num_replicates=3,
    replacement_values=jnp.array([-1, 1]),
    mode="sync",
    packbits=False,
):
    """
    Run multiple replicate of ising model simulations for a given topology and save results.

    Parameters
    ----------
    topo_file : str
        Path to the topology file.
    num_initial_conditions : int, optional
        Number of initial conditions to sample. Defaults to 2**10 if not provided.
    inital_conditions : jax.numpy.ndarray, optional
        Initial condition matrix (rows = individual states). Overrides `num_initial_conditions` if given.
    max_steps : int, optional
        Maximum number of steps to simulate. Defaults to 10 x number of nodes.
    batch_size : int, optional
        Number of samples per batch. Defaults to 2**10.
    save_dir : str, optional
        Directory to store simulation results. Defaults to "IsingSimulResults".
    replacement_values : jax.numpy.ndarray, optional
        Two values used after evaluation: the first replaces elements < 0, the second replaces elements > 0.
        Defaults to [-1, 1]. When saving, all -1 and 0 values are mapped to 0, and 1 stays 1—for compatibility with `packbits`.
    mode : str, optional
        The simulation mode, either "sync" or "async". The default is "sync".
    packbits : bool, optional
        Whether to pack the 0/1 states into bits to reduce memory usage. The default is False.

    Returns
    -------
    None

    Example
    -------
    Run the synchronous simulation for a topology file in three replicates for both the modes:

        >>> run_all_replicates_ising("TOPOS/TS.topo", num_initial_conditions=1000, num_replicates=3, save_dir="IsingResults", mode="sync")
        >>> run_all_replicates_ising("TOPOS/TS.topo", num_initial_conditions=1000, num_replicates=3, save_dir="IsingResults", mode="async")

    This creates a directory called "IsingResults" which has the following directory strucuture:
    IsingResults
    └── 11
        ├── 1
        │   ├──  11_async_ising_results.parquet
        │   └──  11_sync_ising_results.parquet
        ├── 2
        │   ├──  11_async_ising_results.parquet
        │   └──  11_sync_ising_results.parquet
        └── 3
            ├──  11_async_ising_results.parquet
            └──  11_sync_ising_results.parquet

    """

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    # Create a subdirectory for the topology file
    topo_name = topo_file.split("/")[-1].split(".")[0]
    sim_dir = f"{save_dir}/{topo_name}"
    os.makedirs(sim_dir, exist_ok=True)
    # Get the adjacency matrix and node names
    topo_adj, node_names = parse_topo_to_matrix(topo_file)
    print(f"Topology: {topo_file}")
    # Run the simulations for the replicates
    for replicate in range(1, num_replicates + 1):
        # Reating the replicate folder
        replicate_dir = os.path.join(sim_dir, str(replicate))
        os.makedirs(replicate_dir, exist_ok=True)
        # Run the simulation
        result_df = run_ising(
            topo_file=topo_file,
            num_initial_conditions=num_initial_conditions,
            inital_conditions=inital_conditions,
            max_steps=max_steps,
            batch_size=batch_size,
            replacement_values=replacement_values,
            mode=mode,
            packbits=packbits,
        )
        # Saving the results to a parquet file
        result_df.to_parquet(
            os.path.join(replicate_dir, f"{topo_name}_{mode}_ising_results.parquet"),
            index=False,
        )
    return None


if __name__ == "__main__":
    pass
    # # Specify the path to the topo file
    # topo_folder = "TOPOS"

    # # Get the list of all the topo files
    # topo_files = sorted(glob.glob(f"{topo_folder}/*.topo"))
    # print(topo_files)

    # # Specify the replacement values
    # # replacement_values = jnp.array([0, 1])
    # # replacement_values = jnp.array([0, 1])

    # # Specify the number of steps to simulateys free for open source and community projects
    # max_steps = 100
    # print(f"Number of steps: {max_steps}")

    # # Specify the number of initial conditions to simulate
    # num_initial_conditions = 2**14
    # print(f"Number of initial conditions: {num_initial_conditions}")

    # # Specify the batch size for parallel evaluation
    # batch_size = 2**10

    # # Specify the number of replicates
    # num_replicates = 3

    # save_dir = "IsingSimulResults"

    # # Loop over all the topo files
    # for topo_file in topo_files:
    #     run_all_replicates_ising(
    #         topo_file,
    #         num_initial_conditions=num_initial_conditions,
    #         batch_size=batch_size,
    #         save_dir=save_dir,
    #         mode="sync",
    #         packbits=True,
    #     )
    #     run_all_replicates_ising(
    #         topo_file,
    #         num_initial_conditions=num_initial_conditions,
    #         batch_size=batch_size,
    #         save_dir=save_dir,
    #         mode="async",
    #         packbits=True,
    #     )
    #     break
