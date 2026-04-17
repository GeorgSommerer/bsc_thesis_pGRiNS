import glob
from itertools import chain, cycle
import numpy as np
import pandas as pd


# Function to find all the state variables in the logical boolean model
def find_state_variables(bool_lines, bool_logic_ops):
    """
    Processes a list of boolean logic expressions to identify state variables and their indices.

    Args:
        bool_lines (list of str): List of boolean logic expressions as strings.
        bool_logic_ops (dict): Dictionary of boolean logic operations.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping each unique state variable to its index.
            - list: A list of state variable nodes (left-hand side of the expressions).

    Raises:
        ValueError: If a line does not have '=' as the second element.
    """
    # Replace all the parentheses with white spaces
    bool_lines = [line.replace("(", " ").replace(")", " ") for line in bool_lines]
    # Split the lines at white spaces
    bool_lines = [line.split() for line in bool_lines]
    # Make a placeholder list of the nodes for which logical expressions are defined (LHS of the expression)
    statevar_nodes = []
    # Loop through all the lines
    for line in bool_lines:
        # Checking if = is the second element in the line
        if "=" in line[1]:
            # Append the first element of the line to the logicexpr_nodes list
            statevar_nodes.append(line[0])
        else:
            # Raise an error if the line does not have = as the second element
            # Print the index of the line
            raise ValueError(
                f"Logical expression not defined properly ({bool_lines.index(line)})"
            )
    # Get the unique state variables
    all_boolnodes = set(list(chain(*bool_lines)))
    # Remove all boolean operations and "=" from the state variables
    all_boolnodes = all_boolnodes - set(bool_logic_ops.values()) - set("=")
    # Sort the state variables - ensures the order of the state variables is consistent across all runs
    all_boolnodes = sorted(list(all_boolnodes))
    # Make a map of the bool nodes and thier index in the list
    all_boolnodes = {node: i for i, node in enumerate(all_boolnodes)}
    # Return all the state variables and all boolean nodes
    return all_boolnodes, statevar_nodes


# Function to replace all the node names with their index in the list for a given line
def replace_node_names(line, all_boolnodes):
    """
    Replaces node names in a given line with their corresponding indices from a dictionary.

    Args:
        line (str): The input string containing node names to be replaced.
        all_boolnodes (dict): A dictionary where keys are node names and values are their corresponding indices.

    Returns:
        str: The modified line with node names replaced by their indices in the format 'bnode[index]'.
    """
    # Loop through all the nodes
    for key in all_boolnodes.keys():
        # Check if the node is present in the line
        if key in line:
            # Replace the node with its index in the list
            line = line.replace(key, f"bnode[{all_boolnodes[key]}]")
    # Return the line with the node names replaced with their index
    return line


# Function to the read the logical boolean model file
def read_logical_boolmodel(filename, bool_logic_ops):
    """
    Reads and parses a logical boolean model from a file, replacing custom logical operators
    with Python logical operators, and returns the state variables and logical expressions.

    Args:
        filename (str): The path to the file containing the logical boolean model.
        bool_logic_ops (dict): A dictionary mapping custom logical operators to Python logical operators.

    Returns:
        tuple: A tuple containing:
            - all_boolnodes (dict): A dictionary mapping state variable names to their indices.
            - lines (list): A list of logical expressions with state variables replaced by their indices.
    """
    # Open the file and parse it
    with open(filename, "r") as f:
        # Read all the lines of the file
        lines = f.readlines()
        # Strip the lines of any leading or trailing whitespaces
        lines = [line.strip() for line in lines]
        # Strip the lines of newline characters
        lines = [line.replace("\n", "") for line in lines]
        # Replace all other whitespaces with a single whitespace
        lines = [" ".join(line.split()) for line in lines]
        # Replace the logical operators with the python logical operators
        for key, value in bool_logic_ops.items():
            lines = [line.replace(key, value) for line in lines]
        # Find all the state variables in the logical boolean model
        all_boolnodes, statevar_nodes = find_state_variables(lines, bool_logic_ops)
        # Replace all the node names with their index in the list
        lines = [replace_node_names(line, all_boolnodes) for line in lines]
    # Some nodes do not have logical expressions defined (Input nodes)
    # Adding thier expression values as Node = Node
    for node in all_boolnodes.keys():
        if node not in statevar_nodes:
            lines.append(f"bnode[{all_boolnodes[node]}] = bnode[{all_boolnodes[node]}]")
    # Sort the lines according to the state variables index
    lines = sorted(
        lines, key=lambda x: int(x.split("=")[0].replace("bnode[", "").replace("]", ""))
    )
    # Remove all LHS of the logical expressions as we already have ordered them according to the state variables
    # This will make it easier for tracking and updating the state at each step (inplace update can be avoided)
    lines = [line.split("=")[1].strip() for line in lines]
    # Return the state variables and the logical expressions
    return all_boolnodes, lines


# Function to convert the simulation states to a pandas dataframe
def generate_solution_df(simulation_states, node_list):
    """
    Convert the simulation states to a pandas dataframe.

    Parameters:
    simulation_states (list): The list of states of the nodes.
    node_list (list): The list of nodes in the network.

    Returns:
    pandas.DataFrame: The converted dataframe.
    """
    # Convert the simulation states to a pandas dataframe
    simulation_states = pd.DataFrame(simulation_states, columns=["Step"] + node_list)
    # Replace all the True values with 1 and False values with 0 - only applicable for the initial and steady states as they become string and not boolean
    simulation_states[node_list] = simulation_states[node_list].replace(
        {"True": 1, "False": 0}
    )
    # Convert the state of the nodes from boolean to integer
    simulation_states[node_list] = simulation_states[node_list].astype(int)
    # Check if the last step is the steady state if not make it as UnSteady
    # Getting the last row of the simulation states and checking if the state labelled as "Steady"
    if simulation_states["Step"].iloc[-1] != "SteadyState":
        simulation_states.at[simulation_states.index[-1], "Step"] = "UnSteady"
    # Return the simulation states dataframe
    return simulation_states


# Function to generate the final state dictionary
def generate_final_state_dict(simulation_states, all_boolnodes):
    """
    Generate the final state dictionary from the simulation states.

    Parameters:
    simulation_states (list): The list of states of the nodes.
    all_boolnodes (dict): A dictionary mapping state variable names to their indices.

    Returns:
    dict: The final state dictionary.
    """
    # Get the last simulation state
    # Replace the True and False values with 1 and 0
    last_state = [1 if node == "True" else 0 for node in simulation_states[-1][1:]]
    # Make a dictionary of the final state
    last_state = dict(zip(list(all_boolnodes.keys()), last_state))
    # Check if the last step is the steady state if not make it as UnSteady
    if simulation_states[-1][0] != "SteadyState":
        last_state["StateType"] = "UnSteady"
    else:
        last_state["StateType"] = "SteadyState"
    # Add time step to the dictionary
    last_state["Step"] = len(simulation_states) - 1
    # return the dictionary
    return last_state


# Function to update the node state based on the logical expression
def update_node_state(node_index, logical_expresion, bnode):
    """
    Update the state of a node based on a logical expression.

    Args:
        node_index (int): The index of the node to update.
        logical_expresion (list): A list of logical expressions.
        bnode (list): A list representing the current state of nodes.

    Returns:
        list: The updated state of the nodes.
    """
    # Update the node state based on the logical expression
    new_node_state = eval(logical_expresion[node_index])
    # Update the state of the node in bnode
    bnode[node_index] = new_node_state
    # Return the updated state of the node
    return bnode


# Function run the logical boolean model
def run_logicalbool_sync(
    logic_file,
    max_steps=None,
    inital_state=None,
    bool_logic_ops={"AND": "and", "OR": "or", "NOT": "not"},
    time_series=True,
):
    """
    Simulates the synchronous update of a logical boolean model.

    Parameters:
    logic_file (str): Path to the file containing the logical boolean model.
    max_steps (int, optional): Maximum number of steps for the simulation. If not provided, defaults to 100 times the number of state variables.
    inital_state (list of bool, optional): Initial state of the state variables. If not provided, the state variables are randomly initialized.
    bool_logic_ops (dict, optional): Dictionary mapping logical operators to their corresponding string representations. Defaults to {"AND": "and", "OR": "or", "NOT": "not"}.
    time_series (bool, optional): If True, returns the simulation states as a time series DataFrame. If False, returns the final state as a dictionary. Defaults to True.

    Returns:
    dict or pandas.DataFrame: If time_series is False, returns a dictionary with the final state of the state variables, state type and the timestep of the final state. If time_series is True, returns a DataFrame with the simulation states over time.
    """
    # Read the logical boolean model
    all_boolnodes, logical_expressions = read_logical_boolmodel(
        logic_file, bool_logic_ops
    )
    # Get the number of state variables
    num_state_vars = len(all_boolnodes)
    # If the max_steps is not provided, set it to be equal to 100*number of state variables
    if max_steps is None:
        max_steps = 100 * num_state_vars
    # else:
    #     max_steps = max_steps * num_state_vars
    # If the initial state is not provided, randomly initialize the state variables
    # The values will be true or false values
    if inital_state is None:
        inital_state = np.random.choice([True, False], num_state_vars).tolist()
    # Store the states and the step numbers as a list of lists
    simulation_states = []
    # Set the initial state of the state variables
    bnode = inital_state
    # Add the initial state to the simulation states
    simulation_states.append(["InitialState"] + bnode)
    # print("Initial State: ", bnode)
    # Loop through the max_steps
    for i in range(1, max_steps):
        # Map the excec function on the logical expressions list
        new_bnode = list(map(eval, logical_expressions))
        # Check if new state is same as the previous state
        if np.array_equal(new_bnode, bnode):
            # Update the new state as the final state and break the loop
            bnode = new_bnode
            # Append the final state to the simulation states
            simulation_states.append(["SteadyState"] + bnode)
            # print("Final State: ", bnode)
            break
        # Update the state of the state variables
        bnode = new_bnode
        # Append the state of the state variables to the simulation states
        simulation_states.append([i] + bnode)
    if time_series:
        # Return the simulation states dataframe
        return generate_solution_df(simulation_states, list(all_boolnodes.keys()))
    else:
        # Return the final state dictionary
        return generate_final_state_dict(simulation_states, all_boolnodes)


# # Function to check is the previous n states are the same as the current state
# def check_convergence(current_state, simulation_states, min_steps):
#     # Get the last n states of the simulation states
#     last_n_states = simulation_states[:, 1:][-min_steps:]
#     # Check if the last n states are the same and if its the same as the current state
#     if (last_n_states == current_state).all():
#         return True
#     else:
#         return False


# Function to check convergence using one step of sync update to see if the same state is reached
def check_convergence_sync_step(bnode, logical_expressions):
    # Get the state of the nodes after one step of sync update
    new_state = list(map(eval, logical_expressions))
    # Check if bnnode is the same as the new state
    if np.array_equal(bnode, new_state):
        return True
    else:
        return False


# Function to run the logical boolean model in an asynchronous mode
def run_logicalbool_async(
    logic_file,
    max_steps=None,
    inital_state=None,
    update_order=None,
    min_steps=None,
    bool_logic_ops={"AND": "and", "OR": "or", "NOT": "not"},
    time_series=True,
):
    """
    Simulates the asynchronous update of a logical boolean model.

    Parameters:
    logic_file (str): Path to the file containing the logical boolean model.
    max_steps (int, optional): Maximum number of steps for the simulation. Defaults to 100 times the number of state variables.
    inital_state (list of bool, optional): Initial state of the state system. If None, the state variables are randomly initialized. Defaults to None.
    update_order (list of str or str, optional): Order in which the nodes are updated. If None, nodes are randomly sampled each step. If "random_cyclic", nodes are updated in a random cyclic order. Defaults to None.
    min_steps (int, optional): Minimum number of steps for the simulation. Defaults to the number of state variables.
    bool_logic_ops (dict, optional): Dictionary mapping logical operators to their string representations. Defaults to {"AND": "and", "OR": "or", "NOT": "not"}.
    time_series (bool, optional): If True, returns the simulation states as a time series DataFrame. If False, returns the final state, state type and the timestep of the final state as a Dictionary. Defaults to True.

    Returns:
    pandas.DataFrame or dict: If time_series is True, returns a DataFrame containing the simulation states over time. If time_series is False, returns a dictionary containing the final state of the state variables.
    """
    # Read the logical boolean model
    all_boolnodes, logical_expressions = read_logical_boolmodel(
        logic_file, bool_logic_ops
    )
    # Get the number of state variables
    num_state_vars = len(all_boolnodes)
    # If the min_steps is not provided, set it to be equal to number of state variables
    if min_steps is None:
        min_steps = num_state_vars
    # If the max_steps is not provided, set it to be equal to 100*number of state variables
    if max_steps is None:
        max_steps = 100 * num_state_vars
    # else:
    #     max_steps = max_steps * num_state_vars
    # If the initial state is not provided, randomly initialize the state variables
    # The values will be true or false values
    if inital_state is None:
        inital_state = np.random.choice([True, False], num_state_vars).tolist()
    # Set the initial state of the state variables
    bnode = inital_state
    # Initialize the simulation states matrix with the initial state
    simulation_states = np.array([["InitialState"] + bnode])
    # If node cycle is None, randomly sample the nodes from the state variables
    if update_order is None:
        update_order = np.random.choice(range(0, len(all_boolnodes)), max_steps)
    elif update_order == "random_cyclic":
        # Create a order of the ndoes by randomly shuffling the node list
        update_order = np.random.permutation(list(all_boolnodes.keys()))
        # Map the node names to their index in the list
        update_order = [all_boolnodes[node] for node in update_order]
    else:
        # Map the node names to their index in the list
        update_order = [all_boolnodes[node] for node in update_order]
    # Make the node order cyclic
    update_order = cycle(update_order)
    # Initialize the step counter
    step = 1
    # Loop thorugh the cycle of the node order
    for node_to_update in update_order:
        # Break the loop if the step counter exceeds the max_steps
        if step >= max_steps:
            break
        # Update the state of the node
        new_bnode = update_node_state(node_to_update, logical_expressions, bnode)
        # Check if new state is same as the previous state
        # Also check is the current state is the same as the previous
        if step > (min_steps + 1) and np.array_equal(new_bnode, bnode):
            # Now check if synchrnous update will reach the same state
            if check_convergence_sync_step(bnode, logical_expressions):
                # Update the new state as the final state and break the loop
                bnode = new_bnode
                # Append the final state to the simulation states
                simulation_states = np.append(
                    simulation_states, [["SteadyState"] + bnode], axis=0
                )
                break
        # Update the state of the state variables
        bnode = new_bnode
        # Append the state of the state variables to the simulation states
        simulation_states = np.append(simulation_states, [[step] + bnode], axis=0)
        # Increment the step counter
        step += 1
    if time_series:
        simulation_states = generate_solution_df(
            simulation_states, list(all_boolnodes.keys())
        )
        # Return the simulation states
        return simulation_states
    else:
        # Generate the final state dictionary
        final_state = generate_final_state_dict(simulation_states, all_boolnodes)
        # Return the final state dictionary
        return final_state


if __name__ == "__main__":
    # Specifying the folder path
    boolmodel_path = "../Logical_TOPOS/"
    # Get a list of all models in the folder
    model_list = sorted(glob.glob(boolmodel_path + "*.txt"))
    # Loop through all the models
    for model_file in model_list[:1]:
        print(f"Model: {model_file}")
        # Run the logical boolean model
        for i in range(3):
            print(f"Run {i}")
            solu_df = run_logicalbool_async(
                model_file,
                # update_order="random_cyclic",
                time_series=False,
            )
            print(solu_df)
