# bsc_thesis_pGRiNS
My bachelor's thesis at the DILiS group (FU Berlin): https://www.mi.fu-berlin.de/w/DILIS

# Installation:
- Set up a new venv with `pip install -r requirements.txt`
- However, these only work out of the box for Cuda 12.9 GPUs. Make sure to have the correct version of jax installed for your GPUs (e.g. `pip install --upgrade "jax[cuda13]"`) and fix other dependency issues accordingly.
# Usage:
- `exp_data`: whether or not experimental control data should be used to improve the parameter reduction.
- `project_name`: name of the project. Will be used for the name of the output folder.

# Folder Structure:
## Input:
Requires the following `Data` structure:

.
├── Data
│   ├── Experimental
│   │   └── Example_PSeq_Dataset
│   │       └── perturb.h5ad
│   ├── Topos
│   │   ├── example_network_1.topo
│   │   └── example_network_2.topo
│   └── Perts
│       ├── example_pert_list_1.pert
│       └── example_pert_list_2.pert
├── grins
│   └── ...
├── ...
├── pgrins_main.py
└── pgrins_main.sh

If `exp_data` is true, then the directory `Experimental` must contain subdirectories named after the datasets (`AdamsonWeissman2016_GSM2406681_10X010`, `NormanWeissman2019_filtered`, etc.), with the datasets themselved all named `perturb.h5ad`. A useful resource containing many such datasets is `https://zenodo.org/records/7041849`.

The subdirectory `Topos` must contain at least one `.topo` file which lists the directed edges A -> B or A -| C of a directed regulatory network over 3 columns delimited with a single space:
- The first column must have the name `Source` and contain the gene symbol of the outgoing node (A).
- The second column must have the name `Target` and contain the gene symbol of the incoming node (B or C).
- The third column must have the name `Type` and be 1 for an activating edge, and 2 for an inhibiting edge.

The subdirectory `Perts` contains `.pert` files, which have a list of perturbed genes with 3 columns delimited with a single space:
- The first column must have the name `Index` and indicate which perturbation set the perturbed gene belongs to. For example, if this column is [0,1,2,2], then the perturbation of the two last genes will be treated as a double perturbation. 
- The second column must have the name `Gene` and contain the gene symbol of the perturbed gene. This gene must be present in at least one of the files in `Topos`.
- The third column must have the name `Type` and be 1 for overexpression (CRISPRa), 2 for knockdown (CRISPRi), and 3 for knockout (CRISPR KO).

## Output:
.
├── Data
│   ├── Projects
│   │   └── project_name
│   │       ├── project_name.topo
│   │       └── project_name_sinks.topo
│   ├── project_name
│   │   ├── project_name.topo
│   ├── SimulResults_Racipe
│   │   └── project_name
│   │       └── ...
│   ├── SimulResults_Ising
│   │   └── project_name
│   │       └── ...

The outputs will be saved in a subdirectory of `Data` with the same name as the input of `project_name`.
This folder will contain a file called `project_name.topo` which is a combined version of the files in `Data/Topos` without the edges between nodes not in the datasets in `Data/Experimental`.
The simulation results will be saved in `Data/SimulResults_Racipe/project_name` for a RACIPE simulation, or `Data/SimulResults_Ising/project_name` for a Boolean Ising simulation.