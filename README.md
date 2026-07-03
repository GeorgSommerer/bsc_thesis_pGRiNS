# bsc_thesis_pGRiNS
My bachelor's thesis at the DILiS group (FU Berlin): https://www.mi.fu-berlin.de/w/DILIS

## Installation:
```bash
git clone github.com/GeorgSommerer/bsc_thesis_pGRiNS
cd bsc_thesis_pGRiNS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The main branch is relatively lightweight and contains only the tools necessary to run pGRiNS.
The analysis branch also contains methods to perform downstream analysis and ML evaluation, as well as example results.

`requirements.txt` only works out of the box for Cuda 12.9 GPUs. If you use a different GPU, make sure to have the correct version of jax installed for your GPUs (e.g. `pip install --upgrade "jax[cuda13]"`) and fix other dependency issues accordingly.

## Usage:
```bash
python3 pgrins_main.py project_name --options
# Example:
nohup python3 -u pgrins_main.py KeggoRo -eps --num_params 1000 --num_init_conds 100 --batch_size 10000 --max_steps 10000 --pert_factor 100 --tmax 200 --pert_ratio 0.01 --max_num_clusters 0 --no_G_scaling >> Logs/out_KeggoRo.log 2>&1 &
```
Before running this, make sure you have set up correct input folder structure (see below).

Use `python3 pgrins_main.py -h` to display a list of optional commands.
Commonly used arguments are `-e` (if experimental `.h5ad` files are provided), `-p` (if perturbed data should be generated), and `-s` (if the steady state concentration of sink nodes be algebraically calculated instead of simulated, which reduces memory usage and simulation time significantly).
### Steps:
- `A: pgrins_prepare_input.py`: Prepare the input data by creating the project folder, the `.pert` file, and subsetting any experimental data.
- `B1: pgrins_run_grins.py`: Uses GRiNS to simulate unperturbed expression data using the project GRN.
- `C1: pgrins_prepare_output.py`: Turns the simulated unperturbed data into an adata object, adds a missingness filter, and normalizes the data.
- `D: pgrins_cluster.py`: Clusters the unperturbed data to find a homogenous cluster which is used as the control cells.
- `B2: pgrins_run_grins.py`: Uses GRiNS to simulate perturbed expression data using the project GRN.
- `C2: pgrins_prepare_output.py`: Concatenates the simulated control and perturbed data and turns them into an adata object, adds a missingness filter, and normalizes the data.
### Known Issues:
- It is possible that the pipeline crashes at some point due to a lack of available memory. If this happens, simply run the same command again, and the pipeline will restart from the last completed step. If the problem still prevails (especially common for the RACIPE simulation), try lowering `--batch_size`, or even `--num_params` or `--num_init_conds`.
- It is recommended to set the batch size so that the number of simulated cells (which you can calculate as `num_params*num_init_conds*pert_ratio`) is evenly divisible by it. Otherwise, the last batch of the simulation of perturbed cells can take a very long time.

## Folder Structure:
### Input:
Requires the following `Data` structure:
```bash
.
├── Data
│   ├── Experimental
│   │   └── {Example_PSeq_Dataset}
│   │       └── perturb.h5ad
│   ├── Topos
│   │   ├── {example_network_1}.topo
│   │   └── {example_network_2}.topo
│   └── Perts
│       └── {example_pert_list}.pert
├── grins
│   └── ...
├── Prep_Data
│   └── ...
├── Logs
└── pgrins_main.py
```
If `-e` is true, the directory `Experimental` must contain subdirectories named after the datasets (`Norman19`, `Replogle22`, etc.), with adata files themselves all named `perturb.h5ad`. A useful database containing many such datasets is [zenodo.org/records/7041849]https://zenodo.org/records/7041849. `Prep_Data/normalize.py` contains code that automatically normalizes the datasets AdamsonWeissman2016_GSM2406681_10X010, NormanWeissman2019_filtered, and ReplogleWeissman2022_K562_essential when `pgrins_main.py` is called, but it needs to be modified if other datasets were used. Realistically, only experimental control data is needed, although perturbed data would be useful for downstream analysis.

The subdirectory `Topos` must contain at least one `.topo` file which lists the directed edges A -> B or A -| C of a directed regulatory network over 3 columns delimited with a single space:
- The first column must have the name `Source` and contain the gene symbol of the outgoing node (A).
- The second column must have the name `Target` and contain the gene symbol of the incoming node (B or C).
- The third column must have the name `Type` and be 1 for an activating edge, or 2 for an inhibiting edge.
`Prep_Data/grn_to_topo.R` contains code that can be used to turn DoRothEA GRNs and KEGG Pathways PINs into `.topo` files.

The subdirectory `Perts` contains a `.pert` file, which has a list of perturbed genes with 3 columns delimited with a single space:
- The first column must have the name `Index` and indicate which perturbation set the perturbed gene belongs to.
- The second column must have the name `Gene` and contain the gene symbol of the perturbed gene. This gene must be present in at least one of the files in `Topos`.
- The third column must have the name `Type` and be 1 for overexpression (CRISPRa), 2 for knockdown (CRISPRi), and 3 for knockout (CRISPR KO).
For example, the perturbation sets `[(Gene_A : CRISPRa, Gene_B : CRISPRi),(Gene_C : CRISPR_KO),(Gene_A : CRISPRi, Gene_C : CRISPRi)]` would be turned into
```bash
Index Gene Type
0 Gene_A 1
0 Gene_B 2
1 Gene_C 3
2 Gene_A 2
2 Gene_C 2
```
If experimental data is provided, this file is created automatically (unless specified otherwise). Otherwise, it must be specified.

### Output:
```bash
.
├── Data
│   ├── Projects
│   │   └── {project_name}
│   │       ├── {replicate_number}
│   │       │   ├── ctrl_best_cells.pickle
│   │       │   ├── perturb_norm_ctrl.h5ad
│   │       │   └── perturb_norm_full.h5ad
│   │       ├── {project_name}_full.topo
│   │       ├── {project_name}.topo
│   │       └── {project_name}_sinks.topo
│   ├── SimulResults_Racipe
│   │   └── {project_name}
│   │       └── ...
```
The outputs will be saved in a subdirectory of `Data/Projects` with the same name as the input of `{project_name}`.
This folder will contain a file called `{project_name}.topo` which is a combined version of the files in `Data/Topos` without the edges between nodes not in the datasets in `Data/Experimental`. If `-s` is true, then `{project_name}_full.topo` will contain the same edges as `{project_name}.topo` if `-s` was false, `{project_name}.topo` will only contain the edges not pointing towards sink nodes, and `{project_name}_sinks.topo` will contain all edges pointing towards sink nodes.

For each replicate simulated from this network, a folder `{replicate_number}` will be created that contains `perturb_norm_ctrl.h5ad` with all (non-clustered) cells of the unperturbed run, `ctrl_best_cells.pickle`, which contains the row names of `.obs` of `perturb_norm_ctrl.h5ad` belonging to the chosen cluster, and `perturb_norm_full.h5ad`, which combines these chosen control cells with the simulated cells for each perturbation.

In short, `perturb_norm_full.h5ad` is the final output of the pGRiNS pipeline and can be used for downstream analysis.

`Prep_Data/Plots/{project_name}/{replicate_number}` will contain graphs showcasing the results of the PCA and UMAP of the clustering step. Depending on the results, changing some parameters (e.g. the number of PCs used for the UMAP) is advised.

The simulation results will be saved in `Data/SimulResults_Racipe/{project_name}` for a RACIPE simulation, or `Data/SimulResults_Ising/{project_name}` for a Boolean Ising simulation.

## License

[MIT](https://choosealicense.com/licenses/mit/)
