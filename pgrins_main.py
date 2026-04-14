from grins import get_pert_data
import os

topo = "dorothea_tf2_abc"
replicate = "1"
is_racipe = True
if is_racipe:
    path_to_data = f"Data/SimulResults_Racipe/{topo}/{replicate}/{topo}_steadystate_solutions_{replicate}"
else:
    path_to_data = f"Data/SimulResults_Ising/{topo}/{replicate}/{topo}_steadystate_solutions_{replicate}"

if not os.path.isfile(f"{path_to_data}.h5ad"):
    grins_data = get_pert_data.grins_to_adata(path_to_data)
else:
    print("Reading h5ad file...")
    grins_data = sc.read_h5ad(f"{path_to_data}.h5ad")