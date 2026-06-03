from gears import PertData, GEARS
import os
import pickle
import numpy as np
import anndata as ad

def predict_gears(i : int, test_perts : list, filename : str) -> dict[str,list[float]]:
    """
    Uses the GEARS model for the current split to predict the expression vector across all 5000 tested genes for all test perturbations.

    Parameters:
    -----------
    i : int
        The number of the split.
    test_perts : list
        A list of perturbed genes to be predicted.
    filename : str
        The name of the dataset used.

    Returns:
    --------
    gears_preds : dict
        A dictionary with the test_perts as keys and the expression vectors as values.
    """
    out_path = f'Models/Splits/{filename}/GEARS_pred_{i}.pickle'

    if not os.path.exists(out_path):
        proc_data = PertData(f"Data/{filename}")
        if not os.path.isfile(f"Data/{filename}/perturb_processed.h5ad"):
            print("Processing dataset...")
            norm_data = ad.read_h5ad(f"Data/{filename}/perturb_norm.h5ad")
            norm_data.X = norm_data.layers["log1p"].copy()
            """
            In line 242 of gears/pertdata.py, the following line exists:
                dataset_name = dataset_name.lower()
            This is commented out in the source code (path for venv: vsp/lib/python3.12/site-packages/gears/pertdata.py) to conform with already existing Data folder file names.
            Also, this change serves no purpose algorithmically.
            """
            ad.settings.allow_write_nullable_strings = True
            proc_data.new_data_process(dataset_name = "",adata=norm_data)
        proc_data.load(data_path=f"Data/{filename}")
        proc_data.prepare_split(split="custom",split_dict_path=f"Models/Splits/{filename}/split_{i}.pickle")
        proc_data.get_dataloader(batch_size = 32, test_batch_size = 128)
        gears_model = GEARS(proc_data, device = "cpu", 
                            weight_bias_track = False)
        gears_model.model_initialize(hidden_size = 64)

        gears_model.load_pretrained(path=f"Models/GEARS/{filename}/cv_{i}")

        gears_preds = gears_model.predict([[p] for p in test_perts])

        with open(out_path,'wb') as handle:
            pickle.dump(gears_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        gears_preds = pickle.load(open(out_path,'rb'))

    return gears_preds