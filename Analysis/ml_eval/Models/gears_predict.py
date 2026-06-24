import os
import pickle
import numpy as np
import anndata as ad

def predict_gears(it : int, test_perts : list, filename : str, modelname : str, cc : int = None) -> dict[str,list[float]]:
    """
    Uses the GEARS model for the current split to predict the expression vector across all 5000 tested genes for all test perturbations.

    Parameters:
    -----------
    it : int
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
    if cc == None:
        model_append = ""
    else:
        model_append = f"_{cc}"
    out_path = f"../../Data/Experimental/{filename}/{modelname}/Models{model_append}/GEARS/GEARS_pred_{it}.pickle"
    if not os.path.exists(out_path):
        from gears import PertData, GEARS

        proc_data = PertData(f"../../Data/Experimental/{filename}/{modelname}")
        proc_data.load(data_path=f"../../Data/Experimental/{filename}/{modelname}")
        proc_data.prepare_split(split="custom",split_dict_path=f"../../Data/Experimental/{filename}/Splits/split_{it}.pickle")
        proc_data.get_dataloader(batch_size = 32, test_batch_size = 128)
        gears_model = GEARS(proc_data, device = "cpu", 
                            weight_bias_track = False)
        gears_model.model_initialize(hidden_size = 64)

        gears_model.load_pretrained(path=f"../../Data/Experimental/{filename}/{modelname}/Models{model_append}/GEARS/cv_{it}")
        gears_preds = gears_model.predict([p.split("_") for p in test_perts])

        with open(out_path,'wb') as handle:
            pickle.dump(gears_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        gears_preds = pickle.load(open(out_path,'rb'))

    return gears_preds