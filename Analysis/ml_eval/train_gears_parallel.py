from gears import PertData, GEARS
import h5py
import hdf5plugin
import anndata as ad
import sys
import os
import numpy as np
import torch
import scanpy as sc
import torch.multiprocessing as mp

def train_gears_parallel(proc_data : PertData, current_split : int, i : int, lock, filename : str, modelname : str):
    """
    Trains multiple iterations of the GEARS model in parallel for different cross validation splits.
    Using pytorch==2.7.1 for cuda==12.9

    Parameters:
    -----------
    proc_data : PertData
        A GEARS PertData object that has loaded perturb_processed.h5ad.
    current_split : int
        The split that processed in this iteration.
    i : int
        The number of the GPU used in this thread.
    lock : lock
        A multiprocessing lock to prevent multiple files being saved at the same time.
    filename : str
        The name of the file.
    """
    device = f"cuda:{i}"
    torch.cuda.set_device(i) # For each thread, occupy a GPU
    proc_data.prepare_split(split="custom",split_dict_path=f"Models/Splits/{filename}/split_{current_split}.pickle")
    proc_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    with lock:
        print(f"Getting GEARS model for split {current_split}...")
    """
    At some point, a pd.Series containing boolean values is input into scipy._validate_indices, which throws an AttributeError.
    This is because the original code in venv/lib/python3.12/site-packages/scipy/sparse/_index.py, line 401 is:
        index.extend(idx.nonzero())
    However, a pd.Series object needs to be converted with .to_numpy() before .nonzero() can be called.
    Therefore, the source code of scipy is modified at that place to:
        try:
            index.extend(idx.nonzero())
        except AttributeError as e:
            index.extend(idx.to_numpy().nonzero())
    """
    gears_model = GEARS(proc_data, device = device, 
                            weight_bias_track = False)
    gears_model.model_initialize(hidden_size = 64)

    gears_model.train(epochs=20,lr=1e-3)
    with lock: # Lock is necessary so that all files are properly saved
        print(f"Saving {current_split}...")
        os.makedirs(f"../../Data/Experimental/{filename}/{modelname}/Models/GEARS/cv_{current_split}",exist_ok=True)
        gears_model.save_model(path=f"../../Data/Experimental/{filename}/{modelname}/Models/GEARS/cv_{current_split}")   



if __name__ == '__main__':
    """
    Each available GPU is used to train a GEARS model on one iteration of the train/val/test splits from cross validation.
    See train_gears.ipynb for more comments.

    3.6.
    nohup python3 -u train_gears_parallel.py Norman19 experimental KeggoRo_0206 5 >> ../../Logs/GEARS_Norman19_KeggoRo_0206_out.log 2>&1 &
    """
    replicate = 1
    filename = sys.argv[1]
    modelname = sys.argv[2]
    grn_file = sys.argv[3]
    splits_left = int(sys.argv[4])

    os.makedirs(f"../../Data/Experimental/{filename}/{modelname}/{modelname}",exist_ok=True)
    proc_data = PertData(f"../../Data/Experimental/{filename}/{modelname}")
    if not os.path.isfile(f"../../Data/Experimental/{filename}/{modelname}/perturb_processed.h5ad"):
        print("Processing dataset...")
        if modelname == "experimental":
            norm_data = sc.read_h5ad(f"../../Data/Experimental/{filename}/perturb_norm_subset_{grn_file}.h5ad")
        elif modelname == "pGRiNS":
            norm_data = sc.read_h5ad(f"../../Data/Projects/{grn_file}/{replicate:03}/perturb_norm_pert.h5ad")
        norm_data.X = norm_data.layers["log1p"].copy()
        """
        In line 242 of gears/pertdata.py, the following line exists:
            dataset_name = dataset_name.lower()
        This is commented out in the source code (path for venv: vsp/lib/python3.12/site-packages/gears/pertdata.py) to conform with already existing Data folder file names.
        Also, this change serves no purpose algorithmically.
        """
        ad.settings.allow_write_nullable_strings = True
        proc_data.new_data_process(dataset_name = "",adata=norm_data)

    proc_data.load(data_path=f"../../Data/Experimental/{filename}/{modelname}")

    mp.set_sharing_strategy('file_system')
    mp.set_start_method("spawn",force=True) # Necessary for multithreading to work with Torch
    processes = []
    lock = mp.Lock()

    # Divide splits into batches so that each batch occupies as many GPUs as are available
    gpu_count = torch.cuda.device_count()
    print(f"Devices available: {gpu_count}; CV runs: {splits_left}")
    while splits_left > 0:
        splits_iter = splits_left if splits_left < gpu_count else gpu_count
        print(f"Processing {splits_iter} splits...")
        for i in range(splits_iter):
            # Start a number of threads equal to the number of GPUs available
            current_split = splits_left-i-1
            if not os.path.exists(f"Models/GEARS/{filename}/{modelname}/cv_{current_split}"):
                print(f"Split {current_split}:")
                p = mp.Process(target=train_gears_parallel,args=(proc_data,current_split,i,lock,filename,modelname))
                p.start()
                processes.append(p)
                # Join the processes to end them
        for p in processes:
            p.join()
        splits_left -= splits_iter