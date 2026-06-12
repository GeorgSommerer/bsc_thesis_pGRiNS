# ML_evaluation_project

## Name
This code is taken from the Software Project "Gene expression prediction ML models and their evaluation" (2026) at the FU Berlin, in which I participated.

## Project overview
Predicting how cells respond to genetic perturbations is an important challenge in computational
biology. Single-cell CRISPR perturbation experiments allow us to measure changes in gene
expression after specific genes are activated or inhibited. Testing all possible perturbations
experimentally is expensive, time-consuming and not scalable, computational deep learning
models such as GEARS are used to predict the eﬀects of unseen perturbations.
The central question in our project was not only whether GEARS can make accurate predictions,
but also how such predictions should be evaluated. 

In this project we built a reproducible pipeline for preprocessing single-cell perturbation data, 
preparing the data for GEARS, training GEARS models, generating predictions and evaluating the model 
performance against several baseline models. 

## Datasets
We worked with two single-cell CRISPR perturbation datasets:

Adamson: 
Norman: 

## Pipeline
The project pipeline consists of the following main steps:

1. Data preprocessing and filtering
    - Load raw perturbation data (perturb.h5ad)
    - Apply the dataset-specific filtering functions
    - Prepare consistent annotations for downstream analysis
2. GEARS-compatible preprocessing
    - Convert the data into a GEARS-compatible format
    - Save processed data as perturb_processed.h5ad
3. Quality control
    - Compute the QC metrics:
        - total counts per cell
        - number of detected genes per cell
        - percentage of mitochondrial counts
    - Apply MAD-based outlier detection to remove low-quality cells
4. Normalization
    - Normalize cell counts
    - Apply log1p transformation
    - Select highly variable genes
    - Save normalized data as perturb_norm.h5ad
    - Determine genes not in the GO-graph
    - Separate control and perturbed cells
    - Filter perturbations that are not represented in the GEARS Gene Ontology graph and must therefore be skipped
5. Cross-validation splitting
    - Split perturbations into cross-validation folds
    - Use held-out perturbations for testing model generalization
6. Model training and prediction
7. Baseline comparison
    Compare GEARS against:
    - control baseline (negative control)
    - mean baseline (negative baseline)
    - technical duplicate baseline (positive control)
    - interpolated duplicate baseline (positive control)
8. Metrics used for evaluation
    Simple metrics:
    - Means Square Error (MSE)
    - Pearson correlation coefficient
    - Pearson Delta correlation coefficient
    Well calibrated metrics:
    - Weighted Mean Square Error (WMSE)
    - Weighted Pearson Delta
9. Evaluation and visualization
    - Compute the performance metrics for each model
    - Aggregate results across perturbations and CV splits
    - Generate plots for model comparison
    - Parse GEARS training logs to inspect training and validation MSE

## Project structure
```text
project/
├── Data/
│   ├── AdamsonWeissman2016/
│   │   ├── perturb.h5ad
│   │   ├── perturb_norm.h5ad
│   │   └── perturb_processed.h5ad
│   └── NormanWeissman2019/
│       ├── perturb.h5ad
│       ├── perturb_norm.h5ad
│       └── perturb_processed.h5ad
├── Metrics/
│   ├── performance_metrics.py
│   └── weights.py
│
├── Models/
│   ├── GEARS/
│   ├── baselines.py
│   ├── models_predict.py
│   └── Splits/
│
├── Normalization/
│   ├── normalize.py
│   └── split.py
│
├── Results/
│   └── Figures
│
├── full_pipeline.ipynb
├── test.py
├── test_deg.py
├── train_gears_parallel.py
├── performance_metrics.py
├── train_gears_parallel.py
└── README.md
```

## Installation
In order to run this code flawlessly, it is necessary to modify the source code at the following places:

<code>
In line 242 of gears/pertdata.py, the following line exists:
    dataset_name = dataset_name.lower()
This is commented out in the source code (path for venv: vsp/lib/python3.12/site-packages/gears/pertdata.py) to conform with already existing Data folder file names.
Also, this change serves no purpose algorithmically.
<code>


<code>
At some point, a pd.Series containing boolean values is input into scipy._validate_indices, which throws an AttributeError.
This is because the original code in venv/lib/python3.12/site-packages/scipy/sparse/_index.py, line 401 is:
    index.extend(idx.nonzero())
However, a pd.Series object needs to be converted with .to_numpy() before .nonzero() can be called.
Therefore, the source code of scipy is modified at that place to:
    try:
        index.extend(idx.nonzero())
    except AttributeError as e:
        index.extend(idx.to_numpy().nonzero())
<code>


## Usage
It is necessary to save the perturb seq datasets in a directory called `./Data/{filename}/perturb.h5ad`.
Then, run the cells in `full_pipeline.ipynb`, until an exception is raised instructing the user to run `train_gears_parallel.py`.
Follow this instruction and once done continue with `full_pipeline.ipynb` by rerunning the same cell again.

While each cell is designed to loop over all perturb seq filenames, their outputs are generally not synchronized, meaning that e.g. the calculated metrics are overwritten during each iteration of the for loop. Therefore, it is recommended to enter the list of filenames, and then subset them with the desired index: `filenames = [filenames[i]]`.

## Authors and acknowledgment
Code written by Ani, Nelli, Rebecca, and Georg
