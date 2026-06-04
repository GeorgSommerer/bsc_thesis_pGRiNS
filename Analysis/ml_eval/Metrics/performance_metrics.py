# METRICS

# Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def get_weights(norm_data : ad.AnnData, mu_gt_dict : dict[str,list[float]], gt_index_dict : dict[str, list[str]]) -> dict[str,list[float]]:
    """
    Calculates the weights as proposed by Meija et al.
    First, the t-scores for each gene among the ground truth genes are calculated.
    Then, an absolute value transformation and a min-max transformation to [0,1] are applied.
    These weights are then squared and normalized to add up to 1.

    Parameters:
    -----------
    norm_data : ad.AnnData
        The normalized dataset.
    mu_gt_dict : dict[str,list[float]]
        A dictionary containing the test perturbations of the split as keys and the mean expressions of the GT cells of these perturbations as values.
    gt_index_dict : dict[str,list[str]]
        For each perturbed genes as the key, the values are a list of the cells of this perturbation assigned to the ground truth.

    Returns:
    --------
    weights_dict : dict[str,list[float]]
        Has the same shape as mu_gt_dict, but instead of expression values, each gene is assigned a weight for each perturbation.
    """

    gt_cells = np.concatenate(list(gt_index_dict.values())).tolist()
    
    this_norm_data = norm_data[gt_cells]

    sc.tl.rank_genes_groups(
        this_norm_data,
        groupby = 'perturbation',
        method = 't-test_overestim_var',
        reference = 'rest',
        key_added="rgg_Mejia",
        layer="log1p")

    
    weights_dict = {}
    for pert in gt_index_dict.keys():
        # norm_data.uns["rank_genes_groups"] contains the p values for the DEGs of each perturbation
        pert_weights = this_norm_data.uns["rgg_Mejia"]["scores"][pert]
        pert_weights = np.abs(pert_weights) # Absolute value transformation
        pert_weights = (pert_weights-min(pert_weights))/(max(pert_weights)-min(pert_weights)+1e-8) # Min-max transformation to [0,1]
        pert_weights = pert_weights**2 # Squaring
        pert_weights = pert_weights/sum(pert_weights) # Normalization
        pert_weights = pert_weights[np.argsort(this_norm_data.uns["rgg_Mejia"]["names"][pert])] # Change the order so that the corresponding genes are in alphabetical order

        weights_dict[pert] = pert_weights
    
    return weights_dict

def get_metrics(weights : list[float] = None) -> dict:
    """
    Sets the list of metrics to be calculated.

    Parameters:
    weights : list[float], optional
        If none provided, calculate unweighted metrics, else use the weights.
    
    Returns
    -------
    metrics : dict
    A dictionary of the metrics (functions, mostly lambda functions) as values with the names as the keys.
    Each of them has to have exactly the same arguments, even if they are not used in the particular function.
    """

    if weights is None:
        metrics = {
        "MSE":lambda avg_expr_GT_pert, avg_expr_pred, mean_unperturbed : mean_squared_error(avg_expr_GT_pert, avg_expr_pred),
        "Pearson":lambda avg_expr_GT_pert, avg_expr_pred, mean_unperturbed : pearsonr(x=avg_expr_GT_pert, y=avg_expr_pred).correlation,
        "Pearson Delta":lambda avg_expr_GT_pert,avg_expr_pred, mean_unperturbed : 0.0 if (avg_expr_pred == mean_unperturbed).all() else pearsonr(x=avg_expr_GT_pert-mean_unperturbed, y=avg_expr_pred-mean_unperturbed).correlation

        }
    
    else:
        metrics = {
        "WMSE":lambda avg_expr_GT_pert, avg_expr_pred, mean_unperturbed, weights: np.dot(weights,(avg_expr_GT_pert-avg_expr_pred)**2),
        "Weighted Delta R2":lambda avg_expr_GT_pert, avg_expr_pred, mean_unperturbed, weights: 1-(np.dot(weights,(avg_expr_GT_pert-avg_expr_pred)**2))/(np.dot(weights,(avg_expr_GT_pert-mean_unperturbed-np.dot(weights,(avg_expr_GT_pert-mean_unperturbed)))**2)) # 0.0 if (avg_expr_pred == mean_unperturbed).all() else 
        }
    
    return metrics


def compute_metrics_all(mu_gt_dict : dict[str, np.ndarray[np.float64]], 
                    model_list : list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]], 
                    control_baseline : np.ndarray[np.float64],
                    metric_names : list = None,
                    deg_dict : dict[str, list] = None,
                    weight_dict : dict[str,list[float]] = None) -> np.matrix:
    """
    For all models in model_list, compute all metrics (the number of which is equivalent to no_metrics) across all perturbations.
    
    Parameters
    ----------
    mu_gt_dict: dict[str, np.ndarray[np.float64]]
        A dictionary containing the perturbation names as keys and their average expression vectors as values
    model_list: list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]]
        A list containing all models (including the baselines) for which the metrics are to be computed.
        These can be simple numpy arrays, or dicts with the same shape as mu_gt_dict.
    control_baseline: np.ndarray
        The baseline of the control cells. Used for metrics such as Pearson's delta.
    metric_names: list
        A list of the names of the metrics to be computed.
    weight_dict : dict[str,list[float]], optional
        If weighted metrics need to be calculated, a dictionary containing the test perturbations as keys and the weights of the genes as values must be supplied.   
    
    Returns
    -------
    metric_matrix : np.matrix
        A matrix of shape no_models x no_metrics x test_perts containing the metrics calculated for all perturbations for each model.
        The order of models is equivalent to the order in model_list.
        Each model has a corresponding 2D array. 
        The order of metrics (rows of 2D array) is equivalent to the order in metric_names.
        The order of test_pert (columns of 2D array) is equivalent to the order in mu_gt_dict
    """
    if metric_names is None:
        if weight_dict is None:
            metric_names = ["MSE","Pearson","Pearson Delta"]
        else:
            metric_names = ["WMSE", "Weighted Delta R2"]

    no_models = len(model_list)
    no_metrics = len(metric_names)
    no_perts = len(mu_gt_dict)
    perts = list(mu_gt_dict.keys())

    metric_matrix = np.zeros((no_models, no_metrics, no_perts))
    
    for i_pert, pert in enumerate(perts):
        for i_model in range(no_models):
            model = model_list[i_model]
            if isinstance(model_list[i_model], np.ndarray):
                pred = model
                #metric_matrix[i_model] += compute_metrics_pert(mu_gt_dict[pert], model_list[i_model], control_baseline, metric_names)
            elif isinstance(model_list[i_model], dict):
                # If the models prediction is different for each perturbation (i.E. a dict), subset the model with [pert]
                pred = model[pert]
            
            if deg_dict is not None and pert in deg_dict:
                deg_idx = deg_dict[pert]

                gt_input = mu_gt_dict[pert][deg_idx]
                pred_input = pred[deg_idx]
                control_input = control_baseline[deg_idx]
            else:
                gt_input = mu_gt_dict[pert]
                pred_input = pred
                control_input = control_baseline

            # Determine whether or not weights should be used:
            if weight_dict is None:
                weights = None
            else:
                weights = weight_dict[pert]
                    
            metric_matrix[i_model, :, i_pert] = compute_metrics_pert(gt_input, pred_input, control_input, metric_names,weights=weights)
                #metric_matrix[i_model] += compute_metrics_pert(mu_gt_dict[pert], model_list[i_model][pert], control_baseline, metric_names)

    return metric_matrix



def compute_metrics_pert(avg_expr_GT_pert : np.ndarray, 
                    avg_expr_pred : np.ndarray, 
                    mean_unperturbed : np.ndarray,
                    metric_names : list,
                    weights : list = None) -> np.ndarray:
    """
    Computes all Metrics (Pearson, Pearson-Delta, MSE) for a given perturbation.
    
    Parameters
    ----------
    avg_expr_GT: np.ndarray
        A vector containing the average expression values of the ground truth.
    avg_expr_pred: np.ndarray
        A vector containing the average predicted gene expression values for a given model.
    mean_unperturbed: np.ndarray
        A vector of average gene expression values for all control cells.
    metric_names: list
        A list of metrics to be computed.
    weights : list, optional
        A list of weights for the current perturbation.
    
    Returns
    -------
    calc_metrics : np.ndarray
        A vector containing all the metrics calculated for a single model and perturbation.
    """
    
    if (avg_expr_GT_pert.shape != mean_unperturbed.shape) or (avg_expr_pred.shape != mean_unperturbed.shape):
        raise ValueError(f"The shapes of the input vectors do not match ({avg_expr_GT_pert.shape} != {mean_unperturbed.shape}) or ({avg_expr_pred.shape} != {mean_unperturbed.shape}).")
    
    metric_dict = get_metrics(weights)

    if weights is None:
        calc_metrics = np.asarray([metric_dict[metric](avg_expr_GT_pert,avg_expr_pred,mean_unperturbed) for metric in metric_names])
    else:
        calc_metrics = np.asarray([metric_dict[metric](avg_expr_GT_pert,avg_expr_pred,mean_unperturbed,weights) for metric in metric_names])
    
    return calc_metrics


def compute_deg_dict(adata: ad.AnnData, perts: list, n_top: int = 20) -> dict:
    deg_dict = {}

    for pert in perts:
        # Run DEG analysis
        sc.tl.rank_genes_groups(
            adata,
            groupby="perturbation",
            groups=[pert],
            reference="ctrl",   # ref group to compare against
            method="t-test_overestim_var"
        )

        # Extract results
        df = sc.get.rank_genes_groups_df(adata, group=pert)

        # Filter
        df_filtered = df[
            (df["pvals_adj"] < 0.05) &
            (df["logfoldchanges"].notna()) &
            (abs(df["logfoldchanges"]) > 1)   #biological relevance threshold
        ]

        # Take top 20
        top_genes = df_filtered["names"].head(n_top).tolist()

        # Convert gene names -> indices
        gene_indices = [adata.var_names.get_loc(g) for g in top_genes]
        if len(gene_indices)==n_top:
            deg_dict[pert] = gene_indices

    return deg_dict  #key: perturbation name, value: list of indices of top DE genes for this perturbation

def compute_metrics_vs_degs(mu_gt_dict: dict, model_list: list, control_baseline: np.ndarray, 
                            adata: ad.AnnData, perts: list, n_top_list: list = [5,10,20,50,100]):
    """
    Computes Pearson Delta for different numbers of top DE genes.
    
    Returns:
        dict: {n_degs: metric_matrix} where metric_matrix shape = (n_models, n_metrics, n_perts)
    """
    results = {}
    
    for n_top in n_top_list:
        print(f"  Computing for top {n_top} DE genes...")
        deg_dict = compute_deg_dict(adata, perts, n_top=n_top)
        metric_matrix = compute_metrics_all(
            mu_gt_dict=mu_gt_dict,
            model_list=model_list,
            control_baseline=control_baseline,
            metric_names=["Pearson Delta"],
            deg_dict=deg_dict
        )
        
        results[n_top] = metric_matrix
    
    return results

# Plotting metric results
def plot_avg_pearson_delta_vs_degs(results_dict: dict, model_names: list):
    """
    Plot average Pearson Delta vs. number of DE genes for multiple models.
    Includes both line plot (averages) and boxplots (distributions across perturbations).
    
    Parameters
    ----------
    results_dict : dict
        Keys: n_top values, Values: array of shape (n_models, n_metrics, n_perts)
    model_names : list
        Names of models/baselines (e.g., ["Control Baseline", "Mean Baseline", "CPA Model"])
    """
    results_dict_0 = results_dict[0] # Results dict from CV 0, remove if code should be modified for multiple CV iterations

    deg_counts = sorted(results_dict[0].keys())
    n_models = len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create figure with two subplots side by side
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Left plot: Line plot with averages
    for i_model in range(n_models):
        avg_values = []
        for n_top in deg_counts:
            # Average across perturbations (axis=2) and take Pearson Delta (index 0)
            model_avg = np.mean([np.nanmean(results_dict[i][n_top][i_model, 0, :]) for i in range(len(results_dict))]) # Modified to mean over CV iterations as well
            avg_values.append(model_avg)
        
        ax1.plot(deg_counts, avg_values, marker='o', linewidth=2, markersize=8, 
                 label=model_names[i_model], color=colors[i_model])
    
    ax1.set_xlabel("Number of Top DE Genes", fontsize=14)
    ax1.set_ylabel("Average Pearson Delta Correlation", fontsize=14)
    ax1.set_title("Average Pearson Delta vs. DE Genes (Norman19)", fontsize=16)
    ax1.legend(loc="upper right",bbox_to_anchor=(1.45, 0.65))
    ax1.grid(True, alpha=0.3)


            # Add mean and sd error bar
    """
    # Right plot: Boxplots showing distribution across perturbations
    # Prepare data for boxplot: list of lists, each containing values for one n_top
    box_data = []
    box_labels = []
    box_colors = []
    
    for n_top in deg_counts:
        for i_model in range(n_models):
            # Get all perturbation values for this model at this n_top
            values = results_dict_0[n_top][i_model, 0, :]
            box_data.append(values)
            box_labels.append(f"{n_top}\n{model_names[i_model]}")
            box_colors.append(colors[i_model])
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel("Number of Top DE Genes / Model", fontsize=12)
    ax2.set_ylabel("Pearson Delta Correlation", fontsize=12)
    ax2.set_title("Distribution of Pearson Delta vs. DE Genes", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    """
    plt.show()

def data_for_plot(fourD_metric_matrix: np.ndarray, 
                 i_model : int, 
                 i_metric : int,
                 block_num : int,
                 mode: str
                 ) -> tuple[np.ndarray, str]: 
    """
    Helping function for plot_metrics
    Gets data that is used for plotting (in plot_metrics) for a specified model, metric and a selected mode

    Parameters
    ----------
    fourD_metric_matrix: np.ndarray
        A 4D vector containing all the computed metrics in the following dimentions: iteration x model x metric x test_pert
    i_model: int
        Index of the model
    i_metric: int   
        Index of the metric 
    mode: str
        The mode of the plot, wich is set as "per_cv_block" as default
        Can also be set to either "all" for overall results or "iters" for distribution over iterations
    
    Returns
    -------
    data: np.ndarray
        A vector containing the metric computations for specified mode
    title: str
        The title of the plot
    """

    if mode == "all":
        # Mean over iterations: (models, metrics, test_perts)
        #data = fourD_metric_matrix.mean(axis=0)[i_model, i_metric, :]
        #data = np.nanmean(fourD_metric_matrix, axis=0)[i_model, i_metric, :]
        #data = data[~np.isnan(data)]
        
        data = np.concat([fourD_metric_matrix[i,i_model,i_metric,:] for i in range(fourD_metric_matrix.shape[0])],axis=0)
        data = data[~np.isnan(data)]
        title = "Results"

    elif mode == "iters":
        # Mean over test perts: (iterations, models, metrics)
        #data = fourD_metric_matrix.mean(axis=3)[:, i_model, i_metric]
        data = np.nanmean(fourD_metric_matrix, axis=3)[:, i_model, i_metric]
        data = data[~np.isnan(data)]

        title = "Average Test Perturbation Score for each CV"

    else: # If no modes are chosen data is a cv block
        data = fourD_metric_matrix[i_model, i_metric, :]
        data = data[~np.isnan(data)]

        title = f"CV block {block_num}"

    return data, title

# Generates plots for each cv block 
def plot_cv_block(filename : str,
                  fourD_metric_matrix: np.ndarray, 
                  model_list : list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]], 
                  metric_names : list = ["MSE","Pearson","Pearson Delta"]
                  ):
    """
    Generates scatterplots for each metric for every single cross validation block. Each plot is visually divided
    into the data of the three models.

    Parameters
    ----------
    filename: str
        The name of the current dataset.
    fourD_metric_matrix: np.ndarray
        A 4D vector containing all the computed metrics in the following dimentions: iteration x model x metric x test_pert
    model_list: list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]]
        A list containing all models (including the baselines) for which the metrics are to be computed.
        These can be simple numpy arrays, or dicts with the same shape as mu_gt_dict.
    metric_names: list
        A list of metrics to be computed.
    
    Returns
    -------
    None
        This function creates plots
    """
    
    no_blocks = len(fourD_metric_matrix)

    for i_block in range(no_blocks):
        plot_metrics(filename, fourD_metric_matrix[i_block], model_list, "per_cv_block", i_block, False, metric_names)

def plot_metrics(filename : str,
                 fourD_metric_matrix: np.ndarray, 
                 model_list : list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]], 
                 mode: str,
                 num_block: int,
                 add_hue: bool,
                 metric_names : list = ["MSE","Pearson","Pearson Delta"]
                 ): 
    """
    Generates scatterplots for each metric for a specified mode (data distribution). Each plot is visually divided
    into the data of the three models.

    Parameters
    ----------
    filename: str
        The name of the current dataset.
    fourD_metric_matrix: np.ndarray
        A 4D vector containing all the computed metrics in the following dimentions: iteration x model x metric x test_pert
    model_list: list[np.ndarray[np.float64] | dict[str, np.ndarray[np.float64]]]
        A list containing all models (including the baselines) for which the metrics are to be computed.
        These can be simple numpy arrays, or dicts with the same shape as mu_gt_dict.
    metric_names: list
        A list of metrics to be computed.
    mode: str
        The mode of the plot (can be either "all" for overall results over test_perts or "iters" for distribution over iterations)
    
    Returns
    -------
    None
        This function creates plots
    """

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5*len(metric_names), 6))

    _, plottitle = data_for_plot(fourD_metric_matrix, 0, 0, num_block, mode)
    fig.suptitle(f"{plottitle} (Norman19)",fontsize=20)

    no_models = len(model_list)

    for i_metric, metric in enumerate(metric_names):
        ax = axes[i_metric]

        for i_model in range(no_models):
            # Get data
            values = data_for_plot(fourD_metric_matrix, i_model, i_metric, num_block, mode)[0].flatten()
            x = np.full_like(values, fill_value=i_model, dtype=float)

            # Swarmplots (optional for "perts" and "iters": add hue)
            if add_hue:
                hue = np.arange(values.shape[0]) 
                n = values.shape[0]   
                hue_labels = [str(i) for i in range(n)]
             
                # Create swarmplot
                sns.set_theme(style="whitegrid")
                sns.stripplot(x=x, y=values, ax=ax, hue=hue, size=6,legend=False)

                # Add legend manually
                palette = sns.cubehelix_palette(n_colors=n, as_cmap=False)

                handles = [
                    plt.Line2D([0], [0], marker='o', color=palette[i], linestyle='', markersize=8)
                    for i in range(n)
                ]


            else:
                # Create swarmplot
                sns.set_theme(style="whitegrid")
                sns.stripplot(x=x, y=values, ax=ax, size=6)

            # Add mean and sd error bar
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)

            ax.errorbar(x=np.unique(x), y=[mean_val], yerr=[std_val], fmt="o", color="black", capsize=5, markersize=5, zorder=10)
            
        ax.set_title(metric,fontsize=18)

        #ax.set_xlabel("Model",fontsize=20)
        ax.set_xticks(range(len(model_list)))
        ax.set_xticklabels(model_list,ha="right",fontsize=14)
        plt.yticks(fontsize=14)

        #ax.set_ylabel("Metric Value")
        ax.tick_params(axis='x', labelrotation=30)
    if add_hue:
        axes[-1].legend(handles, hue_labels, title="Hue")
    plt.tight_layout()
    plt.savefig(f"Results/Figures/{filename}_metrics_{mode}_{num_block}_{"-".join(metric_names)}")
    plt.show()
