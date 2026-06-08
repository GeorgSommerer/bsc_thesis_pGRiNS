import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import anndata as ad

from scipy.stats import gamma, expon, poisson, shapiro
from scipy import sparse
import statsmodels.api as sm
import importlib
import os

def set_mpl_attributes():
    importlib.reload(mpl)
    mpl.rcParams.update({
        'font.family'         : 'serif',       # Computer Modern — the default LaTeX font
        'font.size'           : 14,            # body text size (most journals use 10 pt)
        'axes.labelsize'      : 14,            # axis-label size matches body text
        'xtick.labelsize'     : 11,             # tick labels one point smaller
        'ytick.labelsize'     : 11,
        'legend.fontsize'     : 11,             # legend text one point smaller
        'figure.titlesize'    : 20,
        
        #'axes.prop_cycle'     : mpl.cycler('color', [   # Okabe–Ito colorblind-safe palette
        #    '#0072B2', '#D55E00', '#009E73',
        #    '#E69F00', '#CC79A7', '#56B4E9',
        #]),
        
        'lines.linewidth'     : 1.5,           # slightly thicker for print clarity
        'axes.linewidth'      : 0.8,           # thinner axis frame
        'xtick.direction'     : 'in',          # inward ticks — journal standard
        'ytick.direction'     : 'in',
        'xtick.minor.visible' : True,          # show minor ticks
        'ytick.minor.visible' : True,
        'xtick.major.size'    : 4,             # longer than the 3.5 default
        'ytick.major.size'    : 4,
        'xtick.minor.size'    : 2,             # half of major — proportional
        'ytick.minor.size'    : 2,
        'xtick.major.width'   : 0.8,           # match axes.linewidth
        'ytick.major.width'   : 0.8,
        'xtick.minor.width'   : 0.6,           # thinner for visual hierarchy
        'ytick.minor.width'   : 0.6,
        'lines.markersize'    : 4,             # smaller markers for print scale
        'errorbar.capsize'    : 3,             # visible end-caps (default is 0)
        'axes.xmargin'        : 0.02,          # hug the data (default is 0.05)
        'axes.ymargin'        : 0.02,
        'legend.frameon'      : False,         # no legend box
        'savefig.bbox'        : 'tight',       # tight bounding box by default
        'savefig.dpi'         : 300,           # publication-quality resolution
    })

    prop_cycle = [c["color"] for c in list(mpl.rcParams["axes.prop_cycle"])]

    return prop_cycle



def plot_pca_results(grn_file,grins_data_clustered,replicate):
    prop_cycle = set_mpl_attributes()

    fig,axs=plt.subplots(1,2,figsize=(15,7))
    max_pca = 50
    axs[1].scatter(range(max_pca),grins_data_clustered.uns["pca"]["variance_ratio"][:max_pca])
    axs[1].set_xlabel("Principal Component")
    axs[1].set_ylabel("Variance Ratio")
    axs[1].set_xticks(range(0,max_pca,5))
    #axs[1].set_ylim(0,0.5)
    #axs[1].set_yscale("log")
    axs[1].set_title("Variance Ratio of the PCA")
    sc.pl.pca(grins_data_clustered,color="n_genes_by_counts",layer="log1p",ax=axs[0],title="PCA colored by number of expressed genes per cell")
    os.makedirs(f"Prep_Data/Plots/{grn_file}/{replicate:03}",exist_ok=True)
    plt.savefig(f"Prep_Data/Plots/{grn_file}/{replicate:03}/pca.png")

    plt.show()



def plot_umap_results(grn_file,grins_data_clustered,replicate):
    prop_cycle = set_mpl_attributes()

    cluster_labels = grins_data_clustered.obs["leiden"]
    cluster_counts = cluster_labels.value_counts()
    clusters = np.array(list(cluster_counts.keys()))

    sample_silhouette_values = np.asarray(grins_data_clustered.obs["silhouette_samples"])
    best_clusters = grins_data_clustered.uns["clustering_results"]["best_clusters"]
    best_cluster_size = sum([cluster_counts[c] for c in best_clusters])
    n_clusters = len(best_clusters)
    mse_scores = grins_data_clustered.uns["clustering_results"]["scores"]

    # Modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    fig, ((ax3,ax2),(ax1, ax0)) = plt.subplots(2, 2)
    fig.set_size_inches(15,13)

    ax1.set_xlim([-0.1, 1])
    ymax = best_cluster_size + (n_clusters + 1) * 10
    ax1.set_ylim([0, ymax])

    y_lower = 10
    cluster_sil_means = []
    best_cluster_i = -1

    for i in (range(len(clusters))):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        if clusters[i] not in best_clusters:
            continue
        else:
            best_cluster_i += 1
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == clusters[i]]
        sample_silhouette_mean = np.mean(ith_cluster_silhouette_values)
        
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(best_cluster_i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        #ax1.axvline(x=sample_silhouette_mean,ymin=y_lower/ymax,ymax=y_upper/ymax,lw=1.2*lw,color="red", linestyle="dotted")

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.08, y_lower + 0.4 * size_cluster_i, str(clusters[i]))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for %d clusters"% n_clusters)
    ax1.set_xlabel("Silhouette score of each cell")
    ax1.set_ylabel("Cluster label")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    # Order clusters and MSEs according to the cluster size
    best_clusters_ordered = clusters[cluster_counts.keys().isin(best_clusters)]
    mse_scores_ordered = [mse_scores[np.where(best_clusters == best_clusters_ordered[i])[0][0]] for i in range(len(best_clusters_ordered))]

    y_pos = np.arange(len(best_clusters_ordered))
    hbars = ax0.bar(y_pos, mse_scores_ordered, align='center',color=[mpl.colors.to_hex(cm.nipy_spectral(float(i) / n_clusters)) for i in y_pos],alpha=0.7)
    ax0.set_xticks(y_pos, labels=best_clusters_ordered)
    ax0.invert_xaxis()
    ax0.set_xlabel('Cluster')
    ax0.set_ylabel('MSE')
    ax0.set_title('MSE between each cluster mean \n and the experimental control mean')

    ax2.scatter(grins_data_clustered.obsm["X_umap"].T[0][~cluster_labels.isin(best_clusters_ordered)],grins_data_clustered.obsm["X_umap"].T[1][~cluster_labels.isin(best_clusters_ordered)],s=0.1,c="grey")
    for i in range(len(best_clusters_ordered)):
        ax2.scatter(grins_data_clustered.obsm["X_umap"].T[0][cluster_labels==best_clusters_ordered[i]],grins_data_clustered.obsm["X_umap"].T[1][cluster_labels==best_clusters_ordered[i]],s=0.1,c = mpl.colors.to_hex(cm.nipy_spectral(float(i) / n_clusters)),label=best_clusters_ordered[i])    

    handles, labels = ax2.get_legend_handles_labels()
    lgnd = ax2.legend(handles[::-1], labels[::-1],frameon=True)
    for handle in lgnd.legend_handles:
        handle.set_sizes([50])
    ax2.set_title("UMAP with 10 best clusters colored")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")

    plt.suptitle(
        "Silhouette analysis for uGRiNS after Leiden clustering"
    )
    sc.pl.umap(grins_data_clustered,s=10,color="n_genes_by_counts",ax=ax3,title="UMAP colored by number of expressed genes per cell")
    plt.tight_layout()
    os.makedirs(f"Prep_Data/Plots/{grn_file}/{replicate:03}",exist_ok=True)
    plt.savefig(f"Prep_Data/Plots/{grn_file}/{replicate:03}/umap.png")
    plt.show()