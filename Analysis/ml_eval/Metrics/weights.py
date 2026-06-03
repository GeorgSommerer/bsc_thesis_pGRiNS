
#%%




import scanpy as sc 
import anndata as ad
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#Things to do: 




#%%

adata = sc.read_h5ad("/home/tilman-woehl/Dokumente/Software Praktikum/übungsaufgabe/AdamsonWeissman2016_GSM2406677_10X005.h5ad")
adata

#%%
#short normalization to test programm 

counts = adata.obs["perturbation"].value_counts()

valid_groups = counts[counts > 1].index

adata_filtered = adata[adata.obs["perturbation"].isin(valid_groups)].copy()
# %%
#this function computes the t score for every cell with respect to every other perturbed cell 
#gets the preprocessed Data as adata object 
#copys the data so the original is not changed 
#computes the t-score using the given function
#saves the result in adata.uns["rank_genes_groups"]
#returns None but the copied adata object was changed 
def tscore(data):
    #datacopy = data.copy()
    sc.tl.rank_genes_groups(
        data,
        groupby = 'perturbation',
        method = 't-test_overestim_var',
        reference = 'rest'
        )
    return None

#%%
#This Function puts the t scores in a data frame with the corresponding gene for later computations 
#gets data after tscore analysis 
#computes a data frame with gene name and tscore 
#is needed becouse rank_genes_groups saves result in form of {(names;t score; pvals);(names;t score;pvals); etc}
#cand work with that and need t score values seperated 
#returns Data frame with gene name; t-score; groupname (have to look up explicit data types)
def toDF(adata) : 
    result = adata.uns["rank_genes_groups"]
    #makes empty data frame 
    dfs = []
    
    #goes through adata.uns["rank_genes_groups"] by the name 
    for group in result["names"].dtype.names:
        #builds a new data frame for every name 
            df = pd.DataFrame({
                #saves name of the name 
                "genes": result["names"][group],
                #saves score that belongs to the name 
                "weight": result["scores"][group]
            })
            #saves group for trackeing where it came from
            df["group"] = group
            #makes a list of all made dataframes 
            dfs.append(df)
     #concludes all data frames from list in one dataframe        
    df_all = pd.concat(dfs)
    
    return df_all
#%%
#ths function performs the absolute value Normalization
#Computes the absolute value of every entry in the dataframe 
#gets data frame 
#computes the absolute value and saves it were the scores were safed 
#returns None but the t-score Column was changed 
def absolutevalue(dataframe):
    #should check if "rank_genes_groups" exists 
    
    dataframe["weight"] = np.abs(dataframe["weight"])
   
    
    return None 
#%%
#this function performs the minmx Normalization 
#gets adata object 
#computes minimum and maximum of the t score column 
#computes min max normalization
#returns None 
def minmax(dataframe): 
    #ließt minimum aus könnte sein das es nicth funktioniert muss überprüft werden 
    min = dataframe["weight"].min()
    max = dataframe["weight"].max()
    
    #changing the  score column to the min max normalization 
    #adding smallest number to prevent dividing by 0

    dataframe["weight"] = (dataframe["weight"] - min) / (max - min + 1e-9)
    
    return None 

#%%
#this function computes the square of every weight
#Data frame 
#computes square of every entry in score column 
#returns None 
def square(dataframe): 
    dataframe["weight"] = dataframe["weight"]**2
    return None

#%%
#this function performs a transformation where every entry is normalized so they all add up to one 
#gets dataframe 
#computes sum of all t score entries 
#divides by sum 
#returns None 
def addone(dataframe): 
    sum = dataframe["weight"].sum()
    
    dataframe["weight"] = dataframe["weight"]/sum
    
    return None 

#%%
#this function is the main function where every function to compute the weights is called 
#gets adata Object 
#computes weights along the steps 
#returns dataframe with gene name as "genes" weights as "weigth" and group name as "group"
def weight(data):
    try:
        tscore(data)
    except ValueError:
        print("Contains perturbation counts of 1 or lower")
    else: 
        df = toDF(data)
        absolutevalue(df)
        minmax(df)
        square(df)
        addone(df)
    
    
    return df