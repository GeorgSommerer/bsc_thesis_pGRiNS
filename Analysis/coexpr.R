library(tidyverse)
# 1+2)
library(WGCNA)
# 3)
library(biomaRt)
library(topGO)
# 4)
library(CoDiNA)
library(reshape2)
setwd("~/Code/bsc_thesis_pGRiNS/Analysis")

# 1) Get adjacency matrix:
get_adj_matrix <- function(ds_list){
  pert_mean_matrix <- read_delim(paste("../Data/Experimental/",ds_list$name,"/",ds_list$name,"-",ds_list$model,"_pert_mean.csv",sep=""),delim=" ")
  
  ds_list$pert_mean_matrix <- pert_mean_matrix
  ds_list$uniqGenesymbols <- colnames(pert_mean_matrix)
  
  # Pearson correlation of genes:
  simMatrix <- cor(pert_mean_matrix)
  
  # For every power, fit the connectivity of the resulting graph to the log power law distribution
  # Higher R^2 means that degree distribution follows power law better, which is more realistic
  # Get the lowest power with R^2>0.85
  powers = c(c(1:10), seq(from = 12, to=30, by=2))
  sft = pickSoftThreshold(pert_mean_matrix, powerVector = powers, verbose = 5)
  
  # Plot R^2 by power
  plot(sft$fitIndices[,1], sft$fitIndices[,2], xlab="Soft Threshold power)", ylab=paste("Scale Free Topology Model Fit,signed R^2"), type="n", main = paste("Scale independence for",ds_list$name));
  text(sft$fitIndices[,1], sft$fitIndices[,2], labels=powers,col="red");
  abline(h=0.85, col="red")
  
  # Plot mean connectivity by power (should not be too low, otherwise clustering does not work well)
  plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab="Soft Threshold (power)", ylab=paste("Mean Connectivity"), type="n", main = paste("Mean connectivity for",ds_list$name))
  text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers,col="red")
  
  # Raise correlation matrix by that power to get weighted adjacency matrix
  beta <- sft$powerEstimate
  adjMatrix <- abs(simMatrix)^beta
  
  ds_list$adjMatrix <- adjMatrix
  
  # Turn matrix into dataframe and filter out all edges with low weights
  tri_adj <- adjMatrix
  tri_adj[lower.tri(tri_adj,diag=TRUE)] <- NA
  network_df <- as_tibble(melt(tri_adj,na.rm=TRUE)) %>% rename(c(Var1="Node.1",Var2="Node.2",value="wTO")) %>% filter(wTO>0.01)
  network_df$Node.1 <- as.character(network_df$Node.1)
  network_df$Node.2 <- as.character(network_df$Node.2)
  
  ds_list$network_df <- network_df
  
  return(ds_list)
}


# 2) Get clusters/modules and use topGO to find pathways associated with modules
get_modules <- function(ds_list){
  # TOM is a measure in [0,1] between each pair of genes that is closer to 1 if the gene with fewer connections and most of its neighbors are all connected to the other gene
  TOM.w <- array(0, dim=c(0, ncol(ds_list$adjMatrix), ncol(ds_list$adjMatrix)))
  TOM.w <- TOMsimilarity(ds_list$adjMatrix)
  
  # Cluster TOM scores using UPGMA
  wTree <- hclust(as.dist(1-TOM.w), method = "average")
  w.clusters = cutreeDynamic(dendro = wTree, distM = 1-TOM.w, deepSplit = 2, cutHeight = 0.995, minClusterSize = 30, pamRespectsDendro = FALSE)
  w.colors = labels2colors(w.clusters)
  
  plotDendroAndColors(wTree, w.colors, "Dynamic Tree Cut", dendroLabels = FALSE, main=paste("Cluster Dendrogram for",ds_list$name),hang = 0.03, addGuide = TRUE, guideHang = 0.05)
  
  # Calculate Eigengenes (1st principal component of each module as defined by w.colors) and merge modules with close Eigengenes
  ME.w<-moduleEigengenes(ds_list$pert_mean_matrix, color=w.colors, softPower=6)
  merged.cluster.w<-mergeCloseModules(ds_list$pert_mean_matrix, w.colors, MEs = ME.w$eigengenes)
  
  new.colors <- labels2colors(merged.cluster.w$colors)
  newME.w <- merged.cluster.w$newMEs
  plotDendroAndColors(wTree, cbind(w.colors, new.colors), c("Unmerged", "Merged"), main=paste("Cluster Dendrogram for",ds_list$name), dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, guideHang = 0.05)
  
  ds_list$modules <- new.colors
  return(ds_list)
}

write_GO_IDs <- function(ds_list,mart){
  # Get GO IDs from HGNC symbols:
  
  GeneGONames = getBM(filters= "hgnc_symbol", attributes= c("hgnc_symbol", "go_id"),values=ds_list$uniqGenesymbols, mart=mart)
  
  # Reformat GO IDs:
  Genes2GO = matrix(,length(ds_list$uniqGenesymbols),2)
  for (i in 1:length(ds_list$uniqGenesymbols))
  {
    temp=GeneGONames[GeneGONames[,1]==ds_list$uniqGenesymbols[i],1:2]
    tempGOs = paste(temp[,2], collapse=",")
    Genes2GO[i,1]=temp[1,1]
    Genes2GO[i,2]=tempGOs
  }
  write.table(Genes2GO, paste("../Data/Experimental/",ds_list$name,"/Genes2Go.txt",sep=""), quote=FALSE, row.names=FALSE, col.names=FALSE, se="\t")
}

GO_enrichment <- function(ds_list,module,ont){
  # Treat the genes in each module as "DEGs" (meaning that they have the factor level 1, and all other genes have 0)
  geneList <- factor(as.integer(ds_list$modules==module))
  names(geneList) <- ds_list$uniqGenesymbols
  
  # Read Gene2GO map:
  Genes2GOmap=readMappings(file = paste("../Data/Experimental/",ds_list$name,"/Genes2Go.txt",sep=""))
  
  # Build the topGO object:
  GOdata = new("topGOdata", ontology = ont, allGenes = geneList, annot = annFUN.gene2GO, gene2GO = Genes2GOmap)
  
  # Run Fisher's exact test:
  resultFisher = runTest(GOdata, algorithm = "classic", statistic = "fisher")
  sigterms = resultFisher@geneData["SigTerms"]
  sigGOIDs = GenTable(GOdata, classicFisher = resultFisher, topNodes = sigterms)
  
  # Apply Benjamini-Hochberg correction:
  qval = p.adjust(sigGOIDs$classicFisher, met='BH')
  sigGOIDscorrected = cbind(sigGOIDs, qval) %>% filter(qval<0.05)
  
  ds_list[["GO_results"]][[module]][[ont]] <- as_tibble(sigGOIDscorrected)
  return(ds_list)
}

# Calculate confusion matrix between experimental and pGRiNS
get_conf_matrix <- function(exp_ds,grins_ds){
  GO_conf_matrix <- matrix(0L, nrow=length(unique(exp_ds$modules)),ncol=unique(pgrins_ds$modules))
  dimnames(GO_conf_matrix) <- list(unique(exp_ds$modules),unique(pgrins_ds$modules))
  
  for (n_module in rownames(GO_conf_matrix)){
    for (p_module in colnames(GO_conf_matrix)){
      for (ont in onts){
        # Get the number of shared GO IDs between exp and pGRiNS for each ont type and each module
        GO_conf_matrix[n_module,p_module] <- GO_conf_matrix[n_module,p_module] + length(intersect(exp_ds$GO_results[[n_module]][[ont]]$GO.ID,pgrins_ds$GO_results[[n_module]][[ont]]$GO.ID))
      }
    }
  }
  heatmap(GO_conf_matrix)
}

# How to measure goodness of confusion matrix?


# 3) On the adjacency matrix: use CoDiNA to compare networks
get_diff_net <- function(exp_ds,grins_ds){
  diff_net <- MakeDiffNet(Data = list(exp_ds$network_df, grins_ds$network_df),Code = c("experimental", "pGRINS"))
  # Phi=="a" are interesting, because alpha edges are categorized as belonging to both networks
  # beta edges have different signs, and gamma edges belong to only 1 network
  DiffNodes = ClusterNodes(diff_net, cutoff.external = 0, cutoff.internal = 1) # Maybe need to clean by Score_Phi_tilde/Score_internal>1?
  
  barplot(table(DiffNodes$Phi_tilde))
  
  common_genes = subset(DiffNodes$Node, DiffNodes$Phi_tilde == 'a')
  exp_genes = subset(DiffNodes$Node, DiffNodes$Phi_tilde == 'g.experimental')
  pgrins_genes = subset(DiffNodes$Node, DiffNodes$Phi_tilde == 'g.pGRINS')
  print(paste("Common genes:",common_genes))
  print(paste("Genes in experimental:",common_genes))
  print(paste("Genes in pGRiNS:",common_genes))
}


######################################################################
# Main:
names = c("Norman19","Replogle22")
models = c("experimental","pGRiNS")

mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
onts <- c("MF","BP","CC")

dataset_lists = list()
for (name in names){
  for (model in models){
    dataset_lists[[name]][[model]] <- list(name=name,model=model)
    
    dataset_lists[[name]][[model]] <- get_adj_matrix(dataset_lists[[name]][[model]])
    
    dataset_lists[[name]][[model]] <- get_modules(dataset_lists[[name]][[model]])
    
    write_GO_IDs(dataset_lists[[name]][[model]],mart)
    
    dataset_lists[[name]][[model]][["GO_results"]] = list()
    for (module in unique(dataset_lists[[name]][[model]]$modules)){
      dataset_lists[[name]][[model]][["GO_results"]][[module]] = list()
      for (ont in onts){
        dataset_lists[[name]][[model]] <- GO_enrichment(dataset_lists[[name]][[model]],module,ont)
      }
    }
  }
  
  get_conf_matrix(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]])
  
  get_diff_net(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]])
}
