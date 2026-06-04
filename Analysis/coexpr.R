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
get_adj_matrix <- function(ds_list,grn_file){
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
  png(file=paste("Plots/",grn_file,"/001/scale_independence_",ds_list$name,"_",ds_list$model,".png",sep=""),width=600, height=350)
  plot(sft$fitIndices[,1], sft$fitIndices[,2], xlab="Soft Threshold power)", ylab=paste("Scale Free Topology Model Fit,signed R^2"), type="n", main = paste("Scale independence for",ds_list$name,"and",ds_list$model));
  text(sft$fitIndices[,1], sft$fitIndices[,2], labels=powers,col="red");
  abline(h=0.85, col="red")
  dev.off()
  # Plot mean connectivity by power (should not be too low, otherwise clustering does not work well)
  png(file=paste("Plots/",grn_file,"/001/mean_connectivity_",ds_list$name,"_",ds_list$model,".png",sep=""),width=600, height=350)
  plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab="Soft Threshold (power)", ylab=paste("Mean Connectivity"), type="n", main = paste("Mean connectivity for",ds_list$name,"and",ds_list$model))
  text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers,col="red")
  dev.off()
  # Raise correlation matrix by that power to get weighted adjacency matrix
  beta <- sft$powerEstimate
  if (is.na(beta)){
    print("No powers resulting in R^2 above 0.85")
    beta <- max(sft$fitIndices["SFT.R.sq"])
  }
  adjMatrix <- abs(simMatrix)^beta
  
  ds_list$adjMatrix <- adjMatrix
  
  # Turn matrix into dataframe and filter out all edges with low weights
  tri_adj <- adjMatrix
  tri_adj[lower.tri(tri_adj,diag=TRUE)] <- NA
  network_df <- as_tibble(melt(tri_adj,na.rm=TRUE)) %>% rename(c(Var1="Node.1",Var2="Node.2",value="wTO"))# %>% filter(wTO>0.01)
  network_df$Node.1 <- as.character(network_df$Node.1)
  network_df$Node.2 <- as.character(network_df$Node.2)
  
  ds_list$network_df <- network_df
  
  return(ds_list)
}


# 2) Get clusters/modules and use topGO to find pathways associated with modules
get_modules <- function(ds_list,grn_file){
  # TOM is a measure in [0,1] between each pair of genes that is closer to 1 if the gene with fewer connections and most of its neighbors are all connected to the other gene
  TOM.w <- array(0, dim=c(0, ncol(ds_list$adjMatrix), ncol(ds_list$adjMatrix)))
  TOM.w <- TOMsimilarity(ds_list$adjMatrix)
  
  # Cluster TOM scores using UPGMA
  wTree <- hclust(as.dist(1-TOM.w), method = "average")
  w.clusters = cutreeDynamic(dendro = wTree, distM = 1-TOM.w, deepSplit = 2, cutHeight = 0.995, minClusterSize = 30, pamRespectsDendro = FALSE)
  w.colors = labels2colors(w.clusters)
  
  #plotDendroAndColors(wTree, w.colors, "Dynamic Tree Cut", dendroLabels = FALSE, main=paste("Cluster Dendrogram for",ds_list$name,"and",ds_list$model),hang = 0.03, addGuide = TRUE, guideHang = 0.05)
  
  # Calculate Eigengenes (1st principal component of each module as defined by w.colors) and merge modules with close Eigengenes
  ME.w<-moduleEigengenes(ds_list$pert_mean_matrix, color=w.colors)
  merged.cluster.w<-mergeCloseModules(ds_list$pert_mean_matrix, w.colors, MEs = ME.w$eigengenes)
  
  new.colors <- labels2colors(merged.cluster.w$colors)
  newME.w <- merged.cluster.w$newMEs
  png(file=paste("Plots/",grn_file,"/001/dendrogram_",ds_list$name,"_",ds_list$model,".png",sep=""),width=900, height=525)
  plotDendroAndColors(wTree, cbind(w.colors, new.colors), c("Unmerged", "Merged"), main=paste("Cluster Dendrogram for",ds_list$name,"and",ds_list$model), dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, guideHang = 0.05)
  dev.off()
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
  write.table(Genes2GO, paste("../Data/Experimental/",ds_list$name,"/coexpr_Genes2Go.txt",sep=""), quote=FALSE, row.names=FALSE, col.names=FALSE, se="\t")
}

GO_enrichment <- function(ds_list,module,ont){
  # Treat the genes in each module as "DEGs" (meaning that they have the factor level 1, and all other genes have 0)
  geneList <- factor(as.integer(ds_list$modules==module))
  names(geneList) <- ds_list$uniqGenesymbols
  
  # Read Gene2GO map:
  Genes2GOmap=readMappings(file = paste("../Data/Experimental/",ds_list$name,"/coexpr_Genes2Go.txt",sep=""))
  
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
get_conf_matrix <- function(exp_ds,pgrins_ds){
  GO_conf_matrix <- matrix(0L, nrow=length(unique(exp_ds$modules)),ncol=length(unique(pgrins_ds$modules)))
  dimnames(GO_conf_matrix) <- list(unique(exp_ds$modules),unique(pgrins_ds$modules))
  
  rownames(GO_conf_matrix) <- unique(exp_ds$modules)
  colnames(GO_conf_matrix) <- unique(pgrins_ds$modules)
  for (n_module in rownames(GO_conf_matrix)){
    for (p_module in colnames(GO_conf_matrix)){
      for (ont in onts){
        # Get the number of shared GO IDs between exp and pGRiNS for each ont type and each module
        GO_conf_matrix[[n_module,p_module]] <- GO_conf_matrix[[n_module,p_module]] + length(intersect(exp_ds$GO_results[[n_module]][[ont]]$GO.ID,pgrins_ds$GO_results[[n_module]][[ont]]$GO.ID))
      }
    }
  }
  return(GO_conf_matrix)
}

# How to measure goodness of confusion matrix?


# 3) On the adjacency matrix: use CoDiNA to compare networks
get_diff_net <- function(exp_ds,grins_ds, grn_ds, grn_file){
  diff_net <- MakeDiffNet(Data = list(exp_ds$network_df, grins_ds$network_df,grn_ds$network_df),Code = c("experimental", "pGRINS","GRN"))
  # Phi=="a" are interesting, because alpha edges are categorized as belonging to both networks
  # beta edges have different signs, and gamma edges belong to only 1 network
  diff_net <- subset(diff_net, diff_net$Score_Phi_tilde/diff_net$Score_internal > 1)
  DiffNodes = ClusterNodes(diff_net, cutoff.external = 0, cutoff.internal = 1) # Maybe need to clean by Score_Phi_tilde/Score_internal>1?
  
  # Plot and save distribution of nodes across categories
  Count <- table(DiffNodes$Phi_tilde)
  res_df <- tibble(Count,"Group"=names(as.list(Count)))
  write_delim(res_df,paste("diffnet_",exp_ds$name,".csv",sep=""),delim=" ")
  ggplot(res_df,aes(x=Group,y=Count))+geom_bar(stat="identity")+ggtitle(paste("Number of genes associated with each dataset after filtering in",exp_ds$name))
  ggsave(paste("Plots/",grn_file,"/001/codina_",exp_ds$name,".png",sep=""))
  
  return(DiffNodes)
}


######################################################################
# Main:
graphics.off()
grn_file = "KeggoRo_0206"
names = c("Norman19","Replogle22")
models = c("experimental","pGRiNS")
if (!exists("dataset_lists")){
  dataset_lists = list()
}
for (name in names){
  print(paste("********************",name,"********************"))
  for (model in models){
    print(paste("********************",model,"********************"))
    dataset_lists[[name]][[model]] <- list(name=name,model=model)
    
    dataset_lists[[name]][[model]] <- get_adj_matrix(dataset_lists[[name]][[model]],grn_file)
    
    dataset_lists[[name]][[model]] <- get_modules(dataset_lists[[name]][[model]],grn_file)
  }
  if (!file.exists(paste("../Data/Experimental/",name,"/coexpr_Genes2Go.txt",sep=""))){ # Get Genes2GO list if not already saved
    mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl",verbose=TRUE))
    write_GO_IDs(dataset_lists[[name]][["experimental"]],mart)
  }
  for (model in models){ 
    dataset_lists[[name]][[model]][["GO_results"]] = list()
    for (module in unique(dataset_lists[[name]][[model]]$modules)){
      dataset_lists[[name]][[model]][["GO_results"]][[module]] = list()
      onts <- c("MF","BP","CC")
      for (ont in onts){
        print(paste("********************",model,module,ont,"********************"))
        dataset_lists[[name]][[model]] <- GO_enrichment(dataset_lists[[name]][[model]],module,ont)
      }
    }
  }
  
  #dataset_lists[[name]][["GO_conf_matrix"]] <- get_conf_matrix(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]])
  
  #get_diff_net(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]])
}


nonsink_df = read_delim(paste("../Data/Projects/",grn_file,"/",grn_file,".topo",sep=""), delim=" ")
sink_df = read_delim(paste("../Data/Projects/",grn_file,"/",grn_file,"_sinks.topo",sep=""), delim=" ")
grn <- bind_rows(nonsink_df,sink_df) %>% filter("Source","Target") %>% mutate("wTO"=1)
for (name in names){
  dataset_lists[[name]][["GRN"]] = list()
  dataset_lists[[name]][["GRN"]][["network_df"]] = grn #filter
  dataset_lists[[name]][["GRN"]][["adjMatrix"]] = grn # turn into adj matrix
  
  dataset_lists[[name]][["GRN"]] <- get_modules(dataset_lists[[name]][["GRN"]])
  
  dataset_lists[[name]][["GRN"]][["GO_results"]] = list()
  for (module in unique(dataset_lists[[name]][["GRN"]]$modules)){
    dataset_lists[[name]][["GRN"]][["GO_results"]][[module]] = list()
    onts <- c("MF","BP","CC")
    for (ont in onts){
      print(paste("********************","GRN",module,ont,"********************"))
      dataset_lists[[name]][["GRN"]] <- GO_enrichment(dataset_lists[[name]][["GRN"]],module,ont)
    }
  }
  
  
  # get_modules -> topGO analysis
}


for (name in names){
  dataset_lists[[name]][["GO_conf_matrix"]] <- get_conf_matrix(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]])
  print(dataset_lists[[name]][["GO_conf_matrix"]])
}

for (name in names){
  # DiffNet on all 3 networks at once???
  dataset_lists[[name]][["DiffNodes_pGRiNS"]] <- get_diff_net(dataset_lists[[name]][["experimental"]]$network_df,dataset_lists[[name]][["pGRiNS"]]$network_df,grn_file)
  
  dataset_lists[[name]][["DiffNodes_GRN"]] <- get_diff_net(dataset_lists[[name]][["experimental"]],dataset_lists[[name]][["pGRiNS"]],dataset_lists[[name]][["GRN"]],grn_file)
}



load("./.RData")
