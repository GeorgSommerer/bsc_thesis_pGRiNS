# library(OmnipathR)
# library(tidyverse)
# trrust <- import_tf_target_interactions(
#   resources = "TRRUST",
#   organism = 9606  
# ) %>% filter(consensus_direction == TRUE)
# head(trrust)


# Dorothea:
library(dorothea)
library(decoupleR)
library(tidyverse)
library(OmnipathR)
library(magrittr)
load("../Data/dorothea_hs.rda") # Taken from https://github.com/saezlab/dorothea/blob/master/data/dorothea_hs.rda
dorothea_filtered <- dorothea_hs %>%
  mutate(mor = as.factor(mor)) %>% 
  rename(Source=tf,Target=target,Type=mor)
levels(dorothea_filtered$Type) <- c("2","1")

write_delim(dorothea_filtered %>% filter(confidence %in% c("A","B","C")) %>% select(!confidence),"../Data/Topos/dorothea_abc.topo",delim=" ")
write_delim(dorothea_filtered %>% filter(confidence %in% c("A","B","C","D")) %>% select(!confidence),"../Data/Topos/dorothea_abcd.topo",delim=" ")
write_delim(dorothea_filtered %>% select(!confidence),"../Data/Topos/dorothea_abcde.topo",delim=" ")


# KEGG:


kegg_pw <- kegg_pathways_download(max_expansion = NULL, simplify = TRUE)
kegg_pw %>% group_by(type) %>% summarise(no_rows = length(type))

print(kegg_pw  %>% group_by(effect) %>% summarise(no_rows = length(effect)), n=100)

# https://www.genome.jp/kegg/xml/docs/
# https://r.omnipathdb.org/reference/kegg_pathways_download.html
# https://omnipathdb.org/

# PPrel: PPIs
# ECrel: enzyme-enzyme successive catalysis (arrow contains numbers, maybe a reaction index?)
# GErel: gene expression
# PCrel: protein-compound


# Idea: Represent the following effects as activating (1): activation, binding/association, compound, expression
  # Represent the following effects as inhibiting (2): dissociation, inhibition, repression
effecttotype <- function(x){
  if (x %in% c("activation", "binding/association", "compound", "expression")){1}
  else{
    if (x %in% c("dissociation","inhibition","repression")){2}
    else{NA}
  }
} 

kegg_filtered <- kegg_pw %>% 
  mutate(Type = mapply(effecttotype,effect)) %>% 
  drop_na(Type) %>% 
  rename(Source=genesymbol_source,Target=genesymbol_target) %>% 
  select(c(Source,Target,Type)) %>% 
  unique()

write_delim(kegg_filtered,"../Data/Topos/kegg.topo",delim=" ")



# # WikiPathways:
# library(rWikiPathways)
# hs.pathways <- listPathways("Homo sapiens")
# wp.hs.gmt <- rWikiPathways::downloadPathwayArchive(format = "gmt")
# wp2gene <- readPathwayGMT("../Data/wikipathways-20260410-gmt-Homo_sapiens.gmt")
