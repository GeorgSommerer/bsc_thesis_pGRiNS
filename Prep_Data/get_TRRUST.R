# library(OmnipathR)
# library(tidyverse)
# trrust <- import_tf_target_interactions(
#   resources = "TRRUST",
#   organism = 9606  
# ) %>% filter(consensus_direction == TRUE)
# head(trrust)

library(dorothea)
library(decoupleR)
library(ggplot2)
library(dplyr)
library(OmnipathR)

load("/home/gesomme/Downloads/dorothea_hs.rda") # Taken from https://github.com/saezlab/dorothea/blob/master/data/dorothea_hs.rda
test <- dorothea_hs %>% filter(confidence!="D") # 454504 -> 438641 entries
dim(filter(test,mor<0))
# 

net <- get_dorothea(levels = c('A', 'B', 'C', 'D')) # Doesnt work
