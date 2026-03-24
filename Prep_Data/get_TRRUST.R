library(OmnipathR)
library(tidyverse)
trrust <- import_tf_target_interactions(
  resources = "TRRUST",
  organism = 9606  
) %>% filter(consensus_direction == TRUE)
head(trrust)
