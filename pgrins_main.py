import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

import os

from Prep_Data import pgrins_prepare_input, pgrins_prepare_output
import pgrins_run_grins


experimental = True
project_name = "Keggoro_abcd"
pgrins_prepare_data.main(experimental,project_name)

is_control = True
is_racipe = True
grn_file = "Keggoro_abcd"
pgrins_run_grins.main(grn_file,is_control,is_racipe)

pgrins_prepare_output.main()