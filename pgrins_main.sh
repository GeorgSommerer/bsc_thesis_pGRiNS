#!/bin/bash

project="Keggoro"

python3 -u Prep_Data/pgrins_prepare_input.py $project -es

python3 -u pgrins_run_grins.py $project Racipe -s

python3 -u Prep_Data/pgrins_prepare_output.py $project Racipe

python3 -u pgrins_narrow_params.py $project -e Norman19

python3 -u pgrins_run_grins.py $project Racipe -sp

python3 -u Prep_Data/pgrins_prepare_output.py $project Racipe -p

python3 -u pgrins_narrow_params.py $project -e Norman19 -p
