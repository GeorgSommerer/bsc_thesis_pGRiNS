# RACIPE Script

## Overview
The `racipe` script provides a default implementation of the RACIPE pipeline. This script is fairly limited in customizablity. If you prefer to customize refer to the [tutorial](./RACIPE_Tutorial.md) and the [RACIPE API](../api/RACIPE.md).

## Usage
```bash 
racipe [-h] [--topodir TOPODIR] [--outdir OUTDIR] [--num_paras NUM_PARAS] [--num_inits NUM_INITS] [--num_reps NUM_REPS] [--num_cores NUM_CORES] [--sampling SAMPLING] [topo]
```
- Run simulations for 100 initial conditions across 1000 parameter sets for all the topo files in the folder `TOPOS`.
```bash
racipe --topodir TOPOS --outdir SimulResults --num_inits 100 --num_paras 1000
```
- Run simulation for 1000 initial conditions across 10000 parameters for the `TS.topo` file.
```bash
racipe --outdir SimulResults --num_inits 1000 --num_paras 10000 TS.topo
```

## Options
-  `-h, --help` 
    Shows the help message with all available options.
-  `--topodir TOPODIR` 
    Directory containing the topology files. Useful if your topology files are stored in a specific directory. Default: `TOPOS`.
-  `--outdir OUTDIR`
    Output directory for simulation results. Default: `SimulResults`.
-  `--num_paras NUM_PARAS`
    Number of parameter sets to generate. This determines the size of the ensemble you want to simulate. Default: `10000`.
-  `--num_inits NUM_INITS`
    Number of initial conditions to use for each simulation. This allows for exploring different starting points in your model. Default: `1000`.
-  `--num_reps NUM_REPS`
    Number of replicates per topofile. Default: `3`.
-  `--num_cores NUM_CORES` 
    Number of cores to use for parallel parameter generation. Default: `All available cores`.
-  `--sampling SAMPLING` 
    Sampling method to generate parameter sets. Choices include: 'Sobol', 'LHS', 'Uniform', 'LogUniform', 'Normal', 'LogNormal'. Default: `Sobol`.