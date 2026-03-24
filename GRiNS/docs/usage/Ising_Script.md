# Ising Boolean Script

## Overview
The `boolise` script provides a default implementation of the Ising Boolean pipeline. This script is fairly limited in customizablity. If you prefer to customize refer to the [tutorial](./Ising_Tutorial.md) and the [Ising API](../api/IsingBoolean.md).

## Usage
```bash 
boolise [-h] [--topodir TOPODIR] [--outdir OUTDIR] [--max_steps MAX_STEPS] [--num_inits NUM_INITS] [--batch_size BATCH_SIZE] [--mode MODE] [--flipvalue FLIPVALUE] [topo]
```
- Run asynchronous boolean simulations with [0,1] formalism for all the topo files in the folder `TOPOS`.
```bash
boolise --topodir TOPOS --outdir IsingSimulResults --mode async --flipvalue 0
```
- Run synchoronous simulation with [-1,1] formalism for the `TS.topo` file.
```bash
boolise --outdir IsingSimulResults --mode sync --flipvalue -1 TS.topo
```

## Options
-  `-h, --help` 
    Shows the help message with all available options.
-  `--topodir TOPODIR` 
    Directory containing the topology files. Useful if your topology files are stored in a specific directory. Default: `TOPOS`.
-  `--outdir OUTDIR`
    Output directory for simulation results. Default: `IsingSimulResults`.
-  `--num_inits NUM_INITS`
    Number of initial conditions to use for each simulation. This allows for exploring different starting points in your model. Default: `16384`.
-  `--max_steps MAX_STEPS`
    Maximum number of steps to simulate. Default: `100`.
-  `--batch_size BATCH_SIZE` 
    Batch size to parallelize over. Optimize based on available VRAM. Default: `1024`
-  `--mode MODE`
    Mode of simulation. Choose between: async: Asynchronous mode (one node updated per step), sync: Synchronous mode (all nodes updates per step). Default: `async`
-  `--flipvalue FLIPVALUE` 
    Replacement values set for flip operations during the simulation. Choose between 0 where [Low=0, High=1] or 1 where [Low=-1, High=1]. Default: `0`
     
     