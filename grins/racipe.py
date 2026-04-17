#!/usr/bin/env python
import glob
import argparse
from multiprocessing import Pool
from grins.racipe_run import gen_topo_param_files, run_all_replicates


def mc_param_gen(
    numCores,
    sim_save_dir,
    num_replicates,
    num_params,
    num_init_conds,
    sampling_method,
    topo_files,
):
    # Start multiprocessing pool
    pool = Pool(numCores)
    print("Generating Parameter and Initial Condition files...")
    # Parallel execution of file generation
    pool.starmap(
        gen_topo_param_files,
        [
            (
                topo_file,
                sim_save_dir,
                num_replicates,
                num_params,
                num_init_conds,
                sampling_method,
            )
            for topo_file in topo_files
        ],
    )
    print("Parameter and Initial Condition files generated.\n")
    # Close multiprocessing pool
    pool.close()
    pool.join()


def main():
    parser = argparse.ArgumentParser(
        prog="racipe",
        description="Run simulation of GRN-ODE model for ensemble of parameters",
    )
    parser.add_argument(
        "topo", type=str, help="topo file name", default="all", nargs="?"
    )
    parser.add_argument(
        "--topodir", type=str, help="topo file directory", default="TOPOS"
    )
    parser.add_argument(
        "--outdir", type=str, help="simulation directory", default="SimulResults"
    )
    parser.add_argument(
        "--num_paras", type=int, help="number of parameters", default=10000
    )
    parser.add_argument(
        "--num_inits", type=int, help="number of initial conditions", default=1000
    )
    parser.add_argument("--num_reps", type=int, help="number of replicates", default=3)
    parser.add_argument(
        "--num_cores",
        type=int,
        help="number of cores for parameter generation",
        default=None,
    )
    parser.add_argument(
        "--sampling",
        type=str,
        help="sampling method. Choices: 'Sobol', 'LHS', 'Uniform', 'LogUniform', 'Normal', 'LogNormal'.",
        default="Sobol",
    )
    args = parser.parse_args()

    # if no topo file is provided, use iterate over all the topo files
    topos = (
        sorted(glob.glob(f"{args.topodir}/*.topo"))
        if args.topo == "all"
        else [args.topo]
    )
    # Print the parameters
    print(f"Number of topology files: {len(topos)}")
    print(f"Number of replicates: {args.num_reps}")
    print(f"Number of parameters: {args.num_paras}")
    print(f"Number of initial conditions: {args.num_inits}\n")
    # Call the parameter generation function
    mc_param_gen(
        args.num_cores,
        args.outdir,
        args.num_reps,
        args.num_paras,
        args.num_inits,
        args.sampling,
        topos,
    )
    for topo_file in topos:
        # Run steady-state simulations
        run_all_replicates(
            topo_file,
            args.outdir,
        )


if __name__ == "__main__":
    main()
