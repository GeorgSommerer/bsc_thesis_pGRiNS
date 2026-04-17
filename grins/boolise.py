#!/usr/bin/env python
import glob
import argparse
import jax.numpy as jnp
from grins.ising_bool import run_all_replicates_ising


def main():
    parser = argparse.ArgumentParser(
        prog="boolise",
        description="Run simulation of GRN-Ising Boolean model",
    )
    parser.add_argument(
        "topo", type=str, help="topo file name", default="all", nargs="?"
    )
    parser.add_argument(
        "--topodir", type=str, help="topo file directory", default="TOPOS"
    )
    parser.add_argument(
        "--outdir", type=str, help="simulation directory", default="IsingSimulResults"
    )
    parser.add_argument("--num_reps", type=int, help="number of replicates", default=3)
    parser.add_argument(
        "--max_steps", type=int, help="number of steps to simulate", default=100
    )
    parser.add_argument(
        "--num_inits", type=int, help="number of initial conditions", default=2**14
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size to parallelize over", default=2**10
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode of simulation. Choose between async, sync",
        default="async",
    )
    parser.add_argument(
        "--flipvalue",
        type=str,
        help="replacement values set. 0 = [Low 0, High 1]; -1 = [Low -1, High 1]",
        default="-1",
    )
    args = parser.parse_args()

    # if no topo file is provided, use iterate over all the topo files
    topos = (
        sorted(glob.glob(f"{args.topodir}/*.topo"))
        if args.topo == "all"
        else [args.topo]
    )
    # Process replacement values
    replacement_dict = {"0": jnp.array([0, 1]), "-1": jnp.array([-1, 1])}
    replacement_values = replacement_dict[args.flipvalue]
    # Print the parameters
    print(f"Number of topology files: {len(topos)}")
    print(f"Number of replicates: {args.num_reps}")
    print(f"Number of initial conditions: {args.num_inits}")
    print(f"Number of steps: {args.max_steps}")
    print(f"Replacement values: {replacement_values}\n")
    for topo_file in topos:
        # Run the simulations
        run_all_replicates_ising(
            topo_file=topo_file,
            num_initial_conditions=args.num_inits,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            replacement_values=replacement_values,
            mode=args.mode,
            packbits=True,
            save_dir=args.outdir,
            num_replicates=args.num_reps,
        )


if __name__ == "__main__":
    main()
