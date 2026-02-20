# -*- coding: utf-8 -*-
import sys

from BDT.flux_simulations.runner import run_simulation

def main():
    iteration = 1

    base_dir = "data"
    pot_dict = "potential_08"
    potential_run = 8

    save_dir = f"{base_dir}/results/{pot_dict}/BDT_SIM_0{potential_run}"

    result = run_simulation(
        iteration=iteration,
        base_dir=base_dir,
        pot_dict=pot_dict,
        potential_run=potential_run,
        save_dir=save_dir,
        # tweak anything here
        N=300,
        temperature=20e-6,
        dt=1e-8,
        totalTime=10.0,
        I_cooling=300.0,
        R_core=0.3193e-6,
    )
    print(result)

if __name__ == "__main__":
    main()