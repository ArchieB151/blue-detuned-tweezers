# -*- coding: utf-8 -*-
from BDT.optical_forces.compute_optical_forces import compute_and_save_forces_for_potential

def main():
    dicts = [
        "potentials/2_minus2",
    ]
    base = "data"


    #convert multiple files in each dict
    start =8
    end = 9


    for d in dicts:
        print(f"Working on {d}")
        for i in range(start, end):
            print(i)

            # ---- fine ----
            pot_fine = f"{base}/{d}/potential_0{i}_fine.npy"
            out_fine = f"{base}/force_fine/potential_0{i}/"
            compute_and_save_forces_for_potential(
                pot_path=pot_fine,
                out_dir=out_fine,
                prefix="force_fine",
                run_i=i,
                span=5e-6,
            )

            # ---- coarse ----
            pot_coarse = f"{base}/{d}/potential_0{i}_coarse.npy"
            out_coarse = f"{base}/force_coarse/potential_0{i}/"
            compute_and_save_forces_for_potential(
                pot_path=pot_coarse,
                out_dir=out_coarse,
                prefix="force_course",
                run_i=i,
                span=25e-6,
            )
    print("Done.")

if __name__ == "__main__":
    main()

