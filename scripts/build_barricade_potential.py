# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp

from BDT.barricade_potential.interp import load_interp_payload
from BDT.barricade_potential.compute import main
from BDT.barricade_potential.config import get_config

def script_main():
    # ---- paths specified ONLY here ----
    path_508 = "data/CaF_lightshifts/energy_shift_vs_intensity_508nm.csv"
    path_780 = "data/CaF_lightshifts/energy_shift_vs_intensity_780nm.csv"

    # ---- params controlled ONLY here ----
    overrides = dict(

        # run type 
        run=8,
        run_type="course", # "course" or "fine" 
        state='|2,-2>', # hyperfine state 

        # wavelengths of light 
        lam1=780e-9,
        lam2=508e-9,

        # objective properties
        NA=0.75,
        f=3e-3, 

        # beam powers in W
        power1=8.769461674e-3,
        power2=5.401138157e-3,

        # optional knobs
        w02_scale=0.2, # scale factor to under fill the tweezer objective 
        workers=32,
        Npts=501,
    )

    cfg = get_config(overrides)
    out_path = f"data/potentials/potential_0{cfg['run']}_11_{cfg['run_type']}.npy"

    _interp_payload = load_interp_payload(path_508, path_780, state=cfg['state'])
    total_potential = main(_interp_payload, cfg)

    np.save(out_path, total_potential)
    print("Saved:", out_path)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Using spawn start method for multiprocessing.")
    except RuntimeError:
        pass
    script_main()