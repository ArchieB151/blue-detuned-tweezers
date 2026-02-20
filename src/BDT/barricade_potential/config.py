# -*- coding: utf-8 -*-
import numpy as np

# ---------------- defaults you can override ----------------
run_type = "fine" # "course" or "fine"
run = 0


Npts = 501 
AX_LIMIT = 15e-6
DTYPE = np.float32
workers = 32
use_numba = False

# physical constants
c = 299792458.0
eps0 = 8.854187817e-12

# beam/lens defaults
lam1 = 780e-9
lam2 = 508e-9
f = 3e-3
NA = 0.75
power1 = 8.769461674e-3
power2 = 5.401138157e-3

# “shape” defaults
w02_scale = 0.2
theta1_n_default = 2401
theta2_n_default = 2801


def get_config(overrides=None):
    overrides = overrides or {}

    # 1) start from defaults
    cfg = dict(
        run_type=run_type,
        run=run,


        Npts=Npts,
        AX_LIMIT=AX_LIMIT,
        DTYPE=DTYPE,
        workers=workers,
        use_numba=use_numba,

        c=c,
        eps0=eps0,

        lam1=lam1,
        lam2=lam2,
        f=f,
        NA=NA,
        power1=power1,
        power2=power2,

        w02_scale=w02_scale,
        theta1_n=theta1_n_default,
        theta2_n=theta2_n_default,
    )

    # 2) apply overrides FIRST (so derived stuff uses final values)
    cfg.update(overrides)

    # 3) derived: SPAN from run_type unless explicitly overridden
    if "SPAN" not in cfg:
        if cfg["run_type"] == "course" or cfg["run_type"] == "coarse":
            cfg["SPAN"] = 25e-6
        else:
            cfg["SPAN"] = 5e-6

    # 4) derived: k, angles, theta grids, waists, pupil amplitudes
    cfg["k1"] = 2 * np.pi / cfg["lam1"]
    cfg["k2"] = 2 * np.pi / cfg["lam2"]

    cfg["alpha1"] = np.arcsin(cfg["NA"])
    cfg["alpha2"] = np.arcsin(cfg["NA"])

    cfg["theta1"] = np.linspace(0, cfg["alpha1"], int(cfg["theta1_n"]), dtype=np.float64)
    cfg["theta2"] = np.linspace(0, cfg["alpha2"], int(cfg["theta2_n"]), dtype=np.float64)

    cfg["w01"] = cfg["f"] * cfg["NA"]
    cfg["w02"] = cfg["f"] * cfg["NA"] * cfg["w02_scale"]

    cfg["E01"] = np.sqrt(4 * cfg["power1"] / (np.pi * cfg["w01"] ** 2 * cfg["c"] * cfg["eps0"]))
    cfg["E02"] = np.sqrt(4 * cfg["power2"] / (np.pi * cfg["w02"] ** 2 * cfg["c"] * cfg["eps0"]))

    return cfg