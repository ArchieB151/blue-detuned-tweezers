# -*- coding: utf-8 -*-
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from functools import partial

from .mp import worker_x_slice, _worker_init

def main(_interp_payload, cfg):
    SPAN    = cfg["SPAN"]
    Npts    = cfg["Npts"]
    AX_LIMIT= cfg["AX_LIMIT"]
    DTYPE   = cfg["DTYPE"]
    workers = cfg["workers"]

    f       = cfg["f"]
    lam1    = cfg["lam1"]
    E01     = cfg["E01"]
    theta1  = cfg["theta1"]
    k1      = cfg["k1"]
    w01     = cfg["w01"]

    lam2    = cfg["lam2"]
    E02     = cfg["E02"]
    theta2  = cfg["theta2"]
    k2      = cfg["k2"]
    w02     = cfg["w02"]

    axis = np.linspace(-SPAN, SPAN, Npts, dtype=np.float64)
    mask_axis = (np.abs(axis) < AX_LIMIT)

    shape = (Npts, Npts, Npts)
    intensity_r = np.zeros(shape, dtype=DTYPE)
    intensity_b = np.zeros(shape, dtype=DTYPE)

    ctx = get_context("spawn")
    work = partial(
        worker_x_slice,
        axis=axis, mask_axis=mask_axis,
        DTYPE=DTYPE,
        f=f,
        lam1=lam1, E01=E01, theta1=theta1, k1=k1, w01=w01,
        lam2=lam2, E02=E02, theta2=theta2, k2=k2, w02=w02
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(_interp_payload,)
    ) as ex:
        futures = [ex.submit(work, xi) for xi in range(Npts)]
        for fut in as_completed(futures):
            xi, r_slice, b_slice = fut.result()
            intensity_r[xi, :, :] = r_slice
            intensity_b[xi, :, :] = b_slice

    total_potential = (intensity_r + intensity_b).astype(DTYPE, copy=False)
    return total_potential