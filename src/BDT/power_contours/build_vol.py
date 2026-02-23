import multiprocessing as mp
import numpy as np
from BDT.barricade_potential.mp import _worker_init, worker_x_slice
from BDT.power_contours.load_data import  BeamParams

def build_potential_volume(
    *,
    axis: np.ndarray,
    interp_payload,
    beam: BeamParams,
    E01: float,
    E02: float,
    dtype=np.float32,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (R, B) where each is [Nx, Ny, Nz] with slices returned by worker_x_slice.
    """
    Nx = Ny = Nz = len(axis)

    # Mask can be used to skip points; keep everything on by default
    mask_axis = np.ones_like(axis, dtype=bool)

    args_common = (
        axis,
        mask_axis,
        dtype,
        beam.f,
        beam.lam1,
        E01,
        beam.theta1,
        beam.k1,
        beam.w01,
        beam.lam2,
        E02,
        beam.theta2,
        beam.k2,
        beam.w02,
    )

    R = np.zeros((Nx, Ny, Nz), dtype=dtype)
    B = np.zeros((Nx, Ny, Nz), dtype=dtype)

    with mp.Pool(
        processes=processes,
        initializer=_worker_init,
        initargs=(interp_payload,),
        maxtasksperchild=50,
    ) as pool:
        # map over x-index
        for xi, r_slice, b_slice in pool.starmap(
            worker_x_slice,
            [(xi, *args_common) for xi in range(Nx)],
            chunksize=max(1, Nx // (4 * (processes or mp.cpu_count()))),
        ):
            R[xi, :, :] = r_slice
            B[xi, :, :] = b_slice

    return R, B


