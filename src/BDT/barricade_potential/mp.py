# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


from .richards_wolf import intensity

# ---------------- worker initialiser & worker -------------------------------
# Globals set in each process
g_interp_508 = None
g_interp_780 = None

def _worker_init(interp_payload):
    global g_interp_508, g_interp_780
    intensities_508, values_508, intensities_780, values_780 = interp_payload
    g_interp_508 = interp1d(intensities_508, values_508, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')
    g_interp_780 = interp1d(intensities_780, values_780, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')

def worker_x_slice(xi, axis, mask_axis, DTYPE,
                   f, lam1, E01, theta1, k1, w01,
                   lam2, E02, theta2, k2, w02):
    Ny = Nz = len(axis)
    r_slice = np.zeros((Ny, Nz), dtype=DTYPE)
    b_slice = np.zeros((Ny, Nz), dtype=DTYPE)

    if not mask_axis[xi]:
        return xi, r_slice, b_slice

    xval = float(axis[xi])
    for y in range(Ny):
        if mask_axis[y]:
            yval = float(axis[y])
            # inner loop can be tight; hoist constants
            for z in range(Nz):
                zval = float(axis[z])
                Ir = intensity(xval, yval, zval, f, lam1, E01, theta1, k1, w01, pol='linear')
                Ib = intensity(xval, yval, zval, f, lam2, E02, theta2, k2, w02, pol='linear')
                r_slice[y, z] = np.float32(g_interp_780(Ir))
                b_slice[y, z] = np.float32(g_interp_508(Ib))
        # else zeros
    return xi, r_slice, b_slice