# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, int32


R2_ZOOM = (5e-6)**2
SPAN_Z = 5e-6
npts_z = 501
SCALE_Z = (npts_z - 1) / (2.0 * SPAN_Z)

R2_COARSE = (25e-6)**2
SPAN_C = 25e-6
npts_c = 501
SCALE_C = (npts_c - 1) / (2.0 * SPAN_C)


@njit(inline='always')
def _inside_cube(dx, dy, dz, span):
    return (abs(dx) <= span) and (abs(dy) <= span) and (abs(dz) <= span)

@njit(inline='always')
def _idx_t(v, span, scale, npts):
    u = (v + span) * scale
    if u <= 0.0:
        return 0, 0.0
    um = npts - 1.0
    if u >= um:
        return npts - 2, 1.0
    i0 = int(u)
    t  = u - i0
    if i0 >= npts - 1:
        i0 = npts - 2
        t  = 1.0
    return i0, t

@njit(inline='always')
def _trilin_component(arr, iy0, ix0, iz0, ty, tx, tz):
    iy1 = iy0 + 1; ix1 = ix0 + 1; iz1 = iz0 + 1
    c000 = arr[iy0, ix0, iz0]
    c100 = arr[iy1, ix0, iz0]
    c010 = arr[iy0, ix1, iz0]
    c110 = arr[iy1, ix1, iz0]
    c001 = arr[iy0, ix0, iz1]
    c101 = arr[iy1, ix0, iz1]
    c011 = arr[iy0, ix1, iz1]
    c111 = arr[iy1, ix1, iz1]

    c00 = c000*(1-ty) + c100*ty
    c10 = c010*(1-ty) + c110*ty
    c01 = c001*(1-ty) + c101*ty
    c11 = c011*(1-ty) + c111*ty

    c0 = c00*(1-tx) + c10*tx
    c1 = c01*(1-tx) + c11*tx

    return c0*(1-tz) + c1*tz

@njit(inline='always')
def _sample_trilin(dx, dy, dz, Fx, Fy, Fz, span, scale, npts):
    if not _inside_cube(dx, dy, dz, span):
        return 0.0, 0.0, 0.0

    iy0, ty = _idx_t(dy, span, scale, npts)
    ix0, tx = _idx_t(dx, span, scale, npts)
    iz0, tz = _idx_t(dz, span, scale, npts)

    fx = _trilin_component(Fx, iy0, ix0, iz0, ty, tx, tz)
    fy = _trilin_component(Fy, iy0, ix0, iz0, ty, tx, tz)
    fz = _trilin_component(Fz, iy0, ix0, iz0, ty, tx, tz)
    return fx, fy, fz

@njit(parallel=True, fastmath=True, cache=True)
def RW_force_trilin_numba(pos, center,
                          Fx_z, Fy_z, Fz_z,
                          Fx_c, Fy_c, Fz_c):

    N = pos.shape[0]
    out = np.zeros((N,3), dtype=pos.dtype)

    for n in prange(N):
        dx = pos[n,0] - center[0]
        dy = pos[n,1] - center[1]
        dz = pos[n,2] - center[2]

        if _inside_cube(dx, dy, dz, SPAN_Z):
            fx, fy, fz = _sample_trilin(dx, dy, dz, Fx_z, Fy_z, Fz_z,
                                        SPAN_Z, SCALE_Z, Fx_z.shape[0])
        elif _inside_cube(dx, dy, dz, SPAN_C):
            fx, fy, fz = _sample_trilin(dx, dy, dz, Fx_c, Fy_c, Fz_c,
                                        SPAN_C, SCALE_C, Fx_c.shape[0])
        else:
            fx = fy = fz = 0.0

        out[n,0] = fx
        out[n,1] = fy
        out[n,2] = fz

    return out