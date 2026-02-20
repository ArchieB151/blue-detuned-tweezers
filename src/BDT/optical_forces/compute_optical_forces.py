# -*- coding: utf-8 -*-
import os
import numpy as np


def load_and_convert_potential(pot_path):
    pot = np.load(pot_path)
    pot *= 1.38e-23
    pot /= 1e3
    return pot


def axis_grid(span, shape):
    # shape is (Ny, Nx, Nz) 
    y = np.linspace(-span, span, shape[0])
    x = np.linspace(-span, span, shape[1])
    z = np.linspace(-span, span, shape[2])
    return x, y, z


def potential_gradient_forces(pot, x, y, z):
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    dV_dy, dV_dx, dV_dz = np.gradient(
        pot,
        dy,  
        dx,  
        dz,  
        axis=(0, 1, 2),
    )
    F_x = -dV_dx
    F_y = -dV_dy
    F_z = -dV_dz
    return F_x, F_y, F_z


def save_forces(out_dir, prefix, run_i, F_x, F_y, F_z):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_x_0{run_i}.npy"), F_x)
    np.save(os.path.join(out_dir, f"{prefix}_y_0{run_i}.npy"), F_y)
    np.save(os.path.join(out_dir, f"{prefix}_z_0{run_i}.npy"), F_z)


def compute_and_save_forces_for_potential(pot_path, out_dir, prefix, run_i, span):
    pot = load_and_convert_potential(pot_path)
    x, y, z = axis_grid(span, pot.shape)
    F_x, F_y, F_z = potential_gradient_forces(pot, x, y, z)
    save_forces(out_dir, prefix, run_i, F_x, F_y, F_z)
    return F_x, F_y, F_z