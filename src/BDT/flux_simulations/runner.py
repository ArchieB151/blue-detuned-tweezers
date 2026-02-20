# -*- coding: utf-8 -*-
import numpy as np
import os

from .forces_io import load_force_fields
from .richard_wolf_force import RW_force_trilin_numba
from .grid import Grid
from .physics import (
    sample_thermal_positions_harmonic,
    sample_thermal_velocities,
    peak_density,
    recycle_far_particles,
    cloud_stats,
    in_tweezer,
    dipole_force,
    Force_scat,
)
from .constants import Gamma_A, nu_A00, lam_L

def run_simulation(
    iteration,
    base_dir,
    pot_dict,
    potential_run,
    save_dir,
    N=300,
    L=0.1,
    temperature=20e-6,
    mass=59*1.66e-27,
    dipole_radius=150e-6,
    dt=1e-8,
    totalTime=10.0,
    I_cooling=300.0,
    R_core=0.3193e-6,
    U0_scale=5.0,
):
    print("loading forces")
    Fx_zoom, Fy_zoom, Fz_zoom, Fx_coarse, Fy_coarse, Fz_coarse = load_force_fields(
        base_dir=base_dir,
        pot_dict=pot_dict,
        potential_run=potential_run
    )
    print("loaded forces")

    Center = np.full(3, L/2)
    U0 = U0_scale * 1.380649e-23 * temperature

    captured = np.zeros(N, dtype=bool)
    inside_prev = np.zeros(N, dtype=bool)

    pos = sample_thermal_positions_harmonic(N, temperature, U0, dipole_radius, Center)
    vel = sample_thermal_velocities(N, temperature, mass)

    grid = Grid(20, L)
    grid.allocateToGrid(pos)
    grid.updateGrid(pos)

    def force_func(p):
        force = dipole_force(p, Center, U0, dipole_radius)
        force += RW_force_trilin_numba(p, Center, Fx_zoom, Fy_zoom, Fz_zoom, Fx_coarse, Fy_coarse, Fz_coarse)
        return force

    acc = force_func(pos) / mass
    timeStep = totalTime / dt
    t = 0.0

    total_unique_load = 0
    total_load = 0

    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/CDT_results_0{potential_run}_{iteration}.txt"

    for i in range(int(timeStep)):

        if i % int(1e6) == 0:
            flux = (total_unique_load / t) if (total_unique_load > 0 and t > 0) else 0.0

            stats = cloud_stats(pos, vel, mass, center=Center, clip_sigma=5)
            w_r_1e2 = stats["sigma_r_radial"]
            w_z_1e2 = stats["w1e2_xyz"][2]

            row = np.array([[ 
                flux,
                stats['n0_cm3'],
                w_r_1e2*1e6,
                w_z_1e2*1e6,
                stats['T_mean']*1e6
            ]])

            file_exists = os.path.exists(save_path)
            with open(save_path, 'ab') as f:
                np.savetxt(
                    f, row,
                    header="" if file_exists else "flux_Hz   n0_cm^-3   w_r_um   w_z_um   T_uK"
                )

        scattering_Force = (
            Force_scat(Gamma_A, nu_A00, I_cooling, lam_L,  1, vel) +
            Force_scat(Gamma_A, nu_A00, I_cooling, lam_L, -1, vel)
        )

        v_half = vel + 0.5*dt*acc
        pos   += dt*v_half
        acc    = (force_func(pos) + scattering_Force) / mass
        vel    = v_half + 0.5*dt*acc
        t += dt

        if total_unique_load >= 100:
            break

        captured, inside_prev, n_new, n_entries = in_tweezer(
            pos,
            w0=R_core,
            centre_vec=Center,
            captured=captured,
            inside_prev=inside_prev
        )

        total_unique_load += n_new
        total_load += n_entries

        if i % 1e4 == 0:
            recycle_far_particles(pos, vel, captured, inside_prev, Center, temperature, U0, dipole_radius, mass)

    return dict(
        total_unique_load=total_unique_load,
        total_load=total_load,
        final_time=t,
        save_path=save_path
    )