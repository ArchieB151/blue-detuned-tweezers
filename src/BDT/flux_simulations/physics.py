# -*- coding: utf-8 -*-
import numpy as np
import os
from numba import njit
from scipy.constants import Boltzmann

from .constants import Kb, h, c, hbar

@njit
def dipole_force(pos, Center, V0, waist):
    displacement = pos - Center
    x = displacement[:, 0]
    y = displacement[:, 1]
    z = displacement[:, 2]

    rho2 = x**2 + y**2 + z**2
    exp_term = np.exp(-2 * rho2/ waist**2)
    prefactor = -4*V0/waist**2

    F_x = prefactor * x * exp_term
    F_y = prefactor * y * exp_term
    F_z = prefactor * z * exp_term
    return np.column_stack((F_x, F_y, F_z))


@njit
def in_tweezer(pos,
               w0, centre_vec,
               captured,           
               inside_prev):       
    """
    Detect molecules crossing INTO the 3-D Gaussian tweezer ellipsoid
        ρ²/w0² + z²/z_R² ≤ 1      (z_R = π w0² / λ).

    Parameters
    ----------
    pos          : (N,3) current positions
    w0           : 1/e² waist  [m]
    lamb         : wavelength  [m]
    centre_vec   : (3,) beam centre
    captured     : (N,) bool,  stays True once molecule EVER entered
    inside_prev  : (N,) bool,  inside/outside state at previous step

    Returns
    -------
    captured        – updated sticky flags
    inside_prev     – updated state for next call
    n_first_time    – # that entered for the first time THIS step
    n_all_entries   – # total entries this step (first-time + re-entries)
    """
    # ---------- geometry ------------------------------------
    # z_R  = 0.3193e-6 
    disp = pos - centre_vec
    rho2 = disp[:, 0]**2 + disp[:, 1]**2
    z2   = disp[:, 2]**2
    # inside = (rho2 / w0**2 + z2 / z_R**2) <= 1.0   # (N,)

    inside = rho2+z2 <= w0**2

    # ---------- who just crossed the boundary ---------------
    entered_now =  inside & (~inside_prev)          # outside  → inside
    n_all_entries = int(entered_now.sum())


    # first-time-ever events
    new_first   = entered_now & (~captured)
    n_first_time = int(new_first.sum())
    captured  |= new_first                         # make sticky

    # ---------- book-keeping for next call ------------------
    inside_prev[:] = inside

    return captured, inside_prev, n_first_time, n_all_entries

def sample_thermal_velocities(N, T, mass):
    sigma_v = np.sqrt(Boltzmann * T / mass)
    velocities = np.random.normal(loc=0.0, scale=sigma_v, size=(N, 3))
    np.random.shuffle(velocities)
    velocities -= np.mean(velocities, axis=0)
    return velocities

def sample_thermal_positions_harmonic(N, T, U0, waist, Center):
    sigma_r = np.sqrt(Boltzmann * T * waist**2 / (4*U0))
    positions = np.random.normal(0, sigma_r, size=(N, 3))
    return positions + Center

def peak_density(N, T, U0, waist):
    """
    Return peak number-density n₀  [m⁻³]
        N      : number of particles
        T      : temperature [K]
        U0     : trap depth  [J]   (same one used for σ_r)
        waist  : Gaussian 1/e² radius of dipole beam [m]
    """
    sigma_r = np.sqrt(Boltzmann * T * waist**2 / (4 * U0))      # rms width
    n0 = N / ((2*np.pi)**1.5 * sigma_r**3)
    n0 /= (100)**3
    return n0, sigma_r

def sample_thermal_positions_harmonic(N, T, U0, waist, Center):

    sigma_r = np.sqrt(Boltzmann * T * waist**2 / (4*U0))
    positions = np.random.normal(0, sigma_r, size=(N, 3))
    return positions + Center



@njit
def _I_sat_twolevel(lam0, tau):
    return np.pi * h * c / (3 * lam0**3 * tau)

# ---------------------------------------------------------------------------
# helper: single-beam scattering rate  R_sc (Hz)
# ---------------------------------------------------------------------------
@njit
def F_scat(Gamma, nu0, I, lam_L,K_sign):
    """
    sign = +1 for the +k beam,  -1 for the –k beam
    """
    lam0  = c / nu0
    s0    = I / (_I_sat_twolevel(lam0, 1/Gamma)*5)

    delta= c/lam_L - nu0           # same for both beams (–26 MHz)
            # cycles per metre
    K = 1.0 / lam_L * K_sign

    
    return ((0.5*Gamma * s0)/3.75) / (1 + s0 + 4*(delta/Gamma)**2)*hbar *K 
@njit
def Beta_scat(Gamma, nu0, I, lam_L,K_sign):

    lam0  = c / nu0
    s0    = I / (_I_sat_twolevel(lam0, 1/Gamma)*5)

    delta= c/lam_L - nu0           # same for both beams (–26 MHz)
            # cycles per metre
    K = 1.0 / lam_L * K_sign

    return - hbar/7.5 *K **2 *4 * (delta / Gamma) * s0 / (1 + s0 + 4*(delta / Gamma) ** 2)**2




def Force_scat(Gamma, nu0, I, lam_L,K_sign,vel):
    """
    Calculate the force from scattering for a single beam.
    sign = +1 for the +k beam,  -1 for the –k beam

    """

    F_slow = (F_scat(Gamma, nu0, I, lam_L,K_sign) - Beta_scat(Gamma, nu0, I, lam_L,K_sign)*vel)
   
    net_force = np.linalg.norm(F_slow, axis=1)
    a = np.random.normal(0.0, 1.0, (len(vel), 3))          # 10 random 3-vectors
    a /= np.linalg.norm(a, axis=1, keepdims=True) 
    F_spon = np.asarray(a) * net_force[:, None]
    Force = F_slow + F_spon



    return Force



def recycle_far_particles(pos, vel, captured, inside_prev,
                          Center, temperature, U0, dipole_radius, mass):
    """
    Re-initialize any particles that wandered beyond the loss boundary.
    Returns number recycled (for stats).
    """
    disp = pos - Center

    # --- spherical boundary ---
    r2 = np.einsum('ij,ij->i', disp, disp)
    R_LOSS = dipole_radius
    lost = r2 > R_LOSS**2

    # --- cubic boundary (outside coarse field cube) ---
    # lost = (np.abs(disp[:,0]) > SPAN_C) | (np.abs(disp[:,1]) > SPAN_C) | (np.abs(disp[:,2]) > SPAN_C)

    if not np.any(lost):
        return 0

    idx = np.flatnonzero(lost)
    n_lost = idx.size

    # respawn with your thermal samplers
    pos[idx] = sample_thermal_positions_harmonic(n_lost, temperature, U0, dipole_radius, Center)
    vel[idx] = sample_thermal_velocities(n_lost, temperature, mass)

    # reset book-keeping
    captured[idx] = False
    inside_prev[idx] = False
    # if you use dwell_counter elsewhere, also reset it:
    # dwell_counter[idx] = 0

    return n_lost


def cloud_stats(pos, vel, mass, center=None, clip_sigma=None, use_com_origin=False):
    """
    Sizes & density from instantaneous positions/velocities.

    pos : (N,3) [m]
    vel : (N,3) [m/s]
    mass: [kg]
    center: (3,) trap center to measure around; if None or use_com_origin=True,
            use instantaneous COM as origin.
    clip_sigma: float or None → drop outliers with |zscore|>clip_sigma along any axis
    use_com_origin: bool → if True, ignore `center` and use COM as origin

    Returns dict with:
      N_eff, r_cm, v_cm, sigma_xyz, fwhm_xyz, w1e2_xyz,
      principal_sigma, principal_axes, n0_m3, n0_cm3,
      Vrms, T_axis, T_mean, lambda_dB, PSD_est, sigma_r_radial
    """
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    N   = pos.shape[0]

    r_cm = pos.mean(axis=0)
    v_cm = vel.mean(axis=0)

    origin = r_cm if (use_com_origin or center is None) else np.asarray(center, float)
    r = pos - origin
    v = vel - v_cm  # remove bulk motion for T

    if clip_sigma is not None and clip_sigma > 0:
        std = r.std(axis=0) + 1e-30
        keep = (np.abs(r / std) <= clip_sigma).all(axis=1)
        r, v = r[keep], v[keep]

    N_eff = len(r)
    var   = r.var(axis=0)                       # <x^2>, <y^2>, <z^2>
    sigma_xyz = np.sqrt(np.maximum(var, 0.0))   # σx,σy,σz   [m]
    fwhm_xyz  = 2*np.sqrt(2*np.log(2))*sigma_xyz
    w1e2_xyz  = 2*sigma_xyz                     # 1/e^2 radii (laser convention)

    # principal axes (in case the cloud is tilted)
    cov   = (r.T @ r) / max(N_eff, 1)
    evals, evecs = np.linalg.eigh(cov)
    principal_sigma = np.sqrt(np.maximum(evals, 0.0))  # ascending
    principal_axes  = evecs

    # Gaussian peak density from second moments
    vol_gauss = (2*np.pi)**1.5 * np.prod(sigma_xyz)
    n0_m3 = (N_eff / vol_gauss) if vol_gauss > 0 else 0.0
    n0_cm3 = n0_m3 / 1e6

    # kinetic temperatures
    Vrms2_axis = (v**2).mean(axis=0)
    T_axis = mass * Vrms2_axis / Kb
    T_mean = T_axis.mean()
    Vrms   = np.sqrt((v**2).sum(axis=1).mean())

    # de Broglie wavelength & PSD estimate
    lambda_dB = h / np.sqrt(2*np.pi*mass*Kb*max(T_mean, 1e-30))
    PSD_est   = n0_m3 * (lambda_dB**3)

    # convenient radial RMS (xy plane)
    sigma_r_radial = np.sqrt((r[:,0]**2 + r[:,1]**2).mean())

    return dict(
        N_eff=N_eff, r_cm=r_cm, v_cm=v_cm,
        sigma_xyz=sigma_xyz, fwhm_xyz=fwhm_xyz, w1e2_xyz=w1e2_xyz,
        principal_sigma=principal_sigma, principal_axes=principal_axes,
        n0_m3=n0_m3, n0_cm3=n0_cm3,
        Vrms=Vrms, T_axis=T_axis, T_mean=T_mean,
        lambda_dB=lambda_dB, PSD_est=PSD_est,
        sigma_r_radial=sigma_r_radial
    )
 

