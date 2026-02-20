# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import j0, j1, jn

c     = 299792458.0 # m/s
eps0  = 8.854187817e-12 

# --------------- helper: Gaussian on the pupil ------------------------------
def apodisation(f_beam, w0_beam, th):
    # vectorised over th
    return np.exp(-(f_beam*np.sin(th))**2 / w0_beam**2)

# ---------------- Richards–Wolf integrals -----------------------------------
def Ivals_linear(rho, z, th, k, f_beam, w0_beam):
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    common = apodisation(f_beam, w0_beam, th) * np.sqrt(cos_th) * np.exp(1j*k*z*cos_th)
    g0 = common * sin_th * (1 + cos_th)
    g1 = common * sin_th**2
    g2 = common * sin_th * (1 - cos_th)
    krst = k*rho*sin_th
    I0 = np.trapezoid(g0 * j0(krst), th)
    I1 = np.trapezoid(g1 * j1(krst), th)
    I2 = np.trapezoid(g2 * jn(2, krst), th)
    return I0, I1, I2

def Ivals_radial(rho, z, th, k, f_beam, w0_beam):
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    common = apodisation(f_beam, w0_beam, th) * np.sqrt(cos_th) * np.exp(1j*k*z*cos_th)
    krst = k*rho*sin_th
    Irho = np.trapezoid(common * 2*sin_th*cos_th * j1(krst), th)
    Iz   = np.trapezoid(common * sin_th**2 * j0(krst), th)
    return Irho, Iz

def Ivals_azimuthal(rho, z, th, k, f_beam, w0_beam):
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    common = apodisation(f_beam, w0_beam, th) * np.sqrt(cos_th)
    phase  = np.exp(1j * k * z * cos_th)
    krst = k*rho*sin_th
    Iphi = np.trapezoid(common * sin_th * j1(krst) * phase, th)
    return Iphi

def field_E(x, y, z, f_beam, lam_beam, E0_beam, th, k_beam, w0_beam, pol='linear'):
    A = np.pi * f_beam / lam_beam * E0_beam
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    if pol == 'linear':
        I0, I1, I2 = Ivals_linear(rho, z, th, k_beam, f_beam, w0_beam)
        Ex = -1j*A*(I0 + I2*np.cos(2*phi))
        Ey = -1j*A*  I2*np.sin(2*phi)
        Ez = -2 *A*  I1*np.cos( phi)

    elif pol == 'radial':
        Irho, Iz = Ivals_radial(rho, z, th, k_beam, f_beam, w0_beam)
        Erho   =  A * Irho
        Ex     =  Erho*np.cos(phi)
        Ey     =  Erho*np.sin(phi)
        Ez     =  2j*A*Iz

    elif pol == 'azimuthal':
        Iphi = Ivals_azimuthal(rho, z, th, k_beam, f_beam, w0_beam)
        Ephi = 2 * A * Iphi
        Ex   = -Ephi * np.sin(phi)
        Ey   =  Ephi * np.cos(phi)
        Ez   =  0.0
    else:
        raise ValueError("pol must be 'linear', 'radial', or 'azimuthal'")

    return Ex, Ey, Ez

def intensity(x, y, z, f_beam, lam_beam, E0_beam, th, k_beam, w0_beam, pol='linear'):
    Ex, Ey, Ez = field_E(x, y, z, f_beam, lam_beam, E0_beam, th, k_beam, w0_beam, pol=pol)
    return 0.5*c*eps0*(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)