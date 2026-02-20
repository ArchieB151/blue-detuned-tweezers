# -*- coding: utf-8 -*-
import scipy.constants
from scipy.constants import Boltzmann

Kb = Boltzmann

c = scipy.constants.c
h = scipy.constants.h
hbar = scipy.constants.hbar
e = scipy.constants.e
eps0 = scipy.constants.epsilon_0
muN = scipy.constants.physical_constants['nuclear magneton'][0]
muB = scipy.constants.physical_constants['Bohr magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
a0 = scipy.constants.physical_constants['atomic unit of length'][0]
DebyeSI = 3.33564e-30
inv_cm_to_Hz = 100 * c

# CaF A-state-ish constants you had
tau_A   = 19.2e-9
Gamma_A = 1 / tau_A
nu_A00  = 4.9447855e14          # 606.28 nm
lam_L   = c / (nu_A00 - 26e6)