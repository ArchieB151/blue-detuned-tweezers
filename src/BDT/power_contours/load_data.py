import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass(frozen=True)
class BeamParams:
    f: float
    NA: float
    lam1: float
    lam2: float
    theta1: np.ndarray
    theta2: np.ndarray
    k1: float
    k2: float
    w01: float
    w02: float


def load_interp_payload(csv_508: str, csv_780: str, state: str):
    df_508 = pd.read_csv(csv_508, index_col=0)
    df_780 = pd.read_csv(csv_780, index_col=0)

    intensities_508 = np.array([float(c) for c in df_508.columns], dtype=float)
    intensities_780 = np.array([float(c) for c in df_780.columns], dtype=float)

    values_508 = np.asarray(df_508.loc[state].values, dtype=float)
    values_780 = np.asarray(df_780.loc[state].values, dtype=float)

    return (intensities_508, values_508, intensities_780, values_780)


def make_beam_params() -> BeamParams:
    f = 3e-3
    NA = 0.75

    lam1 = 780e-9
    lam2 = 508e-9

    k1 = 2 * np.pi / lam1
    k2 = 2 * np.pi / lam2

    alpha1 = np.arcsin(NA)
    alpha2 = np.arcsin(NA)

    theta1 = np.linspace(0, alpha1, 2401)
    theta2 = np.linspace(0, alpha2, 2801)

    w01 = f * NA
    w02 = f * NA * 0.2

    return BeamParams(f=f, NA=NA, lam1=lam1, lam2=lam2, theta1=theta1, theta2=theta2, k1=k1, k2=k2, w01=w01, w02=w02)


def field_amplitudes_from_power(power1_W: float, power2_W: float, w01: float, w02: float) -> tuple[float, float]:
    # Kept identical to your previous script
    c = 299792458.0
    eps0 = 8.854187817e-12

    I01 = 2 * power1_W / (np.pi * w01**2)
    E01 = np.sqrt(2 * I01 / (c * eps0))

    I02 = 2 * power2_W / (np.pi * w02**2)
    E02 = np.sqrt(2 * I02 / (c * eps0))

    return float(E01), float(E02)