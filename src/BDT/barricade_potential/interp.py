# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def load_interp_payload(path_508, path_780, state='|2,-2>'):
    # ---------------- data for interpolation ------------------------------------
    df_508 = pd.read_csv(path_508, index_col=0)
    df_780 = pd.read_csv(path_780, index_col=0)
    state = state

    intensities_508 = np.array([float(col) for col in df_508.columns])
    intensities_780 = np.array([float(col) for col in df_780.columns])

    values_508 = df_508.loc[state].values.astype(np.float64, copy=False)
    values_780 = df_780.loc[state].values.astype(np.float64, copy=False)

    _interp_payload = (intensities_508, values_508, intensities_780, values_780)

    return _interp_payload