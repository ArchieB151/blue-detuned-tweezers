from __future__ import annotations
import numpy as np
from BDT.power_contours.load_data import load_interp_payload, make_beam_params, field_amplitudes_from_power
from BDT.power_contours.build_vol import build_potential_volume
from BDT.power_contours.contours import overlay_contours

# Contour levels
levels_BH = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
levels_TD = [-1]


def main() -> None:
    state = "|2,2>"
    csv_508 = "data/CaF_lightshifts/energy_shift_vs_intensity_508nm.csv"
    csv_780 = "data/CaF_lightshifts/energy_shift_vs_intensity_780nm.csv"

    interp_payload = load_interp_payload(csv_508, csv_780, state)
    beam = make_beam_params()

    # Grid
    Npts = 10
    SPAN = 1.5e-6
    SPANstart = 0.3e-6
    axis = np.linspace(SPANstart, SPAN, Npts)

    # Heatmap scan size 
    size = Npts
    heatmap_BH = np.zeros((size, size), dtype=np.float32)
    heatmap_TD = np.zeros((size, size), dtype=np.float32)


    for n in range(size):
        power1_W = n * 1e-4
        print(n)
        for m in range(size):
            power2_W = m * 1e-4 

            E01, E02 = field_amplitudes_from_power(power1_W, power2_W, beam.w01, beam.w02)

            # Build volume via multiprocessing (heavy bit)
            R, B = build_potential_volume(
                axis=axis,
                interp_payload=interp_payload,
                beam=beam,
                E01=E01,
                E02=E02,
                dtype=np.float32,
                processes=None,
            )
            V = R + B

            # TD: trap depth at origin 
          
            heatmap_TD[m, n] = float(np.nanmin(V))

            # BH: 
            
            heatmap_BH[m , n] = float(np.nanmax(V) - np.nanmin(V))

    np.save("heatmap_BH_2_2_02.npy", heatmap_BH)
    np.save("heatmap_TD_2_2_02.npy", heatmap_TD)

    overlay_contours(heatmap_BH, heatmap_TD, levels_BH=levels_BH, levels_TD=levels_TD)


if __name__ == "__main__":
    main()