# Blue-Detuned Tweezers (BDT)
# Blue-Detuned Tweezers (BDT) — Potentials, Forces, and Flux Simulations

This repo contains a Python package (`BDT`) plus a set of scripts that:
1) build **Richards–Wolf barricade potentials**,  
2) post-process those potentials into **force fields**, and  
3) run **flux / loading simulations** using those precomputed force fields.

Workflow
- Build a 3D barricade potential (Richards–Wolf model)
- Post-process the potential into force fields
- Run flux simulations using the precomputed forces

**High-level sequence:** (A) Build potential → (B) Build forces → (C) Run flux simulations

---

## Requirements

You will need a CSV of light shift vs intensity for your species (the repo includes example CaF files under `data/CaF_lightshifts`).

System
- Python (managed via `uv`)
- A machine with enough RAM/CPU for large 3D arrays (e.g. 501³ grids)
- Access to input CSVs and/or precomputed potential/force .npy files

Python dependencies
Install dependencies via `uv` (see Install below). Core packages:
- `numpy`, `scipy`, `pandas`
- `numba` (simulation kernels)
- `matplotlib` (optional, plotting)
- `tqdm` (optional)

---

## Quickstart — first time

From the repository root:
```bash
uv sync
```

Then follow the three main steps below.

## 1) Build the barricade potential

- Purpose: generate a 3D potential array and save as `.npy`.
- Note: build both a *coarse* and a *fine* potential for the same barricade configuration.

Run:
```bash
uv run python scripts/build_barricade_potential.py
```

Example output (filenames will vary):
- `outputs/potential_08_11_coarse.npy`
- `outputs/potential_08_11_fine.npy`

## 2) Build forces from potentials

- Purpose: load coarse and fine potentials and compute force components (x,y,z).

Run:
```bash
uv run python scripts/build_optical_forces.py
```

Example outputs:
- `data/force_fine/force_fine_x_08.npy`
- `data/force_fine/force_fine_y_08.npy`
- `data/force_fine/force_fine_z_08.npy`
- `data/force_coarse/force_coarse_x_08.npy`
- `data/force_coarse/force_coarse_y_08.npy`
- `data/force_coarse/force_coarse_z_08.npy`

## 3) Run the flux simulation

- Purpose: integrate particle trajectories using precomputed forces and extract flux vs barrier height.
- Tip: run each barrier configuration multiple times (≥10) to gather statistics.

Run:
```bash
uv run python scripts/run_flux_simulations.py
```

Example output:
- `results/BDT_SIM_08/CDT_results_08_1.txt`

---

## Troubleshooting

- **ModuleNotFoundError: No module named 'BDT'**
    - Ensure you run commands from the repository root and that the `src/` layout is active.
    - Quick check:
        ```bash
        uv run python -c "import BDT; print('BDT import OK')"
        ```
    - If the import still fails, confirm `pyproject.toml` contains a package entry (or correct src layout), e.g.:
        - `name = "blue-detuned-tweezersV2"`
        - `packages = ["BDT"]` (or equivalent src configuration)

---

## Adapting to your own species / data

1. Replace the CSV lightshift files in `data/CaF_lightshifts/` with your species' `lightshift_vs_intensity.csv`.
2. Override potential-builder config (arguments or config file) to set:
     - wavelengths: `lam1`, `lam2`
     - optics: `NA`, `f`
     - powers: `power1`, `power2`
     - beam waist scaling: `w02_scale`
3. Adjust simulation parameters in `scripts/run_flux_simulations.py`:
     - `N`, `temperature`, `mass`
     - `dipole_radius`, `U0_scale`
     - integration: `dt`, `totalTime`
     - `R_core`

If you want, I can also: update example config files, add a small runnable example, or run the import check locally.


**(A) Build potential → (B) Build forces → (C) Run flux simulation**

---

## 0. Requirements

a CSV that contains the light shift for a range of different intensities, in this example we take CaF as the species of interest

### System
- Python (managed via **uv**)
- A machine with enough RAM/CPU to build 3D grids (501³ arrays are large)
- Access to the input data files (CSV lightshifts) and/or precomputed potentials/forces

### Python dependencies
Installed via `uv` (see below). Core packages used:
- numpy
- scipy
- pandas
- matplotlib (optional; only needed for plotting)
- numba (used by the simulation kernels)
- tqdm (optional)

---

## 1. Install (first time)

From the repo root:
```bash
uv sync

## 2. Build the barricade potential

- this step generates a 3D potential array (saved as .np)
- you must build both a "fine" and "coarse" potential for each barricade potential.
- to run execute in bash: 
    uv run python scripts/build_barricade_potential.py
- output (example): 
    - outputs/potental_08_11_course.npy

## 3. Build forces from potentials 

- this step loads the potentials "coarse" and "fine" and computes the forces
- to run execute in bash: 
    uv run python scripts/build_forces.py
- output (examples):
    .../force_fine/force_fine_x_08.npy
    .../force_fine/force_fine_y_08.npy
    .../force_fine/force_fine_z_08.npy
    .../force_coarse/force_coarse_x_08.npy
    .../force_coarse/force_coarse_y_08.npy
    .../force_coarse/force_coarse_z_08.npy

## 4. Run the flux simulation

- This step uses the precomputed forces to integrate particle tajectoris in a Monte-Carlo simulation to extract flux vs barrier height
- you will want to run this min 10 times per barrier potential to get good statistics
-  to run execute in bash: 
    uv run python scripts/run_flux_simulations.py 

- Outputs (example):
    .../BDT_SIM_08/CDT_results_08_1.txt

## Common problems

1. ModuleNotFoundError: No module named 'BDT'

solution: 
    - Make sure you are running from the repo root and that src/ layout is active.
    - Run:
        uv run python -c "import BDT; print('BDT import OK')"
    
    - If that fails, check pyproject.toml includes:
	•	name = "blue-detuned-tweezersV2" (any name is fine)
	•	packages = ["BDT"] or uses src layout correctly



## How to adapt to your own data/species
1. Change out the lightshift_vs_intensity.csv files for ones that correspond to your species
2. Use the config override mechanism in the potential builder script:
    •	lam1, lam2
	•	NA, f
	•	power1, power2
	•	w02_scale
3. Change simulation parameters In the simulation runner script:
	•	N, temperature, mass
	•	dipole_radius, U0_scale 
	•	dt, totalTime
	•	R_core