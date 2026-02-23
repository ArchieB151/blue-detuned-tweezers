import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('/Users/archiebaldock/phd/FiggureStyleSheet.mplstyle')

# -------------------- COLOUR PALETTE --------------------
red = '#e31a1a'
orange = '#feb24c'
paleblue = '#9ebcda'
darkblue = '#0c6485'
purple = '#88419d'
pink = '#f748a5'
nicePink = '#DE4080'
green = '#1cac78'
brown = '#8E3238'
teal = '#328A8E'

# ============================================================
#                      CSV LOADERS (MINIMAL)
# ============================================================
def _read_csv_struct(path):
    """
    Reads CSV with headers. Expected columns:
      state,bh_mK,flux_s,flux_err
    rb_flux.csv may also contain red_power_mW, blue_power_mW (ignored).
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    cols = set(data.dtype.names or ())
    required = {"state", "bh_mK", "flux_s", "flux_err"}
    if not required.issubset(cols):
        raise ValueError(f"{path} must have columns {sorted(required)}. Found {sorted(cols)}")
    return data

def load_rb_from_csv(path="rb_flux.csv"):
    d = _read_csv_struct(path)
    # Rb is one "state" in your CSV ("Rb")
    m = (d["state"] == "Rb") | (np.asarray(d["state"], dtype=str) == "Rb")
    bh = np.array(d["bh_mK"][m], float)
    y  = np.array(d["flux_s"][m], float)
    e  = np.array(d["flux_err"][m], float)
    order = np.argsort(bh)
    return bh[order], y[order], e[order]

def load_caf_state_from_csv(path="caf_flux.csv"):
    d = _read_csv_struct(path)
    # Group by state name
    states = np.unique(np.asarray(d["state"], dtype=str))
    out = {}
    for st in states:
        m = np.asarray(d["state"], dtype=str) == st
        bh = np.array(d["bh_mK"][m], float)
        y  = np.array(d["flux_s"][m], float)
        e  = np.array(d["flux_err"][m], float)
        order = np.argsort(bh)
        out[st] = (bh[order], y[order], e[order])
    return out


## FIT FUNCTIONS 

def flux_suppression(Ub_mK, T_mK):
    Ub_mK = np.asarray(Ub_mK, dtype=float)
    x = Ub_mK / float(T_mK)
    x = np.clip(x, 0.0, None)
    erfc = np.vectorize(math.erfc)
    return erfc(np.sqrt(x)) + (2.0 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x)

def fit_flux_suppression(Ub_mK, y, yerr=None, T_bounds=(1e-4, 0.5), n_grid=2000):
    Ub = np.asarray(Ub_mK, float)
    y = np.asarray(y, float)

    if yerr is None:
        yerr = np.ones_like(y, float)
    else:
        yerr = np.asarray(yerr, float)

    m = np.isfinite(Ub) & np.isfinite(y) & np.isfinite(yerr) & (y > 0) & (yerr > 0)
    Ub, y, yerr = Ub[m], y[m], yerr[m]

    if Ub.size < 3:
        raise ValueError("Not enough valid points to fit (need at least 3).")

    w = 1.0 / (yerr**2)

    T_lo, T_hi = T_bounds
    Ts = np.logspace(np.log10(T_lo), np.log10(T_hi), int(n_grid))

    best = (np.inf, None, None)  # (chi2, A, T)

    for T in Ts:
        s = flux_suppression(Ub, T_mK=T)
        denom = np.sum(w * s * s)
        if denom <= 0 or not np.isfinite(denom):
            continue
        A = np.sum(w * y * s) / denom
        yhat = A * s
        chi2 = np.sum(w * (y - yhat)**2)
        if chi2 < best[0]:
            best = (chi2, A, T)

    chi2_best, A_best, T_best = best
    if A_best is None:
        raise RuntimeError("Fit failed — try widening T_bounds.")

    return A_best, T_best, chi2_best


def plot_with_mask(ax, x, y, yerr, fmt, **kwargs):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    ax.errorbar(x[m], y[m], yerr=yerr[m], fmt=fmt, **kwargs)

## Loads Data from CSVs

Rb_Bh_data, rates, error = load_rb_from_csv("data/results/rb_flux.csv")         
caf = load_caf_state_from_csv("data/results/caf_flux.csv")                      


Bh_min = {k: v[0] for k, v in caf.items()}  # just barrier arrays by state name


fig, (ax3, ax1) = plt.subplots(
    1, 2,
    figsize=(2*(3+3/8), (3+3/8)/1.61),
    constrained_layout=True
)

## Left PLOT (Rb) 
ax3.errorbar(
    Rb_Bh_data, rates, yerr=error,
    fmt='o', color=teal, alpha=1,
    ecolor='gray', capsize=3,
    label='Flux'
)

ax3.set_yscale("log")
ax3.set_ylim(0.1, 100)
ax3.set_xlabel(r"Barrier height (mK)")
ax3.set_ylabel(" Flux (s$^{-1}$)", color='black')
ax3.tick_params(axis='y', colors='black')
ax3.title.set_text(r"$^{87}$Rb")
ax3.text(-0.18, 1.13, '(a)', transform=ax3.transAxes,
         verticalalignment='top', horizontalalignment='left')

A_rb, T_rb, chi2_rb = fit_flux_suppression(
    Rb_Bh_data, rates, error,
    T_bounds=(1e-3, 0.5),
    n_grid=3000
)
Rb_line = np.linspace(np.nanmin(Rb_Bh_data), np.nanmax(Rb_Bh_data), 400)
ax3.plot(Rb_line, A_rb * flux_suppression(Rb_line, T_mK=T_rb),
         '--', color='gray', label=fr"Suppression fit ($T$={T_rb:.3f} mK)")
print(f"[Rb] Fit: A = {A_rb:.3g}, T = {T_rb:.5f} mK, chi2 = {chi2_rb:.2f}")

## Right PLOT (CaF)
grey_style = dict(color="0.70", alpha=0.35, ecolor="0.80")
highlight_ecolor = "lightgray"

# Helper to fetch from caf dict
def caf_state(name):
    bh, y, e = caf[name]
    return bh, y, e

#Plots all other states in grey
for st, mk in [
    ("Potentals_2_2", "^"),
    ("Potentals_1_1", "o"),
    ("Potentals_1_minus1", "s"),
    ("Potentals_0_0", "D"),
    ("Potentals_1minus_1", "P"),
    ("Potentals_1_0", "v"),
    ("Potentals_2_minus1", "h"),
    ("Potentals_2_1", ">"),
    ("Potentals_1minus_minus1", "<"),
    ("Potentals_1minus_0", "p"),
    ("Potentals_2_0", "*"),
]:
    if st in caf:
        bh, y, e = caf_state(st)
        plot_with_mask(ax1, bh, y, e, mk, **grey_style, capsize=3, label="_nolegend_")


bh_22, y_22, e_22 = caf_state("Potentals_2_minus2")
plot_with_mask(
    ax1, bh_22, y_22, e_22,
    'X', color='#8E5732', alpha=0.95, ecolor=highlight_ecolor, capsize=3,
    label=r'$|2,-2\rangle$'
)

bh_1m1, y_1m1, e_1m1 = caf_state("Potentals_1minus_1")
plot_with_mask(
    ax1, bh_1m1, y_1m1, e_1m1,
    '8', color='#57328E', alpha=0.95, ecolor=highlight_ecolor, capsize=3,
    label=r'$|1^-,1\rangle$'
)

ax1.set_xlabel(r"Barrier height (mK)")
ax1.set_ylabel(" Flux (s$^{-1}$)", color='black')
ax1.set_yscale("log")
ax1.set_ylim(0.05, 3000)
ax1.set_xlim(-0.009, 0.16)
ax1.tick_params(axis='y', colors='black')
ax1.title.set_text(r"CaF")
ax1.text(-0.18, 1.13, '(b)', transform=ax1.transAxes,
         verticalalignment='top', horizontalalignment='left')

# --- Fits: match your original exactly ---
Ub_fit = bh_22[-5:]
y_fit  = np.array(y_22[-5:], float)
e_fit  = np.array(e_22[-5:], float)

A_caf, T_caf, chi2_caf = fit_flux_suppression(
    Ub_fit, y_fit, e_fit,
    T_bounds=(1e-4, 0.2),
    n_grid=3000
)
Ub_line = np.linspace(0, np.nanmax(Ub_fit) + 0.05, 400)
ax1.plot(Ub_line, A_caf * flux_suppression(Ub_line, T_mK=T_caf),
         '--', color='gray')
print(f"[CaF |2,-2>] Fit: A = {A_caf:.3g}, T = {T_caf:.5f} mK, chi2 = {chi2_caf:.2f}")

Ub_fit2 = bh_1m1[-8:]
y_fit2  = np.array(y_1m1[-8:], float)
e_fit2  = np.array(e_1m1[-8:], float)

A_caf2, T_caf2, chi2_caf2 = fit_flux_suppression(
    Ub_fit2, y_fit2, e_fit2,
    T_bounds=(1e-4, 0.2),
    n_grid=3000
)
Ub_line2 = np.linspace(0, np.nanmax(Ub_fit2) + 0.05, 400)
ax1.plot(Ub_line2, A_caf2 * flux_suppression(Ub_line2, T_mK=T_caf2),
         '--', color='gray')
print(f"[CaF |1^-,1>] Fit: A = {A_caf2:.3g}, T = {T_caf2:.5f} mK, chi2 = {chi2_caf2:.2f}")

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

ax1.minorticks_off()
ax3.minorticks_off()
plt.savefig("finalResults2.pdf")
plt.show()