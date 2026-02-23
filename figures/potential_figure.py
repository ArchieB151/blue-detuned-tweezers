import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
import matplotlib.gridspec as gridspec

plt.style.use("/Users/archiebaldock/blue-detuned-tweezersV2/figures/FiggureStyleSheet.mplstyle")

# ---------------- Load potential ----------------
pot = np.load("Load-your-potential-file-here.npy")  # Replace with your actual file path calculated from scripts/build_barricade_potential.py
x0 = pot.shape[0] // 2

# Axial slice (y–z at fixed x)
plane_axial = pot[:, x0, :].astype(float)


def prep_plane(plane: np.ndarray, sigma_px: float = 0.0) -> np.ndarray:
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=float)

    plane = np.nan_to_num(
        plane,
        copy=False,
        posinf=float(np.nanmax(finite)),
        neginf=float(np.nanmin(finite)),
    )

    if sigma_px > 0:
        plane = gaussian_filter(plane, sigma=sigma_px)

    return plane


plane_axial = prep_plane(plane_axial)


def make_axes_for(plane: np.ndarray, L_um: float = 5.0):
    Ny, Nz = plane.shape
    a = np.linspace(-L_um * 1e-6, L_um * 1e-6, Ny)
    b = np.linspace(-L_um * 1e-6, L_um * 1e-6, Nz)
    A, B = np.meshgrid(a, b, indexing="ij")
    return a, b, A, B


y_a, z_a, Y_a, Z_a = make_axes_for(plane_axial)

norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=0.2)
cmap = plt.get_cmap("plasma")


def downsample(A: np.ndarray, B: np.ndarray, P: np.ndarray, step: int = 2):
    return A[::step, ::step], B[::step, ::step], P[::step, ::step]


Yd_a, Zd_a, Pd_a = downsample(Y_a, Z_a, plane_axial, step=2)


def draw_panel(
    ax,
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    rasterize: bool = True,
):
    facecolors = cmap(norm(P))

    ax.plot_surface(
        A,
        B,
        P,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.85,
    )

    zmin = float(np.nanmin(P))
    zmax = float(np.nanmax(P))
    dz = zmax - zmin if (zmax - zmin) > 0 else 1.0
    z_floor = zmin - 0.10 * dz

    ax.plot_surface(
        A,
        B,
        np.full_like(P, z_floor),
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=1.0,
    )

    a = A[:, 0]
    b = B[0, :]
    Ny, Nz = P.shape
    ia0 = int(np.argmin(np.abs(a - 0.0)))
    ib0 = int(np.argmin(np.abs(b - 0.0)))
    sec_a0 = P[ia0, :]
    sec_b0 = P[:, ib0]

    ax.plot(np.full(Nz, -5.5e-6), b, sec_a0, color="k", lw=1.0, alpha=0.95)
    ax.plot(a, np.full(Ny, 5.5e-6), sec_b0, color="k", lw=1.0, alpha=0.95)

    um = FuncFormatter(lambda v, pos: f"{v * 1e6:.1f}")
    ax.xaxis.set_major_formatter(um)
    ax.yaxis.set_major_formatter(um)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["pane_color"] = (1, 1, 1, 0.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Potential (mK)")
    ax.set_xlim(a.min(), a.max())
    ax.set_ylim(b.min(), b.max())
    ax.view_init(elev=20, azim=-35, roll=0.5)

    if rasterize:
        for coll in ax.collections:
            if coll.__class__.__name__ == "Poly3DCollection":
                coll.set_rasterized(True)
        ax.set_rasterization_zorder(0)

    ax.text2D(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top")
    return z_floor, dz


# ---------------- Combine into one figure ----------------
fig = plt.figure(figsize=(3.375, 5.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 0.3], hspace=0.3)

ax1 = fig.add_subplot(gs[0], projection="3d")
z_floor_a, dz_a = draw_panel(
    ax1,
    Yd_a,
    Zd_a,
    Pd_a,
    ylabel="Axial Position (µm)",
    xlabel="Radial Position (µm)",
    panel_label="(a)",
)

ax1.set_xlim(-5.5e-6, 5.5e-6)
ax1.set_ylim(-5.5e-6, 5.5e-6)
ax1.zaxis._axinfo["juggled"] = (1, 2, 0)

# θ-angle illustration
theta_ray = np.pi / 4
L_ray = 4e-6
z_theta = z_floor_a

ax1.plot([0, L_ray], [0, 0], [z_theta, z_theta], color="#000000", lw=1, zorder=50)
ax1.plot(
    [0, L_ray * np.cos(theta_ray)],
    [0, L_ray * np.sin(theta_ray)],
    [z_theta, z_theta],
    color="#000000",
    lw=1,
    zorder=50,
)

arc_r = 3e-6
theta_arc = np.linspace(0, theta_ray, 60)
ax1.plot(
    arc_r * np.cos(theta_arc),
    arc_r * np.sin(theta_arc),
    np.full_like(theta_arc, z_theta),
    color="k",
    ls="--",
    lw=1.4,
    zorder=60,
)

theta_mid = theta_ray / 2 - 0.19
ax1.text(
    1.2 * arc_r * np.cos(theta_mid),
    2.0 * arc_r * np.sin(theta_mid),
    z_theta,
    r"$\theta$",
    fontsize=10,
    ha="center",
    va="center",
    color="k",
    zorder=100,
)

# (b) Angular barrier height
plane_radial = pot[:, :, x0].astype(float)
plane_radial = prep_plane(plane_radial)

Theta = np.arctan2(Z_a, Y_a)
theta_vals = Theta.ravel()
pot_vals = plane_axial.ravel()

n_bins = 100
bins = np.linspace(-np.pi, np.pi, n_bins)
digitized = np.digitize(theta_vals, bins)

theta_max, pot_max = [], []
for i in range(1, len(bins)):
    mask = digitized == i
    if np.any(mask):
        theta_max.append(0.5 * (bins[i - 1] + bins[i]))
        pot_max.append(np.nanmax(pot_vals[mask]))

theta_max = np.array(theta_max)
pot_max = np.array(pot_max)

theta_wrap = np.concatenate((theta_max, theta_max[:1] + 2 * np.pi))
pot_wrap = np.concatenate((pot_max, pot_max[:1]))
cs = CubicSpline(theta_wrap, pot_wrap, bc_type="periodic")

theta_fine = np.linspace(-np.pi, np.pi, 100)
pot_smooth = cs(theta_fine)


def pi_formatter(x, pos):
    frac = x / np.pi
    if np.isclose(frac, 0):
        return "0"
    if np.isclose(frac, 1):
        return r"$\pi$"
    if np.isclose(frac, -1):
        return r"$-\pi$"
    if np.isclose(frac, 0.5):
        return r"$\pi/2$"
    if np.isclose(frac, -0.5):
        return r"$-\pi/2$"
    return f"{frac:.1f}$\pi$"


ax2 = fig.add_subplot(gs[1])
ax2.plot(theta_fine, np.maximum(0, pot_smooth), "-", lw=1, color="#A90749")
ax2.set_xlabel(r"Polar Angle, $\theta$ (rad)")
ax2.set_ylabel("Barrier Height (mK)")
ax2.set_ylim(-0.00, 0.22)
ax2.set_xlim(-np.pi, np.pi)
ax2.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
ax2.text(-3.1, float(np.nanmax(pot_max)) * 1.4, "(b)", va="top", ha="left")

fig.subplots_adjust(left=0.15, right=0.97, top=0.98, bottom=0.12, hspace=0.1)
plt.savefig("potental_plot.svg", dpi=600)
plt.show()