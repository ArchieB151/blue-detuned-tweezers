import numpy as np
import matplotlib.pyplot as plt

def overlay_contours(heatmap_BH: np.ndarray, heatmap_TD: np.ndarray, levels_BH: list[float], levels_TD: list[float]) -> None:

    ny, nx = heatmap_BH.shape
    x = np.linspace(0, 12.5, nx)
    y = np.linspace(0, 12.5, ny)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 8))
    c1 = plt.contour(X, Y, heatmap_BH, levels=levels_BH, cmap="viridis")
    _ = plt.contour(X, Y, heatmap_TD, levels=levels_TD, cmap="plasma")
    plt.clabel(c1, inline=True, fontsize=8, fmt="%.2f")



    plt.xlabel("Power red (mW)")
    plt.ylabel("Power blue (mW)")
    plt.title("Barrier Height and Trap Depth Contours")
    plt.grid(False)
    plt.legend(frameon=False)
    plt.show()