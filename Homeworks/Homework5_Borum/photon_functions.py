import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## Includes the funtions simulate_photon and animate_photon. ##

def simulate_photon(n_e, sigma, width, max_scatter, rng):

    # convert mean free path from cm to meters.
    mean_free_path = 1.0 / (n_e * sigma) / 100.0

    x, y = 0.0, 0.0
    dx, dy = 1.0, 0.0  # initially traveling into the slab.

    positions = [(x, y)]
    n_scatter = 0
    total_distance = 0.0

    while n_scatter < max_scatter:

        free_path = -mean_free_path * np.log(rng.random())

        # determine distance to the boundary along current direction.
        if dx > 0:
            boundary_distance = (width - x) / dx
            boundary = "right"
        elif dx < 0:
            boundary_distance = -x / dx
            boundary = "left"
        else:
            boundary_distance = np.inf
            boundary = None

        if boundary_distance <= free_path:
            x += boundary_distance * dx
            y += boundary_distance * dy
            total_distance += boundary_distance
            positions.append((x, y))

            return np.array(positions), boundary, n_scatter, total_distance

        # travel to the next scattering event
        x += free_path * dx
        y += free_path * dy
        total_distance += free_path
        positions.append((x, y))

        n_scatter += 1

        # choose a new isotropic direction after scattering
        mu = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)

        dx = mu
        dy = np.sqrt(1.0 - mu ** 2) * np.cos(phi)

    return np.array(positions), "max_scatter", n_scatter, total_distance


def animate_photon(positions, width, output, fps):

    fig, ax = plt.subplots(figsize=(10, 5))

    # slab boundaries
    ax.axvline(0, color="black", linewidth=2)
    ax.axvline(width, color="black", linewidth=2)

    # shade the slab
    ax.axvspan(0, width, color="lightblue", alpha=0.15)

    # photon trajectory
    line, = ax.plot([], [], color="red", linewidth=1.5)
    photon, = ax.plot([], [], "o", color="darkred",
                      markersize=7)

    ax.set_xlim(-0.05 * width, 1.05 * width)

    y_min = min(positions[:, 1])
    y_max = max(positions[:, 1])
    y_range = max(y_max - y_min, 10.0)

    ax.set_ylim(
        y_min - 0.1 * y_range,
        y_max + 0.1 * y_range
    )

    ax.set_xlabel("Position through slab (m)")
    ax.set_ylabel("Transverse displacement (m)")
    ax.set_title("Monte Carlo Photon Scattering")

    ax.grid(alpha=0.3)

    ax.text(
        0.02, 0.95, "Entrance",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.text(
        0.98, 0.95, "Exit",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top"
    )

    def update(frame):
        line.set_data(
            positions[:frame + 1, 0],
            positions[:frame + 1, 1]
        )

        photon.set_data(
            [positions[frame, 0]],
            [positions[frame, 1]]
        )

        return line, photon

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=True,
        repeat=False
    )

    animation.save(output, fps=fps)

    print(f"Animation saved to: {output}")

    plt.show()