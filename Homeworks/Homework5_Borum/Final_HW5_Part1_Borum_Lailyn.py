import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import photon_functions as phf

def main():

    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation of a photon scattering through a stellar slab."
    )

    parser.add_argument(
        "--density",
        type=float,
        default=1e20,
        help="Electron density in cm^-3"
    )

    parser.add_argument(
        "--width",
        type=float,
        default=1000.0,
        help="Slab width in meters"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=6.652e-25,
        help="Thompson cross-section in cm^2"
    )

    parser.add_argument(
        "--max-scatter",
        type=int,
        default=100000,
        help="Maximum number of scattering events"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="photon_path.gif",
        help="Output animation filename (.gif or .mp4)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Animation frames per second"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random number generator seed"
    )

    args = parser.parse_args()

    if args.density <= 0 or args.sigma <= 0:
        parser.error("Density and cross-section must be positive.")

    if args.width <= 0 or args.max_scatter <= 0 or args.fps <= 0:
        parser.error("Width, max-scatter and fps must be positive.")

    # Convert mean free path from cm to m.
    mean_free_path = 1.0 / (args.density * args.sigma)
    mean_free_path /= 100.0

    optical_depth = args.width / mean_free_path

    print("\n--- Photon Scattering Simulation ---")
    print(f"Electron density: {args.density:.3e} cm^-3")
    print(f"Thomson cross-section: {args.sigma:.3e} cm^2")
    print(f"Mean free path: {mean_free_path:.2f} m")
    print(f"Slab width: {args.width:.2f} m")
    print(f"Optical depth: {optical_depth:.3f}")

    rng = np.random.default_rng(args.seed)

    positions, status, n_scatter, total_distance = (
        phf.simulate_photon(
            args.density,
            args.sigma,
            args.width,
            args.max_scatter,
            rng
        )
    )

    print("\n--- Simulation Results ---")
    print(f"Photon outcome: {status}")
    print(f"Number of scatterings: {n_scatter}")
    print(f"Total distance traveled: {total_distance:.2f} m")

    if status == "max_scatter":
        print("Warning: Maximum scattering limit reached.")

    phf.animate_photon(
        positions,
        args.width,
        args.output,
        args.fps
    )


if __name__ == "__main__":
    main()