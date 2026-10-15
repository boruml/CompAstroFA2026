import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse

# Reading in FITS file #

hdul = fits.open('tic0010048801.fits')
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

# Defining Epoch #

mask = (times >= 1843) & (times <= 1855)
t = times[mask]
f = fluxes[mask]

# Fourier Transform #

def analyze_fourier(times, fluxes, keep):

    dt = np.diff(times)
    median_dt = np.median(dt)

    # Subtracting the mean so it does not dominate the spectrum
    mean_flux = np.mean(fluxes)
    centered_flux = fluxes - mean_flux

    # Only nonnegative frequencies are needed
    coefficients = np.fft.rfft(centered_flux)
    frequencies = np.fft.rfftfreq(len(centered_flux), d=median_dt)

    # Power is proportional to the squared coefficient magnitude.
    power = np.abs(coefficients) ** 2

    # Keeping the strongest nonzero-frequency coefficients
    n_nonzero = len(coefficients) - 1
    n_keep = min(keep, n_nonzero)

    selected = np.zeros_like(coefficients)
    if n_keep > 0:
        strongest = np.argsort(np.abs(coefficients[1:]) ** 2)[-n_keep:] + 1
        selected[strongest] = coefficients[strongest]

    # Reconstructing the signal and restoring its mean
    reconstructed = np.fft.irfft(selected, n=len(centered_flux))
    reconstructed += mean_flux

    return frequencies, power, reconstructed, n_keep


def main():
    parser = argparse.ArgumentParser(
        description="Fourier analysis of TIC 10048801 light curve."
    )

    parser.add_argument(
        "--keep", type=int, default=20,
        help="Number of nonzero-frequency coefficients to retain"
    )
    args = parser.parse_args()

    if args.keep < 0:
        parser.error("--keep must be nonnegative")

    frequencies, power, reconstructed, n_keep = analyze_fourier(
        t, f, args.keep
    )

    print(f"Time range: 1843 to 1855")
    print(f"Fourier coefficients retained: {n_keep}")

    # Plotting the light curve and its reconstruction.
    plt.figure(figsize=(10, 5))

    plt.plot(t, f, ".", markersize=3, label="Original")
    plt.plot(t, reconstructed, "-", linewidth=1.5, label=f"Reconstruction ({n_keep} coefficients)")

    plt.xlabel("Truncated Barycentric Julian Date")
    plt.ylabel("Normalized PDC Flux")
    plt.title("TIC 10048801 Light Curve")

    plt.legend()
    plt.tight_layout()


    # Plot power versus frequency, excluding the zero-frequency term.
    plt.figure(figsize=(10, 5))

    plt.plot(frequencies[1:], power[1:])

    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Power (squared flux units)")
    plt.title("Fourier Power Spectrum")
    plt.yscale("log")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
