import argparse
import numpy
from scipy.optimize import newton
import astropy.constants as const
#from astropy.constants import G,

# Constants #

M = 5.972e24  # Earth mass kg
m = 7.348e22  # Moon mass kg
R = 3.844e8  # Distance from Earth to Moon m
G = 6.674e-11  # m3 kg-1 s-2
omega = 2.662e-6  # s-1


def equation(r, G, M, m, R, omega):
  # f(r) = GM/r^2 - Gm/(R - r)^2 - omega^2 * r = 0
  return (G * M / (r**2)) - (G * m / ((R - r) ** 2)) - (omega**2) * r


def derivative(r, G, M, m, R, omega):
  # Derivative f'(r) with respect to r
  return (-2 * G * M / (r**3)) - (2 * G * m / ((R - r) ** 3)) - (omega**2)


def main():
  parser = argparse.ArgumentParser(
      description='Solve for the Earth-Moon L1 Lagrange point distance.'
  )
  parser.add_argument(
      '--r_guess',
      type=float,
      default=3.26e8,
      help='Initial guess for r in meters (default: 3.26e8)',
  )
  args = parser.parse_args()

  # Solve using Newton's method
  r_l1 = newton(
      func=lambda r: equation(r, G, M, m, R, omega),
      x0=args.r_guess,
      fprime=lambda r: derivative(r, G, M, m, R, omega),
      tol=1e-8,
  )

  print(f'Distance r to L1 point: {r_l1:.6e} m')
  print(f'Accurate to 4+ sig figs: {r_l1:.4f} m')


if __name__ == '__main__':
  main()