import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

## Using the Trapezoidal Rule ##

# Defining integrand function #
def f(t):
    return math.exp(-t ** 2)

# Defining trapezoidal rule function #
def trapezoidal_integral(x, num_slices):
    if x == 0:
        return 0.0

    h = x / num_slices

    integral_sum = 0.5 * f(0) + 0.5 * f(x)

    for k in range(1, num_slices):
        integral_sum += f(k * h)

    return integral_sum * h

def main():

    parser = argparse.ArgumentParser(
        description="Calculate the numerical integral E(x) = ∫[0 to x] e^(-t^2) dt using the Trapezoidal Rule."
    )
    parser.add_argument(
        '-s', '--slices',
        type=int,
        default=1000,
        help="Number of slices to use for the numerical integration (default: 1000)."
    )
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help="Generate and show a plot of E(x) versus x."
    )
    args = parser.parse_args()

    x_values = [i / 10.0 for i in range(0, 31)]
    e_values = []

    print(f"{'x':<5} | {'E(x)':<10}")
    print("-" * 18)

    for x in x_values:
        result = trapezoidal_integral(x, args.slices)
        e_values.append(result)
        print(f"{x:<5.1f} | {result:<10.6f}")

    # Plotting the graph if requested
    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(x_values, e_values, marker='o', color='b', linestyle='-', label=r'$E(x) = \int_0^x e^{-t^2} dt$')
        plt.title('Numerical Evaluation of $E(x)$')
        plt.xlabel('x')
        plt.ylabel('E(x)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()