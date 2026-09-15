import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Calculating the time it takes for a ball to hit the ground.")
parser.add_argument("height", type=float, help="Initial height in meters.")
parser.add_argument("-g", "--gravity", type=float, default=9.81, help="Acceleration due to gravity in m/s^2 (default: 9.81).")

args = parser.parse_args()

if args.height < 0:
    print("Error: Height cannot be negative.")


if args.gravity <= 0:
    print("Error: Gravity must be greater than zero.")


time = np.sqrt((2 * args.height) / args.gravity)

print(f"Height: {args.height} meters")
print(f"Gravity: {args.gravity} m/s^2")
print(f"Time to hit the ground: {time:.3f} seconds")