"""
Linear Programming Project
Part 1: Visual / 2D Simplex by hand (graphical)
Part 2: Higher-dimensional Simplex using a library API (scipy.optimize.linprog)
"""

import time
import statistics
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog 


# ==========================
# Part 1 – Visual Simplex in 2D
# ==========================

def objective_2d(x: float, y: float) -> float:
    """
    Objective function for the 2D example:
        Maximize z = 3x + 2y
    """
    return 3 * x + 2 * y


def part1_visual_simplex():
    """
    2-variable LP (precalculus-style geometric interpretation):

        Maximize:   z = 3x + 2y

        Subject to:
            x + y   <= 4
            x + 3y  <= 6
            x >= 0
            y >= 0

    This function:
      * Graphs the constraints.
      * Shades the feasible region.
      * Shows the corner points.
      * Shows a Simplex-style path along the corner points.
    """

    # --- Set up grid for drawing constraint lines ---
    x_vals = np.linspace(0, 6, 400)

    # Constraint lines: treat them as equalities for plotting
    y_line1 = 4 - x_vals           # x + y = 4
    y_line2 = (6 - x_vals) / 3.0   # x + 3y = 6

    fig, ax = plt.subplots()
    ax.set_title("Part 1 – 2D Linear Program and Simplex Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot the constraint lines (only where y >= 0)
    ax.plot(x_vals, y_line1, label="x + y ≤ 4")
    ax.plot(x_vals, y_line2, label="x + 3y ≤ 6")

    # Axes / bounds
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    # --- Feasible region polygon ---
    # By hand we find the intersection points:
    #   (0,0) from x=0, y=0
    #   (0,2) from x=0 and x+3y=6 → y=2
    #   (3,1) from solving x+y=4 and x+3y=6
    #   (4,0) from y=0 and x+y=4 → x=4
    feasible_polygon_x = [0, 0, 3, 4]
    feasible_polygon_y = [0, 2, 1, 0]

    ax.fill(
        feasible_polygon_x,
        feasible_polygon_y,
        alpha=0.3,
        label="Feasible region"
    )

    # --- Simplex-style path along the corner points ---
    # We manually choose a path that always improves the objective:
    #   P0 = (0, 0)
    #   P1 = (0, 2)
    #   P2 = (3, 1)
    #   P3 = (4, 0)  (optimal in this example)
    simplex_path: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.0, 2.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]

    # Plot the corner points and annotate them
    for i, (x, y) in enumerate(simplex_path):
        z_val = objective_2d(x, y)
        ax.scatter(x, y, s=50)
        ax.text(
            x + 0.05,
            y + 0.05,
            f"P{i} ({x:.0f}, {y:.0f})\nz={z_val:.0f}",
            fontsize=8,
        )

    # Draw arrows showing the movement of the Simplex method
    for i in range(len(simplex_path) - 1):
        x0, y0 = simplex_path[i]
        x1, y1 = simplex_path[i + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )

    ax.legend(loc="upper right")

    # Print the Simplex path numerically as well
    print("=== Part 1: 2D Simplex Path (corner points) ===")
    for i, (x, y) in enumerate(simplex_path):
        z_val = objective_2d(x, y)
        print(f"Step {i}: (x, y) = ({x}, {y}),  z = 3x + 2y = {z_val}")

    print("The final point P3 = (4, 0) is the optimal solution in this example.")
    print("You can see the Simplex path moving along the boundary of the feasible region.")

    plt.show()


# ==========================
# Part 2 – Library-based Simplex
# ==========================

def part2_high_dim_simplex_with_timing(num_runs: int = 30):
    """
    Higher-dimensional linear program solved using scipy.optimize.linprog.
    Also includes simple timing-based 'statistical analysis'.

    Example LP (4 variables → 4D problem):

        Maximize:
            z = 5x1 + 4x2 + 3x3 + 7x4

        Subject to:
            2x1 + 1x2 + 1x3 + 3x4 <= 50
            1x1 + 3x2 + 1x3 + 2x4 <= 60
            2x1 + 2x2 + 2x3 + 1x4 <= 40
            x1, x2, x3, x4 >= 0

    SciPy's linprog solves minimization problems, so we instead minimize:
        f = -z = -(5x1 + 4x2 + 3x3 + 7x4)
    """

    # Coefficients for minimizing f = c^T x
    # (negative of profit → solving max as min)
    c = np.array([-5.0, -4.0, -3.0, -7.0])

    # Inequality constraints A_ub x <= b_ub
    A_ub = np.array([
        [2.0, 1.0, 1.0, 3.0],
        [1.0, 3.0, 1.0, 2.0],
        [2.0, 2.0, 2.0, 1.0],
    ])
    b_ub = np.array([50.0, 60.0, 40.0])

    # Bounds for x_i >= 0 (no upper bounds → None)
    bounds = [(0, None), (0, None), (0, None), (0, None)]

    n_vars = len(c)  # dimension of the system (4 here)
    print("\n=== Part 2: High-dimensional Simplex with linprog ===")
    print(f"Number of decision variables (dimension of the system) = {n_vars}")

    # --- Timing analysis ---
    times: List[float] = []
    iterations: List[int] = []

    last_result = None

    for run in range(num_runs):
        t0 = time.perf_counter()
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",  # modern simplex/HiGHS solver
        )
        t1 = time.perf_counter()
        dt = t1 - t0

        times.append(dt)
        iterations.append(result.nit)
        last_result = result

    # Basic statistics on the run times
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.pstdev(times)

    print(f"\nTiming over {num_runs} runs:")
    print(f"  Average time   : {avg_time:.6f} seconds")
    print(f"  Fastest run    : {min_time:.6f} seconds")
    print(f"  Slowest run    : {max_time:.6f} seconds")
    print(f"  Time std dev   : {std_time:.6f} seconds")

    print("\nIterations (number of solver iterations per run):")
    print(f"  Mean iterations: {statistics.mean(iterations):.2f}")
    print(f"  Min iterations : {min(iterations)}")
    print(f"  Max iterations : {max(iterations)}")

    # Print final solution from the last run
    if last_result is None:
        print("Error: no result from linprog.")
        return

    if not last_result.success:
        print("\nSolver failed:")
        print(last_result.message)
        return

    x_opt = last_result.x
    max_profit = -last_result.fun  # negate because we minimized -z

    print("\nFinal solution (from last run):")
    for i, value in enumerate(x_opt, start=1):
        print(f"  x{i} = {value:.4f}")
    print(f"Maximum objective value z* = {max_profit:.4f}")


# ==========================
# Main entry point
# ==========================

def main():
    # ---- Part 1: 2D visual simplex ----
    part1_visual_simplex()

    # Pause so the user can close the plot window before Part 2 runs
    input("\nPress Enter to run Part 2 (library-based Simplex with timing)...\n")

    # ---- Part 2: High-dimensional simplex with linprog + timing ----
    part2_high_dim_simplex_with_timing(num_runs=30)


if __name__ == "__main__":
    main()
    