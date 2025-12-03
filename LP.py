import time
import statistics
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog  # Wrapper for the HiGHS linear optimization C++ library

def score_2d(short_tasks: float, long_tasks: float) -> float:
    """
    The Objective Function (Z).
    
    Goal: Maximize Z = 4x + 3y
    where x = short_tasks, y = long_tasks.
    """
    return 4 * short_tasks + 3 * long_tasks


def part1_visual_simplex():
    """
    Part 1: Geometric Interpretation and Visualization of the Simplex Algorithm.
    
    Demonstrates the fundamental theorem of linear programming:
    the optimal solution lies at a vertex of the feasible region.
    
    System of linear inequalities and bounds:
    1. x + y ≤ 6 — time resource limit
    2. 2x + y ≤ 8 — energy resource limit
    3. x ≥ 0, y ≥ 0 — nonnegativity
    
    This module plots the feasible region and visualizes a traversal along vertices,
    analogous to Simplex pivot steps between adjacent corners.
    """


    # 1. Setup Grid for Plotting
    # Create a vector of x values from 0 to 6 to generate constraint lines
    x_vals = np.linspace(0, 6, 200)

    # 2. Defines Constraints as functions of y (y = mx + c) for plotting
    # Constraint 1, time: x + y = 6  =>  y = -x + 6
    long_from_time_limit = 6 - x_vals
    
    # Constraint 2, energy: 2x + y = 8 => y = -2x + 8
    long_from_energy_limit = 8 - 2 * x_vals

    # Initializes Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Feasible Region & Simplex Traversal")
    ax.set_xlabel("Decision Variable x (Short Tasks)")
    ax.set_ylabel("Decision Variable y (Long Tasks)")

    # 3. Plots Constraint Boundaries
    ax.plot(x_vals, long_from_time_limit, label="Constraint: Time (x + y ≤ 6)", color='blue')
    ax.plot(x_vals, long_from_energy_limit, label="Constraint: Energy (2x + y ≤ 8)", color='red')

    # Sets quadrant limits, Non-negativity constraints x>=0, y>=0
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axhline(0, color='black', linewidth=1) # x-axis
    ax.axvline(0, color='black', linewidth=1) # y-axis

    # 4. Defines the Feasible Region
    # The valid solution space is the intersection of all half-planes defined by inequalities.
    # Vertices calculated via system of equations:
    # Origin: (0,0)
    # Y-Intercept, Time bound: (0,6) - Valid: 2(0)+6=6 <= 8
    # Intersection, Possibly optimal: 6-x = 8-2x => x=2, y=4
    # X-Intercept, Energy bound: (4,0) - Valid: 4+0=4 <= 6
    
    feasible_x = [0, 0, 2, 4]
    feasible_y = [0, 6, 4, 0]

    # Fill the polygon representing the solution space
    ax.fill(feasible_x, feasible_y, color='green', alpha=0.2, label="Feasible Region")

    # 5. Simulates Simplex Algorithm 
    # The algorithm moves from one vertex to an adjacent vertex if the objective function improves.
    # Path: Origin (0,0) -> Edge -> (0,6) -> Edge -> (2,4): Global Max
    simplex_path: List[Tuple[float, float]] = [
        (0.0, 0.0),  # Initial basic feasible solution
        (0.0, 6.0),  # Pivot 1
        (2.0, 4.0),  # Pivot 2 (Optimal)
    ]

    print("-" * 50)
    print("Part 1: Simplex Vertex Traversal")
    print("-" * 50)

    # Plot vertices and calculate Objective Function Z at each step
    for i, (s, l) in enumerate(simplex_path):
        z_val = score_2d(s, l)
        ax.scatter(s, l, s=100, zorder=5, color='black')
        
        # Annotation for clarity
        ax.text(
            s + 0.1, l + 0.1,
            f"Step {i}\n({s},{l})\nZ={z_val:.0f}",
            fontsize=9, weight='bold'
        )
        print(f"Iteration {i}: x={s}, y={l} | Objective Z={z_val}")

    # Draw directed graph edges representing pivots
    for i in range(len(simplex_path) - 1):
        s0, l0 = simplex_path[i]
        s1, l1 = simplex_path[i + 1]
        ax.annotate(
            "", xy=(s1, l1), xytext=(s0, l0),
            arrowprops=dict(arrowstyle="->", linewidth=2, color='purple')
        )

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    print("\n[System] Plot generation complete.")
    plt.show()


def part2_api_example(num_runs: int = 20):
    """
    Part 2: Black-box Solver Integration & Performance Benchmarking.
    
    Uses scipy.optimize.linprog to solve a higher-dimensional
    problem (3 variables). Includes statistical analysis of runtime performance.
    
    Problem Standard Form:
    Maximize Z = 5A + 4B + 3C
    Subject to:
      A + B + C    <= 30: Constraint 1
      2A + B       <= 40: Constraint 2
      B + 2C       <= 30: Constraint 3
      A, B, C      >= 0
    """
    
    # 1. Transforms Objective for Minimization Solver
    # Most LP solvers standard form is minimizing.
    # To maximize f(x), we minimize -f(x).
    # Original: 5A + 4B + 3C  -->  Input: -5A - 4B - 3C
    c = [-5.0, -4.0, -3.0] 

    # 2. Defines Constraints Matrix: A_ub and Vector: b_ub
    # Form: A_ub @ x <= b_ub
    # Row 0: 1A + 1B + 1C <= 30
    # Row 1: 2A + 1B + 0C <= 40
    # Row 2: 0A + 1B + 2C <= 30
    A_ub = [
        [1.0, 1.0, 1.0], 
        [2.0, 1.0, 0.0], 
        [0.0, 1.0, 2.0], 
    ]
    b_ub = [30.0, 40.0, 30.0]

    # Bounds for decision variables: 0 to Infinity
    bounds = [(0, None), (0, None), (0, None)]

    print("\n" + "-" * 50)
    print(f"Part 2: Statistical Benchmarking (N={num_runs})")
    print("-" * 50)

    times: List[float] = []
    last_result = None

    # 3. Execution Loop
    for run in range(num_runs):
        # perf_counter provides high resolution clock for measuring execution time
        t0 = time.perf_counter()
        
        # Invoke the solver
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs", 
        )
        
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_result = result

    # 4. Error Handling
    if last_result is None or not last_result.success:
        print("Runtime Error: Solver failed to converge.")
        if last_result: print(f"Message: {last_result.message}")
        return

    # 5. Result Interpretation
    # Negate the result again to return to maximization domain
    best_score = -last_result.fun
    A_opt, B_opt, C_opt = last_result.x

    print(f"Optimal Vector: [A={A_opt:.2f}, B={B_opt:.2f}, C={C_opt:.2f}]")
    print(f"Optimal Objective Value: {best_score:.2f}")

    # 6. Statistical Analysis
    # Convert seconds to milliseconds for readability
    avg_time = statistics.mean(times) * 1000
    stdev_time = statistics.stdev(times) * 1000
    fastest = min(times) * 1000
    slowest = max(times) * 1000

    print(f"\nPerformance Metrics (ms):")
    print(f"  Mean Runtime: {avg_time:.3f} ms")
    print(f"  Std Dev:      {stdev_time:.3f} ms (Consistency)")
    print(f"  Min/Max:      {fastest:.3f} ms / {slowest:.3f} ms")

if __name__ == "__main__":
    part1_visual_simplex()
    part2_api_example()