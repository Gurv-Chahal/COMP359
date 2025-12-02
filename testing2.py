# timing helper
import time
# simple statistics like mean
import statistics
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# part 1:


# compute productivity score from counts of short and long tasks
def computeTaskScore(shortTasks: float, longTasks: float) -> float:
    return 4 * shortTasks + 3 * longTasks


# run the 2D visual example for short vs long tasks
def runTwoDimVisualExample() -> None:

    # x values from 0 - 6
    shortTaskValues = np.linspace(0, 6, 200)

    # time limit boundary
    timeLimitLongTasks = 6 - shortTaskValues
    # energy limit boundary
    energyLimitLongTasks = 8 - 2 * shortTaskValues

    # create the plot
    figure, axes = plt.subplots()
    axes.set_title("Part 1 – Short vs Long tasks (feasible region)")
    axes.set_xlabel("Short tasks")
    axes.set_ylabel("Long tasks")

    # draw time limit line
    axes.plot(shortTaskValues, timeLimitLongTasks, label="Time limit")
    # draw energy limit line
    axes.plot(shortTaskValues, energyLimitLongTasks, label="Energy limit")

    # x axis range
    axes.set_xlim(0, 6)
    # y axis range
    axes.set_ylim(0, 6)
    # horizontal axis
    axes.axhline(0, color="black", linewidth=1)
    # vertical axis
    axes.axvline(0, color="black", linewidth=1)

    # feasible region corner x values
    cornerShortValues = [0, 0, 2, 4]
    # feasible region corner y values
    cornerLongValues = [0, 6, 4, 0]

    # shade feasible region
    axes.fill(cornerShortValues, cornerLongValues, alpha=0.3, label="Feasible region")

    # sequence of visited corners
    simplexPathPoints: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.0, 6.0),
        (2.0, 4.0),
    ]

    print("Part 1: Simple 2D example (short vs long tasks)")
    print("We choose how many short and long tasks to do in a day.")
    print("The computer walks along corner points of the feasible region:\n")

    # loop over simplex path points
    for stepIndex, (shortCount, longCount) in enumerate(simplexPathPoints):
        # score at this corner
        currentScore = computeTaskScore(shortCount, longCount)
        # plot the corner
        axes.scatter(shortCount, longCount, s=50)
        # label corner with step and score
        axes.text(
            shortCount + 0.05,
            longCount + 0.05,
            f"Step {stepIndex}\n({shortCount:.0f}, {longCount:.0f})\nscore={currentScore:.0f}",
            fontsize=8,
        )
        # print step info
        print(f"  step {stepIndex}: short={shortCount}, long={longCount}, score={currentScore}")

    # loop over consecutive pairs of path points
    for stepIndex in range(len(simplexPathPoints) - 1):
        # starting corner
        startShort, startLong = simplexPathPoints[stepIndex]
        # next corner
        endShort, endLong = simplexPathPoints[stepIndex + 1]
        # arrow from start to next corner
        axes.annotate(
            "",
            xy=(endShort, endLong),
            xytext=(startShort, startLong),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )

    # show legend
    axes.legend(loc="upper right")

    # last corner on path
    bestShortTasks, bestLongTasks = simplexPathPoints[-1]
    # best score
    bestScore = computeTaskScore(bestShortTasks, bestLongTasks)
    print(f"\nFinal corner in the path: short={bestShortTasks}, long={bestLongTasks}")
    print(f"This corner has the highest score = {bestScore}.")
    print("This mimics how Simplex moves along corners in 2D.\n")

    # open the plot window
    plt.show()


# run the 3 variable solver example several times and report results
def runThreeDimSolverExample(runCount: int = 20) -> None:

    # objective coefficients for A, B, C
    objectiveCoefficients = [-5.0, -4.0, -3.0]

    # inequality constraints
    constraintMatrix = [
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0],
    ]
    # constraint right hand sides
    constraintLimits = [30.0, 40.0, 30.0]

    # variables must be non negative
    variableBounds = [(0, None), (0, None), (0, None)]

    # number of decision variables
    variableCount = len(objectiveCoefficients)
    print("\n=== Part 2: 3 variable problem using a solver library ===")
    print(f"Number of decision variables (dimension of the system) = {variableCount}")
    print("Variables are: A, B, C (three different products).")

    # store solver run times
    runTimes: List[float] = []
    # last solver output
    lastSolverResult = None

    # repeat solver call runCount times
    for runIndex in range(runCount):
        # start timer
        startTime = time.perf_counter()
        # call linear programming solver
        solverResult = linprog(
            objectiveCoefficients,
            A_ub=constraintMatrix,
            b_ub=constraintLimits,
            bounds=variableBounds,
            method="highs",
        )
        # end timer
        endTime = time.perf_counter()
        # record elapsed time
        runTimes.append(endTime - startTime)
        # keep last result
        lastSolverResult = solverResult

    # check if solver failed
    if lastSolverResult is None or not lastSolverResult.success:
        print("\nSolver failed or returned no result.")
        if lastSolverResult is not None:
            print("Message:", lastSolverResult.message)
        return

    # convert min value to max score
    bestScore = -lastSolverResult.fun
    # optimal values for A, B, C
    valueA, valueB, valueC = lastSolverResult.x

    print("\nBest mix of products from the solver (last run):")
    print(f"  A = {valueA:.2f}")
    print(f"  B = {valueB:.2f}")
    print(f"  C = {valueC:.2f}")
    print(f"Resulting total score (profit/points) ≈ {bestScore:.2f}")

    # average solver time
    averageTime = statistics.mean(runTimes)
    # fastest solver time
    fastestTime = min(runTimes)
    # slowest solver time
    slowestTime = max(runTimes)

    print(f"\nTiming over {runCount} runs:")
    print(f"  Average time : {averageTime * 1000:.3f} ms")
    print(f"  Fastest run  : {fastestTime * 1000:.3f} ms")
    print(f"  Slowest run  : {slowestTime * 1000:.3f} ms")


# main driver for both parts
def main() -> None:

    # run part 1
    runTwoDimVisualExample()

    # pause before part 2
    input("\nPress Enter to run Part 2 (3 variable solver example)...\n")

    # run part 2 with 20 solver runs
    runThreeDimSolverExample(runCount=20)


if __name__ == "__main__":
    main()
