"""
problems/p09_iterative_linear.py

Problem 9 — Continuous process with recycle (iterative linear system).
Solve using Gauss–Seidel with stopping criterion C (relative, per-component).
Final report rounds variables to two decimals, as requested in the statement.

System written as Ax = b:
    x - 0.6 y - 0.2 z = 1.0
   -0.4 x + y   - 0.1 z = 0.5
   -0.3 x - 0.4 y + 1.0 z = 0.2
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel


def main() -> None:
    coefficient_matrix = np.array([
        [ 1.0, -0.6, -0.2],
        [-0.4,  1.0, -0.1],
        [-0.3, -0.4,  1.0],
    ], dtype=float)

    right_hand_side = np.array([1.0, 0.5, 0.2], dtype=float)

    initial_guess: Sequence[float] = (0.0, 0.0, 0.0)

    # Stricter tolerance to ensure robust rounding to two decimals
    tolerance_for_criterion_c = 1e-4
    maximum_number_of_iterations = 10_000

    print("=== Problem 9 — Continuous process with recycle (Gauss–Seidel) ===")
    print("Function: linear SEAL from statement")
    print("Initial guess:", tuple(f"{value:.3f}" for value in initial_guess))
    print("Stopping criterion: C (relative per-component)")
    print("ε (tolerance for C):", tolerance_for_criterion_c, " | Final values rounded to 2 decimals\n")

    report = gauss_seidel(
        matrix=coefficient_matrix,
        right_hand_side=right_hand_side,
        initial_guess=np.asarray(initial_guess, dtype=float),
        tolerance=tolerance_for_criterion_c,
        max_iterations=maximum_number_of_iterations,
        stopping_criterion="C",
    )

    rounded_solution = np.round(report.solution, 2)
    residual_at_rounded_solution = coefficient_matrix @ rounded_solution - right_hand_side

    print("Converged:", report.iterations < maximum_number_of_iterations)
    print("iterations:", report.iterations)
    print(
        "x* ≈ "
        f"x={rounded_solution[0]:.2f}, y={rounded_solution[1]:.2f}, z={rounded_solution[2]:.2f}"
    )
    print(
        "Ax* - b at rounded x*: ",
        tuple(f"{component:.3e}" if component != 0 else "0" for component in residual_at_rounded_solution),
    )
    print("Final vector (2 decimals):", tuple(f"{value:.2f}" for value in rounded_solution))
    print()

    print(
        "iteration |           x |           y |           z |   absolute_error |   relative_error | residual_infinity_norm"
    )
    print("-" * 118)

    # history elements: (iteration, solution, absolute_error, relative_error, residual_infinity_norm)
    for (
        iteration_number,
        solution_vector,
        absolute_error_value,
        relative_error_value,
        residual_infinity_norm_value,
    ) in report.history:
        current_x, current_y, current_z = solution_vector
        print(
            f"{iteration_number:9d} | "
            f"{current_x:12.6f} | {current_y:12.6f} | {current_z:12.6f} | "
            f"{absolute_error_value:16.6e} | {relative_error_value:16.6e} | {residual_infinity_norm_value:22.6e}"
        )


if __name__ == "__main__":
    main()
