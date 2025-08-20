"""
problems/p01_iterative_linear.py

Solve Problem 1 from the exercise list using the Gauss–Seidel implementation.

Assumptions:
 - Coefficient matrix and RHS are taken from Problem 1 of "Lista de exercícios I - TEB 062 - 2025".
 - Stopping criterion: Criterion C (relative per-component), tolerance 1e-2, as requested in the list.
"""

from __future__ import annotations
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel

def main() -> None:
    # Coefficient matrix A and right-hand side b from Problem 1.
    matrix = np.array([
        [5.1, 1.0, 1.0],
        [3.0, 4.0, 1.0],
        [3.0, 3.0, 6.0],
    ], dtype=float)

    right_hand_side = np.array([5.0, 6.0, 0.0], dtype=float)

    # initial guess (from the exercise)
    initial_guess = np.array([2.0, 1.0, 0.0], dtype=float)

    tolerance = 1e-2

    # Solve using Gauss–Seidel with criterion C and tolerance 1e-2 (0.01)
    report = gauss_seidel(
        matrix=matrix,
        right_hand_side=right_hand_side,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=10_000,
        stopping_criterion="C",
    )

    print()
    print("Exercise 1 - Gauss-Seidel Method")
    print("Coefficient matrix (A):")
    print("A =\n", matrix)
    print()
    print("Right-hand side (b):")
    print("b =", right_hand_side)
    print()
    print("Initial guess:")
    print("x\u207D\u2070\u207E =", initial_guess)
    print()
    print(f"Stopping rule: Criterion C max[(|xi^(k) − xi^(k−1) / xi^(k)| < ε)] with ε = {tolerance:g}")
    print()
    for iteration, solution, absolute_error, relative_error, residual_inf in report.history:
        print(f"Iteration: {iteration}")
        print(f"Solution: {solution}")
        print(f"Absolute error: {absolute_error:.6g}")
        print(f"Relative error: {relative_error:.6g}")
        print(f"Residual (infinity norm): {residual_inf:.6g}")
        print()
    print(f"Solution (approx.): {report.solution}")
    print(f"Iterations performed: {report.iterations}")
    print(f"Final absolute error (criterion B): {report.absolute_error:.6g}")
    print(f"Final relative error (criterion C): {report.relative_error:.6g}")
    print(f"Final residual (infinity norm): {report.residual_infinity_norm:.6g}")

if __name__ == "__main__":
    main()
