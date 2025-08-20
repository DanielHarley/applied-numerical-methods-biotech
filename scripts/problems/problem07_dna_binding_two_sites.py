from __future__ import annotations
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel


def main() -> None:
    # Parameters from the statement
    K1 = 0.5
    K2 = 0.2

    # Build linear system A u = b  with u = (x, y)
    matrix = np.array([
        [1.0, 1.0],
        [1.0 / K1, -1.0 / K2],
    ], dtype=float)
    right_hand_side = np.array([1.0, 0.0], dtype=float)

    # Gauss–Seidel settings for this problem (kept in the problem file, not in the method)
    initial_guess = np.zeros(2, dtype=float)
    tolerance = 1e-8          # ε for Criterion C (relative, per-component)
    max_iterations = 10_000

    print("Problem 7 — Two-site binding equilibrium (Gauss–Seidel)")
    print("System: x + y = 1  and  x/K1 = y/K2")
    print(f"Parameters: K1 = {K1}, K2 = {K2}")
    print("Stopping: Criterion C (relative per-component), ε = 1e-8")
    print("Results rounded to 3 decimals.\n")

    result = gauss_seidel(
        matrix=matrix,
        right_hand_side=right_hand_side,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
        stopping_criterion="C",
    )

    solution = np.array(getattr(result, "solution", result), dtype=float)
    x_value, y_value = float(solution[0]), float(solution[1])

    x_round = round(x_value, 3)
    y_round = round(y_value, 3)

    print(f"Converged: {getattr(result, 'converged', '—')}, iterations: {getattr(result, 'iterations', '—')}")
    print(f"x (site 1) ≈ {x_round:.3f}")
    print(f"y (site 2) ≈ {y_round:.3f}")
    print(f"Check: x + y ≈ {(x_round + y_round):.3f}")
    print(f"Check: x/K1 ≈ {(x_round / K1):.3f} | y/K2 ≈ {(y_round / K2):.3f}")


if __name__ == "__main__":
    main()