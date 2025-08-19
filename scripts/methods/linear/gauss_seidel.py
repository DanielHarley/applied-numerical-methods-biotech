from __future__ import annotations
from typing import Literal
import numpy as np
from dataclasses import dataclass

@dataclass
class GaussSeidelReport:
    solution: np.ndarray            # found solution vector
    iterations: int                 # iterations performed
    absolute_error: float           # ||x^{k} - x^{k-1}||_inf
    relative_error: float           # max_i |(x_i^{k} - x_i^{k-1}) / x_i^{k}|
    residual_infinity_norm: float   # ||b - A x||_inf
    history: list                   # [(iteration, solution, absolute_error, relative_error, residual_infinity_norm)]


def gauss_seidel(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    initial_guess: np.ndarray | None = None,
    tolerance: float = 1e-2,
    max_iterations: int = 10_000,
    stopping_criterion: Literal["B", "C"] = "C",
) -> GaussSeidelReport:
    """
    Solve Ax = b via Gauss–Seidel.

    Parameters
    ----------
    matrix : (n,n) ndarray
        Coefficient matrix A.
    right_hand_side : (n,) ndarray
        Right-hand side vector b.
    initial_guess: (n,) ndarray (initial guess). If None, uses zeros.
    tolerance: tolerance for stopping (Criterion C by default).
    max_iterations: maximum number of iterations.
    stopping_criterion: 'B' for absolute inf-norm (||x^k - x^{k-1}||_inf),
                        'C' for relative per-component criterion.

    Returns
    -------
    GaussSeidelReport dataclass with solution, iterations, errors and history.
    """
    matrix = np.array(matrix, dtype=float)
    right_hand_side = np.array(right_hand_side, dtype=float)
    number_of_equations = matrix.shape[0]
    assert matrix.shape == (number_of_equations, number_of_equations) and right_hand_side.shape == (number_of_equations,), "Invalid dimensions for matrix and right-hand side."

    if initial_guess is None:
        solution = np.zeros(number_of_equations, dtype=float)
    else:
        solution = np.array(initial_guess, dtype=float)

    # light checks for robustness
    if np.any(np.isclose(np.diag(matrix), 0.0)):
        raise ZeroDivisionError("There are zero diagonal element(s) in the coefficient matrix.")

    # simple check about diagonal dominance (sufficient but not necessary for convergence)
    is_row_diagonally_dominant = (
        np.abs(np.diag(matrix)) > (np.sum(np.abs(matrix), axis=1) - np.abs(np.diag(matrix)))
    )
    if not np.all(is_row_diagonally_dominant):
        # Does not block execution; just a warning: it may still converge.
        pass

    history = []
    absolute_error = np.inf
    for iteration in range(1, max_iterations + 1):
        previous_solution = solution.copy()

        # sweep rows applying the Gauss–Seidel update
        for i in range(number_of_equations):
            sum_lower = matrix[i, :i] @ solution[:i]            # contribution from already-updated entries
            sum_upper = matrix[i, i+1:] @ previous_solution[i+1:]  # contribution from not-yet-updated entries
            solution[i] = (right_hand_side[i] - sum_lower - sum_upper) / matrix[i, i]

        # metrics
        difference = solution - previous_solution
        absolute_error = np.max(np.abs(difference))                       # criterion B (inf norm)
        # avoid division by zero: where solution == 0, use denominator 1 to avoid blow-up
        denominator = np.where(np.abs(solution) > 0, np.abs(solution), 1.0)
        relative_error = np.max(np.abs(difference) / denominator)          # criterion C
        residual = right_hand_side - matrix @ solution
        residual_infinity_norm = float(np.max(np.abs(residual)))
        history.append((iteration, solution.copy(), float(absolute_error), float(relative_error), residual_infinity_norm))

        criterion = stopping_criterion.upper()
        if (criterion == "B" and absolute_error < tolerance) or (criterion == "C" and relative_error < tolerance):
            return GaussSeidelReport(
                solution=solution,
                iterations=iteration,
                absolute_error=float(absolute_error),
                relative_error=float(relative_error),
                residual_infinity_norm=residual_infinity_norm,
                history=history,
            )

    # reached here without meeting tolerance
    return GaussSeidelReport(
        solution=solution,
        iterations=max_iterations,
        absolute_error=float(absolute_error),
        relative_error=float(relative_error),
        residual_infinity_norm=residual_infinity_norm,
        history=history,
    )