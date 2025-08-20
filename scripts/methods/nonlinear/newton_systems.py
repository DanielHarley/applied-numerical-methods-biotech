from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Literal
import numpy as np


@dataclass
class NewtonSystemStep:
    iteration: int
    current_estimate: np.ndarray
    function_value: np.ndarray
    stopping_value: float


@dataclass
class NewtonSystemResult:
    converged: bool
    iterations: int
    approximate_solution: np.ndarray
    function_at_solution: np.ndarray
    history: List[NewtonSystemStep]


def newton_system(
    function: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    tolerance: float,
    max_iterations: int,
    stopping_criterion: Literal["C", "B"] = "C",
) -> NewtonSystemResult:
    """
    Perform Newton's method for SEANL.

    Parameters
    ----------
    function : Callable
        Maps x (ℝ^n) to F(x) (ℝ^n).
    jacobian : Callable
        Maps x (ℝ^n) to J(x) (n×n).
    initial_guess : np.ndarray
        Starting vector x^{0}.
    tolerance : float
        Tolerance ε (not hard-coded; must be provided by the problem).
    max_iterations : int
        Maximum number of Newton iterations.
    stopping_criterion : {"C", "B"}
        "C": stop when max_i |Δx_i| < ε;
        "B": stop when max_i |F_i(x_k)| < ε.

    Returns
    -------
    NewtonSystemResult
        Convergence flag, number of iterations performed, last approximation,
        F evaluated at the last approximation, and the full iteration history.
    """
    x_current = np.array(initial_guess, dtype=float)
    history: List[NewtonSystemStep] = []

    for k in range(1, max_iterations + 1):
        F_val = function(x_current)
        J_val = jacobian(x_current)

        # Solve J(x_k) * v_k = -F(x_k)
        try:
            step_direction = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError as exc:
            # Singular Jacobian or ill-conditioned solve -> fail fast
            history.append(
                NewtonSystemStep(
                    iteration=k,
                    current_estimate=x_current.copy(),
                    function_value=F_val.copy(),
                    stopping_value=np.inf,
                )
            )
            return NewtonSystemResult(
                converged=False,
                iterations=k,
                approximate_solution=x_current,
                function_at_solution=F_val,
                history=history,
            )

        x_next = x_current + step_direction

        # Compute stopping value according to the selected criterion.
        if stopping_criterion == "C":
            stopping_value = float(np.max(np.abs(step_direction)))
        elif stopping_criterion == "B":
            stopping_value = float(np.max(np.abs(function(x_next))))
        else:
            raise ValueError("Unsupported stopping_criterion. Use 'C' or 'B'.")

        # Log AFTER computing x^{k+1} and the stopping metric of this iteration.
        history.append(
            NewtonSystemStep(
                iteration=k,
                current_estimate=x_next.copy(),
                function_value=function(x_next).copy(),
                stopping_value=stopping_value,
            )
        )

        # Check stopping condition.
        if stopping_value < tolerance:
            return NewtonSystemResult(
                converged=True,
                iterations=k,
                approximate_solution=x_next,
                function_at_solution=function(x_next),
                history=history,
            )

        x_current = x_next

    # Did not satisfy tolerance within max_iterations
    return NewtonSystemResult(
        converged=False,
        iterations=max_iterations,
        approximate_solution=x_current,
        function_at_solution=function(x_current),
        history=history,
    )