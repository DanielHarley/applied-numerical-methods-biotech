from __future__ import annotations
import numpy as np
from scripts.methods.nonlinear.newton_systems import newton_system

def function(vector: np.ndarray) -> np.ndarray:
    """F(x) for Exercise 6(b)."""
    x1, x2 = vector
    f1 = x2 - np.cosh(x1 / 2.0)
    f2 = 9.0 * x1**2 + 25.0 * x2**2 - 225.0
    return np.array([f1, f2], dtype=float)


def jacobian(vector: np.ndarray) -> np.ndarray:
    """Jacobian J(x) for Exercise 6(b)."""
    x1, x2 = vector
    # ∂f1/∂x1 = - (1/2) * sinh(x1/2);  ∂f1/∂x2 = 1
    # ∂f2/∂x1 = 18*x1;                 ∂f2/∂x2 = 50*x2
    j11 = -0.5 * np.sinh(x1 / 2.0)
    j12 = 1.0
    j21 = 18.0 * x1
    j22 = 50.0 * x2
    return np.array([[j11, j12], [j21, j22]], dtype=float)


def print_header(epsilon: float, max_iterations: int, initial_guess: np.ndarray) -> None:
    """Minimal header required by the project."""
    print("=== Newton method for SEANL (Exercise 6b) ===")
    print("f1(x1, x2) = x2 - cosh(x1/2)")
    print("f2(x1, x2) = 9*x1^2 + 25*x2^2 - 225")
    print(f"Initial guess x(0) = [{initial_guess[0]:.3f}, {initial_guess[1]:.3f}]")
    print("Stopping: Criterion C (absolute), ε =", f"{epsilon:.3f}")
    print(f"Max iterations: {max_iterations}")
    print("Round final x* to 3 decimals; evaluate F at the rounded x*.\n")


def print_summary(converged: bool, iterations: int, x_star: np.ndarray) -> None:
    """Summary lines after the run."""
    x_star_round = np.round(x_star, 3)
    f_at_round = function(x_star_round)
    print(f"Converged: {converged}")
    print(f"iterations: {iterations}")
    print(f"x* ≈ [{x_star_round[0]:.3f}, {x_star_round[1]:.3f}]")
    print(f"F(x*) @ rounded x*: [{f_at_round[0]:.3f}, {f_at_round[1]:.3f}]\n")


def print_history(history) -> None:
    """Iteration table with aligned columns."""
    print("iteration |        x1        |        x2        |    abs_error")
    print("---------------------------------------------------------------")
    for step in history:
        x1, x2 = step.current_estimate
        print(
            f"{step.iteration:9d} | "
            f"{x1:14.6f} | "
            f"{x2:14.6f} | "
            f"{step.stopping_value:12.6f}"
        )


def main() -> None:
    # Exercise parameters (explicit here; never hard-code in the method).
    epsilon = 0.1
    max_iterations = 3
    initial_guess = np.array([2.5, 2.0], dtype=float)

    print_header(epsilon=epsilon, max_iterations=max_iterations, initial_guess=initial_guess)

    result = newton_system(
        function=function,
        jacobian=jacobian,
        initial_guess=initial_guess,
        tolerance=epsilon,
        max_iterations=max_iterations,
        stopping_criterion="C",
    )

    print_summary(
        converged=result.converged,
        iterations=result.iterations,
        x_star=result.approximate_solution,
    )
    print_history(result.history)


if __name__ == "__main__":
    main()