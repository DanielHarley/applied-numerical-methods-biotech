from __future__ import annotations
import math
from scripts.methods.nonlinear.secant import secant


def function_to_solve(x: float) -> float:
    return x * math.exp(x) - 1.0


def main() -> None:
    # Problem-specific configuration (explicit):
    initial_x0 = 0.0
    initial_x1 = 1.0
    tolerance = 0.002
    stopping_criterion = "B"  # |f(x_k)| < ε
    max_iterations = 100

    # Header:
    print("Secant Method — Criterion B (|f(x_k)| < ε)")
    print("Function: f(x) = x*e^x - 1")
    print(f"Initial points: x(0) = {initial_x0:.3f}, x(1) = {initial_x1:.3f}")
    print(f"ε = {tolerance}")
    print("Final result must be rounded to 3 decimals.\n")

    # Run:
    result = secant(
        function=function_to_solve,
        initial_x0=initial_x0,
        initial_x1=initial_x1,
        tolerance=tolerance,
        max_iterations=max_iterations,
        stopping_criterion=stopping_criterion,
    )

    # Final report (evaluate f at the rounded x* as required):
    x_star_rounded = round(result.root_approximation, 3)
    f_at_rounded = function_to_solve(x_star_rounded)

    print(f"Converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"x* ≈ {x_star_rounded:.3f}   f(x*) ≈ {f_at_rounded:.6f}")

    if result.final_pair is not None:
        a, b = result.final_pair
        print(f"Final pair: [{a:.3f}, {b:.3f}]")
    print()

    # Iteration table:
    header = (
        "iteration |   x_{k-1} (prev) |      x_k (curr) |     x_{k+1} (next) |    f(x_{k+1}) |   |f(x_{k+1})|"
    )
    print(header)
    print("-" * len(header))
    for step in result.history:
        print(
            f"{step.iteration:9d} | "
            f"{step.previous_estimate:16.6f} | "
            f"{step.current_estimate:16.6f} | "
            f"{step.next_estimate:16.6f} | "
            f"{step.function_value_at_next:13.6f} | "
            f"{step.stopping_value:13.6f}"
        )


if __name__ == "__main__":
    main()