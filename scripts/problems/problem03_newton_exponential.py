from __future__ import annotations
import math
from scripts.methods.nonlinear.newton import newton_method, NewtonResult

# Problem constants (from the exercise)
kA = 0.0866434
kB = 0.346574
initial_guess_t = 9.0
tolerance = 0.002
maximum_iterations = 10_000


def function(t: float) -> float:
    """f(t) = e^{-kA t} - e^{-kB t} - 0.3"""
    return math.exp(-kA * t) - math.exp(-kB * t) - 0.3


def derivative(t: float) -> float:
    """f'(t) = -kA e^{-kA t} + kB e^{-kB t}"""
    return -kA * math.exp(-kA * t) + kB * math.exp(-kB * t)


def print_header() -> None:
    print()
    print("Exercise 3 - Newton-Rapshon Method")
    print("Function: f(t) = exp(-kA*t) - exp(-kB*t) - 0.3")
    print(f"Parameters: kA = {kA}, kB = {kB}")
    print(f"Initial guess: t0 = {initial_guess_t}")
    print(f"Stopping criterion: C (absolute) with epsilon = {tolerance}")
    print("Report final answer with 3 decimals.\n")


def print_table(result: NewtonResult) -> None:
    print("iteration |         t_k |           f(t_k) |          f'(t_k) |        delta_t |      abs_error")
    print("-----------------------------------------------------------------------------------------------")
    for step in result.history:
        print(
            f"{step.iteration:9d} | "
            f"{step.current_estimate:12.6f} | "
            f"{step.function_value:15.6f} | "
            f"{step.derivative_value:15.6f} | "
            f"{step.update_step:13.6f} | "
            f"{step.stopping_value:13.6f}"
        )


def main() -> None:
    print_header()

    result = newton_method(
        function=function,
        derivative=derivative,
        initial_guess=initial_guess_t,
        tolerance=tolerance,
        stopping="C",
        maximum_iterations=maximum_iterations,
    )

    # Summary (final values rounded to 3 decimals as required)
    estimated_t = result.root_estimate
    rounded_estimated_t = round(estimated_t, 3)
    residual_at_rounded = function(rounded_estimated_t)

    print("\nConverged:", result.converged)
    print("Iterations:", result.iterations)
    print(f"t* ≈ {rounded_estimated_t:.3f}   f(t*) evaluated at rounded t*: {residual_at_rounded:.6f}\n")

    print_table(result)

if __name__ == "__main__":
    main()