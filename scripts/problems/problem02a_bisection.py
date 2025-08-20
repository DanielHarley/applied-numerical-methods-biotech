from __future__ import annotations
import math
from scripts.methods.nonlinear.bisection import bisection

def function_to_solve(x: float) -> float:
    return math.exp(x) - x**3 + 3.0

def main() -> None:
    left_endpoint, right_endpoint = 2.0, 3.0
    tolerance = 1e-1  # epsilon = 0.1, stopping criterion C: |m^(k) - m^(k-1)| < epsilon

    result = bisection(
        function_to_solve,
        left_endpoint,
        right_endpoint,
        tolerance=tolerance,
        maximum_iterations=10_000,
        stopping_criterion="C",
    )

    rounded_root = round(result.root, 3)
    function_at_rounded = function_to_solve(rounded_root)

    print()
    print("Exercise 2(a) — Bisection Method")
    print("Function: f(x) = e^x − x^3 + 3")
    print(f"Initial interval: [{left_endpoint:.3f}, {right_endpoint:.3f}]")
    print(f"Stopping rule: Criterion C (|m^(k) − m^(k−1)| < ε) with ε = {tolerance:g}")
    print("Final answer must be rounded to 3 decimal places.\n")

    header = (
        f"{'iteration':>9} | "
        f"{'left_endpoint':>14} | "
        f"{'right_endpoint':>14} | "
        f"{'midpoint':>14} | "
        f"{'f(midpoint)':>14} | "
        f"{'abs_error':>14}"
    )
    print(header)
    print("-" * len(header))

    for step in result.history:
        abs_err = "—" if step.stopping_value is None else f"{step.stopping_value:.6f}"
        print(
            f"{step.iteration_index:9d} | "
            f"{step.left_endpoint:14.6f} | "
            f"{step.right_endpoint:14.6f} | "
            f"{step.midpoint:14.6f} | "
            f"{step.function_value_at_midpoint:14.6f} | "
            f"{abs_err:>14}"
        )

    print()
    print(f"Converged: {result.converged} in {result.iterations} iterations")
    print(f"x* (3 decimal places) ≈ {rounded_root:.3f}   f(x* rounded) ≈ {function_at_rounded:.6f}")
    print(f"Final interval: [{result.final_interval[0]:.6f}, {result.final_interval[1]:.6f}]\n")

if __name__ == "__main__":
    main()
