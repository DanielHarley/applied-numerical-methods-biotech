from __future__ import annotations
import math
from scripts.methods.nonlinear.bisection import bisection

def function_to_solve(x: float) -> float:
    return math.sin(x) - 1.0 / (x * x)

def main() -> None:
    left_endpoint, right_endpoint = 1.0, 1.4
    tolerance = 2e-2  # epsilon = 0.02, stopping criterion B: |g(m)| < epsilon

    result = bisection(
        function_to_solve,
        left_endpoint,
        right_endpoint,
        tolerance=tolerance,
        maximum_iterations=10_000,
        stopping_criterion="B",
    )

    rounded_root = round(result.root, 3)
    function_at_rounded = function_to_solve(rounded_root)

    print()
    print("Exercise 2(b) — Bisection Method")
    print("Function: g(x) = sin(x) − 1/x^2")
    print(f"Initial interval: [{left_endpoint:.3f}, {right_endpoint:.3f}]")
    print(f"Stopping rule: Criterion B (|g(m)| < ε) with ε = {tolerance:g}")
    print("Final answer must be rounded to 3 decimal places.\n")

    header = (
        f"{'iteration':>9} | "
        f"{'left_endpoint':>14} | "
        f"{'right_endpoint':>14} | "
        f"{'midpoint':>14} | "
        f"{'g(midpoint)':>14} | "
        f"{'criterion_B':>14}"
    )
    print(header)
    print("-" * len(header))

    for step in result.history:
        crit_text = "—" if step.stopping_value is None else f"{step.stopping_value:.6f}"
        print(
            f"{step.iteration_index:9d} | "
            f"{step.left_endpoint:14.6f} | "
            f"{step.right_endpoint:14.6f} | "
            f"{step.midpoint:14.6f} | "
            f"{step.function_value_at_midpoint:14.6f} | "
            f"{crit_text:>14}"
        )

    print()
    print(f"Converged: {result.converged} in {result.iterations} iterations")
    print(f"x* (3 decimal places) ≈ {rounded_root:.3f}   f(x* rounded) ≈ {function_at_rounded:.6f}")
    print(f"Final interval: [{result.final_interval[0]:.6f}, {result.final_interval[1]:.6f}]\n")


if __name__ == "__main__":
    main()
