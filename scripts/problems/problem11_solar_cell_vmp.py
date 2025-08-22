# scripts/problems/problem11_solar_cell_vmp.py
"""
Problem 11 — Solar cell maximum power voltage (nonlinear equations)
TEB062 — Applied Numerical Methods in Biotech

(a) Bisection: find a valid bracket and reduce it to ≤ 0.02 V (criterion C).
(b) Newton: start at V0 = 0.50 V, stop when |ΔV| < 0.005 V (criterion C).
(c) Secant: start at 0.60 V and 0.50 V, stop when |ΔV| < 0.01 V (criterion C).

Printing rules (project spec):
- Header with function, initial info, criterion and ε; round final x* to 3 decimals.
- After run: Converged, iterations, x* (3 decimals), f(x*) at the rounded value, and Final bracket (for bisection).
- Iteration table with aligned columns and numbers using :.6f; stopping column named 'abs_error' (criterion C).
"""

from __future__ import annotations
import math

# === Import generic methods from the project ===
from scripts.methods.nonlinear.bisection import bisection
from scripts.methods.nonlinear.newton import newton_method, NewtonResult
from scripts.methods.nonlinear.secant import secant, SecantResult


# === Physical constants and data from the statement ===
ELEMENTARY_CHARGE_C = 1.6022e-19         # q [C]
BOLTZMANN_CONSTANT_J_PER_K = 1.3806e-23  # k_B [J/K]
TEMPERATURE_K = 297.0                    # T [K]
OPEN_CIRCUIT_VOLTAGE_V = 0.50            # V_oc [V]

ALPHA_PER_VOLT = ELEMENTARY_CHARGE_C / (BOLTZMANN_CONSTANT_J_PER_K * TEMPERATURE_K)  # ≈ 39.0744 1/V


# === Original equation (no algebraic reformulation) ===
def function(voltage: float) -> float:
    """
    g(V) = exp(alpha*V) * (1 + alpha*V) - exp(alpha*V_oc)
    """
    return math.exp(ALPHA_PER_VOLT * voltage) * (1.0 + ALPHA_PER_VOLT * voltage) - math.exp(
        ALPHA_PER_VOLT * OPEN_CIRCUIT_VOLTAGE_V
    )


def derivative(voltage: float) -> float:
    """
    g'(V) = exp(alpha*V) * alpha * (2 + alpha*V)
    """
    return math.exp(ALPHA_PER_VOLT * voltage) * ALPHA_PER_VOLT * (2.0 + ALPHA_PER_VOLT * voltage)


# === Printing utilities (project format) ===
def print_header(title: str, details: str) -> None:
    print(f"=== {title} ===")
    print(details)
    print("Round final x* to 3 decimals.\n")


def fmt_or_dash(value: float | None, width: int = 12) -> str:
    """Format floats with :.6f or print an aligned dash if value is None."""
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:{width}.6f}"


def print_bisection_report(result, epsilon: float) -> None:
    print(f"Converged: {result.converged} with criterion C at epsilon={epsilon} in {result.iterations} iterations.\n")

    root_estimate = round(result.history[-1].midpoint, 3)
    function_at_rounded_root = function(root_estimate)
    final_left = round(result.final_interval[0], 3)
    final_right = round(result.final_interval[1], 3)

    print(f"x* ≈ {root_estimate:.3f}   f(x*) ≈ {function_at_rounded_root:.3f}")
    print(f"Final bracket: [{final_left:.3f}, {final_right:.3f}]\n")

    print(
        f"{'iteration':>9} | {'left_endpoint':>14} | {'right_endpoint':>14} | "
        f"{'midpoint':>10} | {'f(midpoint)':>12} | {'abs_error':>12}"
    )
    print("-" * 86)
    for index, step in enumerate(result.history, start=1):
        print(
            f"{index:9d} | "
            f"{step.left_endpoint:14.6f} | "
            f"{step.right_endpoint:14.6f} | "
            f"{step.midpoint:10.6f} | "
            f"{step.function_value_at_midpoint:12.6f} | "
            f"{fmt_or_dash(step.stopping_value)}"
        )
    print()


def print_newton_report(result: NewtonResult, epsilon: float) -> None:
    print(f"Converged: {result.converged} with criterion C at epsilon={epsilon} in {result.iterations} iterations.\n")

    rounded_root = round(result.root_estimate, 3)
    function_at_rounded_root = function(rounded_root)
    print(f"x* ≈ {rounded_root:.3f}   f(x*) ≈ {function_at_rounded_root:.3f}\n")

    print(f"{'iteration':>9} | {'x_k':>12} | {'f(x_k)':>12} | {'f\'(x_k)':>12} | {'Δx':>12} | {'abs_error':>12}")
    print("-" * 79)
    for index, step in enumerate(result.history, start=1):
        print(
            f"{index:9d} | "
            f"{step.current_estimate:12.6f} | "
            f"{step.function_value:12.6f} | "
            f"{step.derivative_value:12.6f} | "
            f"{fmt_or_dash(step.update_step)} | "
            f"{fmt_or_dash(step.stopping_value)}"
        )
    print()


def print_secant_report(result: SecantResult, epsilon: float) -> None:
    print(f"Converged: {result.converged} with criterion C at epsilon={epsilon} in {result.iterations} iterations.\n")

    rounded_root = round(result.root_approximation, 3)
    function_at_rounded_root = function(rounded_root)
    print(f"x* ≈ {rounded_root:.3f}   f(x*) ≈ {function_at_rounded_root:.3f}\n")

    print(
        f"{'iteration':>9} | {'x_(k-1)':>12} | {'x_k':>12} | {'x_(k+1)':>12} | "
        f"{'f(x_(k+1))':>14} | {'abs_error':>12}"
    )
    print("-" * 88)
    for index, step in enumerate(result.history, start=1):
        print(
            f"{index:9d} | "
            f"{step.previous_estimate:12.6f} | "
            f"{step.current_estimate:12.6f} | "
            f"{step.next_estimate:12.6f} | "
            f"{step.function_value_at_next:14.6f} | "
            f"{fmt_or_dash(step.stopping_value)}"
        )
    print()


# === Main script ===
def main() -> None:
    # ---------- (a) Bisection ----------
    # Investigated bracket: [0, V_oc] -> g(0) < 0 and g(V_oc) > 0 (sign change).
    left_endpoint = 0.0
    right_endpoint = OPEN_CIRCUIT_VOLTAGE_V
    epsilon_bisection = 0.02  # absolute criterion C
    max_iterations_bisection = 10_000

    print_header(
        "Problem 11(a) — Bisection",
        f"Function: g(V) = exp(αV)*(1+αV) - exp(αV_oc)  (α={ALPHA_PER_VOLT:.6f} 1/V)\n"
        f"Initial bracket: [{left_endpoint:.3f}, {right_endpoint:.3f}]\n"
        f"Stopping: criterion C (absolute), ε = {epsilon_bisection}"
    )

    bisection_result = bisection(
        function_to_solve=function,
        left_endpoint=left_endpoint,
        right_endpoint=right_endpoint,
        tolerance=epsilon_bisection,
        maximum_iterations=max_iterations_bisection,
        stopping_criterion="C",
    )
    print_bisection_report(bisection_result, epsilon_bisection)

    # ---------- (b) Newton ----------
    newton_initial_guess = 0.50  # V
    epsilon_newton = 0.005
    max_iterations_newton = 100

    print_header(
        "Problem 11(b) — Newton-Raphson",
        "Function: same g(V)\n"
        f"Initial guess: V0 = {newton_initial_guess:.3f} V\n"
        f"Stopping: criterion C (absolute), ε = {epsilon_newton}"
    )

    newton_result = newton_method(
        function=function,
        derivative=derivative,
        initial_guess=newton_initial_guess,
        tolerance=epsilon_newton,
        stopping="C",
        maximum_iterations=max_iterations_newton,
    )
    print_newton_report(newton_result, epsilon_newton)

    # ---------- (c) Secant ----------
    secant_initial_x0 = 0.60  # V
    secant_initial_x1 = 0.50  # V
    epsilon_secant = 0.01
    max_iterations_secant = 200

    print_header(
        "Problem 11(c) — Secant",
        "Function: same g(V)\n"
        f"Initial points: V0 = {secant_initial_x0:.3f} V, V1 = {secant_initial_x1:.3f} V\n"
        f"Stopping: criterion C (absolute), ε = {epsilon_secant}"
    )

    secant_result = secant(
        function=function,
        initial_x0=secant_initial_x0,
        initial_x1=secant_initial_x1,
        tolerance=epsilon_secant,
        max_iterations=max_iterations_secant,
        stopping_criterion="C",
    )
    print_secant_report(secant_result, epsilon_secant)


if __name__ == "__main__":
    main()
