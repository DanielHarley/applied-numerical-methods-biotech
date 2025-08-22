"""
Problem 8 — Recycled paperboard flowsheet (mass balances) solved with Gauss–Seidel,
tightening the solver tolerance epsilon = 10^(-k) in a loop until the overall
mass IN equals OUT within a strict numerical threshold.

Unknowns order (chosen to favor a good diagonal for Gauss–Seidel):
    x = [A, V, C, E, G, P, B]^T

Linear equations (A x = b):
(1) A + B = 1000
(2) 0.20 A − V = 0
(3) −A + V + C = 0
(4) −0.05 A + E = 0
(5) −C − E + G = 0
(6) P − 33.33 E = 0
(7) −G + P − B = 0
"""

from __future__ import annotations
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel


def solve_once(
    tolerance: float,
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    initial_guess: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Solve A x = b with the user's Gauss–Seidel for a given tolerance."""
    result = gauss_seidel(
        matrix=matrix,
        right_hand_side=right_hand_side,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=100_000,
        stopping_criterion="C",
    )

    try:
        solution = np.asarray(result.solution, dtype=float)
        iterations = getattr(result, "iterations", None)
        converged = getattr(result, "converged", True)
    except AttributeError:
        if isinstance(result, tuple):
            solution = np.asarray(result[0], dtype=float)
            iterations = result[1] if len(result) > 1 else None
            converged = True
        else:
            solution = np.asarray(result, dtype=float)
            iterations = None
            converged = True

    metrics = {
        "iterations": iterations,
        "converged": converged,
        "residual_inf": float(np.linalg.norm(matrix @ solution - right_hand_side, ord=np.inf)),
    }
    return solution, metrics


def fmt(value: float, decimals_when_ge1: int = 3) -> str:
    """
    Formatting rule:
    - If abs(value) < 1: do NOT enforce fixed decimals; trim trailing zeros and dot.
      Examples: 0.050 -> '0.05', 0.500 -> '0.5', ~0 -> '0'
    - If abs(value) >= 1: use thousands separator with fixed 3 decimals.
    """
    tiny = 1e-12
    if abs(value) < tiny:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.{decimals_when_ge1}f}"
    s = f"{value:.{decimals_when_ge1}f}"
    return s.rstrip('0').rstrip('.')


def print_stream_block(
    name: str,
    total: float,
    cellulose: float,
    water: float,
    salt: float,
    glue: float,
) -> None:

    sum_components = cellulose + water + salt + glue
    print(f"— {name} —")
    print(f"  total = {fmt(total)} kg/h")
    print(f"  cellulose = {fmt(cellulose)} | water = {fmt(water)} | salt = {fmt(salt)} | glue = {fmt(glue)}  (kg/h)")
    print(f"  check sum = {fmt(sum_components)} kg/h\n")


def main() -> None:
    # Build linear system A x = b
    # Unknown order: [A, V, C, E, G, P, B]
    matrix = np.array([
        [ 1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00],  # (1) A + B = 1000
        [ 0.20, -1.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # (2) 0.20A - V = 0
        [-1.00,  1.00,  1.00,  0.00,  0.00,  0.00,  0.00],  # (3) -A + V + C = 0
        [-0.05,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],  # (4) -0.05A + E = 0
        [ 0.00,  0.00, -1.00, -1.00,  1.00,  0.00,  0.00],  # (5) -C - E + G = 0
        [ 0.00,  0.00,  0.00, -33.3333333333, 0.00,  1.00,  0.00],  # (6) P - 33.333E = 0 (rescaled)
        [ 0.00,  0.00,  0.00,  0.00, -1.00,  1.00, -1.00],  # (7) -G + P - B = 0
    ], dtype=float)
    right_hand_side = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Tighten tolerance in a loop: epsilon = 10^(-k), k = 2,3,...,
    # until |(F+E) - (V+P)| <= balance_tolerance
    starting_exponent = 2     # 10^-2 as in the exercise list
    maximum_exponent = 14     # safety cap to avoid infinite loops
    balance_tolerance = 1e-12 # numerical closure for overall balance

    current_guess = np.zeros(7, dtype=float)
    final_solution: np.ndarray | None = None
    final_metrics: dict | None = None
    chosen_exponent = starting_exponent

    print("=== Problem 8 — Recycled paper mass balances (Gauss–Seidel with tightening epsilon) ===")
    print("Unknowns order: [A, V, C, E, G, P, B]\n")

    for exponent in range(starting_exponent, maximum_exponent + 1):
        epsilon = 10.0 ** (-exponent)

        solution, metrics = solve_once(
            tolerance=epsilon,
            matrix=matrix,
            right_hand_side=right_hand_side,
            initial_guess=current_guess,
        )

        # Map to named variables (order [A, V, C, E, G, P, B])
        A_flow, V_flow, C_flow, E_flow, G_flow, P_flow, B_flow = solution.tolist()
        overall_in = 1000.0 + E_flow
        overall_out = V_flow + P_flow
        balance_gap = abs(overall_in - overall_out)

        print(
            f"[loop] epsilon = {epsilon:.1e} | IN = {overall_in:.6f} | OUT = {overall_out:.6f} "
            f"| |IN-OUT| = {balance_gap:.3e} | "
            f"residual_inf = {metrics['residual_inf']:.1e} | "
            f"iterations = {metrics['iterations']}"
        )

        # Warm start next iteration
        current_guess = solution.copy()

        # Keep latest as candidate final
        final_solution = solution
        final_metrics = metrics
        chosen_exponent = exponent

        if balance_gap <= balance_tolerance:
            break

    # Safety
    assert final_solution is not None and final_metrics is not None

    # Final reporting (totals and per-stream compositions)
    A_flow, V_flow, C_flow, E_flow, G_flow, P_flow, B_flow = final_solution.tolist()

    # Fixed feed and its composition
    F_flow = 1000.0
    F_cellulose = 0.65 * F_flow
    F_water = 0.30 * F_flow
    F_salt = 0.05 * F_flow
    F_glue = 0.0

    # Streams from splitter keep feed composition
    A_cellulose = 0.65 * A_flow
    A_water = 0.30 * A_flow
    A_salt = 0.05 * A_flow
    A_glue = 0.0

    B_cellulose = 0.65 * B_flow
    B_water = 0.30 * B_flow
    B_salt = 0.05 * B_flow
    B_glue = 0.0

    # Evaporator: pure water V; liquid C retains all non-water from A; water(C) = 0.10*A
    V_cellulose = 0.0
    V_water = V_flow
    V_salt = 0.0
    V_glue = 0.0

    C_cellulose = A_cellulose
    C_water = 0.10 * A_flow
    C_salt = A_salt
    C_glue = 0.0

    # Glue stream E: pure glue
    E_cellulose = 0.0
    E_water = 0.0
    E_salt = 0.0
    E_glue = E_flow

    # Mixer 1: G = C + E
    G_cellulose = C_cellulose
    G_water = C_water
    G_salt = C_salt
    G_glue = E_glue

    # Mixer 2: P = B + G
    P_cellulose = B_cellulose + G_cellulose
    P_water = B_water + G_water
    P_salt = B_salt + G_salt
    P_glue = B_glue + G_glue

    # Diagnostics (use final_metrics consistently)
    residual_inf = final_metrics["residual_inf"]
    iterations = final_metrics["iterations"]
    overall_in = F_flow + E_flow
    overall_out = V_flow + P_flow
    glue_mass_fraction_P = (P_glue / P_flow) if P_flow != 0.0 else np.nan

    # --- Prints (using fmt) ---
    print("\n--- Final report ---")
    print(f"Chosen epsilon = 10^(-{chosen_exponent}) = {10.0 ** (-chosen_exponent):.1e}")
    status = "Converged" if residual_inf < 1e-9 else "Converged (loose)"
    iter_msg = f"in {iterations} iterations" if iterations is not None else ""
    print(f"{status} with criterion C (relative) {iter_msg}. Residual ||Ax-b||_inf = {residual_inf:.2e}\n")

    # Flow totals (including F explicitly)
    print("Flow totals (kg/h):")
    print(f"F = {fmt(F_flow)}")
    print(f"A = {fmt(A_flow)}")
    print(f"B = {fmt(B_flow)}")
    print(f"V = {fmt(V_flow)}")
    print(f"C = {fmt(C_flow)}")
    print(f"E = {fmt(E_flow)}   (glue)")
    print(f"G = {fmt(G_flow)}")
    print(f"P = {fmt(P_flow)}\n")

    # Per-stream breakdown (each stream separately with components)
    print("Per-stream composition breakdown (kg/h):\n")
    print_stream_block("F", F_flow, F_cellulose, F_water, F_salt, F_glue)
    print_stream_block("A", A_flow, A_cellulose, A_water, A_salt, A_glue)
    print_stream_block("B", B_flow, B_cellulose, B_water, B_salt, B_glue)
    print_stream_block("V", V_flow, V_cellulose, V_water, V_salt, V_glue)
    print_stream_block("C", C_flow, C_cellulose, C_water, C_salt, C_glue)
    print_stream_block("E", E_flow, E_cellulose, E_water, E_salt, E_glue)
    print_stream_block("G", G_flow, G_cellulose, G_water, G_salt, G_glue)
    print_stream_block("P", P_flow, P_cellulose, P_water, P_salt, P_glue)

    # Global checks
    print("Checks:")
    print(
        f"Overall balance:  IN = {fmt(overall_in)}   |   OUT = {fmt(overall_out)}   "
        f"| |IN-OUT| = {fmt(abs(overall_in - overall_out))}"
    )
    if np.isnan(glue_mass_fraction_P):
        print("Glue mass fraction in P = NaN (division by zero)")
    else:
        s = f"{glue_mass_fraction_P:.4f}".rstrip('0').rstrip('.')
        print(f"Glue mass fraction in P = {s} (target = 0.03)")


if __name__ == "__main__":
    main()
