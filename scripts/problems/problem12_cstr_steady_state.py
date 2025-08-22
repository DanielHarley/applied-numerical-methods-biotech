# problems/p12_cstr_newton.py
"""
Solve Problem 12 (CSTR with four reactions) via Newton's method for nonlinear systems.
We perform exactly 3 iterations, as requested in the exercise list.

Data (from the statement):
    k1=0.18, k2=0.10, k3=0.30, k4=0.04, V=50, q=20, C_A0=5
Initial guess:
    [C_A(0), C_B(0), C_C(0), C_D(0)] = [0, 1, 2, 3]

Printing pattern:
 - Header with problem description and parameters
 - Iteration table: iteration | Ca | Cb | Cc | Cd | abs_error | rel_error | ||F(x)||_inf
 - Final summary rounded to 3 decimals
"""

from __future__ import annotations
import numpy as np


# ----------------------------- problem parameters -----------------------------

flow_rate = 20.0                 # q
reactor_volume = 50.0            # V
inlet_concentration_A = 5.0      # C_A0

rate_constant_1 = 0.18           # k1  for  A -> 2B
rate_constant_2 = 0.10           # k2  for  2A -> C
rate_constant_3 = 0.30           # k3  for  B -> D + C
rate_constant_4 = 0.04           # k4  for  2B -> C

initial_guess = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)  # [Ca, Cb, Cc, Cd]


# ---------------------------- model: F(x) and J(x) ----------------------------

def reaction_rates(concentration_A: float, concentration_B: float) -> tuple[float, float, float, float]:
    """Volumetric reaction rates r1..r4 defined by mass-action kinetics."""
    r1 = rate_constant_1 * concentration_A
    r2 = rate_constant_2 * concentration_A ** 2
    r3 = rate_constant_3 * concentration_B
    r4 = rate_constant_4 * concentration_B ** 2
    return r1, r2, r3, r4


def residual(vector: np.ndarray) -> np.ndarray:
    """Residual vector F(x) from the steady-state component mass balances."""
    concentration_A, concentration_B, concentration_C, concentration_D = vector
    r1, r2, r3, r4 = reaction_rates(concentration_A, concentration_B)

    F1 = flow_rate * (inlet_concentration_A - concentration_A) - reactor_volume * (r1 + 2.0 * r2)
    F2 = -flow_rate * concentration_B + reactor_volume * (2.0 * r1 - r3 - 2.0 * r4)
    F3 = -flow_rate * concentration_C + reactor_volume * (r2 + r3 + r4)
    F4 = -flow_rate * concentration_D + reactor_volume * (r3)
    return np.array([F1, F2, F3, F4], dtype=float)


def jacobian(vector: np.ndarray) -> np.ndarray:
    """Analytical Jacobian J(x) = dF/dx."""
    concentration_A, concentration_B, concentration_C, concentration_D = vector

    dr1_dCa = rate_constant_1
    dr2_dCa = 2.0 * rate_constant_2 * concentration_A
    dr3_dCb = rate_constant_3
    dr4_dCb = 2.0 * rate_constant_4 * concentration_B

    J = np.zeros((4, 4), dtype=float)

    # dF1/dCa
    J[0, 0] = -flow_rate - reactor_volume * (dr1_dCa + 2.0 * dr2_dCa)
    # dF1/d(other) = 0

    # F2 partials
    J[1, 0] = reactor_volume * (2.0 * dr1_dCa)
    J[1, 1] = -flow_rate + reactor_volume * (-dr3_dCb - 2.0 * dr4_dCb)

    # F3 partials
    J[2, 0] = reactor_volume * dr2_dCa
    J[2, 1] = reactor_volume * (dr3_dCb + dr4_dCb)
    J[2, 2] = -flow_rate

    # F4 partials
    J[3, 1] = reactor_volume * dr3_dCb
    J[3, 3] = -flow_rate

    return J


# ------------------------------ driver (3 steps) ------------------------------

def main() -> None:
    print("=== Problem 12 — CSTR with four reactions (Newton, fixed 3 iterations) ===")
    print(f"q={flow_rate}, V={reactor_volume}, C_A0={inlet_concentration_A}, "
          f"k1={rate_constant_1}, k2={rate_constant_2}, k3={rate_constant_3}, k4={rate_constant_4}")
    print(f"Initial guess [Ca, Cb, Cc, Cd] = {initial_guess.tolist()}\n")

    print("iteration |        Ca |        Cb |        Cc |        Cd |   abs_error |   rel_error |  ||F(x)||_inf")
    print("-------------------------------------------------------------------------------------------------------")

    current = initial_guess.copy()
    previous: np.ndarray | None = None

    for iteration in range(1, 3 + 1):
        F = residual(current)
        J = jacobian(current)

        # Newton step: solve J * v = -F and update x_{k+1} = x_k + v
        update = np.linalg.solve(J, -F)
        next_vector = current + update

        absolute_error = (np.max(np.abs(next_vector - current)) if previous is not None else float("nan"))
        relative_error = (np.max(np.abs((next_vector - current) / next_vector)) if previous is not None else float("nan"))
        residual_inf_norm = np.linalg.norm(residual(next_vector), ord=np.inf)

        print(f"{iteration:9d} | {next_vector[0]:9.6f} | {next_vector[1]:9.6f} | "
              f"{next_vector[2]:9.6f} | {next_vector[3]:9.6f} | "
              f"{absolute_error:11.6f} | {relative_error:11.6f} | {residual_inf_norm:13.6f}")

        current = next_vector
        previous = next_vector

    rounded = np.round(current, 3)
    residual_at_rounded = np.linalg.norm(residual(rounded), ord=np.inf)

    print("\nRan exactly 3 Newton iterations (per the statement).")
    print(f"Ca={rounded[0]:.3f}, Cb={rounded[1]:.3f}, Cc={rounded[2]:.3f}, Cd={rounded[3]:.3f}")
    print(f"Residual at rounded x (∞-norm): {residual_at_rounded:.6f}")


if __name__ == "__main__":
    main()
