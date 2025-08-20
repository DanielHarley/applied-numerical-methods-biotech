from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel, GaussSeidelReport

@dataclass(frozen=True)
class StoichiometryGSResult:
    oxygen_coefficient: float          # a
    ammonia_coefficient: float         # b
    biomass_coefficient: float         # c
    carbon_dioxide_coefficient: float  # d
    water_coefficient: float           # e
    iterations: int
    relative_error: float
    residual_infinity_norm: float
    rq_value: float

def build_linear_system(rq_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Return A, b for the system A x = b with x = [a, c, e]."""
    matrix = np.array([
        [rq_value, 1.0, 0.0],            # Carbon balance: 16 = c + rq*a
        [0.0,      1.06, 2.0],           # Hydrogen balance: 34 = 1.06*c + 2*e
        [2.0 - 2.0 * rq_value, -0.27, -1.0],  # Oxygen balance: 0 = (2-2rq)a - 0.27c - e
    ], dtype=float)
    right_hand_side = np.array([16.0, 34.0, 0.0], dtype=float)
    return matrix, right_hand_side

def solve_with_gauss_seidel_on_normals(
    rq_value: float,
    tolerance: float,
    max_iterations: int,
) -> StoichiometryGSResult:
    """Solve the normal equations with Gauss–Seidel and map back to a,b,c,d,e."""
    matrix, right_hand_side = build_linear_system(rq_value)
    normal_matrix = matrix.T @ matrix
    normal_rhs = matrix.T @ right_hand_side

    report: GaussSeidelReport = gauss_seidel(
        matrix=normal_matrix,
        right_hand_side=normal_rhs,
        initial_guess=np.zeros(3, dtype=float),
        tolerance=tolerance,
        max_iterations=max_iterations,
        stopping_criterion="C",  # relative per-component
    )

    a_value, c_value, e_value = report.solution
    d_value = rq_value * a_value
    b_value = 0.20 * c_value

    return StoichiometryGSResult(
        oxygen_coefficient=float(a_value),
        ammonia_coefficient=float(b_value),
        biomass_coefficient=float(c_value),
        carbon_dioxide_coefficient=float(d_value),
        water_coefficient=float(e_value),
        iterations=report.iterations,
        relative_error=float(report.relative_error),
        residual_infinity_norm=float(report.residual_infinity_norm),
        rq_value=float(rq_value),
    )

def print_header(rq_value: float, tolerance: float) -> None:
    print("Problem 5 — SCP from hexadecane (Gauss–Seidel)")
    print("Reaction: C16H34 + a O2 + b NH3 -> c CH1.66 O0.27 N0.20 + d CO2 + e H2O")
    print(f"Given: RQ = d/a = {rq_value:.2f}")
    print(f"Method: Gauss–Seidel on normal equations (A^T A x = A^T b), criterion C, ε = {tolerance:.1e}")
    print("Coefficients per 1 mol of C16H34 (rounded to 3 decimals):")
    print("-" * 72)

def print_result(result: StoichiometryGSResult) -> None:
    print(f"{'a (O2)':>20} = {result.oxygen_coefficient:10.3f}  mol")
    print(f"{'b (NH3)':>20} = {result.ammonia_coefficient:10.3f}  mol")
    print(f"{'c (Biomass)':>20} = {result.biomass_coefficient:10.3f}  mol")
    print(f"{'d (CO2)':>20} = {result.carbon_dioxide_coefficient:10.3f}  mol")
    print(f"{'e (H2O)':>20} = {result.water_coefficient:10.3f}  mol")
    print("-" * 72)
    print(f"Converged: True  | iterations: {result.iterations} | "
          f"relative_error: {result.relative_error:.2e} | "
          f"||b - A^T A x||_inf: {result.residual_infinity_norm:.2e}")

def main() -> None:
    rq_value = 0.43
    tolerance = 1e-8
    max_iterations = 10000

    print_header(rq_value, tolerance)
    result = solve_with_gauss_seidel_on_normals(
        rq_value=rq_value,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    print_result(result)

if __name__ == "__main__":
    main()