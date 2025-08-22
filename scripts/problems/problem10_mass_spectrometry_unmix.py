"""
problems/p10_mass_spectrometry_gauss_seidel.py

Exercise 10 — Mass spectrometry linear mixture model solved with Gauss–Seidel.

Model:  I = C @ n
  I: vector of peak heights (I1..I5)
  C: matrix of species contributions to each peak (C_ij)
  n: vector of unknown species concentrations [CH4, C2H4, C2H6, C3H6, C3H8]
"""

from __future__ import annotations
import numpy as np
from scripts.methods.linear.gauss_seidel import gauss_seidel, GaussSeidelReport


def main() -> None:
    species_names = ["CH4", "C2H4", "C2H6", "C3H6", "C3H8"]

    contribution_coefficient_matrix = np.array(
        [
            [28.0, 1.0, 0.0, 0.0, 0.1],
            [0.0, 18.0, 12.0, 2.4, 16.0],
            [0.0, 0.0, 10.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 10.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 18.0],
        ],
        dtype=float,
    )

    peak_height_vector = np.array([20.5, 170.0, 49.0, 39.8, 96.3], dtype=float)

    relative_tolerance = 1e-2
    maximum_number_of_iterations = 1_000
    stopping_criterion_choice = "C"  # relative (per-component) criterion

    print("=== Problem 10 — Mass Spectrometry (Gauss–Seidel) ===")
    print(f"Unknowns order: {species_names}")
    print(f"Stopping: criterion {stopping_criterion_choice} (relative), epsilon={relative_tolerance}\n")

    gauss_seidel_report: GaussSeidelReport = gauss_seidel(
        matrix=contribution_coefficient_matrix,
        right_hand_side=peak_height_vector,
        initial_guess=None,
        tolerance=relative_tolerance,
        max_iterations=maximum_number_of_iterations,
        stopping_criterion="C",
    )

    # Summary
    has_converged = (
        gauss_seidel_report.relative_error < relative_tolerance
        if stopping_criterion_choice == "C"
        else gauss_seidel_report.absolute_error < relative_tolerance
    )
    print(
        f"Converged: {has_converged} with criterion {stopping_criterion_choice} "
        f"at epsilon={relative_tolerance} in {gauss_seidel_report.iterations} iterations.\n"
    )

    # Concentrations
    print("Concentrations (same units as the statement):")
    for species_name, concentration_value in zip(species_names, gauss_seidel_report.solution):
        print(f"  {species_name:>5s} = {concentration_value}")

    # Diagnostics
    print(f"\nAbsolute error (inf-norm): {gauss_seidel_report.absolute_error}")
    print(f"Relative error:           {gauss_seidel_report.relative_error}")
    print(f"Residual ||b - A x||_inf: {gauss_seidel_report.residual_infinity_norm}")

    # Quick verification of A x ≈ b (no forced rounding/decimals)
    reconstructed_peak_heights_vector = contribution_coefficient_matrix @ gauss_seidel_report.solution
    print("\nVerification (C @ n):", reconstructed_peak_heights_vector)
    relative_residual = gauss_seidel_report.residual_infinity_norm / np.linalg.norm(peak_height_vector, ord=np.inf)
    print("Relative residual ||b - A x||_inf / ||b||_inf:", relative_residual)


if __name__ == "__main__":
    main()
