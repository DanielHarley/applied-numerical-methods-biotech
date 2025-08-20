from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

@dataclass
class BisectionStep:
    iteration_index: int
    left_endpoint: float
    right_endpoint: float
    midpoint: float
    function_value_at_midpoint: float
    stopping_value: Optional[float]  # value used by the chosen stopping criterion

@dataclass
class BisectionResult:
    """Result of the bisection method."""
    root: float
    iterations: int
    converged: bool
    final_interval: Tuple[float, float]
    function_value_at_root: float
    history: List[BisectionStep]

def bisection(
    function_to_solve: Callable[[float], float],
    left_endpoint: float,
    right_endpoint: float,
    tolerance: float,
    maximum_iterations: int = 10_000,
    *,
    stopping_criterion: str = "C",  # "B" -> |f(m)| < tol   |   "C" -> |m_k - m_{k-1}| < tol
) -> BisectionResult:
    """
    Bisection method with selectable stopping criterion (B or C).

    - Criterion B:  stop when |f(midpoint)| < tolerance
    - Criterion C:  stop when |midpoint_k - midpoint_{k-1}| < tolerance  (absolute)

    Preconditions (Bolzano): opposite signs at endpoints and continuity on [a,b].
    """
    stopping_criterion = stopping_criterion.upper()
    if stopping_criterion not in {"B", "C"}:
        raise ValueError("stopping_criterion must be 'B' or 'C'.")

    function_value_left = function_to_solve(left_endpoint)
    function_value_right = function_to_solve(right_endpoint)

    if function_value_left == 0.0:
        step0 = BisectionStep(0, left_endpoint, right_endpoint, left_endpoint, function_value_left, None)
        return BisectionResult(left_endpoint, 0, True, (left_endpoint, right_endpoint), function_value_left, [step0])

    if function_value_right == 0.0:
        step0 = BisectionStep(0, left_endpoint, right_endpoint, right_endpoint, function_value_right, None)
        return BisectionResult(right_endpoint, 0, True, (left_endpoint, right_endpoint), function_value_right, [step0])

    if function_value_left * function_value_right > 0:
        raise ValueError("Bisection requires opposite signs at the endpoints (Bolzano condition).")

    history: List[BisectionStep] = []
    previous_midpoint: Optional[float] = None

    for iteration_index in range(1, maximum_iterations + 1):
        midpoint = (left_endpoint + right_endpoint) / 2.0
        function_value_midpoint = function_to_solve(midpoint)

        # Update the bracket first so the returned final_interval matches the last decision
        if function_value_left * function_value_midpoint > 0:
            left_endpoint, function_value_left = midpoint, function_value_midpoint
        else:
            right_endpoint, function_value_right = midpoint, function_value_midpoint

        # Compute the stopping value depending on the chosen criterion
        if stopping_criterion == "B":
            stopping_value = abs(function_value_midpoint)
        else:  # "C"
            stopping_value = None if previous_midpoint is None else abs(midpoint - previous_midpoint)

        current_step = BisectionStep(
            iteration_index=iteration_index,
            left_endpoint=left_endpoint,
            right_endpoint=right_endpoint,
            midpoint=midpoint,
            function_value_at_midpoint=function_value_midpoint,
            stopping_value=stopping_value,
        )
        history.append(current_step)

        # Check for convergence
        if function_value_midpoint == 0.0:
            return BisectionResult(midpoint, iteration_index, True, (left_endpoint, right_endpoint), function_value_midpoint, history)

        if stopping_value is not None and stopping_value < tolerance:
            return BisectionResult(midpoint, iteration_index, True, (left_endpoint, right_endpoint), function_value_midpoint, history)

        previous_midpoint = midpoint

    return BisectionResult(midpoint, iteration_index, False, (left_endpoint, right_endpoint), function_value_midpoint, history)
