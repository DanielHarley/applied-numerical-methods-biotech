from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Literal


StoppingCriterion = Literal["B", "C"]


@dataclass
class NewtonStep:
    iteration: int
    current_estimate: float
    function_value: float
    derivative_value: float
    update_step: float
    stopping_value: float


@dataclass
class NewtonResult:
    converged: bool
    iterations: int
    root_estimate: float
    final_stopping_value: float
    history: List[NewtonStep] = field(default_factory=list)


def newton_method(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float,
    stopping: StoppingCriterion,
    maximum_iterations: int,
) -> NewtonResult:
    """
    Run the Newton–Raphson method.

    Parameters
    ----------
    function : Callable[[float], float]
        Nonlinear function f(x).
    derivative : Callable[[float], float]
        Derivative f'(x).
    initial_guess : float
        Starting point x_0.
    tolerance : float
        Epsilon used in the chosen stopping criterion.
    stopping : {"B", "C"}
        "B": stop when |f(x_{k+1})| < epsilon
        "C": stop when |x_{k+1} - x_k| < epsilon  (absolute)
    maximum_iterations : int
        Safety cap on the number of iterations.

    Returns
    -------
    NewtonResult
        Convergence flag, iteration count, final estimate, and detailed history.
    """
    x_current = initial_guess
    history: List[NewtonStep] = []

    for iteration in range(1, maximum_iterations + 1):
        f_value = function(x_current)
        derivative_value = derivative(x_current)

        if derivative_value == 0.0:
            # Cannot proceed if the tangent is horizontal at the current point.
            return NewtonResult(
                converged=False,
                iterations=iteration - 1,
                root_estimate=x_current,
                final_stopping_value=float("inf"),
                history=history,
            )

        x_next = x_current - f_value / derivative_value
        update_step = x_next - x_current

        # Stopping value according to the chosen criterion.
        if stopping == "C":
            stopping_value = abs(update_step)
        else:  # stopping == "B"
            stopping_value = abs(function(x_next))

        history.append(
            NewtonStep(
                iteration=iteration,
                current_estimate=x_current,
                function_value=f_value,
                derivative_value=derivative_value,
                update_step=update_step,
                stopping_value=stopping_value,
            )
        )

        x_current = x_next

        if stopping_value < tolerance:
            return NewtonResult(
                converged=True,
                iterations=iteration,
                root_estimate=x_current,
                final_stopping_value=stopping_value,
                history=history,
            )

    # If we reach here, maximum_iterations was hit.
    return NewtonResult(
        converged=False,
        iterations=maximum_iterations,
        root_estimate=x_current,
        final_stopping_value=history[-1].stopping_value if history else float("inf"),
        history=history,
    )