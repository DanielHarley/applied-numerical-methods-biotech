from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Literal


StoppingCriterion = Literal["B", "C"]


@dataclass
class SecantStep:
    iteration: int
    previous_estimate: float          # x_{k-1}
    current_estimate: float           # x_k
    next_estimate: float              # x_{k+1}
    function_value_at_next: float     # f(x_{k+1})
    stopping_value: float             # |f(x_{k+1})| (B) or |x_{k+1} - x_k| (C)


@dataclass
class SecantResult:
    converged: bool
    iterations: int
    root_approximation: float
    function_value_at_root: float
    history: List[SecantStep] = field(default_factory=list)
    final_pair: tuple[float, float] | None = None  # (x_{k-1}, x_k) at return


def secant(
    function: Callable[[float], float],
    initial_x0: float,
    initial_x1: float,
    tolerance: float,
    max_iterations: int,
    stopping_criterion: StoppingCriterion,
) -> SecantResult:
    if initial_x0 == initial_x1:
        raise ValueError("Secant requires distinct initial guesses: x(0) != x(1).")

    history: List[SecantStep] = []

    previous_estimate = float(initial_x0)  # x_{k-1}
    current_estimate  = float(initial_x1)  # x_k
    f_prev = function(previous_estimate)
    f_curr = function(current_estimate)

    # Safety pre-check on the current available iterate
    if stopping_criterion == "B" and abs(f_curr) < tolerance:
        return SecantResult(
            converged=True,
            iterations=0,
            root_approximation=current_estimate,
            function_value_at_root=f_curr,
            history=history,
            final_pair=(previous_estimate, current_estimate),
        )
    if stopping_criterion == "C" and abs(current_estimate - previous_estimate) < tolerance:
        return SecantResult(
            converged=True,
            iterations=0,
            root_approximation=current_estimate,
            function_value_at_root=f_curr,
            history=history,
            final_pair=(previous_estimate, current_estimate),
        )

    for iteration in range(1, max_iterations + 1):
        denominator = f_curr - f_prev
        if denominator == 0.0:
            # Cannot proceed. If we got here, pre-check above already ruled out convergence.
            return SecantResult(
                converged=False,
                iterations=iteration - 1,
                root_approximation=current_estimate,
                function_value_at_root=f_curr,
                history=history,
                final_pair=(previous_estimate, current_estimate),
            )

        # Secant update
        next_estimate = current_estimate - f_curr * (current_estimate - previous_estimate) / denominator
        f_next = function(next_estimate)

        # Stopping value recorded for this iteration (the exact criterion used)
        if stopping_criterion == "B":
            stopping_value = abs(f_next)
        else:  # "C"
            stopping_value = abs(next_estimate - current_estimate)

        history.append(
            SecantStep(
                iteration=iteration,
                previous_estimate=previous_estimate,
                current_estimate=current_estimate,
                next_estimate=next_estimate,
                function_value_at_next=f_next,
                stopping_value=stopping_value,
            )
        )

        # Shift window to the newest two iterates
        previous_estimate, f_prev = current_estimate, f_curr
        current_estimate,  f_curr = next_estimate,  f_next

        # Standard post-update stopping check on the "latest" x_k
        if (stopping_criterion == "B" and abs(f_curr) < tolerance) or \
           (stopping_criterion == "C" and abs(current_estimate - previous_estimate) < tolerance):
            return SecantResult(
                converged=True,
                iterations=iteration,
                root_approximation=current_estimate,
                function_value_at_root=f_curr,
                history=history,
                final_pair=(previous_estimate, current_estimate),
            )

    # Reached iteration cap
    return SecantResult(
        converged=False,
        iterations=max_iterations,
        root_approximation=current_estimate,
        function_value_at_root=f_curr,
        history=history,
        final_pair=(previous_estimate, current_estimate),
    )