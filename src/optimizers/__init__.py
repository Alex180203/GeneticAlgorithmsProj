from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationResult:
    best_x: np.ndarray
    best_cost: float
    best_history: list = field(default_factory=list)   # best cost after each iteration
    mean_history: list = field(default_factory=list)   # population mean cost per iteration
    positions: list = field(default_factory=list)      # population snapshot per iteration
    n_evals: int = 0
    method: str = ""


from .pso import pso  # noqa: E402
from .ga import ga  # noqa: E402

__all__ = ["OptimizationResult", "pso", "ga"]
