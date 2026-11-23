from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationResult:
    best_x: np.ndarray
    best_cost: float
    best_history: list = field(default_factory=list)
    mean_history: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    n_evals: int = 0
    method: str = ""


from .pso import pso
from .ga import ga

__all__ = ["OptimizationResult", "pso", "ga"]
