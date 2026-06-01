"""Equal Weight (1/N) Portfolio algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class EqualWeightOptimization(Algorithm):
    """Equal Weight (1/N) Portfolio.

    The simplest allocation strategy: assign equal weight to all assets.

        w_i = 1/N for all i

    Despite its simplicity, DeMiguel et al. (2009) showed that 1/N often
    outperforms optimized portfolios out-of-sample due to estimation error
    in more complex methods.
    """

    name = "equal_weight"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        return np.ones(n_assets) / n_assets
