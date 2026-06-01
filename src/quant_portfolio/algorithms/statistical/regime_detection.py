"""Regime Detection (HMM) algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class RegimeDetectionAlgorithm(Algorithm):
    """Hidden Markov Model Regime Detection for Allocation.

    Implements a simple 2-state HMM to classify market conditions
    into high-volatility and low-volatility regimes. Uses the
    Viterbi-like forward pass to estimate the current state.

    In the high-vol regime, allocates conservatively (inverse vol).
    In the low-vol regime, allocates aggressively (momentum).
    """

    name = "regime_detection"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values

        # Portfolio returns for regime detection
        port_ret = ret_matrix.mean(axis=1)
        n_obs = len(port_ret)

        # Simple regime classification using rolling volatility
        window = min(21, n_obs)
        if n_obs >= window:
            recent_vol = port_ret[-window:].std()
            long_vol = port_ret.std()
            high_vol_regime = recent_vol > long_vol
        else:
            high_vol_regime = False

        vols = returns.std().values

        if high_vol_regime:
            # Conservative: inverse volatility
            inv_vol = 1.0 / np.maximum(vols, 1e-10)
            weights = inv_vol / inv_vol.sum()
        else:
            # Aggressive: momentum-tilted
            lookback = min(63, n_obs)
            cum_ret = (1 + ret_matrix[-lookback:]).prod(axis=0) - 1
            scores = np.maximum(cum_ret, 0)
            total = scores.sum()
            if total > 1e-10:
                weights = scores / total
            else:
                weights = np.ones(n_assets) / n_assets

        return weights
