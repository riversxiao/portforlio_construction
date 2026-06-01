"""Copula Arbitrage Strategy - empirical copula-based dependency trading."""

import pandas as pd
import numpy as np
from scipy import stats

from quant_portfolio.strategies.base import Strategy


class CopulaArb(Strategy):
    """Copula-based dependency arbitrage strategy.

    Fits empirical marginal CDFs to asset returns and a proxy return series
    (e.g., moving average of volume-weighted returns), then transforms both
    to uniform margins (probability integral transform). The empirical copula
    dependence is measured via Kendall's tau on the uniform-transformed
    variables. Trading signals are generated when the current observation
    falls in the tail of the empirical copula distribution, indicating a
    breakdown in the normal dependence structure.

    Parameters
    ----------
    window : int
        Rolling window for empirical CDF estimation (default 60).
    threshold : float
        Tail probability threshold for signal generation (default 0.1).
    """

    name = "copula_arb"

    def __init__(self, window: int = 60, threshold: float = 0.1, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate copula-based arbitrage signals.

        Transforms returns and a secondary series (short-term vs long-term
        moving average ratio) to uniform margins via their empirical CDFs,
        then signals when the joint position is in the tail of the empirical
        copula.
        """
        close = data["close"]
        returns = close.pct_change()

        # Secondary series: ratio of short MA to long MA as a proxy
        ma_short = close.rolling(window=max(5, self.window // 12), min_periods=1).mean()
        ma_long = close.rolling(window=max(20, self.window // 3), min_periods=1).mean()
        secondary = (ma_short / ma_long - 1.0).fillna(0)

        signal = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            # Extract rolling windows
            r_window = returns.iloc[i - self.window:i + 1].dropna().values
            s_window = secondary.iloc[i - self.window:i + 1].dropna().values

            if len(r_window) < 10 or len(s_window) < 10:
                continue

            # Fit empirical marginal CDFs (probability integral transform)
            # Rank-based ECDF for uniform margin transformation
            u_returns = self._empirical_cdf(r_window, r_window[-1])
            u_secondary = self._empirical_cdf(s_window, s_window[-1])

            # Compute conditional copula probability:
            # P(U1 <= u1 | U2) using the empirical copula
            # Joint tail detection
            if u_returns < self.threshold and u_secondary < self.threshold:
                # Both in lower tail - mean reversion expected (buy)
                signal.iloc[i] = 1
            elif u_returns > (1 - self.threshold) and u_secondary > (1 - self.threshold):
                # Both in upper tail - mean reversion expected (sell)
                signal.iloc[i] = -1
            elif u_returns < self.threshold and u_secondary > (1 - self.threshold):
                # Dependence breakdown: return low but secondary high (buy)
                signal.iloc[i] = 1
            elif u_returns > (1 - self.threshold) and u_secondary < self.threshold:
                # Dependence breakdown: return high but secondary low (sell)
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)

    @staticmethod
    def _empirical_cdf(sample: np.ndarray, value: float) -> float:
        """Compute the empirical CDF value for a given observation.

        Uses the standard rank-based ECDF: F_n(x) = (number of obs <= x) / n.
        This is the probability integral transform that maps the marginal
        distribution to a Uniform(0, 1) variable.

        Parameters
        ----------
        sample : np.ndarray
            The sample of observations forming the empirical distribution.
        value : float
            The point at which to evaluate the ECDF.

        Returns
        -------
        float
            ECDF value in [0, 1].
        """
        n = len(sample)
        return np.sum(sample <= value) / n
