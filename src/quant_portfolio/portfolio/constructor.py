"""Portfolio constructor combining strategies with optimization algorithms."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm
from quant_portfolio.strategies.base import Strategy


class PortfolioConstructor:
    """Combines strategy signals with algorithm optimization to produce weights.

    The constructor takes multiple strategies and an optimization algorithm,
    then produces final portfolio weights that reflect both signal strength
    and risk/return optimization.

    Parameters
    ----------
    algorithm : Algorithm
        The optimization algorithm to use for weight allocation.
    rebalance_frequency : str
        How often to rebalance: 'daily', 'weekly', 'monthly'.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        rebalance_frequency: str = "monthly",
    ) -> None:
        self.algorithm = algorithm
        self.rebalance_frequency = rebalance_frequency

    def construct(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Construct portfolio weights from signals and returns data.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame of trading signals for multiple assets.
            Columns are asset names, values are signal strengths.
        returns : pd.DataFrame
            DataFrame of asset returns with DatetimeIndex.
        **kwargs : Any
            Additional parameters passed to the optimization algorithm.

        Returns
        -------
        pd.DataFrame
            DataFrame of portfolio weights over time.
            Columns are asset names, values are weights summing to ~1.0.
        """
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(returns.index)

        weights_history = []

        for date in rebalance_dates:
            # Get returns up to this date for optimization
            historical_returns = returns.loc[:date]

            if len(historical_returns) < 2:
                # Not enough data, use equal weights for selected assets
                active_assets = signals.loc[:date].iloc[-1] if len(signals.loc[:date]) > 0 else pd.Series()
                if len(active_assets) == 0:
                    continue
                selected = active_assets[active_assets != 0].index
                if len(selected) == 0:
                    weights = np.zeros(len(returns.columns))
                else:
                    weights = np.zeros(len(returns.columns))
                    mask = returns.columns.isin(selected)
                    weights[mask] = 1.0 / mask.sum()
            else:
                # Filter to assets with active signals
                if date in signals.index:
                    current_signals = signals.loc[date]
                else:
                    # Use the most recent signal
                    valid_signal_dates = signals.index[signals.index <= date]
                    if len(valid_signal_dates) == 0:
                        continue
                    current_signals = signals.loc[valid_signal_dates[-1]]

                selected = current_signals[current_signals != 0].index
                selected = selected.intersection(returns.columns)

                if len(selected) == 0:
                    weights = np.zeros(len(returns.columns))
                else:
                    # Optimize weights for selected assets
                    selected_returns = historical_returns[selected]
                    opt_weights = self.algorithm.optimize(
                        selected_returns, **kwargs
                    )

                    # Map back to full weight vector
                    weights = np.zeros(len(returns.columns))
                    for i, asset in enumerate(selected):
                        col_idx = returns.columns.get_loc(asset)
                        weights[col_idx] = opt_weights[i]

            weights_history.append(
                pd.Series(weights, index=returns.columns, name=date)
            )

        if not weights_history:
            return pd.DataFrame(
                index=returns.index, columns=returns.columns, data=0.0
            )

        weights_df = pd.DataFrame(weights_history)
        # Forward-fill weights to all dates
        weights_df = weights_df.reindex(returns.index, method="ffill").fillna(0)

        return weights_df

    def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> list:
        """Get rebalance dates based on the frequency setting.

        Parameters
        ----------
        index : pd.DatetimeIndex
            The full date index.

        Returns
        -------
        list
            List of rebalance dates.
        """
        if self.rebalance_frequency == "daily":
            return list(index)
        elif self.rebalance_frequency == "weekly":
            # Last trading day of each week
            return list(index.to_series().resample("W").last().dropna().values)
        elif self.rebalance_frequency == "monthly":
            # Last trading day of each month
            return list(index.to_series().resample("ME").last().dropna().values)
        else:
            raise ValueError(
                f"Unknown rebalance_frequency: {self.rebalance_frequency}"
            )
