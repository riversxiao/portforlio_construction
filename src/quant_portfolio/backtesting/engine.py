"""Backtesting engine for strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quant_portfolio.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quant_portfolio.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Container for backtest results.

    Attributes
    ----------
    returns : pd.Series
        Series of portfolio returns.
    equity_curve : pd.Series
        Cumulative portfolio value over time.
    positions : pd.DataFrame
        Position history.
    trades : pd.DataFrame
        Trade log.
    metrics : Dict[str, float]
        Performance metrics dictionary.
    """

    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """Backtesting engine that simulates strategy execution.

    Takes a strategy and price data, simulates trades with transaction costs,
    and computes performance metrics.

    Parameters
    ----------
    strategy : Strategy
        The strategy instance to backtest.
    initial_capital : float
        Starting capital for the simulation.
    commission : float
        Commission rate per trade (e.g., 0.001 for 0.1%).
    slippage : float
        Slippage per trade as a fraction of price.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
    ) -> BacktestResult:
        """Run the backtest on the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Market data with DatetimeIndex and at least a price column.
        price_col : str
            Name of the column to use as the execution price.

        Returns
        -------
        BacktestResult
            Object containing returns, equity curve, positions, trades, and metrics.
        """
        # Generate signals
        signals = self.strategy.generate_signals(data)

        if "signal" not in signals.columns:
            raise ValueError(
                "Strategy must return a DataFrame with a 'signal' column."
            )

        # Align signals with price data
        prices = data[price_col].copy()
        signal_series = signals["signal"].reindex(prices.index).fillna(0)

        # Simulate positions
        positions = signal_series.shift(1).fillna(0)  # Trade on next bar

        # Calculate returns
        price_returns = prices.pct_change().fillna(0)
        strategy_returns = positions * price_returns

        # Apply transaction costs
        trades_mask = positions.diff().fillna(0).abs()
        costs = trades_mask * (self.commission + self.slippage)
        strategy_returns = strategy_returns - costs

        # Build equity curve
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        # Build trade log
        trade_indices = trades_mask[trades_mask > 0].index
        trades_df = pd.DataFrame(
            {
                "date": trade_indices,
                "position_change": positions.diff().loc[trade_indices],
                "price": prices.loc[trade_indices],
            }
        )

        # Compute metrics
        num_trades = int((trades_mask > 0).sum())
        metrics = self._compute_metrics(strategy_returns, num_trades=num_trades)

        return BacktestResult(
            returns=strategy_returns,
            equity_curve=equity_curve,
            positions=positions.to_frame(name="position"),
            trades=trades_df,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        num_trades: int = 0,
    ) -> Dict[str, float]:
        """Compute performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        periods_per_year : int
            Number of trading periods per year.
        num_trades : int
            Number of actual position changes (trades).

        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics.
        """
        return {
            "annualized_return": annualized_return(returns, periods_per_year),
            "annualized_volatility": annualized_volatility(
                returns, periods_per_year
            ),
            "sharpe_ratio": sharpe_ratio(returns, 0.0, periods_per_year),
            "sortino_ratio": sortino_ratio(returns, 0.0, periods_per_year),
            "max_drawdown": max_drawdown(returns),
            "calmar_ratio": calmar_ratio(returns, periods_per_year),
            "total_return": float((1 + returns).prod() - 1),
            "num_trades": num_trades,
        }
