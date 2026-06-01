# Strategy Documentation

This document provides detailed documentation for all 100 trading strategies in the Quant Portfolio library.

## Overview

All strategies extend the `Strategy` base class and implement the `generate_signals(data)` method, which takes an OHLCV DataFrame and returns a DataFrame with a `signal` column containing values of 1 (buy), -1 (sell), or 0 (hold).

```python
from quant_portfolio.strategies.registry import get_strategy

StrategyClass = get_strategy("simple_momentum")
strategy = StrategyClass(lookback=20)
signals = strategy.generate_signals(ohlcv_data)
```

---

## Momentum Strategies (15)

Momentum strategies exploit the tendency of assets that have performed well (or poorly) to continue performing well (or poorly) in the near term.

### simple_momentum
Simple price momentum strategy. Computes returns over a lookback period and generates buy signals when momentum is positive, sell signals when negative.

**Parameters:** `lookback` (default: 20)

### macd_momentum
MACD crossover momentum strategy. Uses the difference between short-term and long-term exponential moving averages to identify momentum shifts.

**Parameters:** `fast_period` (12), `slow_period` (26), `signal_period` (9)

### rsi_momentum
RSI-based momentum strategy. Generates buy signals when RSI indicates upward momentum above a threshold, sell when below.

**Parameters:** `period` (14), `upper_threshold` (60), `lower_threshold` (40)

### dual_momentum
Dual momentum combining absolute and relative momentum. Requires both time-series and cross-sectional momentum to be positive for a buy signal.

**Parameters:** `abs_lookback` (252), `rel_lookback` (126)

### cross_sectional_momentum
Cross-sectional momentum. Ranks assets by their returns and goes long top performers, short bottom performers.

**Parameters:** `lookback` (60), `top_pct` (0.2)

### time_series_momentum
Time-series momentum (TSMOM). Trades each asset based on its own past return regardless of other assets.

**Parameters:** `lookback` (252), `vol_target` (0.15)

### adaptive_momentum
Adaptive momentum with dynamic lookback that adjusts based on recent performance effectiveness.

**Parameters:** `min_lookback` (10), `max_lookback` (60)

### momentum_acceleration
Momentum acceleration strategy. Identifies assets where momentum is increasing in magnitude.

**Parameters:** `lookback` (20), `accel_lookback` (5)

### momentum_reversal
Momentum reversal strategy. Combines short-term reversal with long-term momentum.

**Parameters:** `short_lookback` (5), `long_lookback` (252)

### risk_adjusted_momentum
Risk-adjusted momentum. Scales momentum signals by inverse volatility for better risk-adjusted performance.

**Parameters:** `lookback` (60), `vol_lookback` (20)

### relative_strength
Relative strength vs benchmark using moving average proxy. Compares asset performance to a synthetic benchmark.

**Parameters:** `lookback` (60)

### sector_momentum
Sector momentum rotation. Rotates among sectors based on relative momentum strength.

**Parameters:** `lookback` (20), `holding_period` (20)

### volume_weighted_momentum
Volume-weighted momentum. Weights momentum signals by trading volume for confirmation.

**Parameters:** `lookback` (20)

### idiosyncratic_momentum
Idiosyncratic (residual) momentum. Removes market-beta component and trades on residual momentum.

**Parameters:** `lookback` (60), `market_lookback` (252)

### earnings_momentum
Earnings momentum using volume-price proxy for earnings surprise detection.

**Parameters:** `lookback` (20), `vol_threshold` (1.5)

---

## Mean Reversion Strategies (15)

Mean reversion strategies exploit the tendency of asset prices to revert to a mean or equilibrium level after deviating.

### bollinger_bands
Bollinger Bands mean reversion. Buys when price touches the lower band, sells at the upper band.

**Parameters:** `period` (20), `num_std` (2.0)

### rsi_reversion
RSI-based mean reversion. Buys oversold conditions (RSI < 30), sells overbought (RSI > 70).

**Parameters:** `period` (14), `oversold` (30), `overbought` (70)

### zscore_reversion
Z-score mean reversion. Trades based on the z-score of price relative to its rolling mean.

**Parameters:** `lookback` (20), `entry_z` (2.0), `exit_z` (0.5)

### pairs_trading
Pairs trading using spread z-score. Trades the mean-reverting spread between two correlated assets.

**Parameters:** `lookback` (60), `entry_z` (2.0), `exit_z` (0.5)

### ornstein_uhlenbeck
Ornstein-Uhlenbeck process-based mean reversion. Models price as an OU process and trades deviations.

**Parameters:** `lookback` (60), `entry_threshold` (1.5)

### moving_average_reversion
Moving average distance mean reversion. Trades when price deviates significantly from its moving average.

**Parameters:** `period` (50), `threshold` (0.02)

### half_life_mr
Half-life based mean reversion. Uses the half-life of mean reversion to set optimal holding periods.

**Parameters:** `lookback` (60)

### kalman_filter_mr
Kalman filter-based mean reversion. Uses a Kalman filter to estimate the mean-reverting level.

**Parameters:** `transition_covariance` (0.01)

### cointegration_basket
Basket mean reversion strategy. Constructs a cointegrated basket and trades deviations.

**Parameters:** `lookback` (60), `entry_z` (2.0)

### adf_based
ADF test-based mean reversion. Uses the Augmented Dickey-Fuller test to confirm stationarity before trading.

**Parameters:** `lookback` (60), `pvalue_threshold` (0.05)

### johansen_cointegration
Johansen cointegration-based mean reversion. Multi-asset cointegration testing for spread construction.

**Parameters:** `lookback` (120), `entry_z` (2.0)

### variance_ratio
Variance ratio test-based mean reversion. Tests for mean reversion using the variance ratio statistic.

**Parameters:** `period` (20), `threshold` (0.8)

### hurst_exponent
Hurst exponent regime detection. Uses the Hurst exponent to identify mean-reverting vs trending regimes.

**Parameters:** `lookback` (100), `threshold` (0.4)

### entropy_reversion
Entropy-based mean reversion. Uses information entropy to detect overextended price moves.

**Parameters:** `lookback` (30), `threshold` (1.5)

### mean_reversion_portfolio
Portfolio-level mean reversion. Applies mean reversion logic at the portfolio level rather than individual assets.

**Parameters:** `lookback` (60), `entry_z` (2.0)

---

## Trend Following Strategies (10)

Trend following strategies aim to capture large directional moves by identifying and following established trends.

### moving_average_crossover
Dual moving average crossover. Generates buy when short MA crosses above long MA, sell on cross below.

**Parameters:** `short_window` (20), `long_window` (50)

### adx_trend
ADX-based trend strength. Only trades in the direction of the trend when ADX confirms trend strength.

**Parameters:** `period` (14), `threshold` (25)

### aroon_trend
Aroon indicator trend strategy. Uses Aroon Up/Down to identify trend direction and strength.

**Parameters:** `period` (25)

### ichimoku
Ichimoku Cloud trading strategy. Uses the full Ichimoku system with Tenkan, Kijun, and Cloud.

**Parameters:** `tenkan` (9), `kijun` (26), `senkou_b` (52)

### supertrend
ATR-based Supertrend indicator. Follows price with an adaptive trailing stop based on ATR.

**Parameters:** `period` (10), `multiplier` (3.0)

### turtle_trading
Turtle trading using Donchian channel breakouts. Buys on new highs, sells on new lows.

**Parameters:** `entry_period` (20), `exit_period` (10)

### keltner_channel
Keltner Channel trend following. Uses ATR-based channels around an EMA for trend signals.

**Parameters:** `period` (20), `multiplier` (2.0)

### parabolic_sar
Parabolic SAR trend following. Uses the parabolic stop-and-reverse indicator for entries and exits.

**Parameters:** `af_start` (0.02), `af_max` (0.2)

### heikin_ashi
Heikin-Ashi candle-based trend strategy. Smooths price action using modified candlesticks.

**Parameters:** `confirmation_periods` (3)

### linear_regression_channel
Linear regression channel trend strategy. Trades in the direction of a linear regression slope.

**Parameters:** `period` (50), `num_std` (2.0)

---

## Volatility Strategies (10)

Volatility strategies trade on predicted changes in volatility or use volatility as a signal.

### garch_vol
GARCH-based volatility prediction. Forecasts volatility using a GARCH(1,1) model.

**Parameters:** `lookback` (252)

### vol_breakout
Volatility breakout strategy. Enters positions when realized volatility breaks above recent range.

**Parameters:** `lookback` (20), `multiplier` (1.5)

### vol_mean_reversion
Volatility mean reversion. Trades on the assumption that volatility reverts to its long-term mean.

**Parameters:** `short_lookback` (10), `long_lookback` (60)

### vol_momentum
Volatility momentum strategy. Trades in the direction implied by volatility trends.

**Parameters:** `lookback` (20)

### vol_regime
Volatility regime switching. Identifies high/low volatility regimes and adjusts positions accordingly.

**Parameters:** `lookback` (60), `threshold` (1.5)

### vol_risk_premium
Volatility risk premium harvesting. Captures the spread between implied and realized volatility.

**Parameters:** `lookback` (20)

### vol_surface
Volatility term structure strategy. Trades based on the shape of the volatility term structure.

**Parameters:** `short_lookback` (5), `long_lookback` (20)

### vol_targeting
Volatility targeting strategy. Scales exposure to maintain a constant portfolio volatility.

**Parameters:** `target_vol` (0.15), `lookback` (20)

### implied_realized_spread
Implied vs realized volatility spread. Trades the gap between implied and realized vol.

**Parameters:** `lookback` (20)

### straddle_strategy
Synthetic straddle strategy. Creates synthetic straddle positions using delta-neutral signals.

**Parameters:** `lookback` (20), `threshold` (1.5)

---

## Statistical Arbitrage Strategies (10)

Statistical arbitrage strategies exploit statistical mispricings across related assets.

### copula_arb
Copula-based dependency trading. Models dependency structure and trades breakdowns.

**Parameters:** `lookback` (60), `threshold` (0.05)

### dispersion_trading
Dispersion trading. Trades the difference between index and single-stock volatilities.

**Parameters:** `lookback` (20)

### etf_arb
ETF vs synthetic NAV arbitrage. Exploits deviations between ETF price and its net asset value.

**Parameters:** `lookback` (20), `threshold` (0.01)

### factor_neutral_arb
Factor-neutral statistical arbitrage. Constructs market-neutral portfolios with factor hedging.

**Parameters:** `lookback` (60)

### lead_lag
Lead-lag relationship exploitation. Identifies and trades predictive relationships between assets.

**Parameters:** `lookback` (20), `lag` (1)

### mean_field_arb
Mean-field statistical arbitrage. Uses mean-field theory to model collective market behavior.

**Parameters:** `lookback` (60)

### microstructure_arb
Microstructure-based arbitrage using volume imbalance. Trades on order flow imbalance signals.

**Parameters:** `lookback` (10)

### pca_stat_arb
PCA-based statistical arbitrage. Uses PCA to find mispricings relative to factor structure.

**Parameters:** `lookback` (60), `n_components` (3)

### regime_switching_arb
Regime-dependent spread trading. Adjusts spread trading based on detected market regimes.

**Parameters:** `lookback` (60)

### relative_value
Relative value convergence. Trades pairs expected to converge based on fundamental value.

**Parameters:** `lookback` (60), `entry_z` (2.0)

---

## Factor Strategies (15)

Factor strategies systematically capture risk premia associated with well-known factors.

### value_factor
Value factor using price-based value proxy. Identifies undervalued assets based on price metrics.

**Parameters:** `lookback` (252)

### quality_factor
Quality factor using price stability as quality proxy. Selects assets with consistent returns.

**Parameters:** `lookback` (252)

### size_factor
Size factor using volume as market cap proxy. Tilts toward smaller-cap (lower volume) assets.

**Parameters:** `lookback` (60)

### low_volatility
Low volatility anomaly. Overweights low-volatility assets which historically outperform on risk-adjusted basis.

**Parameters:** `lookback` (60)

### dividend_yield
Dividend yield factor using price stability proxy. Identifies yield-like characteristics from price data.

**Parameters:** `lookback` (252)

### growth_factor
Growth factor using price momentum as growth proxy. Captures growth characteristics.

**Parameters:** `lookback` (126)

### beta_factor
Betting against beta. Goes long low-beta assets and short high-beta assets.

**Parameters:** `lookback` (252)

### liquidity_factor
Liquidity factor using Amihud illiquidity measure. Captures the liquidity premium.

**Parameters:** `lookback` (20)

### investment_factor
Conservative investment factor. Favors assets with conservative investment profiles.

**Parameters:** `lookback` (252)

### accruals_factor
Accruals anomaly using volume-price divergence. Identifies accrual-like patterns.

**Parameters:** `lookback` (60)

### multi_factor
Combined multi-factor strategy. Blends multiple factor signals into a composite score.

**Parameters:** `lookback` (60)

### fundamental_factor
Composite fundamental factor. Combines multiple fundamental-style signals.

**Parameters:** `lookback` (252)

### smart_beta
Smart beta using risk-weighted signals. Alternative index construction methodology.

**Parameters:** `lookback` (60)

### technical_factor
Composite technical factor. Combines multiple technical indicators into a factor score.

**Parameters:** `lookback` (20)

### composite_alpha
Composite alpha combination strategy. Aggregates multiple alpha sources with optimal weights.

**Parameters:** `lookback` (60)

---

## Machine Learning Strategies (10)

ML strategies use machine learning models to predict price movements and generate trading signals.

### logistic_regression
Logistic regression probability strategy. Predicts up/down probability using logistic regression.

**Parameters:** `lookback` (60), `n_features` (5)

### random_forest_strategy
Random Forest classification. Uses ensemble of decision trees for direction prediction.

**Parameters:** `lookback` (60), `n_trees` (10)

### gradient_boosting
Gradient Boosting classification. Sequential weak learners for improved prediction.

**Parameters:** `lookback` (60), `n_estimators` (10)

### svm_strategy
SVM classification strategy. Uses support vector machines with RBF kernel.

**Parameters:** `lookback` (60)

### decision_tree
Decision tree classification. Simple tree-based classification for signal generation.

**Parameters:** `lookback` (60), `max_depth` (5)

### naive_bayes
Gaussian Naive Bayes classification. Fast probabilistic classifier assuming feature independence.

**Parameters:** `lookback` (60)

### kmeans_regime
K-means regime clustering. Identifies market regimes using K-means clustering of features.

**Parameters:** `lookback` (60), `n_clusters` (3)

### pca_strategy
PCA dimensionality reduction strategy. Reduces feature space before classification.

**Parameters:** `lookback` (60), `n_components` (3)

### lstm_strategy
Simple recurrent neural network (numpy-based). Simplified LSTM-like approach without deep learning frameworks.

**Parameters:** `lookback` (60), `hidden_size` (10)

### ensemble_strategy
Ensemble of multiple ML models. Combines predictions from multiple models via voting.

**Parameters:** `lookback` (60)

---

## Technical Strategies (15)

Technical strategies use price and volume-based indicators for signal generation.

### accumulation_distribution
Accumulation/Distribution line. Measures cumulative money flow to identify divergences.

**Parameters:** `lookback` (20)

### cci_strategy
Commodity Channel Index oscillator. Identifies cyclical trends in price.

**Parameters:** `period` (20), `threshold` (100)

### chaikin_oscillator
Chaikin Oscillator. Measures the momentum of the Accumulation/Distribution line.

**Parameters:** `fast_period` (3), `slow_period` (10)

### dmi_strategy
Directional Movement Index. Uses +DI and -DI crossovers for trend direction.

**Parameters:** `period` (14)

### elder_ray
Elder Ray bull/bear power. Measures buying and selling pressure relative to EMA.

**Parameters:** `period` (13)

### fibonacci_retracement
Fibonacci retracement levels. Uses key Fibonacci ratios as support/resistance levels.

**Parameters:** `lookback` (50)

### force_index
Elder Force Index. Combines price change and volume to measure the force behind moves.

**Parameters:** `period` (13)

### money_flow
Money Flow Index (MFI). Volume-weighted RSI indicating buying/selling pressure.

**Parameters:** `period` (14), `oversold` (20), `overbought` (80)

### obv_strategy
On-Balance Volume (OBV). Cumulative volume indicator for confirming trends.

**Parameters:** `lookback` (20)

### pivot_points
Pivot point support/resistance. Uses daily pivots to identify key price levels.

**Parameters:** None

### stochastic_oscillator
Stochastic %K/%D oscillator. Compares closing price to price range over a period.

**Parameters:** `k_period` (14), `d_period` (3)

### trix_strategy
TRIX triple exponential moving average. Measures rate of change of a triple-smoothed EMA.

**Parameters:** `period` (15)

### ultimate_oscillator
Ultimate Oscillator. Multi-timeframe oscillator combining three periods.

**Parameters:** `period1` (7), `period2` (14), `period3` (28)

### vwap_strategy
VWAP deviation strategy. Trades deviations from volume-weighted average price.

**Parameters:** `lookback` (20), `threshold` (1.5)

### williams_r
Williams %R oscillator. Measures overbought/oversold conditions relative to high-low range.

**Parameters:** `period` (14), `oversold` (-80), `overbought` (-20)
