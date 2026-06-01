# Quant Portfolio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Democratizing quantitative finance** - A comprehensive Python library for portfolio construction, backtesting, and optimization using the Baostock API for Chinese A-share market data.

## Features

- **100 Trading Strategies** across 8 categories (Momentum, Mean Reversion, Trend Following, Volatility, Statistical Arbitrage, Factor, Machine Learning, Technical)
- **100 Optimization Algorithms** across 7 categories (Portfolio Optimization, Risk Management, Asset Allocation, Signal Processing, Statistical Methods, Numerical Methods, ML Optimization)
- **Backtesting Engine** with transaction costs, slippage modeling, and comprehensive performance metrics
- **Portfolio Constructor** combining strategy signals with optimization algorithms for weight allocation
- **Risk Management** utilities including VaR, CVaR, beta, tracking error, and more
- **Baostock Integration** for seamless Chinese A-share market data access
- **Modular Architecture** - mix and match strategies, algorithms, and data sources

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/portforlio_construction.git
cd portforlio_construction

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e '.[dev]'
```

## Quick Start

```python
import numpy as np
import pandas as pd
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.portfolio.constructor import PortfolioConstructor

# Generate synthetic data (or use Baostock for real data)
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252, freq="B")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
data = pd.DataFrame({
    "open": close * (1 + np.random.normal(0, 0.005, 252)),
    "high": close * (1 + np.abs(np.random.normal(0, 0.01, 252))),
    "low": close * (1 - np.abs(np.random.normal(0, 0.01, 252))),
    "close": close,
    "volume": np.random.randint(1_000_000, 10_000_000, 252).astype(float),
}, index=dates)

# Apply a strategy
StrategyClass = get_strategy("simple_momentum")
strategy = StrategyClass()
signals = strategy.generate_signals(data)

# Run a backtest
backtester = Backtester(strategy=strategy, initial_capital=1_000_000)
result = backtester.run(data)
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# Optimize portfolio weights
AlgoClass = get_algorithm("equal_weight")
algo = AlgoClass()
returns = data["close"].pct_change().dropna()
multi_returns = pd.DataFrame(np.random.normal(0.0005, 0.02, (251, 5)),
                             index=dates[1:], columns=[f"asset_{i}" for i in range(5)])
weights = algo.optimize(multi_returns)
print(f"Weights: {weights}")
```

## Strategy Catalog

### Momentum Strategies (15)

| Name | Description |
|------|-------------|
| `simple_momentum` | Simple price momentum strategy |
| `macd_momentum` | MACD crossover momentum strategy |
| `rsi_momentum` | RSI-based momentum strategy |
| `dual_momentum` | Dual momentum combining absolute and relative momentum |
| `cross_sectional_momentum` | Cross-sectional momentum strategy |
| `time_series_momentum` | Time-series momentum (TSMOM) strategy |
| `adaptive_momentum` | Adaptive momentum with dynamic lookback |
| `momentum_acceleration` | Momentum acceleration strategy |
| `momentum_reversal` | Momentum reversal strategy |
| `risk_adjusted_momentum` | Risk-adjusted momentum strategy |
| `relative_strength` | Relative strength strategy vs benchmark |
| `sector_momentum` | Sector momentum rotation strategy |
| `volume_weighted_momentum` | Volume-weighted momentum strategy |
| `idiosyncratic_momentum` | Idiosyncratic (residual) momentum strategy |
| `earnings_momentum` | Earnings momentum strategy using volume-price proxy |

### Mean Reversion Strategies (15)

| Name | Description |
|------|-------------|
| `bollinger_bands` | Bollinger Bands mean reversion strategy |
| `rsi_reversion` | RSI-based mean reversion strategy |
| `zscore_reversion` | Z-score mean reversion strategy |
| `pairs_trading` | Pairs trading using spread z-score |
| `ornstein_uhlenbeck` | Ornstein-Uhlenbeck process-based mean reversion |
| `moving_average_reversion` | Moving average distance mean reversion |
| `half_life_mr` | Half-life based mean reversion strategy |
| `kalman_filter_mr` | Kalman filter-based mean reversion |
| `cointegration_basket` | Basket mean reversion strategy |
| `adf_based` | ADF test-based mean reversion strategy |
| `johansen_cointegration` | Johansen cointegration-based mean reversion |
| `variance_ratio` | Variance ratio test-based mean reversion |
| `hurst_exponent` | Hurst exponent regime detection strategy |
| `entropy_reversion` | Entropy-based mean reversion strategy |
| `mean_reversion_portfolio` | Portfolio-level mean reversion strategy |

### Trend Following Strategies (10)

| Name | Description |
|------|-------------|
| `moving_average_crossover` | Dual moving average crossover trend following |
| `adx_trend` | ADX-based trend strength strategy |
| `aroon_trend` | Aroon indicator trend strategy |
| `ichimoku` | Ichimoku Cloud trading strategy |
| `supertrend` | ATR-based Supertrend trend following |
| `turtle_trading` | Turtle trading using Donchian channel breakouts |
| `keltner_channel` | Keltner Channel trend following strategy |
| `parabolic_sar` | Parabolic SAR trend following strategy |
| `heikin_ashi` | Heikin-Ashi candle-based trend strategy |
| `linear_regression_channel` | Linear regression channel trend strategy |

### Volatility Strategies (10)

| Name | Description |
|------|-------------|
| `garch_vol` | GARCH-based volatility prediction strategy |
| `vol_breakout` | Volatility breakout strategy |
| `vol_mean_reversion` | Volatility mean reversion strategy |
| `vol_momentum` | Volatility momentum strategy |
| `vol_regime` | Volatility regime switching strategy |
| `vol_risk_premium` | Volatility risk premium harvesting strategy |
| `vol_surface` | Volatility term structure strategy |
| `vol_targeting` | Volatility targeting strategy |
| `implied_realized_spread` | Implied vs realized volatility spread strategy |
| `straddle_strategy` | Synthetic straddle strategy |

### Statistical Arbitrage Strategies (10)

| Name | Description |
|------|-------------|
| `copula_arb` | Copula-based dependency trading strategy |
| `dispersion_trading` | Dispersion trading strategy |
| `etf_arb` | ETF vs synthetic NAV arbitrage strategy |
| `factor_neutral_arb` | Factor-neutral statistical arbitrage strategy |
| `lead_lag` | Lead-lag relationship exploitation strategy |
| `mean_field_arb` | Mean-field statistical arbitrage strategy |
| `microstructure_arb` | Microstructure-based arbitrage using volume imbalance |
| `pca_stat_arb` | PCA-based statistical arbitrage strategy |
| `regime_switching_arb` | Regime-dependent spread trading strategy |
| `relative_value` | Relative value convergence strategy |

### Factor Strategies (15)

| Name | Description |
|------|-------------|
| `value_factor` | Value factor strategy using price-based value proxy |
| `quality_factor` | Quality factor strategy using price stability |
| `size_factor` | Size factor strategy using volume as market cap proxy |
| `low_volatility` | Low volatility anomaly strategy |
| `dividend_yield` | Dividend yield factor strategy |
| `growth_factor` | Growth factor strategy using price momentum proxy |
| `beta_factor` | Betting against beta factor strategy |
| `liquidity_factor` | Liquidity factor strategy using Amihud illiquidity |
| `investment_factor` | Conservative investment factor strategy |
| `accruals_factor` | Accruals anomaly strategy using volume-price divergence |
| `multi_factor` | Combined multi-factor strategy |
| `fundamental_factor` | Composite fundamental factor strategy |
| `smart_beta` | Smart beta strategy using risk-weighted signals |
| `technical_factor` | Composite technical factor strategy |
| `composite_alpha` | Composite alpha combination strategy |

### Machine Learning Strategies (10)

| Name | Description |
|------|-------------|
| `logistic_regression` | Logistic regression probability strategy |
| `random_forest_strategy` | Random Forest classification strategy |
| `gradient_boosting` | Gradient Boosting classification strategy |
| `svm_strategy` | SVM classification strategy |
| `decision_tree` | Decision tree classification strategy |
| `naive_bayes` | Gaussian Naive Bayes classification strategy |
| `kmeans_regime` | K-means regime clustering strategy |
| `pca_strategy` | PCA dimensionality reduction strategy |
| `lstm_strategy` | Simple recurrent neural network strategy (numpy-based) |
| `ensemble_strategy` | Ensemble of multiple ML models strategy |

### Technical Strategies (15)

| Name | Description |
|------|-------------|
| `accumulation_distribution` | Accumulation/Distribution line strategy |
| `cci_strategy` | CCI oscillator strategy |
| `chaikin_oscillator` | Chaikin Oscillator strategy |
| `dmi_strategy` | Directional Movement Index strategy |
| `elder_ray` | Elder Ray bull/bear power strategy |
| `fibonacci_retracement` | Fibonacci retracement level strategy |
| `force_index` | Elder Force Index strategy |
| `money_flow` | Money Flow Index (MFI) strategy |
| `obv_strategy` | On-Balance Volume (OBV) strategy |
| `pivot_points` | Pivot point support/resistance strategy |
| `stochastic_oscillator` | Stochastic %K/%D oscillator strategy |
| `trix_strategy` | TRIX (triple exponential moving average) strategy |
| `ultimate_oscillator` | Ultimate Oscillator strategy |
| `vwap_strategy` | VWAP deviation strategy |
| `williams_r` | Williams %R oscillator strategy |

## Algorithm Catalog

### Portfolio Optimization (20)

| Name | Description |
|------|-------------|
| `mean_variance` | Markowitz Mean-Variance Optimization |
| `min_variance` | Global Minimum Variance Portfolio |
| `max_sharpe` | Maximum Sharpe Ratio Portfolio |
| `risk_parity` | Risk Parity / Equal Risk Contribution |
| `black_litterman` | Black-Litterman Model |
| `hierarchical_risk_parity` | Hierarchical Risk Parity (HRP) |
| `max_diversification` | Maximum Diversification Portfolio |
| `robust_optimization` | Robust Mean-Variance Optimization |
| `cvar_optimization` | CVaR Portfolio Optimization |
| `omega_ratio` | Omega Ratio Maximization |
| `kelly_criterion` | Kelly Criterion for Optimal Sizing |
| `resampled_efficient_frontier` | Resampled Efficient Frontier (Michaud) |
| `inverse_variance` | Inverse Variance Weighting |
| `equal_weight` | Equal Weight (1/N) Portfolio |
| `most_diversified` | Most Diversified Portfolio |
| `minimum_cvar` | Minimum CVaR Portfolio |
| `target_return` | Target Return Optimization |
| `factor_risk_parity` | Factor Risk Parity |
| `tail_risk_parity` | Tail Risk Parity |
| `entropy_pooling` | Entropy Pooling (Meucci) |

### Risk Management (15)

| Name | Description |
|------|-------------|
| `var_historical` | Historical Value-at-Risk Optimization |
| `var_parametric` | Parametric (Gaussian) VaR Optimization |
| `var_monte_carlo` | Monte Carlo VaR Optimization |
| `cvar_calculation` | Expected Shortfall / CVaR Weights |
| `stress_testing` | Stress Testing Portfolio Optimization |
| `drawdown_control` | Drawdown Control Optimization |
| `position_sizing` | Optimal Position Sizing (Kelly) |
| `stop_loss` | Dynamic Stop-Loss Weighting |
| `correlation_regime` | Correlation Regime Detection |
| `tail_risk` | Tail Risk Measurement (EVT-based) |
| `risk_budgeting` | Risk Budgeting Across Assets |
| `marginal_risk` | Marginal Risk Contribution Weighting |
| `factor_risk_decomposition` | Factor Risk Decomposition |
| `liquidity_risk` | Liquidity-Adjusted Risk Allocation |
| `concentration_risk` | Concentration Risk Minimization |

### Asset Allocation (15)

| Name | Description |
|------|-------------|
| `strategic_allocation` | Strategic Asset Allocation |
| `tactical_allocation` | Tactical Asset Allocation |
| `dynamic_allocation` | Dynamic Asset Allocation |
| `constant_proportion` | CPPI Constant Proportion |
| `target_date` | Target Date / Lifecycle Glide Path |
| `momentum_allocation` | Momentum-Based Allocation |
| `risk_on_off` | Risk-On / Risk-Off Regime |
| `min_correlation` | Minimum Correlation Portfolio |
| `max_decorrelation` | Maximum Decorrelation Portfolio |
| `core_satellite` | Core-Satellite Approach |
| `liability_driven` | Liability-Driven Investment |
| `goal_based` | Goal-Based Allocation |
| `regime_based_allocation` | Regime-Based Conditional Allocation |
| `volatility_weighted` | Volatility-Weighted Allocation |
| `drawdown_based` | Drawdown-Based Dynamic Allocation |

### Signal Processing (10)

| Name | Description |
|------|-------------|
| `kalman_filter` | Kalman Filter for Signal Extraction |
| `wavelet_transform` | Wavelet Transform Denoising |
| `fourier_analysis` | FFT Frequency Decomposition |
| `hodrick_prescott` | Hodrick-Prescott Filter |
| `savitzky_golay` | Savitzky-Golay Smoothing |
| `exponential_smoothing` | Holt-Winters Double Exponential Smoothing |
| `bandpass_filter` | Bandpass Signal Filtering |
| `hilbert_transform` | Hilbert Transform Instantaneous Phase |
| `emd_decomposition` | Empirical Mode Decomposition (EMD) |
| `particle_filter` | Particle Filter State Estimation |

### Statistical Methods (15)

| Name | Description |
|------|-------------|
| `covariance_estimation` | Ledoit-Wolf Shrinkage Covariance |
| `pca_decomposition` | PCA Factor Extraction |
| `independent_component` | ICA Signal Separation |
| `bayesian_estimation` | Bayesian Parameter Estimation |
| `bootstrap_methods` | Bootstrap Confidence Intervals |
| `copula_modeling` | Copula Dependence Modeling |
| `regime_detection` | Hidden Markov Model Regimes |
| `changepoint_detection` | Structural Break Detection |
| `density_estimation` | Kernel Density Estimation |
| `extreme_value` | Extreme Value Theory (EVT) |
| `robust_statistics` | Robust Estimators (MAD, Huber) |
| `time_series_decomposition` | STL Time Series Decomposition |
| `granger_causality` | Granger Causality Testing |
| `dynamic_correlation` | DCC-GARCH Dynamic Correlations |
| `factor_analysis` | Statistical Factor Analysis |

### Numerical Methods (10)

| Name | Description |
|------|-------------|
| `gradient_descent` | Projected Gradient Descent |
| `newton_method` | Newton's Method Optimizer |
| `simulated_annealing` | Simulated Annealing |
| `genetic_algorithm` | Genetic Algorithm Optimization |
| `particle_swarm` | Particle Swarm Optimization (PSO) |
| `differential_evolution` | Differential Evolution |
| `quadratic_programming` | QP Solver Wrapper |
| `linear_programming` | LP Constraint Handling |
| `monte_carlo_simulation` | Monte Carlo Simulation Engine |
| `convex_optimization` | General Convex Optimization |

### Machine Learning Optimization (15)

| Name | Description |
|------|-------------|
| `neural_network_optimizer` | Neural Network Weight Optimization |
| `reinforcement_learning` | Q-Learning Based Allocation |
| `bayesian_optimization` | Bayesian Hyperparameter Optimization |
| `online_learning` | Online / Incremental Learning |
| `transfer_learning` | Transfer Learning Domain Adaptation |
| `attention_weighting` | Attention Mechanism Asset Weighting |
| `autoencoder_features` | Autoencoder Feature Extraction |
| `clustering_allocation` | K-Means Clustering Allocation |
| `manifold_learning` | Manifold-Based Dimensionality Reduction |
| `graph_based` | Graph/Network-Based Allocation |
| `meta_learning` | Meta-Learning Across Market Regimes |
| `multi_objective` | Multi-Objective Optimization (Pareto) |
| `adversarial_robustness` | Adversarial Robustness in Optimization |
| `variational_inference` | Variational Inference for Uncertainty |
| `gaussian_process` | GP Regression for Return Prediction |

## Backtesting

```python
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.backtesting.engine import Backtester

# Get a strategy from the registry
StrategyClass = get_strategy("dual_momentum")
strategy = StrategyClass()

# Configure the backtester
backtester = Backtester(
    strategy=strategy,
    initial_capital=1_000_000,
    commission=0.001,    # 0.1% commission
    slippage=0.0005,     # 0.05% slippage
)

# Run the backtest
result = backtester.run(data)

# Access results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
print(f"Calmar Ratio: {result.metrics['calmar_ratio']:.3f}")
```

## Portfolio Construction

```python
from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.portfolio.constructor import PortfolioConstructor

# Choose an optimization algorithm
AlgoClass = get_algorithm("risk_parity")
algo = AlgoClass()

# Build the portfolio constructor
constructor = PortfolioConstructor(
    algorithm=algo,
    rebalance_frequency="monthly",  # 'daily', 'weekly', 'monthly'
)

# Construct portfolio from signals and returns
weights = constructor.construct(signals_df, returns_df)
print(weights.tail())
```

## Project Structure

```
portforlio_construction/
├── src/quant_portfolio/
│   ├── __init__.py
│   ├── strategies/           # 100 trading strategies
│   │   ├── base.py          # Strategy abstract base class
│   │   ├── registry.py      # Auto-discovery registry
│   │   ├── momentum/        # 15 momentum strategies
│   │   ├── mean_reversion/  # 15 mean reversion strategies
│   │   ├── trend_following/ # 10 trend following strategies
│   │   ├── volatility/      # 10 volatility strategies
│   │   ├── stat_arb/        # 10 statistical arbitrage strategies
│   │   ├── factor/          # 15 factor strategies
│   │   ├── ml/              # 10 machine learning strategies
│   │   └── technical/       # 15 technical strategies
│   ├── algorithms/           # 100 optimization algorithms
│   │   ├── base.py          # Algorithm abstract base class
│   │   ├── registry.py      # Auto-discovery registry
│   │   ├── optimization/    # 20 portfolio optimization
│   │   ├── risk/            # 15 risk management
│   │   ├── allocation/      # 15 asset allocation
│   │   ├── signal_processing/ # 10 signal processing
│   │   ├── statistical/     # 15 statistical methods
│   │   ├── numerical/       # 10 numerical methods
│   │   └── ml_optimization/ # 15 ML optimization
│   ├── backtesting/          # Backtesting engine
│   │   ├── engine.py        # Backtester class
│   │   └── metrics.py       # Performance metrics
│   ├── portfolio/            # Portfolio construction
│   │   ├── constructor.py   # PortfolioConstructor class
│   │   └── risk.py          # Risk analysis utilities
│   └── data/                 # Data layer
│       ├── baostock_provider.py  # Baostock API integration
│       └── data_utils.py    # Data utilities
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── docs/                     # Documentation
├── pyproject.toml
├── CONTRIBUTING.md
└── LICENSE
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new strategies, algorithms, and other contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
