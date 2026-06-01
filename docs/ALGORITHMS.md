# Algorithm Documentation

This document provides detailed documentation for all 100 optimization algorithms in the Quant Portfolio library.

## Overview

All algorithms extend the `Algorithm` base class and implement the `optimize(returns)` method, which takes a DataFrame of asset returns and returns a numpy array of portfolio weights.

```python
from quant_portfolio.algorithms.registry import get_algorithm

AlgoClass = get_algorithm("risk_parity")
algo = AlgoClass()
weights = algo.optimize(returns_df)
# weights is a numpy array summing to approximately 1.0
```

---

## Portfolio Optimization (20)

Classical and modern portfolio optimization methods for computing optimal asset weights.

### mean_variance
**Markowitz Mean-Variance Optimization**

Maximizes expected return for a given level of risk (or minimizes risk for a given return). The foundational model of modern portfolio theory.

Mathematical basis: min w'Cw - lambda * w'mu, subject to sum(w) = 1

### min_variance
**Global Minimum Variance Portfolio**

Finds the portfolio with the lowest possible variance regardless of expected returns. Useful when return estimates are unreliable.

### max_sharpe
**Maximum Sharpe Ratio Portfolio**

Finds the portfolio on the efficient frontier with the highest risk-adjusted return (Sharpe ratio).

### risk_parity
**Risk Parity / Equal Risk Contribution**

Allocates weights such that each asset contributes equally to total portfolio risk. Avoids concentration risk.

### black_litterman
**Black-Litterman Model**

Combines market equilibrium with investor views to produce stable, intuitive portfolio weights.

### hierarchical_risk_parity
**Hierarchical Risk Parity (HRP)**

Uses hierarchical clustering of the correlation matrix to build diversified portfolios without matrix inversion. By Marcos Lopez de Prado.

### max_diversification
**Maximum Diversification Portfolio**

Maximizes the diversification ratio (weighted average volatility divided by portfolio volatility).

### robust_optimization
**Robust Mean-Variance with Uncertainty Sets**

Accounts for estimation error in expected returns by optimizing over a set of plausible parameter values.

### cvar_optimization
**CVaR (Conditional Value-at-Risk) Optimization**

Minimizes expected losses in the tail of the return distribution rather than variance.

### omega_ratio
**Omega Ratio Maximization**

Maximizes the ratio of gains to losses relative to a threshold return.

### kelly_criterion
**Kelly Criterion for Optimal Sizing**

Determines position sizes that maximize the expected logarithm of wealth (geometric growth rate).

### resampled_efficient_frontier
**Resampled Efficient Frontier (Michaud, 1998)**

Averages weights across multiple simulated efficient frontiers to reduce estimation sensitivity.

### inverse_variance
**Inverse Variance Weighting**

Weights assets inversely proportional to their variance. Simple but effective diversification.

### equal_weight
**Equal Weight (1/N) Portfolio**

Assigns equal weight to all assets. Surprisingly competitive benchmark due to diversification benefits.

### most_diversified
**Most Diversified Portfolio**

Maximizes the diversification ratio defined as the ratio of weighted volatilities to portfolio volatility.

### minimum_cvar
**Minimum CVaR Portfolio**

Finds the portfolio that minimizes Conditional Value-at-Risk at a specified confidence level.

### target_return
**Target Return Optimization**

Finds the minimum-variance portfolio that achieves a specified target return.

### factor_risk_parity
**Factor Risk Parity**

Equalizes risk contributions from underlying risk factors rather than individual assets.

### tail_risk_parity
**Tail Risk Parity**

Similar to risk parity but uses tail risk measures (CVaR) instead of volatility for risk contributions.

### entropy_pooling
**Entropy Pooling (Meucci, 2008)**

Updates a prior distribution to incorporate investor views while minimizing information loss (relative entropy).

---

## Risk Management (15)

Algorithms focused on measuring, managing, and allocating risk within portfolios.

### var_historical
**Historical Value-at-Risk**

Estimates potential losses using historical return percentiles. Non-parametric approach.

### var_parametric
**Parametric (Gaussian) VaR**

Assumes normally distributed returns to compute Value-at-Risk analytically.

### var_monte_carlo
**Monte Carlo VaR**

Simulates thousands of scenarios to estimate the tail loss distribution.

### cvar_calculation
**Expected Shortfall / CVaR**

Computes the average loss in the worst alpha-percent of scenarios. More coherent than VaR.

### stress_testing
**Stress Testing Portfolio Optimization**

Optimizes portfolios considering performance under stress scenarios.

### drawdown_control
**Drawdown Control Optimization**

Constrains portfolio weights to limit maximum drawdown exposure.

### position_sizing
**Optimal Position Sizing**

Uses volatility-adjusted Kelly criterion for position size determination.

### stop_loss
**Dynamic Stop-Loss Weighting**

Adjusts portfolio weights based on dynamic stop-loss triggers.

### correlation_regime
**Correlation Regime Detection**

Identifies correlation regimes and adjusts allocation to manage regime-dependent risks.

### tail_risk
**Tail Risk Measurement (EVT-based)**

Uses Extreme Value Theory to model and allocate based on tail risk.

### risk_budgeting
**Risk Budgeting Across Assets**

Allocates risk budgets to assets and determines weights that achieve the budget.

### marginal_risk
**Marginal Risk Contribution Weighting**

Weights assets based on their marginal contribution to portfolio risk.

### factor_risk_decomposition
**Factor Risk Decomposition**

Decomposes portfolio risk into factor exposures and allocates accordingly.

### liquidity_risk
**Liquidity-Adjusted Risk Allocation**

Incorporates liquidity constraints and costs into the optimization.

### concentration_risk
**Concentration Risk Minimization**

Minimizes portfolio concentration using entropy or HHI-based measures.

---

## Asset Allocation (15)

Strategic and tactical approaches to distributing capital across asset classes.

### strategic_allocation
**Strategic Asset Allocation**

Long-term target weights based on risk/return assumptions and investor objectives.

### tactical_allocation
**Tactical Asset Allocation**

Short-term tilts around strategic weights based on market conditions.

### dynamic_allocation
**Dynamic Asset Allocation**

Time-varying allocation that adapts to changing market conditions.

### constant_proportion
**CPPI (Constant Proportion Portfolio Insurance)**

Maintains exposure as a multiple of the cushion (portfolio value minus floor).

### target_date
**Target Date / Lifecycle Glide Path**

Adjusts risk allocation over time as the target date approaches.

### momentum_allocation
**Momentum-Based Allocation**

Overweights assets with positive momentum, underweights those with negative.

### risk_on_off
**Risk-On / Risk-Off Regime**

Binary regime switching between risk-seeking and risk-averse allocations.

### min_correlation
**Minimum Correlation Portfolio**

Minimizes the average pairwise correlation of portfolio constituents.

### max_decorrelation
**Maximum Decorrelation Portfolio**

Maximizes the decorrelation benefit among portfolio assets.

### core_satellite
**Core-Satellite Approach**

Combines a diversified core with concentrated satellite positions for alpha.

### liability_driven
**Liability-Driven Investment (LDI)**

Matches asset allocation to liability structure and duration.

### goal_based
**Goal-Based Allocation**

Allocates to meet specific investment goals with defined probability thresholds.

### regime_based_allocation
**Regime-Based Conditional Allocation**

Adjusts allocation based on detected market regimes (bull/bear/neutral).

### volatility_weighted
**Volatility-Weighted Allocation**

Weights inversely proportional to each asset's volatility for risk equalization.

### drawdown_based
**Drawdown-Based Dynamic Allocation**

Reduces exposure dynamically as drawdown increases, rebuilds as recovery occurs.

---

## Signal Processing (10)

Algorithms that filter, denoise, and extract signals from noisy financial time series.

### kalman_filter
**Kalman Filter for Signal Extraction**

Recursive Bayesian filter that estimates the hidden state (true signal) from noisy observations.

### wavelet_transform
**Wavelet Transform Denoising**

Multi-resolution analysis that separates signal from noise at different frequency scales.

### fourier_analysis
**FFT Frequency Decomposition**

Identifies dominant cycles and periodicities in price data using Fast Fourier Transform.

### hodrick_prescott
**Hodrick-Prescott Filter**

Decomposes time series into trend and cyclical components. Common in macroeconomics.

### savitzky_golay
**Savitzky-Golay Smoothing**

Polynomial smoothing filter that preserves higher moments better than simple moving averages.

### exponential_smoothing
**Holt Double Exponential Smoothing**

Captures both level and trend with exponentially decaying weights.

### bandpass_filter
**Bandpass Signal Filtering**

Extracts cycles within a specific frequency band, filtering out both long-term trend and high-frequency noise.

### hilbert_transform
**Hilbert Transform Instantaneous Phase**

Computes instantaneous amplitude and phase to identify trend direction changes.

### emd_decomposition
**Empirical Mode Decomposition (EMD)**

Data-driven decomposition into intrinsic mode functions. Adaptive to non-stationary data.

### particle_filter
**Particle Filter State Estimation**

Sequential Monte Carlo method for non-linear, non-Gaussian state estimation.

---

## Statistical Methods (15)

Statistical estimation, testing, and modeling algorithms for portfolio decisions.

### covariance_estimation
**Ledoit-Wolf Shrinkage Covariance**

Shrinks the sample covariance matrix toward a structured target to reduce estimation error.

### pca_decomposition
**PCA Factor Extraction**

Extracts principal components as statistical risk factors for dimensionality reduction.

### independent_component
**ICA Signal Separation**

Separates mixed signals into statistically independent sources.

### bayesian_estimation
**Bayesian Parameter Estimation**

Updates prior beliefs about parameters using observed data via Bayes' theorem.

### bootstrap_methods
**Bootstrap Confidence Intervals**

Resamples observed data to construct empirical confidence intervals for portfolio metrics.

### copula_modeling
**Copula Dependence Modeling**

Models the dependence structure separately from marginal distributions.

### regime_detection
**Hidden Markov Model Regimes**

Identifies latent market states (regimes) using Hidden Markov Models.

### changepoint_detection
**Structural Break Detection**

Identifies points where the statistical properties of returns change abruptly.

### density_estimation
**Kernel Density Estimation**

Non-parametric estimation of the return distribution for improved risk assessment.

### extreme_value
**Extreme Value Theory (EVT)**

Models the tail of the return distribution for rare event risk management.

### robust_statistics
**Robust Estimators (MAD, Huber)**

Uses robust statistical estimators that are less sensitive to outliers.

### time_series_decomposition
**STL-like Time Series Decomposition**

Decomposes returns into trend, seasonal, and residual components.

### granger_causality
**Granger Causality Testing**

Tests whether one time series helps predict another for lead-lag allocation.

### dynamic_correlation
**DCC-GARCH Dynamic Correlations**

Models time-varying correlations using Dynamic Conditional Correlation GARCH.

### factor_analysis
**Statistical Factor Analysis**

Identifies latent factors that explain the correlation structure of returns.

---

## Numerical Methods (10)

General-purpose optimization algorithms adapted for portfolio problems.

### gradient_descent
**Projected Gradient Descent**

Iterative first-order optimization projected onto the simplex constraint.

### newton_method
**Newton's Method Optimizer**

Second-order optimization using the Hessian for faster convergence near optima.

### simulated_annealing
**Simulated Annealing**

Global optimization via probabilistic acceptance of worse solutions at high "temperatures".

### genetic_algorithm
**Genetic Algorithm Optimization**

Evolves a population of candidate portfolios through selection, crossover, and mutation.

### particle_swarm
**Particle Swarm Optimization (PSO)**

Swarm intelligence where particles explore the weight space guided by personal and global bests.

### differential_evolution
**Differential Evolution**

Population-based optimizer using vector differences for mutation. Good for non-convex problems.

### quadratic_programming
**QP Solver Wrapper**

Efficient solver for quadratic objectives with linear constraints (standard portfolio problems).

### linear_programming
**LP Constraint Handling**

Handles linear portfolio constraints and transaction cost minimization.

### monte_carlo_simulation
**Monte Carlo Simulation Engine**

Evaluates portfolio performance across thousands of simulated scenarios.

### convex_optimization
**General Convex Optimization**

Wrapper for disciplined convex programming suitable for a wide range of portfolio formulations.

---

## Machine Learning Optimization (15)

Modern ML-based approaches to portfolio weight determination.

### neural_network_optimizer
**Neural Network Weight Optimization**

Simple feedforward neural network that learns asset weights from return features.

### reinforcement_learning
**Q-Learning Based Allocation**

Learns optimal allocation policy through trial-and-error interaction with returns.

### bayesian_optimization
**Bayesian Optimization for Portfolio Selection**

Uses Gaussian processes to efficiently search the weight space for optimal configurations.

### online_learning
**Online / Incremental Learning**

Continuously updates weights as new data arrives without full retraining.

### transfer_learning
**Transfer Learning for Domain Adaptation**

Adapts allocation models trained on one market regime to work in another.

### attention_weighting
**Attention Mechanism for Asset Weighting**

Uses attention-like scoring to dynamically weight assets based on feature relevance.

### autoencoder_features
**Autoencoder Feature Extraction**

Compresses return data into lower-dimensional features for allocation decisions.

### clustering_allocation
**K-Means Clustering Allocation**

Groups assets by similarity and allocates across/within clusters.

### manifold_learning
**Manifold-Based Dimensionality Reduction**

Maps high-dimensional asset space to a lower-dimensional manifold for allocation.

### graph_based
**Graph/Network-Based Allocation**

Models assets as nodes in a network and uses graph metrics for centrality-based allocation.

### meta_learning
**Meta-Learning Across Market Regimes**

Learns to quickly adapt allocation to new market regimes from few observations.

### multi_objective
**Multi-Objective Optimization (Pareto Front)**

Simultaneously optimizes multiple competing objectives (return, risk, drawdown, turnover).

### adversarial_robustness
**Adversarial Robustness in Optimization**

Trains allocation models to be robust against adversarial perturbations in input data.

### variational_inference
**Variational Inference for Uncertainty**

Approximates posterior distributions over weights to quantify allocation uncertainty.

### gaussian_process
**GP Regression for Return Prediction**

Uses Gaussian Process regression to predict returns with uncertainty estimates for allocation.
