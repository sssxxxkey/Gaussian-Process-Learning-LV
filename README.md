# Nonlinear Dynamical System Parameter Identification

This repository studies parameter identification in nonlinear dynamical systems using noisy observations.

The current experiments focus on the Lotka–Volterra predator–prey model and compare different approaches for estimating the unknown system parameters from noisy trajectory data.

## Project Overview

The main workflow is:

Noisy observations
        ↓
Derivative estimation
        ↓
Feature construction
        ↓
Parameter identification
        ↓
Model evaluation

Two approaches are currently implemented:

1. Savitzky–Golay (SG) smoothing + Ridge regression
2. Gaussian Process (GP) derivative inference + Gaussian Process Learning (GPL)

The goal is to understand how different derivative estimation and regression techniques affect parameter recovery under noisy observations.

---

## 1. Dynamical System

The experiments use the Lotka–Volterra predator–prey model:

$$
\frac{dx}{dt} = \alpha x - \beta xy
$$

$$
\frac{dy}{dt} = \delta xy - \gamma y
$$

where:

- $x(t)$ represents the prey population
- $y(t)$ represents the predator population
- $\alpha$ represents the prey growth rate
- $\beta$ represents the interaction rate between prey and predators
- $\delta$ represents the predator growth rate due to prey consumption
- $\gamma$ represents the predator decay rate

The current synthetic experiment uses:

$$
\alpha = 1.5,\quad
\beta = 1,\quad
\delta = 1,\quad
\gamma = 3
$$

The system is numerically simulated using `scipy.integrate.solve_ivp`.

---

# 2. Method I — Savitzky–Golay + Ridge Regression

The first approach uses Savitzky–Golay smoothing to estimate derivatives from noisy observations.

### Pipeline

```text
Clean trajectory
      ↓
Add observational noise
      ↓
Savitzky–Golay smoothing
      ↓
Estimate dx/dt and dy/dt
      ↓
Construct feature matrix
      ↓
Ridge regression
      ↓
Estimate α, β, δ, γ
