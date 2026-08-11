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

# 3. Method II — Gaussian Process Learning (GPL)

The second approach uses Gaussian Process Regression to reconstruct the underlying dynamics and estimate their derivatives from noisy observations.

Unlike the Savitzky–Golay approach, which directly smooths the observations before differentiating, the GP approach provides a probabilistic representation of the latent trajectory and derives derivative information from the GP kernel.

## 3.1 Gaussian Process Regression

For each state variable, a separate Gaussian Process is trained:

$$
x(t) \sim GP(m_x(t), k_x(t,t'))
$$

$$
y(t) \sim GP(m_y(t), k_y(t,t')).
$$

The current implementation uses an RBF kernel:

$$
k(t,t')
=
\sigma_f^2
\exp
\left(
-\frac{(t-t')^2}{2\ell^2}
\right).
$$

The GP hyperparameters are optimized using BFGS.

The observation noise is also estimated during GP optimization, with bounded noise variance.

---

## 3.2 GP Derivative Inference

The derivative of a Gaussian Process is also a Gaussian Process.

For the training observations $Y$ at locations $t$, the implementation constructs:

$$
K_{uu}
=
K(t,t)+\sigma_n^2I
$$

where $K_{uu}$ is the covariance matrix of the noisy observations.

The covariance between derivatives and function values is computed as:

$$
K_{du}
=
\frac{\partial K(t,t')}{\partial t}.
$$

The derivative covariance is obtained from:

$$
K_{dd}
=
\frac{\partial^2 K(t,t')}
{\partial t\partial t'}.
$$

The posterior mean of the derivative is then calculated as:

$$
\hat d
=
K_{du}K_{uu}^{-1}Y.
$$

The conditional derivative covariance is:

$$
\Sigma_d
=
K_{dd}
-
K_{du}K_{uu}^{-1}K_{du}^{T}.
$$

The implementation uses the inverse of this covariance as the weighting matrix:

$$
R_{dd}=\Sigma_d^{-1}.
$$

Therefore, the GP provides not only an estimate of the derivative, but also information about derivative uncertainty.

---

## 3.3 GPL Parameter Identification

Once the GP derivative estimates are obtained, the Lotka–Volterra system can be written as a linear regression problem in the unknown parameters.

For the prey equation:

$$
\dot{x}
=
\alpha x-\beta xy
$$

we define

$$
G_x=
\begin{bmatrix}
x & -xy
\end{bmatrix},
\qquad
\theta_x=
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}.
$$

Therefore,

$$
\dot{x}=G_x\theta_x.
$$

Similarly, for the predator equation:

$$
\dot{y}
=
\delta xy-\gamma y,
$$

we define

$$
G_y=
\begin{bmatrix}
xy & -y
\end{bmatrix},
\qquad
\theta_y=
\begin{bmatrix}
\delta\\
\gamma
\end{bmatrix}.
$$

The GPL estimator is:

$$
\hat{\theta}
=
\left(
G^TR_{dd}G+\lambda I
\right)^{-1}
G^TR_{dd}\hat d.
$$

where:

- $G$ is the feature matrix
- $R_{dd}$ is the inverse derivative covariance
- $\hat d$ is the GP derivative estimate
- $\lambda$ is the regularization parameter

The regularization parameter is selected from a predefined candidate set in the current experiment.

---

## 3.4 Parameter Recovery

The estimated parameters are compared with the known parameters used to generate the synthetic trajectory:

$$
\alpha=1.5,\qquad
\beta=1,\qquad
\delta=1,\qquad
\gamma=3.
$$

The relative parameter error is calculated as:

$$
\text{Error}(\%)
=
\frac{|\hat{\theta}-\theta_{\text{true}}|}
{|\theta_{\text{true}}|}
\times100\%.
$$

This provides a quantitative measure of how accurately the GPL method recovers the underlying Lotka–Volterra dynamics.
