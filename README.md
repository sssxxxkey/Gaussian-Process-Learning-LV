# Nonlinear Dynamical System Parameter Identification

> Comparing Savitzky–Golay/Ridge regression and Gaussian Process Learning for parameter estimation in nonlinear dynamical systems.

This repository studies parameter identification in nonlinear dynamical systems using noisy observations.

The current experiments focus on the Lotka–Volterra predator–prey model and investigate how different derivative estimation and regression methods recover the underlying system parameters.

---

## Project Overview

The main workflow is:

```text
Noisy observations
        ↓
Derivative estimation
        ↓
Feature construction
        ↓
Parameter identification
        ↓
Model evaluation
```

Two approaches are currently implemented:

1. **Savitzky–Golay (SG) smoothing + Ridge regression**
2. **Gaussian Process (GP) derivative inference + Gaussian Process Learning (GPL)**

The main goal is to compare these approaches for parameter identification from noisy observations.

---

# 1. Dynamical System

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
- $\beta$ represents the prey–predator interaction rate
- $\delta$ represents the predator growth rate due to prey consumption
- $\gamma$ represents the predator decay rate

The current synthetic experiment uses:

$$
\alpha = 1.5,\qquad
\beta = 1,\qquad
\delta = 1,\qquad
\gamma = 3
$$

The system is numerically simulated using `scipy.integrate.solve_ivp`.

---

# 2. Method I — Savitzky–Golay + Ridge Regression

The first approach uses Savitzky–Golay smoothing to estimate derivatives from noisy observations.

## 2.1 Pipeline

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
```

The Lotka–Volterra equations are linear in the unknown parameters:

$$
\frac{dx}{dt}
=
\alpha x-\beta xy
$$

and

$$
\frac{dy}{dt}
=
\delta xy-\gamma y.
$$

Therefore, the feature matrices are constructed as:

$$
G_x =
\begin{bmatrix}
x & -xy
\end{bmatrix}
$$

and

$$
G_y =
\begin{bmatrix}
xy & -y
\end{bmatrix}.
$$

The unknown parameters are then estimated using Ridge regression.

This method serves as a baseline for comparison with the Gaussian Process approach.

---

# 3. Method II — Gaussian Process Learning (GPL)

The second approach uses Gaussian Process Regression to reconstruct the underlying dynamics and estimate their derivatives from noisy observations.

Unlike the Savitzky–Golay approach, which directly smooths the observations before differentiating, the GP approach models the latent trajectory probabilistically and obtains derivative information from derivatives of the GP kernel.

---

## 3.1 Gaussian Process Regression

A separate Gaussian Process is trained for each state variable:

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

The observation noise variance is also optimized within bounded constraints.

---

## 3.2 GP Derivative Inference

The derivative of a Gaussian Process is also a Gaussian Process.

For the training observations, the implementation constructs the following kernel matrices.

### Observation covariance

$$
K_{uu}
=
K(t,t)+\sigma_n^2 I
$$

where $\sigma_n^2$ represents the observation noise variance.

### Derivative–observation covariance

$$
K_{du}
=
\frac{\partial K(t,t')}
{\partial t}
$$

### Derivative covariance

$$
K_{dd}
=
\frac{\partial^2 K(t,t')}
{\partial t\partial t'}
$$

The posterior mean of the derivative is calculated as:

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

The implementation then uses the inverse derivative covariance:

$$
R_{dd}=\Sigma_d^{-1}
$$

as the weighting matrix in the parameter estimation step.

Therefore, the GP approach provides both:

- an estimate of the derivative
- information about derivative uncertainty

---

## 3.3 GPL Parameter Identification

Once the GP derivative estimates are obtained, the Lotka–Volterra equations can be written as linear regression problems in the unknown parameters.

For the prey equation:

$$
\dot{x}
=
\alpha x-\beta xy
$$

we define:

$$
G_x=
\begin{bmatrix}
x & -xy
\end{bmatrix}
$$

and

$$
\theta_x=
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}.
$$

Therefore:

$$
\dot{x}=G_x\theta_x.
$$

For the predator equation:

$$
\dot{y}
=
\delta xy-\gamma y
$$

we define:

$$
G_y=
\begin{bmatrix}
xy & -y
\end{bmatrix}
$$

and

$$
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
G^T R_{dd}G+\lambda I
\right)^{-1}
G^T R_{dd}\hat d.
$$

where:

- $G$ is the feature matrix
- $R_{dd}$ is the inverse derivative covariance
- $\hat d$ is the GP derivative estimate
- $\lambda$ is the regularization parameter

In the implementation, the resulting linear system is solved using `numpy.linalg.solve`.

---

## 3.4 Regularization Parameter

The current implementation evaluates several candidate values of $\lambda$:

```text
1e-6
1e-5
1e-4
1e-3
1e-2
0.1
```

The current experiment selects the value producing the smallest parameter error relative to the known synthetic ground truth.

This is currently used for benchmarking the implementation.

---

## 3.5 Parameter Recovery

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
\frac{|\hat{\theta}-\theta_{\mathrm{true}}|}
{|\theta_{\mathrm{true}}|}
\times100\%.
$$

The implementation reports the estimated values and relative errors for all four parameters.

---

# 4. Comparison of the Two Methods

The two approaches differ mainly in how derivative information is obtained.

| Method | Derivative Estimation | Parameter Estimation | Main Feature |
|---|---|---|---|
| SG + Ridge | Savitzky–Golay filtering | Ridge regression | Simple numerical baseline |
| GP + GPL | Gaussian Process derivative inference | Weighted regularized regression | Probabilistic derivative estimation |

### SG + Ridge

```text
Noisy observations
        ↓
Smoothing
        ↓
Numerical derivative
        ↓
Feature matrix
        ↓
Ridge regression
        ↓
Parameter estimates
```

### GP + GPL

```text
Noisy observations
        ↓
Gaussian Process Regression
        ↓
Kernel derivatives
        ↓
Derivative mean + covariance
        ↓
GPL
        ↓
Parameter estimates
```

The SG approach relies on numerical smoothing and differentiation.

The GP approach models the underlying trajectory probabilistically and obtains derivative information from the GP kernel.

---

# 5. Experimental Setup

The current synthetic experiment uses:

| Setting | Value |
|---|---:|
| Dynamical system | Lotka–Volterra |
| Simulation interval | $[0,30]$ |
| Initial condition | $(1,1)$ |
| Total simulation points | 3000 |
| Training interval | $[0,10]$ |
| Number of training points | 200 |
| Noise level | 10% |
| GP kernel | RBF |
| GP optimizer | BFGS |
| Random seed | 42 |

The clean trajectory is first generated using numerical integration.

Observational noise is then added to the training data, and a subset of observations is randomly selected for parameter identification.

---

# 6. Results

The current implementation evaluates:

- GP derivative estimates
- True derivatives
- Estimated parameters
- Relative parameter errors

The derivative estimates are compared against the true derivatives generated from the known Lotka–Volterra equations.

The parameter estimation output has the following form:

```text
==============================
 GPL LV Parameter Identification
==============================

Param     True      GPL        Error%
---------------------------------------------
alpha     1.5000    ...
beta      1.0000    ...
delta     1.0000    ...
gamma     3.0000    ...
```

The implementation also produces derivative comparison plots for:

$$
\frac{dx}{dt}
$$

and

$$
\frac{dy}{dt}.
$$

---

# 7. Repository Structure

```text
Gaussian-Process-Learning-LV/
│
├── 01-SG-based-parameter-estimation/
│   │
│   ├── 01_generate_lv_model.py
│   ├── 02_define_sg_derivative.py
│   ├── 03_generate_data.py
│   ├── 04_estimate_derivatives.py
│   ├── 05_construct_feature_matrix.py
│   ├── 06_ridge_parameter_estimation.py
│   └── 07_visualization.py
│
├── 02-GP-based-parameter-estimation/
│   │
│   └── GPL_LV_parameter_estimation.py
│
└── README.md
```

---

# 8. Technologies

The project currently uses:

- Python
- NumPy
- SciPy
- GPy
- Matplotlib
- Savitzky–Golay filtering
- Gaussian Process Regression
- Ridge regression

---

# 9. Future Work

Planned extensions include:

- More systematic hyperparameter selection
- Parameter uncertainty quantification
- Bayesian parameter posterior estimation
- Posterior sampling
- Trajectory reconstruction using estimated parameters
- Experiments under different noise levels
- Experiments with different training data sizes
- Extension to other nonlinear dynamical systems
- Comparison with SINDy-based system identification
- Improved numerical stability using Cholesky-based methods
- More systematic comparison between SG, GP/GPL, and SINDy

---

# 10. Motivation

This project explores data-driven identification of nonlinear dynamical systems from noisy observations.

The broader objective is to understand how different approaches to derivative estimation and parameter inference affect the recovery of underlying governing equations.

The current work focuses on three potential directions:

```text
Numerical smoothing
        │
        └── SG + Ridge

Probabilistic inference
        │
        └── Gaussian Process Learning

Sparse system identification
        │
        └── SINDy
```

The long-term goal is to compare these approaches systematically and investigate their robustness under noisy and limited observations.
