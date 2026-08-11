# Nonlinear Dynamical System Parameter Identification

This repository contains implementations for parameter estimation of nonlinear dynamical systems from noisy observations.

The current study focuses on the Lotka-Volterra system and investigates different approaches for derivative estimation and parameter identification.

## Project Overview

The current project includes two approaches:

1. Savitzky-Golay smoothing and Ridge regression
2. Gaussian Process based derivative estimation and parameter identification

The overall workflow is:

```text
Dynamical system
        |
        v
Generate trajectory data
        |
        v
Add observational noise
        |
        v
Estimate derivatives
        |
        v
Construct feature matrix
        |
        v
Estimate system parameters
        |
        v
Compare with ground truth
```

The project is currently under development. The methodology and experimental design will be further refined based on the research framework and guidance from the project supervisor.

---

## 1. Dynamical System

The current experiments use the Lotka-Volterra predator-prey model.

The system is defined by:

```text
dx/dt = alpha * x - beta * x * y

dy/dt = delta * x * y - gamma * y
```

where:

- x and y are the state variables
- alpha, beta, delta, and gamma are the unknown system parameters

The current synthetic experiment uses:

```text
alpha = 1.5
beta  = 1.0
delta = 1.0
gamma = 3.0
```

The trajectory is generated numerically using SciPy.

---

## 2. Method I - Savitzky-Golay + Ridge Regression

The first approach uses Savitzky-Golay smoothing to estimate derivatives from noisy observations.

### Pipeline

```text
Clean trajectory
        |
        v
Add observational noise
        |
        v
Savitzky-Golay smoothing
        |
        v
Derivative estimation
        |
        v
Feature matrix construction
        |
        v
Ridge regression
        |
        v
Parameter estimation
```

This method is currently used as a baseline approach for comparison.

The implementation is organized into several steps:

```text
01. Generate Lotka-Volterra model
02. Define Savitzky-Golay derivative estimation
03. Generate noisy observations
04. Estimate derivatives
05. Construct feature matrix
06. Perform Ridge regression
07. Visualize results
```

---

## 3. Method II - Gaussian Process Learning

The second approach uses Gaussian Process Regression to estimate the underlying trajectory and its derivatives from noisy observations.

The current implementation uses an RBF kernel.

### Pipeline

```text
Noisy observations
        |
        v
Gaussian Process Regression
        |
        v
Kernel derivative calculation
        |
        v
Derivative estimation
        |
        v
Derivative covariance
        |
        v
GPL parameter estimation
        |
        v
Parameter recovery
```

For each state variable, a separate Gaussian Process is trained.

The current implementation explicitly constructs the kernel matrices required for derivative inference, including:

```text
Kuu
Kdu
Kdd
```

The derivative estimate and derivative covariance are then used in the parameter estimation procedure.

The detailed mathematical formulation of this method will be refined according to the research methodology provided by the project supervisor.

---

## 4. Current Experimental Setup

The current synthetic experiment uses the following settings:

| Setting | Value |
|---|---|
| Dynamical system | Lotka-Volterra |
| Simulation interval | 0 to 30 |
| Initial condition | (1, 1) |
| Number of simulation points | 3000 |
| Training interval | 0 to 10 |
| Number of training points | 200 |
| Noise level | 10% |
| GP kernel | RBF |
| Random seed | 42 |

These settings are part of the current implementation and may be modified in later experiments.

---

## 5. Repository Structure

```text
Gaussian-Process-Learning-LV/
|
+-- 01-SG-based-parameter-estimation/
|   |
|   +-- 01_generate_lv_model.py
|   +-- 02_define_sg_derivative.py
|   +-- 03_generate_data.py
|   +-- 04_estimate_derivatives.py
|   +-- 05_construct_feature_matrix.py
|   +-- 06_ridge_parameter_estimation.py
|   +-- 07_visualization.py
|
+-- 02-GP-based-parameter-estimation/
|   |
|   +-- GPL_LV_parameter_estimation.py
|
+-- README.md
```

---

## 6. Software and Libraries

The current implementation uses:

- Python
- NumPy
- SciPy
- GPy
- Matplotlib

---

## 7. Current Status

This repository is an ongoing research project.

The current implementation is intended for methodological exploration and comparison rather than as a final software package.

The following parts are currently being developed:

- Derivative estimation
- Gaussian Process based parameter estimation
- Regularization
- Numerical stability
- Experimental comparison
- Visualization

Further sections of the methodology, mathematical formulation, results, and comparisons will be added after the research framework is finalized.

---

## 8. Future Development

The project may be extended to include:

- More systematic parameter estimation experiments
- Different noise levels
- Different amounts of training data
- Additional nonlinear dynamical systems
- Comparison with SINDy
- Parameter uncertainty analysis
- More systematic evaluation of model robustness

The exact experimental design will be updated as the research progresses.
