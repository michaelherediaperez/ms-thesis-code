# ms-thesis-code

This repository contains the code used in my master thesis titled:

**"A state-of-the-art review of the Bouc-Wen model and hysteresis characterization through sparse regression techniques"**.

## Table of Contents

1. [Auxiliary functions](#auxiliary-functions)
2. [Bouc-Wen models and hysteresis behaviors](#bouc-wen-models-and-hysteresis-behaviors)
3. [SINDy implementation](#sindy-implementation)
4. [Complementary figures](#complementary-figures)

## Auxiliary Functions

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [functions.py](functions.py)       | Helper functions for plotting and data management.                         |
| [dynamics_functions.py](dynamics_functions.py) | Functions for hysteresis simulations.                                      |
| [sindy_implementation_functions.py](sindy_implementation_functions.py) | Functions for SINDy implementation in hysteresis identification.           |

## Bouc-Wen Models and Hysteresis Behaviors

### Bouc-Wen Class Models

| Model File                         | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [bw_model.py](bw_model.py)         | Standard Bouc-Wen model. Simulates symmetric hysteresis with softening/hardening. |
| [bw_analysis.py](bw_analysis.py)   | Sensitivity analysis of the parameters of the standard Bouc-Wen model.      |
| [mbwbn_model.py](mbwbn_model.py)   | Modified Bouc-Wen-Baber-Noori model. Simulates pinched hysteresis, stiffness degradation, and strength degradation. |
| [hys-sym-deg-pinc.py](hys-sym-deg-pinc.py) | Generates figures for the modified Bouc-Wen-Baber-Noori model (mbwbn_model.py). |
| [gbw_model.py](gbw_model.py)       | Generalized Bouc-Wen model. Simulates asymmetric hysteresis.               |
| [fsbw_model.py](fsbw_model.py)     | Flag-shape Bouc-Wen model. Simulates flag-shaped hysteresis.                |
| [bw_citation_evolution.py](bw_citation_evolution.py) | Displays the evolution of the citations of the Bouc-Wen class model across the years. |

### Other Hysteresis Behaviors

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [hys-s-shaped.py](hys-s-shaped.py) | S-shaped hysteresis.                                                        |
| [hys-backslash-like.py](hys-backslash-like.py) | Backslash-like hysteresis.                                                  |
| [hys-butterfly.py](hys-butterfly.py) | Double loop hysteresis.                                                     |
| [hys-lonely-stroke.py](hys-lonely-stroke.py) | Hysteresis with lonely-stroke.                                              |


## SINDy Implementation

The Sparse Identification of Nonlinear Dynamics (SINDy) technique is used for hysteresis characterization. The implementation can be found in the following files:

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [sindy_hys_01.py](sindy_hys_01.py) | Case 1: Exponential growth external force.                                  |
| [sindy_hys_02.py](sindy_hys_02.py) | Case 2: Linear-time growth external force.                                  |
| [sindy_hys_01_control.py](sindy_hys_01_control.py) | Control implementation for Case 1 (exponential growth external force).      |


## Complementary Figures

The following codes are used to generate complementary figures for the thesis:

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [2d_damped_oscillator.py](2d_damped_oscillator.py) | 2D damped oscillator simulation.                                           |
| [abs_smooth_approximation.py](abs_smooth_approximation.py) | Smooth approximation of the absolute value.                                |
| [pnorm.py](pnorm.py)               | Plots for the l0-norm, l1-norm, and l2-norm.                               |
| [sign_approx_tanh.py](sign_approx_tanh.py) | Plots the approximations used for the SINDy implementation, including smooth absolute value and sign approximations. |

