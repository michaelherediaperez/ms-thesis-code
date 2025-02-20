# A state-of-the-art review of the Bouc-Wen model and hysteresis characterization through sparse regression techniques

This repository contains the code used in my master thesis.

![](results_plots/bw--time-response-hyst-loop.svg)

## Table of contents

1. [Auxiliary functions](#auxiliary-functions)
2. [Bouc-Wen models and hysteresis behaviors](#bouc-wen-models-and-hysteresis-behaviors)
3. [SINDy implementation](#sindy-implementation)
4. [Complementary figures](#complementary-figures)
5. [Folders structure](#folders-structure)

## Auxiliary functions

| File name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [functions.py](functions.py)       | Helper functions for plotting and data management.                         |
| [dynamics_functions.py](dynamics_functions.py) | Functions for hysteresis simulations.                                      |
| [sindy_implementation_functions.py](sindy_implementation_functions.py) | Functions for SINDy implementation in hysteresis identification.           |

## Bouc-Wen models and hysteresis Behaviors

### Bouc-Wen class models

| Model file                         | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [bw_model.py](bw_model.py)         | Standard Bouc-Wen model. Simulates symmetric hysteresis with softening/hardening. |
| [bw_analysis.py](bw_analysis.py)   | Sensitivity analysis of the parameters of the standard Bouc-Wen model.      |
| [mbwbn_model.py](mbwbn_model.py)   | Modified Bouc-Wen-Baber-Noori model. Simulates pinched hysteresis, stiffness degradation, and strength degradation. |
| [hys-sym-deg-pinc.py](hys-sym-deg-pinc.py) | Generates figures for the modified Bouc-Wen-Baber-Noori model (mbwbn_model.py). |
| [gbw_model.py](gbw_model.py)       | Generalized Bouc-Wen model. Simulates asymmetric hysteresis.               |
| [fsbw_model.py](fsbw_model.py)     | Flag-shape Bouc-Wen model. Simulates flag-shaped hysteresis.                |
| [bw_citation_evolution.py](bw_citation_evolution.py) | Displays the evolution of the citations of the Bouc-Wen class model across the years. |

### Other hysteresis behaviors

| File name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [hys-s-shaped.py](hys-s-shaped.py) | S-shaped hysteresis.                                                        |
| [hys-backslash-like.py](hys-backslash-like.py) | Backslash-like hysteresis.                                                  |
| [hys-butterfly.py](hys-butterfly.py) | Double loop hysteresis.                                                     |
| [hys-lonely-stroke.py](hys-lonely-stroke.py) | Hysteresis with lonely-stroke.                                              |

![](zz_screnshots/Screenshot_20250219_163427.png)

## SINDy implementation

The sparse identification of nonlinear dynamics (SINDy) technique is used for hysteresis characterization. The implementation can be found in the following files:

| File name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [sindy_hys_01.py](sindy_hys_01.py) | Case 1: Exponential growth external force.                                  |
| [sindy_hys_02.py](sindy_hys_02.py) | Case 2: Linear-time growth external force.                                  |
| [sindy_hys_01_control.py](sindy_hys_01_control.py) | Control implementation for Case 1 (exponential growth external force).      |

![](sindy_plots_1c/poly_c.svg)
![](sindy_plots_1c/four_c.svg)

## Complementary figures

The following codes are used to generate complementary figures for the thesis:

| File name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [2d_damped_oscillator.py](2d_damped_oscillator.py) | 2D damped oscillator simulation.                                           |
| [abs_smooth_approximation.py](abs_smooth_approximation.py) | Smooth approximation of the absolute value.                                |
| [pnorm.py](pnorm.py)               | Plots for the l0-norm, l1-norm, and l2-norm.                               |
| [sign_approx_tanh.py](sign_approx_tanh.py) | Plots the approximations used for the SINDy implementation, including smooth absolute value and sign approximations. |
| [terms_growth.py](terms_growth.py) | Presents how the increase in the order of a polynomial feature library leds to a exponential growth tendency. |
| [pareto_front.py](pareto_front.py) | A mockup to ilustrate the Pareto curve. |


![](zz_screnshots/Screenshot_20250219_163915.png)
![](results_plots/p-norm.svg)
![](results_plots/pareto_curve.svg)

## Folders structure

The results and data used are organized into the following folders:


| Folder Name         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **data**            | Contains data used in the generation of hysteresis models.                  |
| **outputs**         | Contains `.txt` outputs from the SINDy implementation.                           |
| **results_data**    | Contains data generated from hysteresis simulations using Bouc-Wen (BW) models. |
| **results_plots**   | Contains plots from BW simulations, hysteresis behaviors, and complementary figures. |
| **scopus**          | Contains citation data for the Bouc-Wen class model.                        |
| **scopus_plots**    | Contains plots showing the evolution of the Bouc-Wen class model over the years. |
| **sindy_plots_1**   | Contains plots for Case 1 (exponential growth external force), including single and composed libraries. |
| **sindy_plots_1c**  | Contains plots for Case 1 with control.                                 |
| **sindy_plots_2**   | Contains plots for Case 2 (linear-time growth external force).           |