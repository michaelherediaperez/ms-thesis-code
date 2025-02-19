"""
Testing multiple utilities of PySINDy (Python package for the Sparse 
Identification of Nonlinear Dynamical Systems) to capture the dynamics of 
hysteretic structural systems.

Focus on: Feature Library

Coder: 
------
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

References:
-----------
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing 
    equations from data by sparse identification of nonlinear dynamical systems.
    Proceedings of the national academy of sciences, 113(15), 3932-3937.
    
    de Silva, B. M., Champion, K., Quade, M., Loiseau, J. C., Kutz, J. N., & 
    Brunton, S. L. (2020). Pysindy: a python package for the sparse 
    identification of nonlinear dynamics from data. arXiv preprint 
    arXiv:2004.08424.

    Kaptanoglu, A. A., de Silva, B. M., Fasel, U., Kaheman, K., Goldschmidt, 
    A. J., Callaham, J. L., ... & Brunton, S. L. (2021). PySINDy: A 
    comprehensive Python package for robust sparse system identification. arXiv 
    preprint arXiv:2111.08481.
 
    Oficial documentation at: 
    https://pysindy.readthedocs.io/en/latest/index.html

    Oficial GitHub repository at: 
    https://github.com/dynamicslab/pysindy/tree/master

Date:
-----
    January 2025
"""

import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

# Custom functions
from functions import plot_hysteresis, plot_comparatives_3, save_image
from abs_smooth_approximation import abs_smooth_approximation

import time


def simulate_sindy_model(X, X0, t, ft_lb_key, ft_lb_val):
    """
    This functions generates a SINDy model from the data and performes the 
    simulation based on given initial conditions. The feature library 
    information for the modeling process is given by a dictionary entry. 
    The algorithm is the STLSQ with a fixed values, and the method for 
    numerical differentiation is the default one. 
    
    Args:
    -----
        X (array): 
            State vector.
        X0 (array): 
            Initial conditions.
        t (array): 
            Time vector.
        ft_lb_key (string):
            Dictionary key, it is the name of the feature library configuration. 
            It is used to name the plot. 
        ft_lb_val (array): 
            Dictionary value, it is the feature library configuration used to 
            build the SINDy model.
    """

    # Build the SINDy model with the defined feature library.
    model = ps.SINDy(feature_library=ft_lb_val)
    
    # Define the optimization method for the model.
    model.optimizer = ps.STLSQ(threshold=0.01, alpha=0.05)
    #model.optimizer = Lasso(alpha=2, max_iter=2000, fit_intercept=False)
    
    # Fit the model and compute the simulation for the initial conditions.
    model.fit(X, t=t)
    X_sindy = model.simulate(X0, t) 
        
    # Repor the model, its score and the features implemented.
    print("\n-----\n")
    print(f"Used library: {ft_lb_key}")
    print("\nThe model:"); model.print()
    print("\nThe score:"); print(model.score(X, t=t))
    print("\nThe names:"); print(model.get_feature_names())

    # Plot the results.
    fig = plot_comparatives_3(
        X.T, X_sindy.T, t, xylabels=["x(t)", "fr(t)", "t"], title=ft_lb_key
        )
    # --------------------------------------------------
    # COMMENT THE FOLLOWING LINE SO IT DOES NOT OVERDOES
    # --------------------------------------------------
    #save_image(fig, ft_lb_key, directory="sindy_plots_2")

    return model


def run_simulations_over_feature_library(feature_library) -> None: 
    """Optimization to run the simulation the construction and simulation over 
    a set of features library defined as a dictionary.

    Args:
    -----
        feature_library (dic): 
            Its keys are the coded names of the feature library.
            Its values are the ps.FeatureLibrary structure with its own 
            constraints.
    """
    for key, val in feature_library.items():
        simulate_sindy_model(X, X0, t, key, val)

# -----
# Load the data and work with it.

# Load the .txt file and read it with Numpy.
file_path = "./results_data/hys--simulation-02.txt"
data = np.loadtxt(file_path, skiprows=1)

# Get half the information.
num_measurements = data.shape[0]
half_data = data[:num_measurements // 2, :]

x  = half_data[:, 0]     # displacement [mm] 
fr = half_data[:, 1]     # restoring force [kN] 
t  = half_data[:, 2]     # time [s]

# Display the experimental hysteresis.
#plot_hysteresis(x, fr)

# Build the state variables vector.
X = np.column_stack([x, fr])
# Set the initial conditions.
X0 = np.array([0, 0])

# =======================
# FEATURE LIBRARY TESTING
# =======================

# Argument options for Polynomial Library
pos_pol_degress = range(1,4)            # Possible degrees
pos_bias =  [True, False]               # Possible bias 
pos_interaction = [True, False]         # Possible terms interaction

# Arguments options for the Fourier Library
pos_freq = range(1,5)                   # Possible frequencies

# Functions for Custom Library
pos_custom_functions = [                   # Names, if not given above. 
    lambda x: abs_smooth_approximation(x), # f1(.) --> Absolute value.
    lambda x: np.tanh(10*x),               # f2(.) --> Sign function approx.
    lambda x: np.exp(x),                   # f3(.) --> Exponential function. 
    #lambda x: 1/x,                         # f4(.) --> 1/x  
    lambda x, y: np.sin(x + y),            # f5(.) --> Sin of A + B 
    lambda x, y: np.cos(x + y),            # f6(.) --> Cos of A + B 
]

pos_custom_fun_names = [
    lambda x: "abs(" + x + ")",
    lambda x: "sgn(" + x + ")",
    lambda x: "exp(" + x + ")",
    #lambda x: "1/" + x,
    lambda x, y: "sin(" + x + "," + y + ")",
    lambda x, y: "cos(" + x + "," + y + ")"
]

# Define the possible polynomial feature libraries
poly_feature_libraries = {}

# Store the possible combinations of polynomial libraries.
for i in pos_interaction:
    for j in pos_bias:
        for n in pos_pol_degress:
            poly_feature_libraries[f"FL_pol_n{n}_b{j}_i{i}"] = ps.PolynomialLibrary(
                degree=n, 
                include_interaction=i, 
                include_bias=j
                )

# Define the possible polynomial feature libraries
four_feature_libraries = {}

# Considers both sine and cosine terms.
for freq in pos_freq:
    four_feature_libraries[f"FL_fr_n{freq}"] = ps.FourierLibrary(
        n_frequencies=freq
        )

# Define the possible custom functions without a given name.
cust_feature_library = {} 
cust_feature_library[f"FL_cus"] = ps.CustomLibrary(
    library_functions=pos_custom_functions,
    function_names=pos_custom_fun_names,
    interaction_only=False # to consider all types of cominations
    )

# -----
# Star to measure time.
time_0 = time.time()

# -----
# Iterate over all the possibilities:

run_simulations_over_feature_library(poly_feature_libraries) 
run_simulations_over_feature_library(four_feature_libraries)
run_simulations_over_feature_library(cust_feature_library)

print("\n----- ENDS -----")

# -----
# Stops timer.
time_f = time.time()
total_time = time_f-time_0
print(f"TIME USED: {total_time} seconds")

# Fin :)