"""
Testing multiple utilities of PySINDy (Python package for the Sparse 
Identification of Nonlinear Dynamical Systems) to capture the dynamics of 
hysteretic structural systems.

Focus on: Control

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

# -----
# Star to measure time.
time_0 = time.time()
# -----

def simulate_sindy_model(X, X0, t, ft_lb_key, ft_lb_val):
    """_summary_

    Args:
    -----
        X (_type_): _description_
        X0 (_type_): _description_
        t (_type_): _description_
        ft_lb_key (_type_): _description_
        ft_lb_val (_type_): _description_
    """
    # In this case, the external force is considered to be a external input as 
    # well a state variable becasue we hace measurements of it.  
    u = X[:, 1]

    # Find the SINDy model.
    model = ps.SINDy(feature_library=ft_lb_val)
    
    # Build the optimizer.
    model.optimizer = ps.STLSQ(threshold=0.01, alpha=0.05)
    #model.optimizer = Lasso(alpha=0.1, max_iter=2000, fit_intercept=False)
    #model.optimizer = ps.SR3(threshold=0.1, thresholder="l1")
    # WE FOUND GOOD PERFORMANCE WITH THE STLSQ ALGORITHM


    
    model.fit(X, t=t, u=u)
    # Compute the model simulation.
    X_sindy = model.simulate(X0, t, u=u) 

    # Repor the model, its score and the features implemented.
    print("\n-----\n")
    print(f"Used library: {ft_lb_key}")
    print("\nThe model:"); model.print()
    print("\nThe score:"); print(model.score(X, t=t, u=u))
    print("\nThe names:"); print(model.get_feature_names())

    # FIX THE PROBLEM OF SIZE:
    X_sindy = np.vstack([X_sindy, X_sindy[-1,:]])

    # Plot the results.
    fig = plot_comparatives_3(
        X.T, X_sindy.T, t, xylabels=["x(t)", "Fr(t)", "t"], title=ft_lb_key
        )
    # COMMENT THE FOLLOWING LINE SO IT DOES NOT OVERDOES
    #save_image(fig, ft_lb_key, directory="sindy_plots_c")


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
# Create a dictionary with hysteresis data for easy access.
data = {}
 
# RPL walls.
data["rpl_01"] = "./data/rpl-wall-daniel/rpl_wall_01.txt"
data["rpl_02"] = "./data/rpl-wall-daniel/rpl_wall_02.txt"
data["rpl_03"] = "./data/rpl-wall-daniel/rpl_wall_03.txt"

# BW models
data["hys_01"] = "./results_data/hys--simulation-01.txt"
data["hys_02"] = "./results_data/hys--simulation-02.txt"

# -----
# Load the data and work with it.

# Load the .txt file.
file_path = data["hys_01"]

# Read the file, in this case the columns are separated by spaces.
#data = pd.read_csv(file_path, delim_whitespace=True)  
data = pd.read_csv(file_path, sep='\s+')

# Extract columns. 		 
x  = data.iloc[:, 0].values     # displacement [mm] 
Fr = data.iloc[:, 1].values     # restoring force [kN] 
t  = data.iloc[:, 2].values     # time [s]

# Display the experimental hysteresis.
#plot_hysteresis(x, Fr)

# Build the state variables vector.
X = np.column_stack([x, Fr])

# Set the initial conditions.
X0 = np.array([0, 0])


# -----
# Build the feature library, we already know what functions to use:
# Polynomials up to second degree, with default settings (bias and interaction)
# Fouriers up to third frequencies.
# Custom functions: abs, sign, exp, sin(a+b), cos(a+b)
feature_libraries = {}

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

# Define the feature libraries (include the control _c).
feature_libraries["poly_c"] = ps.PolynomialLibrary(degree=2)
feature_libraries["four_c"] = ps.FourierLibrary(n_frequencies=3)
feature_libraries["cust_c"] = ps.CustomLibrary(
    library_functions=pos_custom_functions,
    function_names=pos_custom_fun_names,
    interaction_only=False # to consider all types of cominations
    )

# -----
# Perform the test.

run_simulations_over_feature_library(feature_libraries) 

# -----
# Concatenation and tensor product of libraries.







print("\n----- ENDS -----")

# -----
# Stops timer.
time_f = time.time()
total_time = time_f-time_0
print(f"TIME USED: {total_time} seconds")
# -----


# Fin :)