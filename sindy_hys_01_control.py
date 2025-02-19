"""
Testing multiple utilities of PySINDy (Python package for the Sparse 
Identification of Nonlinear Dynamical Systems) to capture the dynamics of 
hysteretic structural systems.

Focus on: Control

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

References:
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
    January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pysindy as ps
from sklearn.linear_model import Lasso

# Custom functions
from abs_smooth_approximation import abs_smooth_approximation
from sindy_implementation_functions import simulate_sindy_model

import time

# -----
# Load the data and work with it.

# Load the .txt file.
file_path = "./results_data/hys--simulation-01.txt"

# Read the file, in this case the columns are separated by spaces.
#data = pd.read_csv(file_path, delim_whitespace=True)  
data = pd.read_csv(file_path, sep='\s+')

# Extract columns. 		 
x  = data.iloc[:, 0].values     # displacement [mm] 
fr = data.iloc[:, 1].values     # restoring force [kN] 
t  = data.iloc[:, 2].values     # time [s]

# Build the state variables vector.
X = np.column_stack([x, fr])

# Set the initial conditions.
X_0 = np.array([0, 0])

# -----
# Build the feature library. 
# We already know what functions to use:
# - Polynomials up to 2nd degree, with default settings (bias and interaction)
# - Fouriers up to third frequencies.
# - Custom functions: abs, sign, exp, sin(a+b), cos(a+b)
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

# Star to measure time.
time_0 = time.time()

print("\n----- BEGINS -----")
simulate_sindy_model(
    X, X_0, t, feature_libraries, optimizer="stlsq", control_u=fr, store=False
    )
print("\n----- ENDS -----")

# Stops timer.
time_f = time.time()
total_time = time_f-time_0
print(f"TIME USED: {total_time} seconds")


# Fin :)