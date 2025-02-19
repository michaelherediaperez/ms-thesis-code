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
import pysindy as ps

# Custom functions
from sindy_implementation_functions import simulate_sindy_model
from sindy_implementation_functions import build_pos_custom_library


# -----
# Load the data and work with it.

# Load the .txt file and read the data
file_path = "./results_data/hys--simulation-01.txt"
data      = np.loadtxt(file_path, skiprows=1)

# Unpack the state measurements.
x   = data[:, 0]     # displacement [mm] 
f_r = data[:, 1]     # restoring force [kN] 
t   = data[:, 2]     # time [s]

# Build the state variables vector.
X = np.column_stack([x, f_r])

# Set the initial conditions.
X_0 = np.array([0, 0])

# -----
# Build the feature library. 
# We already know what functions to use:
# - Polynomials up to 2nd degree, with default settings (bias and interaction)
# - Fouriers up to third frequencies.
# - Custom functions: abs, sign, exp, sin(a+b), cos(a+b)
feature_libraries = {}

# Build the custom library.
custom_library = build_pos_custom_library()

# Define the feature libraries (include the control _c).
feature_libraries["poly_c"] = ps.PolynomialLibrary(degree=2)
feature_libraries["four_c"] = ps.FourierLibrary(n_frequencies=3)
feature_libraries["cust_c"] = custom_library["FL_cus"]
 
# Perform the test.
print("\n----- BEGINS -----")
simulate_sindy_model(
    X, X_0, t, feature_libraries, optimizer="stlsq", control_u=f_r
    )
print("\n----- ENDS -----")


# Fin :)