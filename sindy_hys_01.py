"""
Testing multiple utilities of PySINDy (Python package for the Sparse 
Identification of Nonlinear Dynamical Systems) to capture the dynamics of 
hysteretic structural systems.

Focus on: Feature Library

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
Fr = data.iloc[:, 1].values     # restoring force [kN] 
t  = data.iloc[:, 2].values     # time [s]

# Build the state variables vector.
X = np.column_stack([x, Fr])

# Set the initial conditions.
X_0 = np.array([0, 0])


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
# Iterate over all the possibilities:

print("\n----- BEGINS -----")
simulate_sindy_model(X, X_0, t, poly_feature_libraries, optimizer="stlsq") 
simulate_sindy_model(X, X_0, t, four_feature_libraries, optimizer="stlsq")
simulate_sindy_model(X, X_0, t, cust_feature_library, optimizer="stlsq")
print("\n----- ENDS -----")

# -----
# Concatenation and tensor product of libraries.

# It is observed in the experimentation that the polynomial feature lirbary 
# requieres the interaction and bias terms for better performance (as set 
# by default in PySINDy). Also, up to degree n=2 shows good performance, n>2 
# shows ill-conditioning of the problem as the performance degrades.  
#
# The fourier libraries shows good performances with n_freq=3 and n_freq=4. 
# However, the former has a frequency offset, the latter tries to adjust this 
# offset but decreases the amplitude of the response of the variables.
#  
# Custum library works fine.

# COMMENT OUT THIS ANALYSIS TO PERFORM THE FIRST SIMULATIONS.
filtered_featured_libraries = {}
filtered_featured_libraries["poly"] = ps.PolynomialLibrary(degree=2)
filtered_featured_libraries["four"] = ps.FourierLibrary(n_frequencies=3)
 
# Define the concatenation and the tensorization of polynomial and fourier 
# feature libraries.
concat_poly_four_custom = {}
tensor_poly_four_custom = {}

for key_cust, val_cust in cust_feature_library.items():
    for key_filt, val_filt in filtered_featured_libraries.items():
        concat_poly_four_custom[f"{key_cust}_{key_filt}_c"] = val_cust + val_filt
        tensor_poly_four_custom[f"{key_cust}_{key_filt}_t"] = val_cust * val_filt

concat_poly_four_custom["FL_poly_four_c"] = filtered_featured_libraries["poly"] + filtered_featured_libraries["four"]
tensor_poly_four_custom["FL_poly_four_t"] = filtered_featured_libraries["poly"] * filtered_featured_libraries["four"]

# -----
# Sun the simulations

# Start timer.
time_0 = time.time()

print("\n----- BEGINS -----")
simulate_sindy_model(X, X_0, t, concat_poly_four_custom, optimizer="stlsq")
simulate_sindy_model(X, X_0, t, tensor_poly_four_custom, optimizer="stlsq")
print("\n----- ENDS -----")

# Stops timer.
time_f = time.time()
total_time = time_f-time_0
print(f"TIME USED: {total_time} seconds")


# Fin :)