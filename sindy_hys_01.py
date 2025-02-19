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

import pysindy as ps
from sklearn.linear_model import Lasso

from sindy_implementation_functions import simulate_sindy_model
from sindy_implementation_functions import build_pos_polynomial_libraries
from sindy_implementation_functions import build_pos_fourier_libraries
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
# Testing possible feature libraries, singles.

# Build the possible feature libraries.
poly_feature_libraries = build_pos_polynomial_libraries()
four_feature_libraries = build_pos_fourier_libraries()
cust_feature_library = build_pos_custom_library()

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

# --- COMMENT OUT THIS ANALYSIS TO PERFORM THE FIRST SIMULATIONS. ----
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

# Run the simulations
print("\n----- BEGINS -----")
simulate_sindy_model(X, X_0, t, concat_poly_four_custom, optimizer="stlsq")
simulate_sindy_model(X, X_0, t, tensor_poly_four_custom, optimizer="stlsq")
print("\n----- ENDS -----")


# Fin :)