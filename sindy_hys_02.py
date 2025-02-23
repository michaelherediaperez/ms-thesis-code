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

from sindy_implementation_functions import simulate_sindy_model
from sindy_implementation_functions import build_pos_polynomial_libraries
from sindy_implementation_functions import build_pos_fourier_libraries
from sindy_implementation_functions import build_pos_custom_library


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

# Build the state variables vector.
X = np.column_stack([x, fr])

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


# Fin :)