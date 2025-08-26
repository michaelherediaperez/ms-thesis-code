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
import pandas as pd

# Custom functions
from sindy_implementation_functions import simulate_sindy_model
from sindy_implementation_functions import build_pos_custom_library

import matplotlib as mpl
# Configure matplotlib for STIX font - comprehensive setup
mpl.rcParams.update({
    # Primary font configuration
    "font.family": "serif",              # Use serif family
    "font.serif": ["STIX", "STIXGeneral", "STIX Two Text"], # STIX font priority
    "mathtext.fontset": "stix",          # Math expressions in STIX
    
    # Explicit font specification for all text elements
    "axes.labelsize": 18,
    "axes.titlesize": 18, 
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.size": 16,
    
    # Line properties
    "lines.linewidth": 1.5
})

# -----
# Load the data and work with it.

# Load the .txt file and read the data
file_path = "./results_data/hys--simulation-01.txt"
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

# Build the custom library.
custom_library = build_pos_custom_library()

# Define the feature libraries (include the control "_c").
feature_libraries["poly_c"] = ps.PolynomialLibrary(degree=2)
feature_libraries["four_c"] = ps.FourierLibrary(n_frequencies=3)
feature_libraries["cust_c"] = custom_library["FL_cus"]
 
# Perform the test.
print("\n--- Simulation begins")
simulate_sindy_model(
    X, X_0, t, feature_libraries, th=0.01, control_u=fr, store="sindy_hys_01_control"
   )
print("\n--- Simulation ends")


# Fin :)