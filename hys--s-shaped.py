"""
This code creates a graph for the complex behavior of hysteresis with 
s-shape.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    January 2025

Bibliography:
    Lu, Y., Xiong, F., & Zhong, J. (2022). Uniaxial hysteretic spring models 
    for static and dynamic analyses of self-centering rocking shallow 
    foundations. Engineering Structures, 272, 114995.

Commentary:
    - Data shared by Lu, Y., Xiong, F., & Zhong, J.    
    - AI platform Claude.ai was used in this code to help with programming 
      language conversion and to solve problems.  
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from functions import store_data, save_image, plot_hysteresis

# Load the `.mat` file
mat_data = loadmat('data/s-shaped-FSV10.mat')

# ------
# Understand the data.

# Let's first print the shapes and types of the key variables.
print("M1 shape:", mat_data['M1'].shape)
print("Mu shape:", mat_data['Mu'].shape)
print("thf shape:", mat_data['thf'].shape)
print("theta_s shape:", mat_data['theta_s'].shape)

# Print a few values to see what we're working with.
print("\nFirst few values of M1:", mat_data['M1'][:5])
print("Mu value:", mat_data['Mu'])
print("First few values of thf:", mat_data['thf'][:5])
print("theta_s value:", mat_data['theta_s'])

# Calculate the ratios.
y_ratio = mat_data['M1'] / mat_data['Mu']
x_ratio = mat_data['thf'] / mat_data['theta_s']

# Print the shapes of our ratios.
print("\nShape of x_ratio:", x_ratio.shape)
print("Shape of y_ratio:", y_ratio.shape)

# Print a few values of our ratios.
print("\nFirst few x_ratio values:", x_ratio[:5])
print("First few y_ratio values:", y_ratio[:5])

# The data is in a different shape than we initially assumed. The `M1` and `thf`
# are arrays with 41000 points, while `Mu` and `theta_s` are single values. 
# The arrays are also wrapped in an extra dimension: shape is (1, 41000) instead
# of just (41000)

# -----
# Create the plots.

# Calculate the ratios and flatten the arrays to 1D. Using `.item()` to get the 
# single value.
y_ratio = mat_data['M1'].flatten() / mat_data['Mu'].item()  
x_ratio = mat_data['thf'].flatten() / mat_data['theta_s'].item()

fig = plot_hysteresis(x_ratio, y_ratio)
# save_image(fig, "hys--s-shape")
# t = np.zeros(x_ratio.shape)
# store_data("hys--s-shape", t, x_ratio, y_ratio)

# Fin :)    