"""
Adapted.
This code creates a graph for the complex behavior of double-loop 
(butterfly-like) hysteresis. 

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

References:
    Chandra, A., Daniels, B., Curti, M., Tiels, K., Lomonova, E. A., & 
    Tartakovsky, D. M. (2023). Discovery of sparse hysteresis models for 
    piezoelectric materials. Applied Physics Letters, 122(21).

    Original code: Exp7_Butterfly.py, from 
    https://github.com/chandratue/SmartHysteresis

Date:
    January 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from functions import plot_hysteresis, store_data, save_image

# -----
# Define the model function.
def model(y, t):
    dydt = 16*0.4*np.abs(np.cos(t))*np.sin(t) - 4*0.85*np.abs(np.cos(t))*y +\
          4*0.2*(np.cos(t))
    return dydt

# -----
# Model simulation.

npoints = 1000          # Number of points to calculate.

t = np.linspace(0, 10*np.pi, npoints) # Generate time array.

x  = 4*np.sin(t) # Define input signal.
dx = np.gradient(x) # Calculate the derivative of the input signal.
y0 = 0 # Set initial condition and solve ODE.
y  = odeint(model, y0, t) # Solve the ODE.

# Calculate derivative of output signal.
y  = np.reshape(y, npoints)
dy = np.gradient(y, t)

# -----
# Plot and store data.

fig = plot_hysteresis(x, y**2)
# save_image(fig, "hys--butterfly")
# store_data("hys--butterfly", x, y**2, t)

# Fin :)