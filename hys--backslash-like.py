"""
Adapted.
This code creates a graph for the complex behavior of backlash-like hysteresis.
 
Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

References:
    Chandra, A., Daniels, B., Curti, M., Tiels, K., Lomonova, E. A., & 
    Tartakovsky, D. M. (2023). Discovery of sparse hysteresis models for 
    piezoelectric materials. Applied Physics Letters, 122(21).

    Original code: Exp1_Duhem_Simulated.py, from 
    https://github.com/chandratue/SmartHysteresis

Data:
    January 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from functions import plot_hysteresis, store_data, save_image


# -----
# Define the model function.
def model(y, t):
    dydt = 0.4*np.abs(t*np.cos(t) + np.sin(t))*t*np.sin(t) - \
        0.5*np.abs(t*np.cos(t) + np.sin(t))*y + 0.25*(t*np.cos(t) + np.sin(t))
    return dydt

# -----
# Model simulation. 

npoints = 1000                        # Number of points to calculate.

t  = np.linspace(0, 7*np.pi, npoints) # Generate time array.
x  = t*np.sin(t)                      # Define input signal.
dx = np.gradient(x, t)                # Calculate derivative of input signal.
y0 = 0                                # Set initial condition and solve ODE.
y  = odeint(model, y0, t)             # Solve the ODE.

# Calculate derivative of output signal.
y  = np.reshape(y, npoints)
dy = np.gradient(y, t)

# -----
# Plot and store data.

fig = plot_hysteresis(x, y)
# save_image(fig, "hys--backslash-like")
# store_data("hys--backlash-like", x, y, t)

# Fin :)