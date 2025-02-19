"""
This code creates a graph for the complex behavior of backlash-like hysteresis 

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    January 2025

References:
    Chandra, A., Daniels, B., Curti, M., Tiels, K., Lomonova, E. A., & 
    Tartakovsky, D. M. (2023). Discovery of sparse hysteresis models for 
    piezoelectric materials. Applied Physics Letters, 122(21).

    Original code: Exp1_Duhem_Simulated.py, from 
    https://github.com/chandratue/SmartHysteresis

Commentary:
    AI platform Claude.ai was used in this code to solve problems.  
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functions import plot_hysteresis, store_data, save_image

# -----
# Obtain the data by solving the model. 

# Number of points to calculate.
npoints = 1000

# Generate time array.
t = np.linspace(0, 7*np.pi, npoints)

# Define input signal.
x = t*np.sin(t)
#x = 1.0 * (1 - np.exp(-0.01 * t)) * np.sin( * t)

# Calculate derivative of input signal.
dx = np.gradient(x, t)

# Define the model function.
def model(y, t):
    dydt = 0.4*np.abs(t*np.cos(t) + np.sin(t))*t*np.sin(t) - \
        0.5*np.abs(t*np.cos(t) + np.sin(t))*y + 0.25*(t*np.cos(t) + np.sin(t))
    return dydt

# Set initial condition and solve ODE.
y0 = 0
y = odeint(model, y0, t)

# Calculate derivative of output signal.
y = np.reshape(y, npoints)
dy = np.gradient(y, t)

# -----
# Plot and store data.

fig = plot_hysteresis(x, y)
save_image(fig, "hys--backslash-like")
store_data("hys--backlash-like", x, y, t)

# Fin :)