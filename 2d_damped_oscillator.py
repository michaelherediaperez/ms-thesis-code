"""
Recreating the two-dimensional damped oscillator example form SINDY

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

References:
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing 
    equations from data by sparse identification of nonlinear dynamical systems. 
    Proceedings of the national academy of sciences, 113(15), 3932-3937.

Date:
    January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps

from functions import plot_data, plot_comparatives_3, save_image

def damped_oscillator(t, state, a, b, c, d, n):
    """
    Simulate a two dimensional damped oscillator with multiple complexity nonlinearity.  

    dx = a x^n + b y^n
    dy = c x^n + d y^n 
    """ 

    # Extract the state
    x, y = state

    # Define the system 
    dx = a * x**n + b * y**n
    dy = c * x**n + d * y**n

    return [dx, dy] 

def simulate(params):
    """
    Simulate the damped oscillator with solve_ivp
    """
    # Extract the parameters.
    dt         = params["dt"]      # Time step
    t_span     = params["t_span"]   # time span for the simulation.
    X0         = params["X0"]       # Initial state. 
    a, b, c, d = params["abcd"]     # System parameters.
    n          = params["n"]        # nonlinearity degree

    # Solve using solve_ivp
    solution = solve_ivp(
        damped_oscillator, t_span, X0, 
        args=(a, b, c, d, n), 
        dense_output=True, 
        # method = "LSODA", recommended by pysindy Oficial Doc.
        method="RK45", 
        rtol=1e-10, 
        atol=1e-10
        )
     
    # Create time array
    t = np.arange(t_span[0], t_span[1], dt)
    
    # Get solution at specified times
    sol_dense = solution.sol(t)
    x = sol_dense[0]  
    y = sol_dense[1]  

    return x, y, t

# -----
# Run examples
params = {
    "X0"     : [2, 0],
    "t_span" : [0, 25],
    "dt"     : 0.01,   
    "abcd"   : [-0.1, 2, -2, -0.1],
    "n"      : [1, 3]
    }

figures_names = ["2d_oscillator_n1", "2d_oscillator_n3"]

# Iterate over each `n` in params["n"]
for i, n in enumerate(params["n"]):
    params2 = params.copy()
    params2["n"] = n  

    # Simulate the data.
    x, y, t = simulate(params2)

    # Build the state vector.
    X = np.column_stack([x, y])

    # Define the SINDy algorithm parameters.
    poly_order = 3
    threshold  = 0.03

    # Build the SINDy model and displayit in screen.
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(X, t=t)
    model.print()

    # Simulate with the SINDy model.
    X_sindy = model.simulate(np.array(params["X0"]).T, t)

    # Presents the results.
    fig = plot_comparatives_3(X.T, X_sindy.T, t)
    # Save the plots.
    # save_image(fig, figures_names[i])
    # print("\n")

