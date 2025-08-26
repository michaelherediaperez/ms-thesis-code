"""
Ploting multiple hysteretic behaviors: symmetric, degradation and pinching. 

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

Date:
    Januray 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from functions import save_image, store_data, read_parameter_set, plot_hysteresis
from bw_mbwbn_model import mbwbn_model
from dynamics_functions import std_restoring_force, simulate_ODE


# Define the time span and initial values.
params = {}
params["t_span"] = [0, 70]
params["X_0"]    = np.zeros(4) 

# The following parameter sets are to be considered. 
# The keys are the rows chosen form the file "./data/bw_parameters.csv".
# The values are the behaviors they represent. 
rows = {
    0 : "hys--symmetric",
    1 : "hys--deg-strength",
    2 : "hys--deg-stiffness",
    5 : "hys--pinching"  
}

for row, behavior in rows.items():
    
    # Read the parameters.
    params_row = read_parameter_set("./data/bw_parameters.csv", row_index=row)
    # Join the parameters fromthe .csv file to the fixed ones.
    params = params | params_row
    
    # Run the simulation.
    t, sol = simulate_ODE(params=params, ode_system=mbwbn_model)
    x, v, z, e = sol

    # Calculate the standard restoring force.
    f_r = std_restoring_force(params, x, z) 

    # Present the results.
    fig = plot_hysteresis(x, f_r)
    
    # Store the results.
    # save_image(fig, behavior)
    # store_data(behavior, x, f_r, t)