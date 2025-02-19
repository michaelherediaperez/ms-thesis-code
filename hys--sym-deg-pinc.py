"""
Creating multiple hysteretic behaviors. 

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
from mbwbn_model import simulate_mbwbm, std_restoring_force


# Define the time span and initial values.
params = {}
params["t_span"] = [0, 70]
params["X_0"] = np.zeros(4) 

# The following parameter sets are to be considered.
rows = {
    0 : "hys--symmetric",
    1 : "hys--deg-strength",
    2 : "hys--deg-stiffness",
    5 : "hys--pinching"  
}

for row, behavior in rows.items():
    # Read the parameters.
    params_row = read_parameter_set("./data/bw_parameters.csv", row_index=row)
    params = params | params_row
    
    # Run the simulation.
    x, v, z, e = simulate_mbwbm(params)

    # Calculate the standard restoring force.
    f_r = std_restoring_force(params, x, z) 

    # Present the results.
    fig = plot_hysteresis(x, f_r)
    
    # Store the results.
    save_image(fig, behavior)
    # t = np.linspace(params["t_span"][0], params["t_span"][1], x.shape[0])
    # store_data(behavior, x, f_r, t)