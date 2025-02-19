"""
This code creates a graph for the complex behavior of butterfly hysteresis. 

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    January 2025

Reference:
    Dong, H., Wang, C., Han, Q., & Du, X. (2024). Re-centering capability of 
    partially self-centering structures with flag-shaped hysteretic behavior 
    subjected to near-fault pulsed ground motion. Soil Dynamics and Earthquake 
    Engineering, 186, 108892.

Date.
    January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from functions import plot_hysteresis, save_image, store_data


def sc_restoring_force(x, z, params):
    """
    Computes the restoring force of a self-centering system. 
    Bases on Dong et al. 2024. 

    Args:
        x (numpy.ndarray):
            Displacement vector.
        z (numpy.ndarray):
            Hysteretic displacement vector.
        params (dict): 
            Dictionary with the system parameters.

    Returns:
        numpy.ndarray: restoring force vector.
    """    
    # Unpack the parameters. 
    alpha = params["alpha"]
    ke    = params["ke"]
    ks1   = params["ks1"]
    ks2   = params["ks2"]
    usy   = params["usy"]
    
    # Hysteretic force component.
    fe = alpha * ke * x + (1 - alpha) * ke * z  
    
    # Self-centering force component.
    fs = np.where(
        abs(x) < usy,
        ks1 * x,
        ks1 * usy * np.sign(x) + ks2 * (x - usy * np.sign(x))
    )
        
    # Combined restoring force
    f_r = fe + fs
    
    return f_r


def bw_fs_model(t, X, params):
    """
    Differential equations governing the Bouc-Wen Flag-Shaped (BW-FS) 
    hysteresis model.
    
    Args:
        t (numpy.ndarray): 
            Time vector.
        X (numpy.ndarray): 
            State vector [x, v, z].
        params (dict): 
            Dictionary containing system parameters.
    
    Returns:
        dydt: Time derivatives [dx/dt, dv/dt, dz/dt]
    """
    # Unpacl the state variables.
    x, v, z = X  

    # Unpack system parameters
    m     = params["m"]     # Mass
    c     = params["c"]     # Damping coefficient
    beta  = params["beta"]  # Bouc-Wen shape parameter
    gamma = params["gamma"] # Bouc-Wen shape parameter
    n     = params["n"]     # Bouc-Wen exponent
    omega = params["omega"] # Frequency of external forcing
    
    # Hysteretic displacement evolution. Neither degradation nor pinching 
    # are considered. 
    dzdt = beta * abs(v) * (1 - abs(z)**n) * np.sign(v) - gamma * z * abs(v)

    # Restoring force for a self-centering element
    f_r = sc_restoring_force(x, z, params)
    
    # External forcing (sinusodial with exponential growth).
    f_ext = t*np.exp(-0.03*t)*np.sin(omega* t)

    # Equation of motion: m*\ddot{x} + c*\dot{x} + f_r = f_ext
    dvdt = (f_ext - c * v - f_r) / m

    return [v, dvdt, dzdt]


# -----
# Run test
if __name__ == "__main__": 
    # Define system parameters
    params = {
        "m"     : 1.0,   # Mass
        "c"     : 0.01,  # Damping coefficient
        "ke"    : 2,     # Initial stiffness
        "alpha" : 0,     # Post-yield stiffness ratio
        "beta"  : 0.5,   # Bouc-Wen shape parameter
        "gamma" : 0.5,   # Bouc-Wen shape parameter
        "n"     : 4,     # Bouc-Wen exponent
        "ks1"   : 10.0,  # Initial stiffness of self-centering system
        "ks2"   : 0.5,   # Post-yield stiffness after yielding
        "usy"   : 0.5,   # Yield displacement
        "omega" : 2.0    # Frequency of excitation
    }

    # Initial conditions: [x(0), v(0), z(0)]
    y0 = [0.0, 0.0, 0.0]

    # Time span, simulate for 50 seconds.
    t_span = (0, 60) 
    # Time steps.
    t_eval = np.linspace(t_span[0], t_span[1], 1000) 

    # Solve the system with solve_ivp.
    solution = solve_ivp(
        bw_fs_model, t_span, y0, 
        args=(params,), t_eval=t_eval, method="RK45", dense_output=True
        )

    # Extract results
    t = solution.t
    x = solution.y[0]   # Displacement
    z = solution.y[2]   # Hysteretic displacement
    
    # Compute the restoring force.
    f_r = sc_restoring_force(x, z, params)

    # Plot results.
    fig = plot_hysteresis(x, f_r)
    
    # Store the results.
    # save_image(fig, "hys--flag-shape")
    # store_data("hys--flag-shape", x, f_r, t_eval)

# Fin :)