"""
Generalized Bouc-Wen (GBW) model of hysteresis.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

References:
    Song, J., & Der Kiureghian, A. (2006). Generalized Bouc-Wen model for 
    highly asymmetric hysteresis. Journal of engineering mechanics, 132(6), 
    610-618.

Date:
    Januray 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from functions import store_data, save_image, plot_hysteresis

from dynamics_functions import external_force
from dynamics_functions import std_restoring_force 
from dynamics_functions import simulate_ODE 


def calculate_psi(X, betas):
    """
    This function calculates the shape functions psi.

    Args:
        X (numpy.ndarray): 
            State matrix of the system:
            X[0] = x, displacement.
            X[1] = v, velocity.
            X[2] = z, hysteretic displacement
		betas (list): 
            A list with the beta_i parameters of the shape function for the GBW.

    Returns:
        psi (numpy.ndarray): 
            Vector with the values for the psi function for each instant.
    """
    # UNpack the state
    x, v, z = X
    
    # Calculate shape function.
    psi = betas[0]*np.sign(v*z) + betas[1]*np.sign(x*v) +\
          betas[2]*np.sign(x*z) + betas[3]*np.sign(v) + betas[4]*np.sign(z) +\
          betas[5]*np.sign(x) 
          
    return psi


def gbw_model(t, X, params):
    """
    This functions returns the ODE of the generalized Bouc-Wen model by Son 
    and Der Kiureghian (2006). The restoring force is considered to be the standard one: the paralel 
    action of a elastic spring and a hysteretic one.
    
    f_r(t) = akx + (1-a)kz.

    Args:
        t (numpy.ndarray):
            Time vector.
		X (numpy.ndarray): 
            State matrix of the system:
            X[0] = x, displacement.
            X[1] = v, velocity.
            X[2] = z, hysteretic displacement
		f (numpy.ndarray): 
            Normalized mass external excitation.
		params (dict): 
            Dictionary of the parameters of the model. 

	Returns:
 	    dxdt (numpy.ndarray): ODE of the model.
    """
    
    # Unpack the states
    x, v, z = X
    # Timse span of the simulation.
    t_span  = params["t_span"]  
    
    # Unpack the parameters.
    m     = params["m"] 
    c     = params["c"] 
    k     = params["k"] 
    alpha = params["alpha"] 
    betas = params["betas"] 
    n     = params["A"] 
    A     = params["n"] 
    
    # The external force.
    f0    = params["f0"]
    omega = params['omega']
    b     = params["b"]
    f     = external_force(t_span, f0, omega, b, kind=1)
    
    # Interpolate the force at time t.
    t_values = np.linspace(t_span[0], t_span[1], len(f))
    f_t = np.interp(t, t_values, f)
        
    # Calculate psi
    psi = calculate_psi(X, betas)
    
    # System equations
    dxdt = v
    dvdt = (f_t - c*v - std_restoring_force(params, x, z)) / m
    dzdt = v * (A - abs(z)**n * psi)
    
    return [dxdt, dvdt, dzdt]


# Run example.
if __name__ == "__main__":

    # This parameters are extracted from the original WGBW paper.
    # params = {
    #     "m": 100,          
    #     "c": 0.1,           
    #     "k": 49.2,          
    #     "f0": 100,          
    #     "omega": 0.63,      
    #     "b": 0.1,            
    #     "t_span": [0, 100],  
    #     "alpha": 0.9,      
    #     "n": 1,           
    #     "A": 1.0,           
    #     "betas": [           
    #         0.470,           
    #         -0.118,           
    #         0.0294,           
    #         0.115,           
    #         -0.121,           
    #         -0.112,               
    #     ],           
    #     "x0": 0.0,          
    #     "v0": 0.0,          
    #     "z0": 0.0,          
    # }
    
    params = {
        "m"      : 1,          
        "c"      : 0.1,           
        "k"      : 1,          
        "f0"     : 2,          
        "omega"  : 1,      
        "b"      : 0.1,            
        "t_span" : [0, 60],  
        "alpha"  : 0.01,      
        "n"      : 1,           
        "A"      : 1.0,           
        "betas"  : [           
             0.470,           
            -0.118,           
             0.029,           
             0.115,           
            -0.121,           
            -0.112,               
        ],           
        "X_0"    : np.zeros(3) 
    }

    # Simulate the GBW model.
    t, sol = simulate_ODE(params=params, ode_system=gbw_model)
    x, v, z = sol

    # Calculate the restoring force
    f_r = std_restoring_force(params, x, z)
    
    # Plot the hysteresis
    fig = plot_hysteresis(x, f_r)
    # save_image(fig, "hys--asymmetric")
    # store_data("hys--asymmetric", x, f_r, t)
    
# Fin :)
