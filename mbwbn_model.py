"""
Modified Bouc-Wen-Baber-Noori (m-BWBN) model of hysteresis.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

References:
    Foliente, Greg. C. (1995). Hysteresis modeling of wood joints and structural 
    systems. Journal of Structural Engineering, 121(6), 1013-1022.

Date:
    Januray 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics_functions import external_force
from dynamics_functions import std_dissipated_energy_derivative
from dynamics_functions import std_restoring_force 
from dynamics_functions import simulate_ODE

from functions import plot_hysteresis


def mbwbn_model(t, X, params):
    """
    This function returns the set of equations of the modified 
    Bouc-Wen-Baber-Noori model (m-BWBN) by Foliente (1995). This model 
    works pinching with Folientes's pinching equation.
    
    The restoring force is considered to be the standard one: the paralel 
    action of a elastic spring and a hysteretic one.
    
    f_r(t) = akx + (1-a)kz.
        
    The dissipated energy is considered to be the standard one: the area 
    enclosed by the hysteretic component of the restoring force over the 
    hysteresis graph. 
    
    varepsilon = (1-alpha) k int_{0}^{t}x(tau)\dot{x}(tau).

 	Args:
        t (numpy.ndarray):
            Time vector.
		X (numpy.ndarray): 
            State matrix of the system:
            X[0] = x, displacement.
            X[1] = v, velocity.
            X[2] = z, hysteretic displacement
            X[3] = e, dissipated energy.
		f (numpy.ndarray): 
            Normalized mass external excitation.
		params (dict): 
            Dictionary of the parameters of the model. 

	Returns:
 	    dxdt (numpy.ndarray): ODE of the model.
    """        
    
    # Extract the desired calculations to be considered (True/False).
    degradation = params.get("degradation")
    pinching = params.get("pinching")
    
    x = X[0]    # Displacement.
    v = X[1]    # Velocity.
    z = X[2]    # Hysteretic displacement
    e = X[3]    # Dissipated energy
    
    t_span  = params["t_span"]  # time interval of the simulation.
    X_0     = params["X_0"]     # Initial state.
        
    # Extract the physical-related parameters.
    m      = params["m"]
    c      = params["c"]
    k      = params["k"]
    
    # Extract standard Bouc-Wen model related parameters.
    alpha  = params["alpha"]
    gamma  = params["gamma"]
    beta   = params["beta"]
    n      = params["n"]
    A      = params["A"]
           
    # m-BWBN with strength/stiffness degradation.
    if degradation:
        nu_0      = params.get("nu_0", 1)      
        delta_nu  = params.get("delta_nu", 1)
        A_0       = A      
        delta_A   = params.get("delta_A", 1)
        eta_0     = params.get("eta_0", 1)       
        delta_eta = params.get("delta_eta", 1)

        # Calculates the degradation parameters.
        nu_eps  = nu_0  + delta_nu * e  
        A_eps   = A_0   - delta_A  * e
        eta_eps = eta_0 + delta_eta * e
    else:
        # Set the degradation parameters innactive.
        nu_eps  = 1
        A_eps   = A   
        eta_eps = 1   
        
    # m-BWBN with pinching.
    if pinching:
        p 	      = params.get("p", 1)
        zeta_0     = params.get("zeta_0", 1)
        psi_0     = params.get("delta_psi", 1)
        delta_psi = params.get("delta_psi", 1)
        lambda_   = params.get("lambda_", 1)
        q         = params.get("q", 1)
        
        # Calculate the pinching parameters.
        zu   = (1/(nu_eps*(beta + gamma)))**(1/n)
        zeta_1 = (1 - np.exp(-p*e)) * zeta_0
        zeta_2 = (psi_0 + delta_psi*e) * (lambda_ + zeta_1)
        
        # Calculate the Foliente's pinching function.
        h = 1 - zeta_1 * np.exp(-((z*np.sign(v) - q*zu)**2) / (zeta_2**2))
    else:
        # Set the pinching function innactive
        h = 1
    
    # The external force.
    f0    = params["f0"]
    omega = params['omega']
    b     = params["b"]
    f = external_force(t_span, f0, omega, b, kind=1)
    
    # Interpolate the force at time t.
    t_values = np.linspace(t_span[0], t_span[1], len(f))
    f_t = np.interp(t, t_values, f)
       
    # System equations.        
    dxdt = np.array([
        v,
        (f_t - std_restoring_force(params, x, z) - c*v) / m,
        h*(A_eps*v - nu_eps*(beta*abs(v)*(abs(z)**(n-1))*z + gamma*v*(abs(z)**n))) / eta_eps,
        std_dissipated_energy_derivative(v, z, alpha, k)
    ]).T
    
    # Assuming dxdt is a numpy array, verify that dxdt has no empty nor inf 
    # data.
    if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
        print(__file__ + ": Check out! There are NaNs or Infs in the calculation")
        print(dxdt)
        
    return dxdt 


# -----
# Execute test.
if __name__ == "__main__":

    # Parameter related to the external force and the physics of the sytem.
    phys_f_params = {
        "m"      : 1,          # mass [kg]
        "k"      : 1,          # stiffness [kN/mm]
        "c"      : 0.10,       # damping coefficient
        "f0"     : 2.00,       # force amplitude
        "omega"  : 1.00,       # forcing frequency
        "b"      : 0.03,       # force exponential growth rate. 
        "t_span" : [0, 60],    # time span  
        "X_0"    : np.zeros(4) # Initial state
    } 
    
    # Wether to condier degradation or pinching.
    choice_params = {
        "degradation" :  False, 
        "pinching"    :  False
    }
    
    # Parameters related to the model.
    bw_params = {
        "alpha"       :  0.01,    # Ratio of post- to pre-yield sitffness.
        "beta"        :  0.5,    # Bouc-Wen parameter.    
        "gamma"       :  0.5,    # Bouc-Wen parameter.
        "n"           :  1.0,    # Bouc-Wen parameter.
        "nu_0"        :  0.4092, # strength degradation
        "delta_nu"    :  3.2397, # strength degradation parameter
        "A"           :  1.0000, # hysteresis amplitude
        "delta_A"     :  0.8155, # control parameter of 'A' with respect to the energy
        "eta_0"       :  3.4873, # stiffness degradation
        "delta_eta"   : -0.8492, # stiffness degradation parameter
        "p"           :  9.7076, # parameter which controls initial pinching
        "zeta_0"      :  1.0962, # pinching severity
        "psi_0"       :  1.2478, # pinching parameter
        "delta_psi"   : -3.6339, # parameter which controls change of pinching
        "lambda_"     : -3.2035, # pinching parameter
        "q"           :  1.5662  # pinching parameter 
    }
       
    # Build the whole set of parameter
    params = phys_f_params | choice_params | bw_params
    
    # Solve the system and unpack the state variables.
    t, sol = simulate_ODE(params=params, ode_system=mbwbn_model)
    x, y, z, e = sol
    
    # Calculate the restoring force
    f_r = std_restoring_force(params, x, z)

    # Plot the results.
    fig = plot_hysteresis(x, f_r)
    
    
# Fin :)