"""
Set of functions to use in the modeling of Bouc-Wen class simulations

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

Date:
    Januray 2025
"""


import numpy as np
from scipy.integrate import solve_ivp


def external_force(t_span, f0, omega, b, kind):
    """
    External force for the of hysteresis. The same as the restoring force in 
    experimental measurements.
    
    Args:
        t_span (list): [t_0, t_f]
            Time span for the analysis.
        f0 (float):
            Amplitud coefficient.
        omega (float):
            Forcing frequency.
        b (float):
            Regularization parameter for the growth rate
            Consider b = 0 for kind 1.
        kind (int):
            Kind of external excitation, two kinds are considered:
            kind = 1 : Sinusoidal kind with exponential growth.
            kind = 2 : Sinusoidal kind with linear growth.  
        
    Returns:
        f (numpy.ndarray):
            External excitation array.
    """
    # Define the time array.
    t = np.linspace(t_span[0], t_span[1], 1000)
    
    # Sinusoidal kind with exponential growth.
    if kind == 1: 
        f = f0 * (1 - np.exp(-b * t)) * np.sin(omega * t)
    # Sinusoidal kind with linear growth.
    elif kind == 2:
        f = f0 * t*np.sin(omega * t)
    else:
        raise Exception("Not valid external force formulation")
    return f

def std_restoring_force(params, x, z):
    """
    Computes the standard restoring force for the Bouc-Wen model, as the paralel 
    action of a elastic spring and a hysteretic one:
    
    $f_r(t) = akx + (1-a)kz$.
    
    Args:
        params (dict):
            Set of system parameters. Include `alpha` and `k`. 
        x (numpy.ndarray):
            Displacement history array.
        z (numpy.ndarray):
            Hysteretic displacement history array.
        
    Returns:
        f_r (numpy.ndarray):
            Restoring force history array.
    """
    # Unpack the parameters
    alpha = params["alpha"] # Ratio to post- to pre-yield stiffness.
    k     = params["k"]     # System stiffness. 
    f_r = alpha*k*x + (1-alpha)*k*z
    return f_r


def std_dissipated_energy_derivative(v, z, alpha, k):
    """
    This functios return the derivative of the standard dissipated energy, 
    given by:

    $varepsilon = (1-alpha) k int_{0}^{t}x(tau)\dot{x}(tau)$.
    
    Args:
        v (numpy.ndarray):
            Velocity vector (derivative of the displacement).
        z (numpy.ndarray):
            Hysteretic displacement vector.
        alpha (float):
            Ratio to post- to pre-yield stiffness.
        k (float):
            System stiffness.
        
    Returns:
        dot_varepsilon (numpy.ndarray): 
            Derivative of the dissipated energy.
    """
    # Calculate the derivative of the dissipated energy. 
    # The derivative cancels the integral.
    dot_varepsilon = (1-alpha)*k*z*v
    return dot_varepsilon


def simulate_ODE(params, ode_system, method="LSODA"):
    """
    Solve the ODE system using solve_ivp function. This function is created to 
    fit the neccesities of the Bouc-Wen class model.
    
    Args:
        params (dict): 
            Dictionary containing all necessary parameters.
        ode_system (numpy.ndarray):
            RHS of the state-space form of the ODE representing the system.
        method (str): default to "LSODA".
            Numerical method to be used in the solve_ivp function.
    
    Returns:
        t (numpy.ndarray): 
            Time vector where the solution is computed. 
        sol_dense (numpy.ndarray): 
            Dense solution matrix from solve_ivp. [x_1, x_2, ... x_n]
    """
    # Extract time span and initial conditions
    t_span = params['t_span']
    X_0    = params['X_0']
    
    # Set up the solver
    solution = solve_ivp(ode_system, t_span, X_0, 
        args=(params,),  
        method=method,      # Numerical method ("LSODA" or "Radau")
        rtol=1e-6,
        atol=1e-6,
        dense_output=True
    )
    
    if solution.sol is None:
        raise ValueError("Error: solve_ivp() did not generate a dense output \
            solution. Check solver settings.")

    # Create time array
    t = np.linspace(t_span[0], t_span[1], 1000)
    
    # Get solution at specified times
    sol_dense = solution.sol(t)
    
    return t, sol_dense