"""
Standard Bouc-Wen model of hysteresis.

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia
    
Considering the eq. of motion for a SDOF as:

    m\ddot{x} + c\dot{x} + f_r(t) = f(t)
    
where the restoring force f_r(t) is given by:
    
    f_r(t) = akx + (1-a)kz

and the hystretic displacement z is as follows:  

- Equation:
    dz      dx        | dx |       {n-1}         dx | dz | ^ {n}
    -- =  A -- - beta | -- | z |z|^     - gamma  -- | -- |
    dt      dt        | dt |                     dt | dt |


- LaTeX:
    \dot{z}=A\dot{x}-\beta|\dot{x}|z|z|^{n-1}-\gamma\dot{x}|z|^{n}.
    
Date:
    Januray 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Custom functions
from functions import store_data, save_image, plot_hysteresis


def external_force_sin(t, F0, omega, growth=False):
    """
    External force for the Bouc Wen model of hysteresis. For the moment, it is 
    a Sinusoidal excitation.
    """
    if growth:
        alpha = growth 
        f = F0 * (1 - np.exp(-alpha * t)) * np.sin(omega * t)
    else:
        f= F0 * t*np.sin(omega * t)
    return f

def restoring_force(alpha, k, x, z):
    """
    Coputes the restoring force as f_r(t) = akx + (1-a)kz.
    """
    f_r = alpha*k*x + (1-alpha)*k*z
    return f_r

def bouc_wen_system(t, state, m, c, k, alpha, gamma, beta, n, A, F0, omega):
    """
    Bouc-Wen model equations
    state[0] = x (displacement)
    state[1] = v (velocity)
    state[2] = z (hysteretic variable)
    """
    x, v, z = state
    
    # External force (sinusoidal)
    f = external_force_sin(t, F0, omega, growth=0.03)    
    #f = external_force_sin(t, F0, omega)

    # System equations
    dxdt = v
    dvdt = (f- c*v - restoring_force(alpha, k, x, z)) / m
    dzdt = A*v - beta*abs(v)*z*abs(z)**(n-1) - gamma*v*abs(z)**n 
    
    return [dxdt, dvdt, dzdt]

def simulate_bouc_wen(params, derivatives=False):
    """
    Generate simulation data using solve_ivp
    """
    # Extract parameters
    m      = params["m"]
    c      = params["c"]
    k      = params["k"]
    alpha  = params["alpha"]
    gamma  = params["gamma"]
    beta   = params["beta"]
    n      = params["n"]
    A      = params["A"]
    F0     = params["F0"]
    omega  = params["omega"]
    t_span = params["t_span"]
    x0     = params["x0"]
    v0     = params["v0"]
    z0     = params["z0"]
    
    # Initial conditions
    initial_state = [x0, v0, z0]
    
    # Solve using solve_ivp
    solution = solve_ivp(bouc_wen_system, t_span, initial_state,
                        args=(m, c, k, alpha, gamma, beta, n, A, F0, omega),
                        dense_output=True,
                        method="RK45",
                        rtol=1e-10,
                        atol=1e-10)
    
    # Create time array
    t = np.linspace(t_span[0], t_span[1], 1000)
    
    # Get solution at specified times
    sol_dense = solution.sol(t)
    x = sol_dense[0]  # displacement
    v = sol_dense[1]  # velocity
    z = sol_dense[2]  # hysteretic variable

    # Calculate the derivatives of the velocity and the hysteretic 
    # displacement. We know the exact solution to this derivative.
    if derivatives:
        # Caclulate the external force, again.
        f = external_force_sin(t, F0, omega)
        # a = dvdt
        a = (f- c*v - alpha*k*x - (1-alpha)*k*z) / m
        # dz = dzdt. 
        dz = A*v - beta*abs(v)*z*abs(z)**(n-1) - gamma*v*abs(z)**n      
        return t, x, v, z, a, dz
    else:
        return t, x, v, z

# -----
# Run examples.
if __name__ == "__main__":

    # System parameters
    params = {
        "m": 1.0,       # mass
        "c": 0.1,       # damping coefficient
        "k": 1.0,       # stiffness
        "alpha": 0.01,  # ratio of post to pre-yield stiffness
        "gamma": 0.5,   # controls shape of hysteresis
        "beta": 0.5,    # controls shape of hysteresis
        "n": 1.0,       # controls smoothness of transition
        "A": 1,         # controls amplitude of hysteresis
        "F0": 2,        # force amplitude
        "omega": 1,     # forcing frequency
        "t_span": [0, 40],
        "x0": 0.0,      # initial displacement
        "v0": 0.0,      # initial velocity
        "z0": 0.0       # initial hysteretic variable
    }

    # Generate data
    t, x, v, z = simulate_bouc_wen(params)
    # Calculate the restoring force
    f_r = restoring_force(params["alpha"], params["k"], x, z) 

    # Create plots
    fig = plt.figure(figsize=(15, 5))

    # Time series
    plt.subplot(1, 3, 1)
    plt.plot(t, x, "b-", label="Displacement")
    plt.plot(t, z, "r--", label="Hysteretic Variable")
    plt.xlabel("Time")
    plt.ylabel("Amplitud")
    plt.grid(True, alpha=0.5)
    plt.legend()

    # Time series
    plt.subplot(1, 3, 2)
    plt.plot(t, f_r, "b-")
    plt.xlabel("Time")
    plt.ylabel("Restoring force")
    plt.grid(True, alpha=0.5)

    # Hysteresis loop
    plt.subplot(1, 3, 3)
    plt.plot(x, f_r, "b-")
    plt.plot(0, 0, "b*")
    plt.xlabel("Displacement")
    plt.ylabel("Restoring force")
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # -----
    # Store the data
    # store_data("hys--simulation-02", x, f_r, t)
    # save_image(fig, "hys--simulation-01")
    
    # For hardening.
    # fig = plot_hysteresis(x, f_r)
    # save_image(fig, "hys--hardening")
    # store_data("hys--hardening", x, f_r, t)
# Fin :)