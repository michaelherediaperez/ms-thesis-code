"""
Analysis of the the orignal Bouc-Wen model of hysteresis.

Requirements:
    bw_model.py

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia
    
Date:
    Januray 2025
"""


import matplotlib.pyplot as plt
from bw_model import simulate_bouc_wen, restoring_force
from functions import save_image

# System parameters
params = {
        'm': 1.0,       # mass
        'c': 0.1,       # damping coefficient
        'k': 1.0,       # stiffness
        'alpha': 0.01,  # ratio of post to pre-yield stiffness
        'gamma': 0.5,   # controls shape of hysteresis
        'beta': 0.5,    # controls shape of hysteresis
        'n': 1.0,       # controls smoothness of transition
        'A': 1,         # controls amplitude of hysteresis
        'F0': 2,        # force amplitude
        'omega': 1,     # forcing frequency
        't_span': [0, 70],
        'x0': 0.0,      # initial displacement
        'v0': 0.0,      # initial velocity
        'z0': 0.0       # initial hysteretic variable
    }

# Generate data.
t, x, v, z = simulate_bouc_wen(params)
# Calculate the restoring force
f_r = restoring_force(params['alpha'], params['k'], x, z) 

# -----
# Single plot of the hysteresis and the time response.

# Create plots.
fig_1 = plt.figure(figsize=(15, 5))

# Time series.
plt.subplot(1, 2, 1)
plt.plot(t, x, 'b-', label='Displacement')
plt.plot(t, z, 'r--', label='Hysteretic Variable')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time Response')
plt.grid(True)
plt.legend()

# Hysteresis loop.
plt.subplot(1, 2, 2)
plt.plot(x, f_r, 'b-', label='Hysteresis Loop')
plt.plot(0, 0, 'b*')
plt.xlabel('Displacement')
plt.ylabel('Restoring force')
plt.title('Hysteresis Loop')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Save the figure
# save_image(fig_1, "bw--time-response-hyst-loop")

# -----
# Influence of each parameter in the hysteresis shape. 

# Create parameter variations.
variations = {
'base case': params.copy(),
'high beta': {**params, 'beta': 2.0},
'high gamma': {**params, 'gamma': 2.0},
'high n': {**params, 'n': 2.0},
'low alpha': {**params, 'alpha': 0.1},
'high A': {**params, 'A': 2.0}
}

# Create plots
fig_2 = plt.figure(figsize=(15, 10))

for i, (name, params) in enumerate(variations.items(), 1):
    
    # Run simulation
    t, x, v, z = simulate_bouc_wen(params)
    # Calculate the hysteretic force
    f_r = restoring_force(params['alpha'], params['k'], x, z) 
    
    # Plot hysteresis loop
    plt.subplot(2, 3, i)
    plt.plot(x, f_r, 'b-', label=name)
    plt.plot(0, 0, 'b*')
    plt.xlabel('Displacement')
    plt.ylabel('Restoring force')
    plt.title(f'{name}\n' + 
            f'β={params["beta"]}, γ={params["gamma"]},\n' +
            f'n={params["n"]}, α={params["alpha"]}, A={params["A"]}')
    plt.grid(True)
    
    plt.tight_layout()

plt.show()

# Save the figure
# save_image(fig_2, "bw--parameter-influence")

# Print explanation of effects
print("\nParameter Effects on Hysteresis Behavior:")
print("\n1. Beta:")
print("   - Higher beta: Makes the loop more pinched")
print("   - Lower beta: Makes the loop more rounded")
print("\n2. Gamma:")
print("   - Higher gamma: Increases the nonlinearity of the hysteresis")
print("   - Lower gamma: Makes the behavior more linear")
print("\n3. Exponent (n):")
print("   - Higher n: Sharpens the transitions")
print("   - Lower n: Smooths the transitions")
print("\n4. Alpha:")
print("   - Higher alpha: More linear elastic behavior")
print("   - Lower alpha: More hysteretic behavior")
print("\n5. Parameter A:")
print("   - Higher A: Increases the size of the hysteresis loop")
print("   - Lower A: Decreases the sizThe parameters of the standard Bouc-Wen model from Eq.~\eqref{eq:bouc-wen-model-sign} play a fundamental role in shaping the hysteresis behavior, each influencing the system's response in distinct ways. To better ilustrate this, a simulation was carried out solving the hysteretic system defined by Eqs.~\eqref{eq:bouc-wen-model-sign},~\eqref{eq:restoring-force} and~\ref{eq:sdof-eqm-nonlinear-dyn-system--c-constant}, taking into accoun the following parameters: mass of $m=1.0\,\mathrm{kg}$, a damping coefficient of $c=0.1\,\mathrm{Ns/m}$, and a stiffness of $k=1.0\,\mathrm{N/m}$. The external excitation was a sinusoidal force in the form of $$f = f_0 (1 - \exp(-b t)) \sin(\omega t)$$ with an amplitude of $f_0 =1.0\,\mathrm{N}$, a forcing frequency of $\omega = 1.0\,\mathrm{rad/s}$ and a exponential growth rate $b = 0.03\, s^{-1}$. The time span of the simulation was $0$ to $50\,\mathrm{s}$, with initial conditions set to zero for displacement, velocity, and the hysteretic variable.e of the hysteresis loop")


# -----
# Sensitivity analysis of the parameters.

# To see variations in each inidividual parameter, a shorter time is desired.
t_span_for_sensitivity = [0, 20] 

# Parameter variations to study
parameter_variations = {
    'n': [1, 2, 3, 4],
    'beta': [0.1, 0.5, 1.0, 2.0],
    'gamma': [0.1, 0.5, 1.0, 2.0],
    'alpha': [0.1, 0.3, 0.5, 0.7],
    'A': [0.5, 1.0, 1.5, 2.0]
}

# Create a plot for each parameter
fig_3, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (param_name, param_values) in enumerate(parameter_variations.items()):
    if i < len(axes):
        for value in param_values:
            # Create new parameters with this variation
            current_params = params.copy()
            current_params['t_span'] = t_span_for_sensitivity
            current_params[param_name] = value
            
            # Run simulation
            t, x, v, z = simulate_bouc_wen(current_params)
            # Calculate the restoring force.
            f_r = restoring_force(params['alpha'], params['k'], x, z) 
            
            # Plot
            axes[i].plot(x, f_r, label=f'{param_name}={value}')
            plt.plot(0, 0, 'b*')
            axes[i].set_xlabel('Displacement')
            axes[i].set_ylabel('Restoring force')
            axes[i].set_title(f'Effect of {param_name}')
            axes[i].grid(True)
            axes[i].legend()

# Remove empty subplot
if len(parameter_variations) < len(axes):
    fig_3.delaxes(axes[-1])
{param_name}
plt.tight_layout()
plt.show()

# Save the figure
# save_image(fig_3, "bw--sensitivity-analysis")

# Print explanation of parameter effects
print("\nParameter Effects on Hysteresis Behavior:")
for param_name in parameter_variations.keys():
    print(f"\n{param_name.upper()} variations:")
    if param_name == 'n':
        print("- Higher values lead to sharper transitions")
        print("- Lower values produce smoother, more rounded loops")
    elif param_name == 'beta':
        print("- Higher values create more pinched loops")
        print("- Lower values result in wider loops")
    elif param_name == 'gamma':
        print("- Higher values increase nonlinearity")
        print("- Lower values make behavior more linear")
    elif param_name == 'alpha':
        print("- Higher values increase linear elastic component")
        print("- Lower values enhance hysteretic behavior")
    elif param_name == 'A':
        print("- Higher values increase loop size")
        print("- Lower values decrease loop size")


# Fin :)