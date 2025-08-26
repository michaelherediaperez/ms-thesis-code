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
import itertools
                    
from bw_model import simulate_bouc_wen, restoring_force
from functions import save_image

import matplotlib as mpl
# Configure matplotlib for STIX font - comprehensive setup
mpl.rcParams.update({
    # Primary font configuration
    "font.family": "serif",              # Use serif family
    "font.serif": ["STIX", "STIXGeneral", "STIX Two Text"], # STIX font priority
    "mathtext.fontset": "stix",          # Math expressions in STIX
    
    # Explicit font specification for all text elements
    "axes.labelsize": 18,
    "axes.titlesize": 18, 
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.size": 16,
    
    # Line properties
    "lines.linewidth": 1.5
})

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
# Single plot of the hysteresis and the time response. (with colors)

def single_plot_of_hysteresis_with_response_time(color=True):
    """
    This function creates a single plot showing the time response and the 
    hysteresis loop, both with colors if `color` is True, or in black and white 
    if False (this last one for the Springer publication).

    Args:
        color (bool, optional): To use colors blue and red. Defaults to True.
    
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plots.
    """
    # Create plots.
    fig_1 = plt.figure(figsize=(15, 5))

    # Time series.
    plt.subplot(1, 2, 1)
    if color:
        plt.plot(t, x, 'b-', label='Displacement')
        plt.plot(t, z, 'r--', label='Hysteretic Variable')
    else:
        plt.plot(t, x, 'k-', label=r'Displacement $x(t)$')  # Black line
        plt.plot(t, z, 'k--', label=r'Hysteretic Variable $z(t)$')  # Dashed black
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Time Response')
    plt.grid(True, alpha=0.5)
    plt.legend()

    # Hysteresis loop.
    plt.subplot(1, 2, 2)
    if color:
        plt.plot(x, f_r, 'b-', label='Hysteresis Loop')
        plt.plot(0, 0, 'b*')
    else:
        plt.plot(x, f_r, 'k-', label='Hysteresis Loop')  # Black line
        plt.plot(0, 0, 'ko')  # Black circle at origin
    plt.xlabel('Displacement')
    plt.ylabel('Restoring force')
    plt.title('Hysteresis Loop')
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return fig_1

# Create the figure with colors.
fig_1_color = single_plot_of_hysteresis_with_response_time(color=True)    
# Create the figure in black and white for Springer publication.
fig_1_black = single_plot_of_hysteresis_with_response_time(color=False)
# Save the figure
save_image(fig_1_color, "bw--time-response-hyst-loop-color")
save_image(fig_1_black, "bw--time-response-hyst-loop-black") 

# -----
# Influence of each parameter in the hysteresis shape. 

# Create parameter variations.
variations = {
'base case': params.copy(),
'high beta': {**params, 'beta': 2.0},
'high gamma': {**params, 'gamma': 2.0},
'high n': {**params, 'n': 2.0},
'high alpha': {**params, 'alpha': 0.1},
'high A': {**params, 'A': 2.0}
}

def influence_of_each_parameter_in_hysteresis_shape(color=True):
    """
    This function creates a single plot showing the influence of each parameter 
    in the hysteresis shape. It generates subplots for each variation of the 
    standard Bouc-Wen mdoel, both with colors if `color` is True, or in black 
    and white if False (this last one for the Springer publication).

    Args:
        color (bool, optional): To use colors blue and red. Defaults to True.
    
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plots.
    
    """
    # Create plots
    fig_2 = plt.figure(figsize=(15, 10))

    for i, (name, params) in enumerate(variations.items(), 1):
        
        # Run simulation
        t, x, v, z = simulate_bouc_wen(params)
        # Calculate the hysteretic force
        f_r = restoring_force(params['alpha'], params['k'], x, z) 
        
        # Plot hysteresis loop
        plt.subplot(2, 3, i)
        if color:
            plt.plot(x, f_r, 'b-', label=name)
            plt.plot(0, 0, 'b*')
        else:
            plt.plot(x, f_r, 'k-', label=name)
            plt.plot(0, 0, 'ko')
        plt.xlabel('Displacement [m]')
        plt.ylabel('Restoring force [N]')
        plt.title(f'{name}\n'
                rf'$\beta={params["beta"]}, \gamma={params["gamma"]},$' '\n'
                rf'$n={params["n"]}, \alpha={params["alpha"]}, A={params["A"]}$')
        plt.grid(True)
        
        plt.tight_layout()

    plt.show()
    
    return fig_2

# Create the figure with colors.
fig_2_color = influence_of_each_parameter_in_hysteresis_shape(color=True)    
# Create the figure in black and white for Springer publication.
fig_2_black = influence_of_each_parameter_in_hysteresis_shape(color=False)
# Save the figure
save_image(fig_2_color, "bw--parameter-influence-color")
save_image(fig_2_black, "bw--parameter-influence-black")

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
print("   - Lower A: Decreases the size of the hysteresis loop")


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

def sensitivity_analysis_of_parameters(color=True):
    """
    This function performs a sensitivity analysis of the parameters of the 
    standard Bouc-Wen model. It generates subplots for each parameter variation,
    both with colors if `color` is True, or in black and white if False (this 
    last one for the Springer publication).

    Args:
        color (bool, optional): To use colors blue and red. Defaults to True.
    
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plots.
    """

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
                if color:
                    axes[i].plot(x, f_r, label=f'{param_name}={value}')
                    plt.plot(0, 0, 'k*')
                else:
                    # Cycle through different line styles
                    linestyles = itertools.cycle(['-', '--', ':', '-.'])  
                    # Get the next linestyle
                    linestyle = next(linestyles)  
            
                    axes[i].plot(x, f_r, color='black', linestyle=linestyle, label=rf'{param_name} = {value}')
                    axes[i].plot(0, 0, 'ko')
                
                axes[i].set_xlabel('Displacement [m]')
                axes[i].set_ylabel('Restoring force [N]')
                axes[i].set_title(f'Effect of {param_name}')
                axes[i].grid(True)
                axes[i].legend()

    # Remove empty subplot
    if len(parameter_variations) < len(axes):
        fig_3.delaxes(axes[-1])
    {param_name}
    plt.tight_layout()
    plt.show()

    return fig_3

# Create the figure with colors.
fig_3_color = sensitivity_analysis_of_parameters(color=True)    
# Create the figure in black and white for Springer publication.
fig_3_black =  sensitivity_analysis_of_parameters(color=False)
# Save the figure
save_image(fig_3_color, "bw--sensitivity-analysis-color")
save_image(fig_3_black, "bw--sensitivity-analysis-black")

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

# -----
# Fin :)