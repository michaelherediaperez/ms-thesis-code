# -*- coding: utf-8 -*-

"""
Ploting hysteresis behaviors from previous simulations. 
  

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

April 2025
"""


import os
import numpy as np

from functions import plot_hysteresis, save_image

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


# Path to directory.
data_dir = "results_data"

# Iterate over .txt files in the results directory.
for file_name in os.listdir(data_dir):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_dir, file_name)
        
        # Load the data, skipping the header line.
        try:
            data = np.loadtxt(file_path, skiprows=1)
            x, fr = data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Could not read {file_name}: {e}")
            continue
        
        # Extrating the file name without extension.
        base_name = os.path.splitext(file_name)[0]

        # Call plotting and image saving for color plots.
        fig = plot_hysteresis(x, fr, labels=True, color=True)
        save_image(fig, f"{base_name}-color", directory="hysteresis_plots")
        
        # Call plotting and image saving for color plots.
        fig = plot_hysteresis(x, fr, labels=True, color=False)
        save_image(fig, f"{base_name}-black", directory="hysteresis_plots")
        

# Fin :)