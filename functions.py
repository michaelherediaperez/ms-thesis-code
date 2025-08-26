"""
Series of functions to optimize plotting and data production.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia

January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple


def plot_hysteresis(x, fr, labels=False, color=True):
    """
    Plot the hysteresis with or without labels. The only lables considered will be "Restoring force" and "Displacement". 

    Args:
        x (numpy.ndarray or list or tuple): 
            Displacement.
        fr (numpy.ndarray or list or tuple):
            Restoring force. 
        labels (bool, optional): Defaults to True.
            Wether or not to plot labels. 
    
    Returns:
        fig (matplotlib.figure.Figure)
    """
    # Set the figure size and put the lables if asked for.
    if labels:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set(xlabel="Displacement", ylabel="Restoring force")
    else:
        # A square plot.
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the hysteresis.
    if color:
        ax.plot(x, fr, 'b', linewidth=1)
        # Plot the initial point.
        ax.plot(0, 0, 'b*')
    else:
        ax.plot(x, fr, 'k', linewidth=1)
        # Plot the initial point.
        ax.plot(0, 0, 'k*')
    # Configure the plot aesthetic (no numbers and grid).
    ax.set_xticklabels([])          
    ax.set_yticklabels([])
    plt.grid(True, alpha=0.3)
    
    plt.show()
    return fig


def plot_data(x, y, t):
    """
    Plots the simulation.
    """
    # Create the figure to store the axes.
    # figsize 15/5 = 3 identical plots.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].plot(t, x, "b-")  
    axs[1].plot(t, y, "b-")  
    axs[2].plot(x, y, "b-")  

    # Define the titles and labels
    x_labels = ["t"]*3
    y_labels = ["x", "y", "y"] 
    for i, ax in enumerate(axs):
        ax.set(xlabel=x_labels[i], ylabel=y_labels[i])  
        ax.grid()

    # Adjust layout for better visualization, and present.
    plt.tight_layout()
    plt.show()

    return fig


def plot_comparatives_3(X, X_s, t, xylabels=["x", "y", "t"], title=""):
    """
    This function creates a graph with three plots showing the phase-state plot 
    of the hysteresis simulation and the identified model.
    
    Args:
        X (numpy.ndarray):
            State matrix of the system [x(t), v(t), z(t)]
        X_s (numpy.ndarray):
            SINDy output.
        t (numpy.ndarray):
            Time vector.
        xylabels (list): default ["x", "y", "t"].
            Lables to put to the state variables
        title (str): default "".
            Title for the plot.
            
    Returns:
        fig (matplotlib.figure.Figure):
            The plot.
    """
    # Extract the data
    x, y     = X
    x_s, y_s = X_s

    # Create the figure to store the axes.
    # figsize 15/5 = 3 identical plots.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].plot(t, x, "b-", label="data")  
    axs[1].plot(t, y, "b-", label="data")  
    axs[2].plot(x, y, "b-", label="data")  

    axs[0].plot(t,   x_s, "r--", label="sindy")  
    axs[1].plot(t,   y_s, "r--", label="sindy")  
    axs[2].plot(x_s, y_s, "r--", label="sindy")  

    # Define the titles and labels
    x_labels = [xylabels[2], xylabels[2], xylabels[0]] # t, t, x
    y_labels = [xylabels[0], xylabels[1], xylabels[1]] # x, y, y

    for i, ax in enumerate(axs):
        ax.set(xlabel=x_labels[i], ylabel=y_labels[i])  
        ax.grid()
        ax.legend()
    
    if title != "":
        fig.suptitle(title, fontsize=16)

    # Adjust layout for better visualization, and present.
    plt.title = title
    plt.tight_layout()
    plt.show()

    return fig


def plot_comparatives(t, X_dot, X_dot_s, hysteresis):
    """
    This function creates a graph with the data of the system measurements and 
    the simulation performed with the SINDy algorithm. The comparison of the 
    derivatives of the state variables and the constructed hysteresis are 
    presented, each one its individual plot.

    Args:
        t (array): dimension (n_samples)
            time data.
        
        X_dot (array of arrays): dimensions (n_samples, n_input_features) 
            The system simulated or real dynamics. 

            >>>X_dot = [dxdt, dvdt, dzdt]
        
        X_dot_s (array of arrays): dimensions (n_samples, n_input_features)
            The dynamics obtained with the SINDy (_s) algorithm.

            >>>X_dot_s  = [dxdt_s, dvdt_s, dzdt_s]
        
        hystresis (list of arrays): dimensions (n_samples, 4) 
            Data to build the hystresis, first pair is the 
            simulation or real, second pair is the SINDy results.

            >>>hysteresis = [x, F_r, x_s, F_r_s]    
    
    Returns:
        fig (matplotlib.figure.Figure)
    """

    # A List with the titles and ylables for each subplot, except for the 
    # hystresis one
    titles = ["Displacement evolution", 
              "Velocity evolution",
              "Hysteretic variable evolution",
              ""]
    y_labels = ["dx/dt", "dv/dt", "dx/dt", ""]

    # Create the figure to store the axes.
    fig = plt.figure(figsize=(10, 6))
    
    # Set the grid: Three rows, two columns.
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1])  

    # Asign axes to each cell inside the grid.
    ax1 = fig.add_subplot(grid[0, 0])  # Top left
    ax2 = fig.add_subplot(grid[1, 0])  # Middle left
    ax3 = fig.add_subplot(grid[2, 0])  # Bottom left
    ax4 = fig.add_subplot(grid[:, 1])  # Right column (spanning all rows)

    # Creata a list of axes.
    axes = [ax1, ax2, ax3, ax4]

    # Iterate over the axes ploting for each one.
    for i, ax in enumerate(axes):
        # The last plot, ax4, is the hystresis plot. So:
        if i == 3:
            ax.plot(hysteresis[0], hysteresis[1], 'b-', label='Actual')
            ax.plot(hysteresis[2], hysteresis[3], 'r--', label='Predicted')
            ax.set_xlabel('Displacement x(t) [m]')
            ax.set_ylabel('Restoring force F_r(t) [kN]')
            ax.set_title("Hysteresis")
            ax.legend()
        else:
            ax.plot(t, X_dot[:, i], 'b-')
            ax.plot(t, X_dot_s[:, i], 'r--')
            ax.set_ylabel(y_labels[i])
            ax.set_title(titles[i])
            # Write the xlabel just for the last plot of the left column.
            if i == 2:
                ax.set_xlabel("Time [s]")     
        # Set a grid for each graph.  
        ax.grid()

    # Adjust layout for better visualization, and present.
    plt.tight_layout()
    plt.show()

    return fig


def store_data(file_name, x, y, t, results_dir="results_data", voltage=False):
    """
    Store simulation results into a text file with a specified format.

    Args:
        file_name (str): 
            Base name for the output file (without extension).
        x (numpy.ndarray): 
            Data for the second column (e.g., force or voltage).
        y (numpy.ndarray): 
            Data for the third column (e.g., displacement).
        t (numpy.ndarray): 
            Time array.
        results_dir (str, optional): Default "results_data".
            Directory to save the output file. 
        voltage (bool, optional): Default False.
            If True, uses voltage-related headers. 

    Raises:
        ValueError: If the input arrays (t, x, y) do not have the same length.
    
    Prints:
        Saving confirmation. "Data successfully saved to: <output_file>"
    """

    # Ensure the directory exists.
    os.makedirs(results_dir, exist_ok=True)

    # Validate input array lengths.
    if not (len(t) == len(x) == len(y)):
        raise ValueError("All input arrays (t,x,y) must have the same length.")

    # Build the file path.
    output_file = os.path.join(results_dir, f"{file_name}.txt")

    # Determine header based on voltage flag.
    if voltage:
        header = "displacement \tvoltage \ttime"
    else:
        header = "displacement \tforce \ttime"

    # Combine the data into a single array
    data = np.column_stack((x, y, t))
    # Save the data to a text file
    np.savetxt(output_file, data, delimiter='\t', header=header, comments='')
    print(f"Data saved to: {output_file}")


def save_image(fig, file_name, directory="results_plots", formats=("pdf","eps"), dpi=300):
    """
    Save the given Matplotlib figure in multiple formats.

    Args:
        fig (matplotlib.figure.Figure): 
            The Matplotlib figure to save.
        file_name (str): 
            The base name of the file (without extension).
        directory (str): Default 300
            The directory where the files will be saved.
        formats (tuple): Default ("pdf", "eps").  
            The formats to save the figure.
        dpi (int): Default 300.
            The resolution of the saved images in dots per inch (DPI).
            
    Prints:
        Saving confirmation. "Figure saved to: <output_file>"
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Save the figure in all specified formats
    for fmt in formats:
        full_path = os.path.join(directory, f"{file_name}.{fmt}")
        fig.savefig(full_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to: {full_path}")


def read_parameter_set(csv_file, row_index):
    """
    Reads a specific row from a CSV file and converts it into a parameter dictionary.

    Args:
        csv_file (str): 
            Path to the CSV file.
        row_index (int): 
            The index of the row to extract (0-based index).

    Returns:
        params (dict): 
            A dictionary containing the parameter set from the specified row.
    """
    # Read CSV file as DataFrame.
    df = pd.read_csv(csv_file)
    # Check if the requested row exists.
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} is out of bounds. File has {len(df)} rows.")
    # Extract the row and convert it to a dictionary.
    params = df.iloc[row_index].to_dict()
    return params

def get_txt_files_from_folder(folder_path: str) -> Tuple[List[str], List[str]]:
    """
    Access a folder and retrieve the names and full paths of all .txt files.

    Args:
		folder_path : str
			Relative or absolute path to the folder containing the .txt files.

    Returns:
		Tuple : [List[str], List[str]]
			A tuple containing:
			- A list of filenames (without extension) to be used for saving 
   			plots.
	        - A list of full file paths for reading the files.
    """
    all_files = os.listdir(folder_path)
    txt_files = [f for f in all_files if f.endswith('.txt')]
    full_paths = [os.path.join(folder_path, f) for f in txt_files]
    file_names = [os.path.splitext(f)[0] for f in txt_files]
    return file_names, full_paths
# -----
# Fin :)