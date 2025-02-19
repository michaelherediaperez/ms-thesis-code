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


def plot_hysteresis(x, f_r, labels=False):
    """
    Plot the hysteresis with or without labels. Theonly lables considered will 
    be "RF" and "D" for "Restoring force" and "Displacement", respectively. 

    Args:
        x (numpy.ndarray or list or tuple): 
            Displacement.
        f_r (numpy.ndarray or list or tuple):
            Restoring force. 
        labels (bool, optional): Defaults to True.
            Wether or not to plot labels. 
    
    Returns:
        fig (matplotlib.figure.Figure)
    """
    # Set the figure size and put the lables if asked for.
    if labels:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set(xlabel="D", ylabel="RF")
    else:
        # A square plot.
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the hysteresis.
    ax.plot(x, f_r, "b", linewidth=1)
    # Plot the initial point.
    ax.plot(0, 0, "b*")
    
    # Configure the plot aesthetic (no numbers and grid).
    ax.set_xticklabels([])          
    ax.set_yticklabels([])
    plt.grid(True, alpha=0.3)
    
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
    # Extract the data.
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
    np.savetxt(output_file, data, delimiter="\t", header=header, comments="")
    print(f"Data saved to: {output_file}")


def save_image(
    fig, file_name, directory="results_plots", formats=("pdf", "svg"), dpi=300
    ):
    """
    Save the given Matplotlib figure in multiple formats.

    Args:
        fig (matplotlib.figure.Figure): 
            The Matplotlib figure to save.
        file_name (str): 
            The base name of the file (without extension).
        directory (str): Default 300
            The directory where the files will be saved.
        formats (tuple): Default ("pdf", "svg").  
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
    Reads a specific row from a CSV file and converts it into a parameter 
    dictionary.

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
        raise IndexError(f"Row index {row_index} is out of bounds. File has \
            {len(df)} rows.")
    # Extract the row and convert it to a dictionary.
    params = df.iloc[row_index].to_dict()
    return params


# Fin :)