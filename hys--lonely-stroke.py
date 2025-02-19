"""
This code creates a graph for the complex behavior of hysteresis with 
lonely-stroke.

Coder:
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia
    170001 Manizales, Colombia
    January 2025

References:
    Konda, R., & Zhang, J. (2022). Hysteresis with lonely stroke in artificial 
    muscles: Characterization, modeling, and inverse compensation. Mechanical 
    Systems and Signal Processing, 164, 108240.

Commentary:
    Matlab code and data shared by Konda, R., & Zhang, J.    
    
Date:
    January 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from functions import store_data, plot_hysteresis, save_image


def input_disc(level):
    """
    Creates a matrix of input pairs between -1 and 1 for generating state 
    sequences.
    
    This function generates a triangular grid of points. For example, with 
    level=3: It creates points: [-1,-1], [-1,0], [-1,1], [0,0], [0,1], [1,1]
    The number of pairs generated is level*(level+1)/2.
    
    Args:
        level (int): 
            Number of points to create in each dimension. Determines the 
            resolution of the grid.
    
    Returns:
        pair (numpy.ndarray): 
            A matrix of shape (level*(level+1)/2, 2) containing pairs of values 
            where each row is [alpha, beta] used as thresholds in cell_state.
    
    Example:
        >>> input_disc(3)
        >>> array([[-1., -1.],
        >>>        [ 0., -1.],
        >>>        [ 0.,  0.],
        >>>        [ 1., -1.],
        >>>        [ 1.,  0.],
        >>>        [ 1.,  1.]])
    """
    
    umax = 1
    umin = -1
    
    # Calculate step size to create evenly spaced points.
    step        = (umax - umin) / (level - 1)
    input_range = np.arange(umin, umax + step, step)
    
    # Initialize output matrix.
    pair = np.zeros((level * (level + 1) // 2, 2))
    
    # Generate pairs in triangular pattern.
    k = 0
    for i in range(len(input_range)):
        for j in range(i + 1):
            pair[k] = [input_range[i], input_range[j]]
            k += 1
            
    return pair

def cell_state(pair, input_data):
    """
    Implements a hysteresis function that creates a state sequence based on 
    input values and threshold pairs.
    
    For each input value, the state is determined by:
    - If input >= alpha: state becomes 1.
    - If input <= beta: state becomes -1.
    - If beta < input < alpha: keep previous state (hysteresis).
    
    Args:
        pair (numpy.ndarray): 
            Array of [alpha, beta] thresholds where alpha >= beta.
        input_data (numpy.ndarray): 
            Array of input values to process.
    
    Returns:
        (numpy.ndarray): 
            Array of same length as input_data containing states (-1 or 1).
    
    Example:
        >>> cell_state([0.5, -0.5], [0, 0.6, 0.3, -0.6])
        >>> array([-1.,  1.,  1., -1.])
    """
    
    # Initialize all states to -1.
    cell  = np.full(len(input_data), -1.0)
    alpha = pair[0]   # Upper threshold.
    beta  = pair[1]   # Lower threshold.
    
    # Handle first element specially.
    if input_data[0] > alpha:
        cell[0] = 1
    elif input_data[0] <= beta:
        cell[0] = -1
    
    # Process remaining elements with hysteresis.
    for i in range(1, len(input_data)):
        if input_data[i] >= alpha:
            cell[i] = 1
        elif input_data[i] <= beta:
            cell[i] = -1
        else:
            # Maintain previous state if input is between thresholds.
            cell[i] = cell[i-1]
    
    return cell

def main():
    """
    Main function that processes input data, generates states, and creates 
    visualization.
    
    The function performs these steps:
    1. Loads input data from MATLAB file.
    2. Generates additional random data points.
    3. Normalizes the combined data.
    4. Creates multiple state sequences using different threshold pairs.
    5. Combines states with random weights to produce output.
    6. Visualizes the results.
    
    Requirements:
        - "lonely-stroke.mat" file in the ./data/ directory.
        - NumPy, SciPy, and Matplotlib installed.
    """

    # Load data from MATLAB file.
    mat_data = loadmat("data/lonely-stroke.mat")
    L2       = mat_data["L2"].flatten()  # Convert to 1D array.
    
    # Parameters for data processing.
    scale  = 1
    offset = 0.2
    
    # Extract specific portion of input data.
    L_dash = L2[156:518]  # Python uses 0-based indexing.
    L2     = L_dash
    
    # Generate random points and interpolate between them.
    ran = np.round(9 * np.random.rand(9))
    L3  = []
    
    # Create points between consecutive random numbers.
    for i in range(1, len(ran)):
        least = ran[i-1]
        most  = ran[i]
        lll   = []
        
        # Interpolate with 0.1 step size.
        if least <= most:
            in_val = least
            while in_val <= most:
                in_val += 0.1
                lll.append(in_val)
        else:
            in_val = least
            while in_val >= most:
                in_val -= 0.1
                lll.append(in_val)
                
        L3.extend(lll)
    
    # Convert to numpy array and combine data.
    L3 = np.array(L3)
    L  = np.concatenate([L2, L3])
    
    # Normalize input data.
    maxi       = 10
    maxinput   = 0.7
    input_data = (L - maxi/2) / maxi + offset
    input_data = input_data / maxinput
    
    # Generate state sequences.
    level = 300
    pair  = input_disc(level)  # Generate threshold pairs.
    
    # Initialize state matrix and random weights.
    muopt = np.random.rand(level*(level+1)//2 + 1)
    state = np.zeros((level*(level+1)//2 + 1, len(L)))
    state[-1, :] = 1  # Set bias row to 1.
    
    # Generate state sequences for each threshold pair.
    for i in range(level*(level+1)//2):
        state[i, :] = cell_state(pair[i, :], input_data)
    
    # Calculate weighted sum of states.
    output = np.dot(muopt, state)
    
    # Normalize output.
    maxoutput = 2.265878742287655e+04
    minoutput = -0.835550131429351
    output    = output / maxoutput
    output    = output - minoutput
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.plot(L[:92], output[:92], "r-", label="Original Data")
    plt.plot(L[92:], output[92:], "b-", label="Generated Data")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlim([0, 9])
    plt.gcf().set_size_inches(10, 8)
    plt.legend()
    plt.title("Data Visualization with State Sequences")
    plt.show()

    return L, output


if __name__ == "__main__":
    
    # Run the simulation.
    x, z = main()
    # Plot the results.
    fig = plot_hysteresis(x, z)
    
    # Store the image and the data.
    # save_image(fig, "hys--lonely-stroke")
    # t = np.zeros(z.shape) # build a mockup of the time
    # store_data("hyst--lonely-stroke", x, z, t)
    
# Fin :)