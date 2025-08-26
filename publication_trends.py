"""
Publication trends of different topics over the years. 

In this file, the number of publications over the years for the Bouc-Wen model 
and the SINDy algorithm is plotted, based on different databases.  

Coder: 
    Michael Heredia Pérez
    mherediap@unal.edu.co
    Universidad Nacional de Colombia 
    170001 Manizales, Colombia

Date:
    May 2025
    
Notes:
    The following color schemes are followed:
    - For my mater thesis: both solid lines, first blue and then red. 
    - For the Springer paper: both black lines, first dotted and then solid. 
    This must be set manually in the code. 
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from functions import save_image
from functions import get_txt_files_from_folder

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

def plot_publication_trend(data_series, title, filename, directory="publication_trends_plots"):
    """
    Plot publication trends over time for multiple data series.
    
    Creates a line plot showing publication counts by year for one or more 
    datasets. Saves the plot to file and displays it.
    
    Args:
		data_series : list of tuples
			List of (data, label, linestyle) tuples where:
			- data: DataFrame with 'Year' and 'Count' columns
			- label: str, legend label for the series
			- linestyle: str, matplotlib linestyle ('-', '--', ':', etc.)
		title : str
			Plot title
		filename : str
			Output filename (without extension)
		directory : str, optional
			Output directory path (default: "springer_plots")
		
    Returns:
		None
			Displays plot and saves to file
    """
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each data series
    for data, label, linestyle, color in data_series:
        ax.plot(data["Year"], data["Count"], 
                label=label, marker="o", linewidth=2.5, 
                color=color, linestyle=linestyle)
    
    # Set axis labels
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    
    # Configure tick parameters
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels
    ax.tick_params(axis="y")
    
    # Apply tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Add legend and grid for multiple series
    if len(data_series) > 1:
        ax.legend()
    
    # Add grid and title
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title(title)
    
    # Save figure to file
    save_image(fig, filename, directory=directory)
    
    # Display and close figure
    plt.show()
    plt.close(fig)
    

# -----
# Load data from the CSV files.

# Call the input folder.
input_folder = "publication_registry"

# Get the names and paths.
file_names, file_paths = get_txt_files_from_folder(input_folder)

# Sort them in alphabetical order.
file_names.sort()
file_paths.sort()

# For the Bouc-Wen model.
elsevier_bw_general = pd.read_csv(file_paths[0], delimiter=",")  
elsevier_bw_spcific = pd.read_csv(file_paths[1], delimiter=",")  
springer_bw_general = pd.read_csv(file_paths[3], delimiter=",")  
springer_bw_spcific = pd.read_csv(file_paths[4], delimiter=",")  

# For the SINDy algorithm
elsevier_sindy = pd.read_csv(file_paths[2], delimiter=",")  

# -----
# Get the total summation for Bouc-Wen model.

df_general_total = pd.merge(
    elsevier_bw_general, springer_bw_general, on="Year", how="outer", suffixes=('_elsevier', '_springer')
).fillna(0)
df_general_total["Count"] = df_general_total["Count_elsevier"] + df_general_total["Count_springer"]

df_spcific_total = pd.merge(
    elsevier_bw_spcific, springer_bw_spcific, on="Year", how="outer", suffixes=('_elsevier', '_springer')
).fillna(0)
df_spcific_total["Count"] = df_spcific_total["Count_elsevier"] + df_spcific_total["Count_springer"]

# -----
# Plot the data, individual and merged figures.

# Plot 1: General search for Bouc-Wen
plot_publication_trend(
    data_series=[
        (elsevier_bw_general, "Science Direct", "solid", "b"),
        (springer_bw_general, "Springer Nature Link", "solid", "r")
    ],
    title="Overview of the publication trends for general search on databases",
    filename="publication_trends_general_bw"
)

# Plot 2: Specific search for Bouc-Wen
plot_publication_trend(
    data_series=[
        (elsevier_bw_spcific, "Science Direct", "solid", "b"),
        (springer_bw_spcific, "Springer Nature Link", "solid", "r")
    ],
    title="Overview of the publication trends for specific search on databases",
    filename="publication_trends_specific_bw"
)

# Plot 3: General search for SINDy
plot_publication_trend(
    data_series=[
        (elsevier_sindy, "Science Direct", "solid", "b")
    ],
    title="Overview of the publication trends for general search on ScienceDirect",
    filename="publication_trends_sindy"
)

# Plot 4: total number of publications for the general search for Bouc-Wen
plot_publication_trend(
    data_series=[
        (df_general_total, " ", "solid", "b")
    ],
    title="Overview of the publication trends for general search on databases",
    filename="publication_trends_general_bw_total"
)

# Plot 5: total number of publications for the specific search for Bouc-Wen
plot_publication_trend(
    data_series=[
        (df_spcific_total, " ", "solid", "b")
    ],
    title="Overview of the publication trends for specific search on databases",
    filename="publication_trends_specific_bw_total"
)

# Fin :)