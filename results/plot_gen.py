"""
Module used for plotting results
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_and_save_plots(directory_path):
    """
    Function that generates plots from csv files
    """

    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Filter CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    # Loop through each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)

        # Read CSV file into a Pandas DataFrame
        data_frame = pd.read_csv(file_path)

        print(f"Max rollout {csv_file[:len(csv_file)-4]} is: ", np.max(data_frame['Value']))
        print(f"Last {csv_file[:len(csv_file)-4]} of the rollout is: ", \
              data_frame['Value'][len(data_frame['Value']) - 1])
        print('---------------------')
        # Do some processing on the DataFrame (replace this with your processing steps)
        # For example, let's plot a histogram of a column
        sns.set(style="whitegrid")

        label = csv_file[:len(csv_file)-4]
        # Create the line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(y=data_frame['Value'], x=data_frame['Step'], label=label, linewidth=2)

        # Customize the plot
        plt.title("Beautiful Line Plot with Seaborn")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        sns.despine()

        # Set custom colors
        palette = sns.color_palette("husl", 1)  # Choose the number of colors you need
        sns.set_palette(palette)

        plt.savefig(f"{directory_path}/{label}.svg", format="svg")

        # Close the plot to release resources
        plt.close()

def main():
    """
    Main function
    """

    parser = argparse.ArgumentParser(description="Process CSV files and save plots.")
    parser.add_argument("--directory-path", type=str, help="Path to the dir containing CSV files")
    args = parser.parse_args()

    process_and_save_plots(args.directory_path)

if __name__ == "__main__":
    main()
