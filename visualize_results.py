import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import sys

def plot_results(results_dir):
    # Find laptimes csv
    laptimes_files = glob.glob(os.path.join(results_dir, "*_laptimes.csv"))
    if not laptimes_files:
        print(f"No laptimes file found in {results_dir}")
        return

    laptimes_path = laptimes_files[0]
    print(f"Reading {laptimes_path}")
    df = pd.read_csv(laptimes_path)

    # The CSV likely has columns: lap, driver1, driver2, ...
    # We need to melt it for seaborn: lap, driver, time
    
    # Check columns
    print("Columns:", df.columns)
    
    # Assuming 'lap' is a column and others are drivers
    if 'lap' not in df.columns:
        print("Error: 'lap' column not found")
        return

    # Melt the dataframe
    df_melted = df.melt(id_vars=['lap'], var_name='driver', value_name='lap_time')
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melted, x='lap', y='lap_time', hue='driver')
    plt.title("Lap Times per Driver")
    plt.xlabel("Lap")
    plt.ylabel("Lap Time (s)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = "lap_times.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Use the specific directory from the user's session
    results_dir = "logs/RaceStrategy-v2/results/2025-11-19_10-48_10000b"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        
    plot_results(results_dir)
