import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_path = "/home/hice1/mgulati30/Final Project/runs/exp_pr/precision_recall.csv"
prec_recall_data = pd.read_csv(results_path)

#filter for epochs
epoch_to_plot = 5
class_to_plot = 0 
data = prec_recall_data[(prec_recall_data['epoch'] == epoch_to_plot) & (prec_recall_data['class'] == class_to_plot)]


if data.empty:
    print(f"No data available for epoch {epoch_to_plot} and class {class_to_plot}")
else:
    # Plot precision vs recall curve for the specified epoch and class
    plt.figure(figsize=(8, 6))
    plt.plot(data['recall'], data['precision'], marker='o', label=f'Epoch {epoch_to_plot}, Class {class_to_plot}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Define the file path where you want to save the plot
    plot_save_path = f"/home/hice1/mgulati30/Final Project/runs/exp_pr/precision_recall_epoch{epoch_to_plot}_class{class_to_plot}.png"
    
    # Save the plot to the specified path
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved at: {plot_save_path}")
    
    # Show the plot
    plt.show()