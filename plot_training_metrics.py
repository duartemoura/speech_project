import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

def parse_log_file(log_file):
    """Parse a training log file and extract metrics."""
    epochs = []
    train_losses = []
    valid_losses = []
    valid_errors = []
    learning_rates = []
    test_loss = None
    test_error = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match the pattern in the log file for training/validation
            match = re.match(r'Epoch: (\d+), lr: ([\d.e+-]+) - train loss: ([\d.e+-]+) - valid loss: ([\d.e+-]+), valid error: ([\d.e+-]+)', line)
            if match:
                epoch, lr, train_loss, valid_loss, valid_error = match.groups()
                epochs.append(int(epoch))
                learning_rates.append(float(lr))
                train_losses.append(float(train_loss))
                valid_losses.append(float(valid_loss))
                valid_errors.append(float(valid_error))
            
            # Match the pattern for test metrics (single value at the end)
            test_match = re.match(r'Epoch loaded: \d+ - test loss: ([\d.e+-]+), test error: ([\d.e+-]+)', line)
            if test_match:
                test_loss, test_error = test_match.groups()
                test_loss = float(test_loss)
                test_error = float(test_error)
    
    return {
        'epochs': epochs,
        'learning_rates': learning_rates,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_errors': valid_errors,
        'test_loss': test_loss,
        'test_error': test_error
    }

def moving_average(data, window_size=3):
    """Calculate moving average for smoothing curves."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_experiment_summary(metrics_dict, exp_name, save_dir='plots'):
    """Create a summary plot for a single experiment."""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    window = 3
    epochs_ma = metrics_dict['epochs'][window - 1:]

    # Plot: Losses
    axes[0].plot(epochs_ma, moving_average(metrics_dict['train_losses'], window), label='Train Loss (MA)', color='blue')
    axes[0].plot(epochs_ma, moving_average(metrics_dict['valid_losses'], window), label='Valid Loss (MA)', color='red')
    axes[0].set_title('Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot: Validation Error
    axes[1].plot(epochs_ma, moving_average(metrics_dict['valid_errors'], window), label='Valid Error (MA)', color='green')
    axes[1].set_title('Validation Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error Rate')
    axes[1].legend()
    axes[1].grid(True)

    # Summary Table
    final_metrics = {
        'Final Train Loss': f"{metrics_dict['train_losses'][-1]:.4f}",
        'Final Valid Loss': f"{metrics_dict['valid_losses'][-1]:.4f}",
        'Final Valid Error': f"{metrics_dict['valid_errors'][-1]:.4f}",
        'Best Valid Error': f"{min(metrics_dict['valid_errors']):.4f}",
        'Best Valid Loss': f"{min(metrics_dict['valid_losses']):.4f}"
    }
    
    # Add test metrics if available
    if metrics_dict['test_loss'] is not None:
        final_metrics['Test Loss'] = f"{metrics_dict['test_loss']:.4f}"
        final_metrics['Test Error'] = f"{metrics_dict['test_error']:.4f}"
    
    table_fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    cell_text = [[k, v] for k, v in final_metrics.items()]
    ax.table(cellText=cell_text, colWidths=[0.5, 0.3], cellLoc='left', loc='center')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{exp_name}_summary.png'), dpi=300, bbox_inches='tight')
    table_fig.savefig(os.path.join(save_dir, f'{exp_name}_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_comparison_summary(experiment_paths, save_dir='plots'):
    """Create comparison plots for multiple experiments."""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Experiment Comparison', fontsize=16)

    all_metrics = {}
    for exp_path in experiment_paths:
        exp_name = os.path.basename(os.path.dirname(exp_path))
        all_metrics[exp_name] = parse_log_file(exp_path)

    window = 3

    # Plot: Validation Loss Comparison
    for exp_name, metrics in all_metrics.items():
        epochs_ma = metrics['epochs'][window-1:]
        valid_loss_ma = moving_average(metrics['valid_losses'], window)
        axes[0,0].plot(epochs_ma, valid_loss_ma, label=exp_name)
    axes[0,0].set_title('Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # Plot: Validation Error Comparison
    for exp_name, metrics in all_metrics.items():
        epochs_ma = metrics['epochs'][window-1:]
        valid_error_ma = moving_average(metrics['valid_errors'], window)
        axes[0,1].plot(epochs_ma, valid_error_ma, label=exp_name)
    axes[0,1].set_title('Validation Error')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Error Rate')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Plot: Test Loss Comparison (Bar Chart)
    test_losses = []
    exp_names = []
    for exp_name, metrics in all_metrics.items():
        if metrics['test_loss'] is not None:
            test_losses.append(metrics['test_loss'])
            exp_names.append(exp_name)
    
    if test_losses:
        bars = axes[1,0].bar(exp_names, test_losses)
        axes[1,0].set_title('Test Loss Comparison')
        axes[1,0].set_xlabel('Experiment')
        axes[1,0].set_ylabel('Test Loss')
        axes[1,0].grid(True)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[1,0].get_xticklabels(), rotation=45, ha='right')

    # Summary Table
    summary_metrics = []
    for exp_name, metrics in all_metrics.items():
        row = [
            exp_name,
            f"{metrics['train_losses'][-1]:.4f}",
            f"{metrics['valid_losses'][-1]:.4f}",
            f"{metrics['valid_errors'][-1]:.4f}",
            f"{min(metrics['valid_errors']):.4f}",
            f"{min(metrics['valid_losses']):.4f}",
        ]
        
        # Add test metrics if available
        if metrics['test_loss'] is not None:
            row.extend([
                f"{metrics['test_loss']:.4f}",
                f"{metrics['test_error']:.4f}"
            ])
        else:
            row.extend(['N/A', 'N/A'])
            
        summary_metrics.append(row)

    columns = [
        'Experiment', 'Final Train Loss', 'Final Valid Loss',
        'Final Valid Error', 'Best Valid Error', 'Best Valid Loss',
        'Test Loss', 'Test Error'
    ]

    df = pd.DataFrame(summary_metrics, columns=columns)

    axes[1,1].axis('off')
    axes[1,1].table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    df.to_csv(os.path.join(save_dir, 'experiment_metrics.csv'), index=False)


def main():
    # Example usage
    results_dir = 'results/speaker_id'
    
    # Get all experiment directories
    experiment_paths = []
    for exp_dir in os.listdir(results_dir):
        log_path = os.path.join(results_dir, exp_dir, 'train_log.txt')
        if os.path.exists(log_path):
            experiment_paths.append(log_path)
            # Create individual summary plots
            metrics = parse_log_file(log_path)
            plot_experiment_summary(metrics, exp_dir, save_dir='plots/individual')
    
    # Create comparison plots
    if experiment_paths:
        plot_comparison_summary(experiment_paths, save_dir='plots/comparison')

if __name__ == '__main__':
    main() 