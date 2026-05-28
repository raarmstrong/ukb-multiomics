import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

def calculate_net_benefit(y_true, y_probs, thresholds, target_prevalence=None):
    """
    Calculate Net Benefit for a model across a range of thresholds.
    """
    n = len(y_true)
    nb_list = []
    
    # Calculate actual prevalence in this sample
    sample_prevalence = np.mean(y_true)
    rho = target_prevalence if target_prevalence is not None else sample_prevalence

    for pt in thresholds:
        if pt >= 1.0:
            nb_list.append(0.0)
            continue
        
        y_pred = (y_probs >= pt).astype(int)
        
        # We use the Sens/Spec method which is robust to case-control sampling
        # if target_prevalence (rho) is provided.
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        nb = (sens * rho) - ((1 - spec) * (1 - rho) * (pt / (1 - pt)))
        nb_list.append(nb)
        
    return np.array(nb_list)

def calculate_net_benefit_bootstrap(y_true, y_probs, thresholds, target_prevalence=None, n_bootstrap=100):
    """
    Calculate bootstrapped Net Benefit confidence intervals.
    """
    boot_nbs = []
    indices = np.arange(len(y_true))
    
    for i in range(n_bootstrap):
        # Resample indices
        boot_idx = resample(indices, replace=True, random_state=i)
        y_true_boot = y_true[boot_idx]
        y_probs_boot = y_probs[boot_idx]
        
        nb = calculate_net_benefit(y_true_boot, y_probs_boot, thresholds, target_prevalence=target_prevalence)
        boot_nbs.append(nb)
        
    boot_nbs = np.array(boot_nbs)
    lower = np.percentile(boot_nbs, 2.5, axis=0)
    upper = np.percentile(boot_nbs, 97.5, axis=0)
    mean_nb = np.percentile(boot_nbs, 50, axis=0)
    
    return mean_nb, lower, upper

def calculate_treat_all_net_benefit(y_true, thresholds, target_prevalence=None):
    """Calculate Net Benefit for 'Treat All' strategy."""
    rho = target_prevalence if target_prevalence is not None else np.mean(y_true)
    # Filter thresholds to avoid division by zero or negative NB that breaks plots
    safe_t = np.where(thresholds < 1.0, thresholds, 0.99)
    return rho - (1 - rho) * (safe_t / (1 - safe_t))

def calculate_treat_none_net_benefit(thresholds):
    """Calculate Net Benefit for 'Treat None' strategy (always 0)."""
    return np.zeros_like(thresholds)

def plot_dca(models_results_dict, thresholds=None, target_prevalence=None, title="Decision Curve Analysis", 
             output_path=None, show=True, x_limits=None, y_limits=None, n_bootstrap=0):
    """
    Plot Decision Curve Analysis comparing multiple models.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 100)
    
    plt.figure(figsize=(10, 7))
    
    first_y_true = None
    all_nb_values = []
    
    # Plot each model
    for label, data in models_results_dict.items():
        y_probs, y_true = data
        if first_y_true is None:
            first_y_true = y_true
            
        if n_bootstrap > 0:
            print(f"  Bootstrapping CIs for {label}...")
            nb, lower, upper = calculate_net_benefit_bootstrap(y_true, y_probs, thresholds, target_prevalence, n_bootstrap)
            line, = plt.plot(thresholds, nb, label=label, linewidth=2)
            plt.fill_between(thresholds, lower, upper, color=line.get_color(), alpha=0.15)
        else:
            nb = calculate_net_benefit(y_true, y_probs, thresholds, target_prevalence=target_prevalence)
            plt.plot(thresholds, nb, label=label, linewidth=2)
            
        all_nb_values.extend(nb)
    
    # Plot Treat All and Treat None
    if first_y_true is not None:
        nb_all = calculate_treat_all_net_benefit(first_y_true, thresholds, target_prevalence=target_prevalence)
        nb_none = calculate_treat_none_net_benefit(thresholds)
        
        plt.plot(thresholds, nb_all, label='Treat All', linestyle='--', color='gray', alpha=0.7)
        plt.plot(thresholds, nb_none, label='Treat None', linestyle='-', color='black', alpha=0.5)
        
        # Set limits
        if x_limits:
            plt.xlim(x_limits)
        
        if y_limits:
            plt.ylim(y_limits)
    
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
