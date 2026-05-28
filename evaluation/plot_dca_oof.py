import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import glob
import sys

# Add parent directory to path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our modular DCA utility
from evaluation.dca import plot_dca, calculate_net_benefit

def generate_paper_figures(config_path, n_bootstrap=100):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    RESULTS_DIR = config['resultsdir']
    OOF_DIR = os.path.join(RESULTS_DIR, 'oof_predictions_YYYYMMDD')
    OUTPUT_DIR = os.path.join(RESULTS_DIR, 'paper_figures_oof', 'dca')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all OOF prediction files
    prediction_files = glob.glob(os.path.join(OOF_DIR, 'oof_predictions_*.csv'))
    
    if not prediction_files:
        print(f"No prediction files found in {OOF_DIR}. Have you run the generation scripts?")
        return

    # Standard thresholds for clinical utility 
    thresholds = np.linspace(0.0, 0.5, 500)

    # POPULATION PREVALENCE ESTIMATES
    POPULATION_PREVALENCE = {
        'af': 0.04, # https://pmc.ncbi.nlm.nih.gov/articles/PMC8099514/
        'aki': 0.18, # https://pmc.ncbi.nlm.nih.gov/articles/PMC10709241/
        'ami': 0.004, # https://pmc.ncbi.nlm.nih.gov/articles/PMC9960189/
        'delirium': 0.07, # https://www.bjanaesthesia.org/article/S0007-0912(26)00057-7/fulltext
        'stroke': 0.12, # https://pubmed.ncbi.nlm.nih.gov/39723887/
        'ssi': 0.11 # https://www.sciencedirect.com/science/article/pii/S1743919121002715
    }

    for file_path in prediction_files:
        filename = os.path.basename(file_path)
        parts = filename.replace('oof_predictions_', '').replace('.csv', '').split('_')
        comp = parts[0]
        dataset = "_".join(parts[1:])

        print(f"\n--- Processing DCA for {comp} ({dataset}) ---")
        df_oof = pd.read_csv(file_path)
        
        # --- LOOK FOR TRANSFER PREDICTIONS ---
        ds_variants = [dataset]
        if dataset in ['infl_1', 'infl_1+2', 'prot_all']:
            ds_variants.append(f"prot_{dataset}")
            if dataset == 'prot_all':
                ds_variants.append('prot_prot_all')

        transfer_patterns = []
        for dsv in ds_variants:
            transfer_patterns.extend([
                os.path.join(RESULTS_DIR, 'predictions', f'data_{dsv}_with_predictions_ae_{comp}_newcontrols_newprepro.csv'),
                os.path.join(RESULTS_DIR, f'data_{dsv}_with_predictions_ae_{comp}_newcontrols_newprepro.csv'),
                os.path.join(RESULTS_DIR, f'data_{dsv}_with_predictions_{comp}_ratio_10.csv')
            ])
        
        df_transfer = None
        for p in transfer_patterns:
            if os.path.exists(p):
                print(f"  Found transfer predictions: {os.path.basename(p)}")
                df_t = pd.read_csv(p)
                # find the prob column
                prob_col = [c for c in df_t.columns if 'prob' in c and comp in c]
                if prob_col:
                    df_transfer = df_t[['eid', prob_col[0]]].rename(columns={prob_col[0]: 'prob_transfer'})
                    break

        # --- ALIGNMENT (Inner Merge) ---
        if df_transfer is not None:
            df = pd.merge(df_oof, df_transfer, on='eid', how='inner')
            print(f"  Aligned OOF and Transfer data. N={len(df)}")
        else:
            df = df_oof
            print(f"  No transfer predictions found for {comp} {dataset}. Plotting Baseline vs Direct only.")

        y_true = df['case'].values
        prevalence = POPULATION_PREVALENCE.get(comp, np.mean(y_true))
        
        models_results = {
            'Clinical Baseline': (df['prob_baseline'].values, y_true),
            'Direct Learning (Omics)': (df['prob_omics'].values, y_true)
        }
        
        if df_transfer is not None:
            models_results['Transfer Learning (Omics)'] = (df['prob_transfer'].values, y_true)
        
        # Calculate Limits
        all_model_nbs = []
        zero_crossings = []
        for label, data in models_results.items():
            p, y = data
            nb_vals = calculate_net_benefit(y, p, thresholds, target_prevalence=prevalence)
            all_model_nbs.extend(nb_vals)
            positive_indices = np.where(nb_vals >= 0)[0]
            if len(positive_indices) > 0:
                zero_crossings.append(thresholds[positive_indices[-1]])
        
        combined_nb = all_model_nbs + [0]
        y_lims = (min(combined_nb) - 0.05, max(combined_nb) + 0.05)
        x_max = min(max(zero_crossings) + 0.05 if zero_crossings else 0.5, 0.5)

        # Output Plot
        output_plot = os.path.join(OUTPUT_DIR, f'dca_oof_combined_{comp}_{dataset}.png')
        plot_dca(
            models_results,
            thresholds=thresholds,
            target_prevalence=prevalence,
            title=f"Decision Curve Analysis: {comp.upper()} ({dataset.upper()})\n(Aligned Comparison, Prevalence: {prevalence:.2%})",
            output_path=output_plot,
            show=False,
            x_limits=(0, x_max),
            y_limits=y_lims,
            n_bootstrap=n_bootstrap
        )
        print(f"Saved plot to: {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config.yaml")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of bootstrap iterations for 95% CI")
    args = parser.parse_args()
    generate_paper_figures(args.config, n_bootstrap=args.bootstrap)
