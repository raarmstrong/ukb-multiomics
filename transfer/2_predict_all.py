import os
import pandas as pd
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model  

import argparse
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from Preprocessors import binarize_cci

# Ensure TensorFlow doesn't flood the logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loaded libraries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default='../../config.yaml', help="Path to config.yaml")
    config_arg = parser.parse_args()

    config_path = config_arg.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    # Load the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

else:
# if running interactive
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

DATA_DIR = config['datadir']
RESULTS_DIR = config['resultsdir']
MODEL_DIR = config['modeldir']

print("Configuration loaded.")

# --- Helper Functions for Reconstruction ---
from utils import load_metab_pipeline, reconstruct_pipeline
from utils import predict_external_probs, sort_external_data

# --- Example Usage ---

# 1. Load external dataset (must be pre-merged like 'data_all' training script)
# Ensure columns renamed (e.g. 'cci_at_diag' -> 'score') to match training

metab_data_path = os.path.join(DATA_DIR, 'metabolomics_reduced.csv.zip')
prot_data_path = os.path.join(DATA_DIR, 'proteomics_all.csv.gz')
external_cohort_data_path = os.path.join(DATA_DIR, 'all_cc_me_cp.csv.gz')

print('Loading external cohort data...')
external_cohort_data = pd.read_csv(external_cohort_data_path, usecols=['eid', 'admimeth_uni', 'opcat', 'age_surgery', 'sex', 'score', 'comp', 'case'])

input_dict = {
    'metab': (metab_data_path, None),
    'multi_infl_1': (metab_data_path, prot_data_path),
    'multi_infl_1+2': (metab_data_path, prot_data_path),
    'multi_prot_all': (metab_data_path, prot_data_path),
    'prot_infl_1': (None, prot_data_path),
    'prot_infl_1+2': (None, prot_data_path),
    'prot_prot_all': (None, prot_data_path)
}

i=0

print('Starting predictions for all datasets...')

for dataset in input_dict.keys():
    i+=1
    print(f'Processing dataset {i}/{len(input_dict)}: {dataset}')

    data_all = sort_external_data(dataset, *input_dict[dataset], external_cohort_data, DATA_DIR)

    for cn in ['af', 'aki', 'ami', 'delirium', 'stroke', 'ssi']:
        print(f'Predicting from {dataset} for complication: {cn}')

        # 2. Run the predictor
        TRAINING_DATE = 'YYYYMMDD'
        model_file = os.path.join(MODEL_DIR, f'{TRAINING_DATE}/final_model_{cn}_en_{dataset}_ae_16_False_nonsurg_newcontrols_newprepro.joblib')
        payload_file = os.path.join(MODEL_DIR, f'{TRAINING_DATE}/final_model_payload_{cn}_en_{dataset}_ae_16_False_nonsurg_newcontrols_newprepro.joblib')
        
        comp_col = 'comp_orig' if 'comp_orig' in data_all.columns else 'comp'

        model_data = data_all[data_all[comp_col] == cn].drop(comp_col, axis=1).copy()

        if cn in ['stroke', 'delirium']:
            model_data = model_data[model_data['age_surgery'] >= 60]

        try:
            probabilities = predict_external_probs(model_file, payload_file, model_data)
            
            # 3. Save results
            model_data[f'predicted_prob_{cn}'] = probabilities
            print(model_data['predicted_prob_' + cn].describe())
            cols_to_save = ['eid', 'case', 'admimeth_uni', 'opcat', f'predicted_prob_{cn}']
            export_data = model_data[cols_to_save]
            export_data.to_csv(f'{RESULTS_DIR}/predictions/data_{dataset}_with_predictions_ae_{cn}_newcontrols_newprepro.csv', index=False)
            print("Predictions saved for dataset/complication:",  dataset, cn)

        except ValueError as e:
            print(e)