import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from imblearn.ensemble import BalancedBaggingClassifier
import argparse
import yaml
import sys

# Add parent directory to path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom modules from root
from NeuralNetClasses import AutoencoderTransformerProt
from Classifiers import get_classifier_and_cv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use

def generate_oof_predictions(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    DATA_DIR = config['datadir']
    RESULTS_DIR = config['resultsdir']
    
    # Load Data
    data_cc = pd.read_csv(f'{DATA_DIR}/all_cc_me_cp.csv.gz')
    data_prot = pd.read_csv(f'{DATA_DIR}/proteomics_all.csv.gz')
    
    # proteomics has a 'comp' column, rename clinical one
    data_cc.rename(columns={'comp': 'comp_orig'}, inplace=True)
    
    # Merge
    data_all = pd.merge(data_cc, data_prot, on='eid', how='inner')
    
    # Filter clinical criteria
    data_all = data_all[data_all['admimeth_uni'].isin(['elective', 'emergency'])]
    data_all['opcat'] = data_all['opcat'].apply(lambda x: 'complex' if x.endswith('com') else '0')
    
    # Inflammation panels
    olink_key_inflammation = pd.read_csv(f'{DATA_DIR}/olink_explore_inflammation.csv', header=None)
    olink_key_inflammation_ii = pd.read_csv(f'{DATA_DIR}/olink_explore_inflammation_ii.csv', header=None)

    columns_inflammation = [col.lower() for col in olink_key_inflammation[0] if col.lower() in data_prot.columns]
    columns_inflammation_ii = [col.lower() for col in olink_key_inflammation_ii[0] if col.lower() in data_prot.columns]
    
    columns_base = ['age_surgery', 'sex', 'score', 'admimeth_uni', 'opcat']
    columns_prot_all = [col for col in data_prot.columns if col not in ['eid', 'spectrometer']]
    
    # Prepare Output Dir
    OOF_DIR = os.path.join(RESULTS_DIR, 'oof_predictions_YYYYMMDD')
    os.makedirs(OOF_DIR, exist_ok=True)

    complications = ['af', 'aki', 'ami', 'delirium', 'stroke', 'ssi']
    dataset_list = ['infl_1', 'infl_1+2', 'prot_all']

    for cn in complications:
        print(f"\n--- Processing Complication: {cn} ---")
        age_limit = 60 if cn in ['delirium', 'stroke'] else 18
        
        # Filter for complication and age
        data_comp = data_all[data_all['comp_orig'] == cn].copy()
        
        if cn == 'ami':
            if 'tretspef_uni' in data_comp.columns:
                data_comp = data_comp[~data_comp['tretspef_uni'].isin(["Cardiology", "Cardiac surgery", "Cardiothoracic surgery"])]
            
        data_comp = data_comp[data_comp['age_surgery'] >= age_limit]
        
        for ds_type in dataset_list:
            print(f"  Dataset Type: {ds_type}")
            
            if ds_type == 'infl_1':
                columns_now = columns_inflammation
            elif ds_type == 'infl_1+2':
                columns_now = columns_inflammation + columns_inflammation_ii
            elif ds_type == 'prot_all':
                columns_now = columns_prot_all
                
            # Drop excessive missingness for this subset
            data_now = data_comp[columns_base + columns_now + ['eid', 'case']].copy()
            data_now = data_now.dropna(thresh=0.8 * (len(columns_base) + len(columns_now)), axis=0)
            
            if data_now.empty:
                print(f"    Skipping {ds_type} due to no data.")
                continue

            X = data_now.drop(['eid', 'case'], axis=1)
            y = data_now['case']
            eids = data_now['eid']
            
            print(f"    Sample size: {len(y)} (Cases: {y.sum()})")

            # --- PREPROCESSING SETUP ---
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant')),
                ('encoder', OneHotEncoder(handle_unknown='error', drop='if_binary'))
            ])
            def binarize_cci(X):
                return (X >= 2).astype(int)
            zero_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('binarizer', FunctionTransformer(binarize_cci, validate=False))
            ])

            # --- 1. CLINICAL BASELINE PIPELINE ---
            clinical_prepro = ColumnTransformer([
                ('num', num_pipeline, ['age_surgery']),
                ('cat', cat_pipeline, ['sex', 'admimeth_uni', 'opcat']),
                ('zero', zero_pipeline, ['score'])
            ], remainder='drop')

            # --- 2. CLINICAL + PROTEOMICS (AE) PIPELINE ---
            dropout_trial = False if ds_type == 'infl_1' else True
            ae_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('ranknorm', QuantileTransformer(output_distribution='normal', n_quantiles=len(y), subsample=1e5, random_state=42)),
                ('clipper', FunctionTransformer(lambda x: np.clip(x, -10, 10))),
                ('autoencoder', AutoencoderTransformerProt(encoding_dim=16, epochs=20, dropout=dropout_trial))
            ])
            
            combined_prepro = ColumnTransformer([
                ('num', num_pipeline, ['age_surgery']),
                ('cat', cat_pipeline, ['sex', 'admimeth_uni', 'opcat']),
                ('zero', zero_pipeline, ['score']),
                ('ae', ae_pipeline, columns_now)
            ], remainder='drop')

            # --- MODEL & CV ---
            _, model_obj, cv_raw = get_classifier_and_cv(y.sum())
            
            # For cross_val_predict to work, we need a simple partition (not repeated)
            from sklearn.model_selection import StratifiedKFold
            cv_partition = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # --- GENERATE OOF PREDICTIONS ---
            print("    Generating OOF predictions for Clinical Baseline...")
            pipeline_baseline = make_pipeline(clinical_prepro, model_obj)
            probs_baseline = cross_val_predict(pipeline_baseline, X, y, cv=cv_partition, method='predict_proba', n_jobs=60)[:, 1]
            
            print("    Generating OOF predictions for Clinical + Proteomics (AE)...")
            pipeline_omics = make_pipeline(combined_prepro, model_obj)
            probs_omics = cross_val_predict(pipeline_omics, X, y, cv=cv_partition, method='predict_proba', n_jobs=60)[:, 1]
            
            # --- SAVE ---
            results_oof = pd.DataFrame({
                'eid': eids,
                'case': y,
                'prob_baseline': probs_baseline,
                'prob_omics': probs_omics
            })
            
            output_file = os.path.join(OOF_DIR, f'oof_predictions_{cn}_{ds_type}.csv')
            results_oof.to_csv(output_file, index=False)
            print(f"    Saved OOF predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config.yaml")
    args = parser.parse_args()
    generate_oof_predictions(args.config)
