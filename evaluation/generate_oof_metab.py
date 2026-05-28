import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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
from Preprocessors import PERCENTAGE_FIELDS, find_percentage_indices, metabolomic_transforms
from NeuralNetClasses import AutoencoderTransformer
from Classifiers import get_classifier_and_cv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use

def generate_oof_predictions(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    DATA_DIR = config['datadir']
    RESULTS_DIR = config['resultsdir']
    
    # Load Data
    data_cc = pd.read_csv(f'{DATA_DIR}/all_cc_me_cp.csv.gz')
    data_metab = pd.read_csv(f'{DATA_DIR}/metabolomics_all_clean.csv.gz')
    metab_info = pd.read_csv(f'{DATA_DIR}/metabolomics_info_clean.csv.gz')
    data_metab = pd.merge(data_metab, metab_info[['eid', 'spectrometer']], on='eid')
    
    # Merge
    data_all = pd.merge(data_cc, data_metab, on='eid', how='inner')
    
    # Filter clinical criteria
    data_all = data_all[data_all['admimeth_uni'].isin(['elective', 'emergency'])]
    data_all['opcat'] = data_all['opcat'].apply(lambda x: 'complex' if x.endswith('com') else '0')
    
    columns_base = ['age_surgery', 'sex', 'score', 'admimeth_uni', 'opcat']
    columns_metab = [col for col in data_metab.columns if col not in ['eid', 'spectrometer']]
    
    # Prepare Output Dir
    OOF_DIR = os.path.join(RESULTS_DIR, 'oof_predictions_YYYYMMDD')
    os.makedirs(OOF_DIR, exist_ok=True)

    complications = ['af', 'aki', 'ami', 'delirium', 'stroke', 'ssi']

    for cn in complications:
        print(f"\n--- Processing Complication: {cn} ---")
        age_limit = 60 if cn in ['delirium', 'stroke'] else 18
        
        # Filter for complication and age
        data_comp = data_all[data_all['comp'] == cn].copy()
        
        if cn == 'ami':
            data_comp = data_comp[~data_comp['tretspef_uni'].isin(["Cardiology", "Cardiac surgery", "Cardiothoracic surgery"])]
            
        data_comp = data_comp[data_comp['age_surgery'] >= age_limit]
        
        # Drop excessive missingness
        data_comp = data_comp.dropna(thresh=0.8 * (len(columns_base) + len(columns_metab)), axis=0)
        
        X = data_comp[columns_base + columns_metab + ['spectrometer']]
        y = data_comp['case']
        eids = data_comp['eid']
        
        print(f"Sample size: {len(y)} (Cases: {y.sum()})")

        # --- PREPROCESSING SETUP ---
        percentage_indices = find_percentage_indices(X[columns_metab], PERCENTAGE_FIELDS)
        
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
        clinical_cols = ['age_surgery', 'sex', 'admimeth_uni', 'opcat']
        clinical_cat = ['sex', 'admimeth_uni', 'opcat']
        clinical_num = ['age_surgery']
        
        clinical_prepro = ColumnTransformer([
            ('num', num_pipeline, clinical_num),
            ('cat', cat_pipeline, clinical_cat),
            ('zero', zero_pipeline, ['score'])
        ], remainder='drop')

        # --- 2. CLINICAL + METAB (AE) PIPELINE ---
        metab_transformer = FunctionTransformer(
            lambda X: metabolomic_transforms(X, percentage_indices=percentage_indices),
            validate=False
        )
        ae_pipeline = Pipeline([
            ('metab_preprocess', metab_transformer),
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('clipper', FunctionTransformer(lambda x: np.clip(x, -10, 10))),
            ('autoencoder', AutoencoderTransformer(encoding_dim=16, epochs=20, dropout=False))
        ])
        
        combined_prepro = ColumnTransformer([
            ('num', num_pipeline, ['age_surgery']),
            ('cat', cat_pipeline, ['sex', 'admimeth_uni', 'opcat', 'spectrometer']),
            ('zero', zero_pipeline, ['score']),
            ('ae', ae_pipeline, columns_metab)
        ], remainder='drop')

        # --- MODEL & CV ---
        _, model_obj, cv_raw = get_classifier_and_cv(y.sum())
        
        # For cross_val_predict to work, we need a simple partition (not repeated)
        from sklearn.model_selection import StratifiedKFold
        cv_partition = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # --- GENERATE OOF PREDICTIONS ---
        print("Generating OOF predictions for Clinical Baseline...")
        pipeline_baseline = make_pipeline(clinical_prepro, model_obj)
        probs_baseline = cross_val_predict(pipeline_baseline, X, y, cv=cv_partition, method='predict_proba', n_jobs=60)[:, 1]
        
        print("Generating OOF predictions for Clinical + Metab (AE)...")
        pipeline_omics = make_pipeline(combined_prepro, model_obj)
        probs_omics = cross_val_predict(pipeline_omics, X, y, cv=cv_partition, method='predict_proba', n_jobs=60)[:, 1]
        
        # --- SAVE ---
        results_oof = pd.DataFrame({
            'eid': eids,
            'case': y,
            'prob_baseline': probs_baseline,
            'prob_omics': probs_omics
        })
        
        output_file = os.path.join(OOF_DIR, f'oof_predictions_{cn}_metab.csv')
        results_oof.to_csv(output_file, index=False)
        print(f"Saved OOF predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config.yaml")
    args = parser.parse_args()
    generate_oof_predictions(args.config)
