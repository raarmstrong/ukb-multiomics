# import libraries
import numpy as np
import pandas as pd
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use

from datetime import datetime
# moving other packages up here as they are being imported multiple times
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# for autoencoder
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from Preprocessors import PERCENTAGE_FIELDS, find_percentage_indices, metabolomic_transforms
from scipy.special import logit
from NeuralNetClasses import AutoencoderTransformer, AutoencoderTransformerProt
from Classifiers import get_classifier_and_cv

from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score
from collections import Counter

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import argparse
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config.yaml")
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

args = []
args.insert(0, 'multi_late_only')

datadir = DATA_DIR
  
data_cc = pd.read_csv(f'{datadir}/all_cc_me_cp.csv.gz')
data_prot = pd.read_csv(f'{datadir}/proteomics_all.csv.gz')

# there is a proteomics column called comp so chnage original
data_cc.rename(columns={'comp': 'comp_orig'}, inplace=True)

# metab data
data_metab = pd.read_csv(f'{datadir}/metabolomics_all_clean.csv.gz')
metab_info = pd.read_csv(f'{datadir}/metabolomics_info_clean.csv.gz')
data_metab = pd.merge(data_metab, metab_info[['eid', 'spectrometer']], on='eid')
del(metab_info)

# merge everything
data_multiomic = pd.merge(data_metab, data_prot, how = 'left', on = 'eid')
data_all = pd.merge(data_cc, data_multiomic, how = 'left', on = 'eid')
data_all.shape
#data_all.to_csv(f'{datadir}/data_all.csv.gz', index=False)

# define datasets - base demographics, clinical, bloods, pgs, metabolomic, gneotype

#columns_base = ['case', 'age_surgery', 'score', 'tretspef_uni', 'admimeth_uni', 'opcat']
# filter to admimeth elective or emergency
data_all = data_all[data_all['admimeth_uni'].isin(['elective', 'emergency'])]

# converting opcat to binary
data_all['opcat'] = data_all['opcat'].apply(lambda x: 'complex' if x.endswith('com') else '0')

columns_base = ['case', 'age_surgery', 'sex', 'score', 'admimeth_uni', 'opcat']
columns_prot = [col for col in data_prot.columns if col not in ['eid']]
columns_metab = [col for col in data_metab.columns if col not in ['eid']]

# inflammation panels
olink_key_inflammation = pd.read_csv(f'{datadir}/olink_explore_inflammation.csv', header=None)
olink_key_inflammation_ii = pd.read_csv(f'{datadir}/olink_explore_inflammation_ii.csv', header=None)

columns_inflammation = [col.lower() for col in olink_key_inflammation[0] if col.lower() in data_prot.columns]
columns_inflammation_ii = [col.lower() for col in olink_key_inflammation_ii[0] if col.lower() in data_prot.columns]


data_all = data_all[(data_all['eid'].isin(data_prot['eid'])) & (data_all['eid'].isin(data_metab['eid']))]
data_all.shape
data_all = data_all.drop('eid', axis=1) 

# complication loop

#compnames = ['af', 'aki']
# swap complication loop to age loop
for cn in ['af', 'aki', 'ami', 'delirium', 'stroke', 'ssi']:

    age = 60 if cn in ['delirium', 'stroke'] else 18

    start_comp = datetime.now()
    print('Starting', cn, ' ', args[0], ' at ', start_comp)

    # set complication
    data = data_all[data_all['comp_orig'] == cn]

    if cn == 'ami':
        print('Dropping cardiac for ami')
        print(f'Before: {data.shape[0]}')
        data = data[~data['tretspef_uni'].isin(["Cardiology", "Cardiac surgery", "Cardiothoracic surgery"])]
        print(f'After: {data.shape[0]}')
        
    data = data.drop('comp_orig', axis=1)

    # set age
    data = data[data['age_surgery'] >= age]

    # set dataset loop
    dataset_list = ['infl_1', 'infl_1+2', 'prot_all']

    for dataset in dataset_list:
        print('Dataset:', dataset)
        # set dataset
        if dataset == 'infl_1':
            columns_now = columns_inflammation
        elif dataset == 'infl_1+2':
            columns_now = columns_inflammation + columns_inflammation_ii
        elif dataset == 'prot_all':
            columns_now = columns_prot
        data_now = data[['case'] + columns_metab + columns_now]

        # remove individuals with >=80% missing values
        data_now = data_now.dropna(thresh=0.8*data_now.shape[1], axis=0)

        # drop >10% missing values
        predrop = data_now.columns

        data_now = data_now.dropna(thresh=0.9*len(data_now), axis=1)
        postdrop = data_now.columns
        print('Columns dropped:', [col for col in predrop if col not in postdrop])
        print('Number of columns dropped:', len([col for col in predrop if col not in postdrop]))

        X = data_now.drop(['case'], axis=1)
        y = data_now['case']

        print('Cases and controls:', y.value_counts())

        # preprocessing

        columns_to_exclude = ['count_all', 'count_ins', 'mi', 'chf', 'pvd', 'cevd', 'dementia', 'cpd', 'rheumd', 'pud', 'mld', 'diab', 'diabwc', 'hp', 'rend', 'canc', 'msld', 'metacanc', 'aids', 'score', 'spectrometer'] 
        columns_to_scale = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col not in columns_to_exclude]
        columns_to_encode = [col for col in X.columns if X[col].dtype in ['object'] and col not in columns_to_exclude]
        columns_to_zero = ['score']

        metab_columns_for_ae = [col for col in columns_metab if col not in ['spectrometer'] and col in X.columns]
        prot_columns_for_ae = [col for col in columns_prot if col not in ['spectrometer'] and col in X.columns]
        columns_for_ae = metab_columns_for_ae + prot_columns_for_ae
        
        columns_to_scale = [col for col in columns_to_scale if col not in columns_for_ae]

        percentage_indices = find_percentage_indices(X[metab_columns_for_ae], PERCENTAGE_FIELDS)
        # Create FunctionTransformer for metabolomic preprocessing
        metab_transformer = FunctionTransformer(
            lambda X: metabolomic_transforms(X, percentage_indices=percentage_indices),
            validate=False
        )

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
            ])
        # categorical columns
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant')),
            ('encoder', OneHotEncoder(handle_unknown='error', drop='if_binary'))])
        # columns to zero
        def binarize_cci(X):
            return (X >= 2).astype(int)

        zero_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('binarizer', FunctionTransformer(binarize_cci, validate=False))
            ])
      
        dim_trial = 16
        dropout_trial = True
        if dataset == 'infl_1':
            dropout_trial = False
        samples = X.shape[0]

        print('Dropout:', dropout_trial)
        print('Dimensionality:', dim_trial)

        ae_pipeline_metab = Pipeline([
            ('metab_preprocess', metab_transformer),
            ('scaler', StandardScaler()), # Calculates mean/std while ignoring NaNs
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Impute with Mean (0)
            ('clipper', FunctionTransformer(lambda x: np.clip(x, -10, 10))), # Safety Winsorization for AE
            ('autoencoder', AutoencoderTransformer(encoding_dim=dim_trial, epochs=20, dropout=dropout_trial))
        ])

        ae_pipeline_prot = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('ranknorm', QuantileTransformer(output_distribution='normal', n_quantiles=samples, subsample=1e5, random_state=42)),
            ('clipper', FunctionTransformer(lambda x: np.clip(x, -10, 10))), # Safety Winsorization for AE
            ('autoencoder', AutoencoderTransformerProt(encoding_dim=dim_trial, epochs=20, dropout=dropout_trial))
        ])

            # combine
        data_pipeline_ae = ColumnTransformer([
            #('num', num_pipeline, columns_to_scale),
            ('ae_metab', ae_pipeline_metab, metab_columns_for_ae),
            ('ae_prot', ae_pipeline_prot, prot_columns_for_ae),
            #('cat', cat_pipeline, columns_to_encode),
            #('zero', zero_pipeline, columns_to_zero)
            ],
            n_jobs=1,
            remainder='drop'
            )
        
        n_cases = y.sum()
    
        base_lr, model, cv_strategy = get_classifier_and_cv(n_cases)
        
        args.insert(1, 'bagbal') 
        models = []
        models.append((
            args[1],
            model
            ))

        for name, model in models:

            imba_pipeline = make_pipeline(data_pipeline_ae, model)

            start_mod = datetime.now()
            print('Starting fit ', name, ' for ', cn, ' at ', start_mod)
            
            scores = {'recall': 'recall',
                        'balanced_accuracy': 'balanced_accuracy',
                        'roc_auc': 'roc_auc',
                        'precision': 'precision',
                        'f1': 'f1',
                        'accuracy': 'accuracy',
                        'recall_macro': 'recall_macro',
                        'precision_macro': 'precision_macro',
                        'f1_macro': 'f1_macro',
                        'average_precision': 'average_precision'}

            nested = cross_validate(
                imba_pipeline, 
                X = X, 
                y = y, 
                cv=cv_strategy, 
                return_train_score=True,
                scoring=scores,
                return_estimator=False, 
                n_jobs=60, 
                verbose=3)


            end_mod = datetime.now()
            print('Finished ', name, 'for ', cn, ' at ', end_mod, '. Time elapsed: ', end_mod - start_mod)
            print('Results for ', name, 'for ', cn, ' in dataset ', args[0])

            results_df = pd.DataFrame(nested).drop(['fit_time', 'score_time'], axis=1)
            results_df.loc['mean'] = results_df.mean()
            results_df.loc['std'] = results_df.std()
            results_df['model'] = name
            results_df['cn'] = cn
            results_df['dataset'] = args[0]
            results_df['encoding_dim'] = dim_trial
            results_df['dropout'] = dropout_trial
            results_df['date'] = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_df['age'] = age
            results_df['n_cases'] = y.sum()
            results_df['n_controls'] = len(y) - y.sum()

            print(results_df[['test_recall', 'test_balanced_accuracy', 'test_roc_auc']])

            # make a results folder for current date
            current_date_dir = datetime.now().strftime("%Y%m%d")
            full_output_dir = os.path.join(RESULTS_DIR, current_date_dir)
            os.makedirs(full_output_dir, exist_ok=True)

            results_filename = f'results_{cn}_{name}_{args[0]}_{dataset}_ae_{dim_trial}_{dropout_trial}_{age}_fixed_newae_prepro.csv'
            results_file_path = os.path.join(full_output_dir, results_filename)
            results_df.to_csv(results_file_path, index=True)
                                
            end_comp = datetime.now()
            print(f'Finished {cn} {args[0]} {dim_trial} {dropout_trial} {age} in {end_comp - start_comp}')

    print('Done')