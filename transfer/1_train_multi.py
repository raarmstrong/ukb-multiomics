# multi_better_controls
# now with ratio

# import libraries
import numpy as np
import pandas as pd
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logging
from joblib import dump

from datetime import datetime
# moving other packages up here as they are being imported multiple times
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

#from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# for autoencoder
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from utils import save_multi_pipeline, load_multi_pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Preprocessors import PERCENTAGE_FIELDS_TRANSFER, find_percentage_indices, metabolomic_transforms, binarize_cci, metab_prepro_func, clipper_func
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
    parser.add_argument("--config", default='../../config.yaml', type=str, required=False, help="Path to config.yaml")
    parser.add_argument("--ratio", default=None, type=int, required=False, help="Ratio for case-control sampling")
    parser.add_argument("--config_ae", default='yes', type=str, required=False, help="Whether to run autoencoder (yes/no)")
    parser.add_argument("--config_fitfinal", default='yes', type=str, required=False, help="Whether to fit final model (yes/no)")
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
args.insert(0, 'multi')

datadir = DATA_DIR
  
# this is new case control data using incidence density sampling
# eid, cci_at_diag, age_at_diag, sex, comp, case
if config_arg.ratio:
    data_cc = pd.read_csv(f'{datadir}/transfer_all_cc_multi_ratio_{config_arg.ratio}.csv')
else:
    data_cc = pd.read_csv(f'{datadir}/transfer_all_cc_multi.csv')
# rename to match the surgical dataset
data_cc = data_cc.rename(columns={'cci_at_diag': 'score',
                                'age_at_diag': 'age_surgery'})
# need to turn sex into Female/Male same as surgical dataset
data_cc['sex'] = data_cc['sex'].map({0: 'Female', 1: 'Male'})

data_prot = pd.read_csv(f'{datadir}/proteomics_all.csv.gz')
columns_prot = [col for col in data_prot.columns if col not in ['eid']]

# there is a proteomics column called comp so chnage original
data_cc.rename(columns={'comp': 'comp_orig'}, inplace=True)

data_metab = pd.read_csv(f'{datadir}/metabolomics_reduced.csv.zip')
data_metab = data_metab.rename(columns={'Participant ID': 'eid',
                                        'Spectrometer': 'spectrometer'})
columns_metab = [col for col in data_metab.columns if col not in ['eid', 'spectrometer']]

# inflammation panels
olink_key_inflammation = pd.read_csv(f'{datadir}/olink_explore_inflammation.csv', header=None)
olink_key_inflammation_ii = pd.read_csv(f'{datadir}/olink_explore_inflammation_ii.csv', header=None)

columns_inflammation = [col.lower() for col in olink_key_inflammation[0] if col.lower() in data_prot.columns]
columns_inflammation_ii = [col.lower() for col in olink_key_inflammation_ii[0] if col.lower() in data_prot.columns]

# merge everything
data_multiomic = pd.merge(data_prot, data_metab, on='eid', how='inner')
data_all = data_cc.merge(data_multiomic, on='eid', how='left', validate='m:1')
data_all = data_all.drop('eid', axis=1)

columns_base = ['case', 'age_surgery', 'sex', 'score', 'spectrometer']

# loop through complications
# using value count to do in order of frequency
comp_loop = ['ssi'] if config_arg.ratio else ['af', 'aki', 'ami', 'delirium', 'stroke']

for cn in comp_loop:

    start_comp = datetime.now()
    print('Starting', cn, ' ', args[0], ' at ', start_comp)

    # set dataset by complication
    data_now = data_all[data_all['comp_orig'] == cn].copy()
    # set dataset by age
    if cn in ['stroke', 'delirium']:
        data_now = data_now[data_now['age_surgery']>=60]

    for dataset in ['infl_1', 'infl_1+2', 'prot_all']:
        print('Dataset:', dataset)
        # filter down to dataset of interest

        if dataset != 'prot_all':
            if dataset == 'infl_1':
                selected_columns = columns_inflammation
            elif dataset == 'infl_1+2':
                selected_columns = columns_inflammation + columns_inflammation_ii
            else:
                raise ValueError(f"Unknown dataset name: {dataset}")
        elif dataset == 'prot_all':
            selected_columns = columns_prot
        else:
            raise ValueError(f"Unknown dataset name: {dataset}")
        
        data_now_selected = data_now[columns_base + columns_metab + selected_columns].copy()

        # remove individuals with >=80% missing values
        data_now_selected = data_now_selected.dropna(thresh=0.8*data_now_selected.shape[1], axis=0)

        # drop >10% missing values
        predrop = data_now_selected.columns

        data_now_selected = data_now_selected.dropna(thresh=0.9*len(data_now_selected), axis=1)
        postdrop = data_now_selected.columns
        print('Columns dropped:', [col for col in predrop if col not in postdrop])
        print('Number of columns dropped:', len([col for col in predrop if col not in postdrop]))

        X = data_now_selected.drop('case', axis=1)
        y = data_now_selected['case']

        print('Cases and controls:', y.value_counts())

        # preprocessing

        columns_to_exclude = ['count_all', 'count_ins', 'mi', 'chf', 'pvd', 'cevd', 'dementia', 'cpd', 'rheumd', 'pud', 'mld', 'diab', 'diabwc', 'hp', 'rend', 'canc', 'msld', 'metacanc', 'aids', 'score', 'spectrometer'] 
        columns_to_scale = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col not in columns_to_exclude]
        columns_to_encode = [col for col in X.columns if X[col].dtype in ['object'] and col not in columns_to_exclude] + ['spectrometer']
        columns_to_zero = ['score']

        run_ae = True if config_arg.config_ae.lower() == 'yes' else False

        if run_ae == True:
            metab_columns_for_ae = [col for col in columns_metab if col not in ['spectrometer'] and col in X.columns]
            prot_columns_for_ae = [col for col in columns_prot if col not in ['spectrometer'] and col in X.columns]
            columns_for_ae = metab_columns_for_ae + prot_columns_for_ae
            percentage_indices = find_percentage_indices(X[metab_columns_for_ae], PERCENTAGE_FIELDS_TRANSFER)

        print('Columns to scale:', columns_to_scale)
        print('Columns to encode:', columns_to_encode)
        print('Columns to zero:', columns_to_zero)
        if run_ae == True:
            print('Columns for AE:', columns_for_ae)

        # categorical columns
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant')),
            ('encoder', OneHotEncoder(handle_unknown='error', drop='if_binary'))])
        zero_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('binarizer', FunctionTransformer(binarize_cci, validate=False))
            ])
        
        if run_ae == True:

            dim_trial = 16
            dropout_trial = True
            if dataset == 'infl_1':
                dropout_trial = False
            samples = X.shape[0]

        # multiomics prepro
            metab_transformer = FunctionTransformer(metab_prepro_func, validate=False, kw_args={'percentage_indices': percentage_indices})

            metab_preprocessor = Pipeline([
                ('metab_preprocess', metab_transformer),
                ('scaler', StandardScaler()), # Calculates mean/std while ignoring NaNs
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ])

            prot_preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('ranknorm', QuantileTransformer(output_distribution='normal', n_quantiles=samples, subsample=1e5, random_state=42))
            ])

            # combine the multiomic preprocessors
            multiomic_preprocessor = ColumnTransformer([
                ('metab', metab_preprocessor, metab_columns_for_ae),
                ('prot', prot_preprocessor, prot_columns_for_ae)
            ])
                
            ae_pipeline = Pipeline([
                ('multiomic', multiomic_preprocessor),
                ('clipper', FunctionTransformer(clipper_func)), # Safety Winsorization for AE
                ('autoencoder', AutoencoderTransformerProt(encoding_dim=dim_trial, epochs=20, dropout=dropout_trial))
            ])# combine
            data_pipeline_ae = ColumnTransformer([
                ('num', num_pipeline, columns_to_scale),
                ('ae', ae_pipeline, columns_for_ae),
                ('cat', cat_pipeline, columns_to_encode),
                ('zero', zero_pipeline, columns_to_zero)],
                n_jobs=1,
                remainder='drop'
                )
        else:
            data_pipeline_ae = ColumnTransformer([
                ('num', num_pipeline, columns_to_scale),
                ('cat', cat_pipeline, columns_to_encode),
                ('zero', zero_pipeline, columns_to_zero)],
                n_jobs=1,
                remainder='drop'
                )

        n_cases = y.sum()
    
        base_lr, model, cv_strategy = get_classifier_and_cv(n_cases)

        # adjust base classifier to do balanced class weights
        base_lr.set_params(class_weight='balanced',
                           n_jobs = 80)
    
        args.insert(1, 'en') 
        models = []
        models.append((
            args[1],
            base_lr
            ))
        
        for name, model in models:

            imba_pipeline = make_pipeline(data_pipeline_ae, model)

            if config_arg.config_fitfinal.lower() == 'no':
                grid_imba = GridSearchCV(imba_pipeline, param_grid=params, cv=inner_cv, scoring='roc_auc', return_train_score=True, n_jobs=16, verbose=3)

                start_mod = datetime.now()
                print('Starting fit ', name, ' for ', cn, ' at ', start_mod)
                
                scores = {'recall': 'recall',
                            'balanced_accuracy': 'balanced_accuracy',
                            'roc_auc': 'roc_auc',
                            'precision': 'precision',
                            'f1': 'f1',
                            'recall_macro': 'recall_macro',
                            'precision_macro': 'precision_macro',
                            'f1_macro': 'f1_macro',
                            'average_precision': 'average_precision'}

                if name=='en':
                    jobs=5
                else:
                    jobs=5

                nested = cross_validate(grid_imba, X = X, y = y, cv=outer_cv, return_train_score=True, 
                                        scoring=scores,
                                        return_estimator=False, n_jobs=jobs, verbose=3)


                end_mod = datetime.now()
                print('Finished ', name, 'for ', cn, ' at ', end_mod, '. Time elapsed: ', end_mod - start_mod)
                print('Results for ', name, 'for ', cn, ' in dataset ', args[0])

                #for n in nested['estimator']:
                #    print(f'{n.best_score_}, {n.best_params_}')

                #besties = [n.best_params_ for n in nested['estimator']]
                #besties_df = pd.DataFrame(besties)

                #results_df = pd.DataFrame(nested).drop(['fit_time', 'score_time', 'estimator'], axis=1)
                
                results_df = pd.DataFrame(nested).drop(['fit_time', 'score_time'], axis=1)
                results_df.loc['mean'] = results_df.mean()
                results_df.loc['std'] = results_df.std()
                results_df['model'] = name
                results_df['cn'] = cn
                results_df['dataset'] = f'{args[0]}_{dataset}'
                results_df['date'] = datetime.now().strftime("%Y%m%d-%H%M%S")
                if run_ae == True:
                    results_df['encoding_dim'] = 16
                    results_df['dropout'] = False
                results_df['age'] = ''
                results_df['n_cases'] = y.sum()
                results_df['n_controls'] = len(y) - y.sum()
                results_df['spectro'] = 'newcontrols'

                print(results_df[['test_roc_auc', 'test_recall', 'test_balanced_accuracy']])

                # make a results folder for current date
                current_date_dir = datetime.now().strftime("%Y%m%d")
                full_output_dir = os.path.join(RESULTS_DIR, current_date_dir)
                os.makedirs(full_output_dir, exist_ok=True)

                if run_ae == True:
                    if config_arg.ratio:
                        results_filename = f'results_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_ratio_{config_arg.ratio}.csv'
                    else:
                        results_filename = f'results_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_newcontrols.csv'
                else:
                    if config_arg.ratio:
                        results_filename = f'results_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_ratio_{config_arg.ratio}.csv'
                    else:
                        results_filename = f'results_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_newcontrols.csv'
                results_file_path = os.path.join(full_output_dir, results_filename)
                #besties_filename = f'best_params_{cn}_{name}_{args[0]}_ae_{dim_trial}_{dropout_trial}_{age}_final.csv'
                #besties_file_path = os.path.join(full_output_dir, besties_filename)
                
                results_df.to_csv(results_file_path, index=True)
                #besties_df.to_csv(besties_file_path, index=True)
                
                #model_output_dir = os.path.join(MODEL_DIR, current_date_dir)
                #os.makedirs(model_output_dir, exist_ok=True)

                #model_filename = f'model_{cn}_{name}_{args[0]}_ae_{dim_trial}_{dropout_trial}_{age}_newpipeline_newestbaseline.joblib'
                #model_file_path = os.path.join(model_output_dir, model_filename)
                #dump(nested, model_file_path)

            else:
                # fit final model on all data
                start_mod = datetime.now()
                print('Starting final fit ', name, ' for ', cn, ' at ', start_mod)

                imba_pipeline.fit(X, y)

                end_mod = datetime.now()
                print('Finished final fit ', name, ' for ', cn, ' at ', end_mod, '. Time elapsed: ', end_mod - start_mod)

                # retrieve and save final model
                final_model = imba_pipeline
                print('Final model parameters:', final_model.get_params())

                # save final model and params

                # output
                current_date_dir = datetime.now().strftime("%Y%m%d")
                model_output_dir = os.path.join(MODEL_DIR, current_date_dir)
                os.makedirs(model_output_dir, exist_ok=True)

                # save the pipeline
                if run_ae == True:
                    if config_arg.ratio:
                        model_filename = f'final_model_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_ratio_{config_arg.ratio}.joblib'
                        param_filename = f'final_model_params_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_ratio_{config_arg.ratio}.yaml'
                    else:
                        model_filename = f'final_model_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_newcontrols_newprepro.joblib'
                        param_filename = f'final_model_params_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_newcontrols_newprepro.yaml'
                else:
                    if config_arg.ratio:
                        model_filename = f'final_model_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_ratio_{config_arg.ratio}.joblib'
                        param_filename = f'final_model_params_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_ratio_{config_arg.ratio}.yaml'
                    else:
                        model_filename = f'final_model_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_newcontrols_newprepro.joblib'
                        param_filename = f'final_model_params_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_newcontrols_newprepro.yaml'
                model_file_path = os.path.join(model_output_dir, model_filename)
                param_file_path = os.path.join(model_output_dir, param_filename)
                
                # save model
                save_multi_pipeline(final_model, model_file_path)
                # save best params
                with open(param_file_path, 'w') as param_file:
                    yaml.dump(final_model.get_params(), param_file)
            
                print('Final model saved to ', model_file_path)
                print('Final model parameters saved to ', param_file_path)

                # and do one other version that includes expected columns in case of problems
                model_payload = {
                    'model': final_model,
                    'expected_features': X.columns.tolist()
                }
                if run_ae == True:
                    if config_arg.ratio:
                        model_filename_payload = f'final_model_payload_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_ratio_{config_arg.ratio}.joblib'
                    else:
                        model_filename_payload = f'final_model_payload_{cn}_{name}_{args[0]}_{dataset}_ae_16_False_nonsurg_newcontrols_newprepro.joblib'
                else:
                    if config_arg.ratio:
                        model_filename_payload = f'final_model_payload_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_ratio_{config_arg.ratio}.joblib'
                    else:
                        model_filename_payload = f'final_model_payload_{cn}_{name}_{args[0]}_{dataset}_noae_nonsurg_newcontrols_newprepro.joblib'
                model_file_path_payload = os.path.join(model_output_dir, model_filename_payload)
                save_multi_pipeline(model_payload, model_file_path_payload)
                print('Final model payload saved to ', model_file_path_payload)
                            
        end_comp = datetime.now()
        print(f'Finished {cn} {args[0]} in {end_comp - start_comp}')

print('Done')