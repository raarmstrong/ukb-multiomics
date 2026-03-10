# set up for AE currently

# import libraries
import numpy as np
import pandas as pd
import sys
import os

from datetime import datetime
# moving other packages up here as they are being imported multiple times
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier

#from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score
from collections import Counter

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Classifiers import get_classifier_and_cv

import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../../config.yaml', type=str, required=False, help="Path to config.yaml")
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
datadir = DATA_DIR
RESULTS_DIR = config['resultsdir']
MODEL_DIR = config['modeldir']

for dataset in ['metab', 'multi_infl_1', 'multi_infl_1+2', 'multi_prot_all', 'prot_infl_1', 'prot_infl_1+2', 'prot_prot_all']:
  
    for cn in ['af', 'aki', 'ami', 'delirium', 'stroke', 'ssi']:
        # this is the new dataset which includes omics and predicted probability etc.
        cols_to_use = ['eid', 'case', 'admimeth_uni', 'opcat', f'predicted_prob_{cn}']
        data_all = pd.read_csv(f'{RESULTS_DIR}/predictions/data_{dataset}_with_predictions_ae_{cn}_newcontrols_newprepro.csv', usecols=cols_to_use)
        
        print(f'Processing complication: {cn} for dataset: {dataset}')

        # read in ful data so ami can filter by specialty
        if cn == 'ami':
            data_full = pd.read_csv(f'{DATA_DIR}/all_cc_me_cp.csv.gz', usecols=['eid', 'tretspef_uni', 'comp'])
            data_ami = data_full[data_full['comp'] == 'ami'][['eid', 'tretspef_uni']]
            data_all = data_all.merge(data_ami, on='eid', how='left', validate='m:1')
            # filter to specialties relevant to ami
            print('Dropping cardiac for ami')
            print(f'Before: {data_all.shape[0]}')
            data_all = data_all[~data_all['tretspef_uni'].isin(["Cardiology", "Cardiac surgery", "Cardiothoracic surgery"])]
            print(f'After: {data_all.shape[0]}')
            data_all = data_all.drop('tretspef_uni', axis=1)

        # filter to admimeth elective or emergency
        data_all = data_all[data_all['admimeth_uni'].isin(['elective', 'emergency'])]

        # converting opcat to binary
        data_all['opcat'] = data_all['opcat'].apply(lambda x: 'complex' if x.endswith('com') else '0')

        # convert predicted prob to log odds
        epsilon = 1e-15
        data_all[f'predicted_prob_{cn}'] = data_all[f'predicted_prob_{cn}'].clip(epsilon, 1 - epsilon)
        data_all[f'predicted_logodds_{cn}'] = np.log(data_all[f'predicted_prob_{cn}'] / (1 - data_all[f'predicted_prob_{cn}']))
        data_all = data_all.drop(columns=[f'predicted_prob_{cn}'])

        print('Missing values:')
        print(data_all.isna().sum())

        start_comp = datetime.now()
        print('Starting', cn, ' ', dataset, ' at ', start_comp)
        data_now = data_all.copy()

        # remove individuals with >=80% missing values
        data_now = data_now.dropna(thresh=0.8*data_now.shape[1], axis=0)

        # drop >10% missing values
        predrop = data_now.columns

        data_now = data_now.dropna(thresh=0.9*len(data_now), axis=1)
        postdrop = data_now.columns
        print('Columns dropped:', [col for col in predrop if col not in postdrop])
        print('Number of columns dropped:', len([col for col in predrop if col not in postdrop]))

        X = data_now.drop(['eid', 'case'], axis=1)
        y = data_now['case']

        print('Cases and controls:', y.value_counts())

        # preprocessing

        columns_to_scale = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        columns_to_encode = [col for col in X.columns if X[col].dtype in ['object']]

        print('Columns to scale:', columns_to_scale)
        print('Columns to encode:', columns_to_encode)

        # categorical columns
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant')),
            ('encoder', OneHotEncoder(handle_unknown='error', drop='if_binary'))])
        data_pipeline = ColumnTransformer([
            ('num', num_pipeline, columns_to_scale),
            ('cat', cat_pipeline, columns_to_encode)],
            n_jobs=1,
            remainder='drop'
            )

        n_cases = y.sum()
    
        base_lr, model, cv_strategy = get_classifier_and_cv(n_cases, transfer=True)
        
        models = []
        models.append((
            'bagbal',
            model
            ))

        for name, model in models:

            imba_pipeline = make_pipeline(data_pipeline, model)

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
            print('Results for ', name, 'for ', cn, ' in dataset ', dataset)
            
            results_df = pd.DataFrame(nested).drop(['fit_time', 'score_time'], axis=1)
            results_df.loc['mean'] = results_df.mean()
            results_df.loc['std'] = results_df.std()
            results_df['model'] = name
            results_df['cn'] = cn
            results_df['dataset'] = dataset
            results_df['date'] = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_df['age'] = ''
            results_df['n_cases'] = y.sum()
            results_df['n_controls'] = len(y) - y.sum()

            print(results_df[['test_roc_auc', 'test_recall', 'test_balanced_accuracy']])

            # make a results folder for current date
            #current_date_dir = datetime.now().strftime("%Y%m%d")
            full_output_dir = os.path.join(RESULTS_DIR, 'transfer')
            os.makedirs(full_output_dir, exist_ok=True)

            results_filename = f'results_{cn}_{name}_{dataset}_ae_withriskscore_newcontrols_newprepro.csv'
            results_file_path = os.path.join(full_output_dir, results_filename)
            
            results_df.to_csv(results_file_path, index=True)
        
        end_comp = datetime.now()
        print(f'Finished {cn} {dataset} in {end_comp - start_comp}')

print('Done')