import os
from joblib import dump, load
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def save_metab_pipeline(payload, base_path):
    pipeline = payload['model'] if isinstance(payload, dict) else payload
    joblib_path = base_path
    keras_path = base_path.replace('.joblib', '_encoder.h5')

    try:
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        
        # Save the ENCODER specifically (the part that does the transforming)
        if hasattr(ae_step_wrapper, 'encoder'):
            ae_step_wrapper.encoder.save(keras_path)
            print(f"Encoder weights saved to: {keras_path}")
            
            # Nullify BOTH to allow pickling
            ae_step_wrapper.encoder = None
            if hasattr(ae_step_wrapper, 'autoencoder'):
                ae_step_wrapper.autoencoder = None
        
        dump(payload, joblib_path)
        print(f"Main payload saved to: {joblib_path}")
        
    except Exception as e:
        print(f"Standard save triggered (No AE found or error: {e})")
        dump(payload, joblib_path)

def load_metab_pipeline(base_path):
    """ Helper to reload the split model later """
    payload = load(base_path)
    
    keras_path = base_path.replace('.joblib', '_encoder.h5')
    if os.path.exists(keras_path):
        pipeline = payload['model'] if isinstance(payload, dict) else payload
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        ae_step_wrapper.model = load_model(keras_path)
        
    return payload

def save_prot_pipeline(payload, base_path):
    pipeline = payload['model'] if isinstance(payload, dict) else payload
    joblib_path = base_path
    keras_path = base_path.replace('.joblib', '_encoder.h5')

    try:
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        
        # Save the ENCODER specifically (the part that does the transforming)
        if hasattr(ae_step_wrapper, 'encoder'):
            ae_step_wrapper.encoder.save(keras_path)
            print(f"Encoder weights saved to: {keras_path}")
            
            # Nullify BOTH to allow pickling
            ae_step_wrapper.encoder = None
            if hasattr(ae_step_wrapper, 'autoencoder'):
                ae_step_wrapper.autoencoder = None
        
        dump(payload, joblib_path)
        print(f"Main payload saved to: {joblib_path}")
        
    except Exception as e:
        print(f"Standard save triggered (No AE found or error: {e})")
        dump(payload, joblib_path)

def load_prot_pipeline(base_path):
    """ Helper to reload the split model later """
    payload = load(base_path)
    
    keras_path = base_path.replace('.joblib', '_encoder.h5')
    if os.path.exists(keras_path):
        pipeline = payload['model'] if isinstance(payload, dict) else payload
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        ae_step_wrapper.model = load_model(keras_path)
        
    return payload


def save_multi_pipeline(payload, base_path):
    pipeline = payload['model'] if isinstance(payload, dict) else payload
    joblib_path = base_path
    keras_path = base_path.replace('.joblib', '_encoder.h5')

    try:
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        
        # Save the ENCODER specifically (the part that does the transforming)
        if hasattr(ae_step_wrapper, 'encoder'):
            ae_step_wrapper.encoder.save(keras_path)
            print(f"Encoder weights saved to: {keras_path}")
            
            # Nullify BOTH to allow pickling
            ae_step_wrapper.encoder = None
            if hasattr(ae_step_wrapper, 'autoencoder'):
                ae_step_wrapper.autoencoder = None
        
        dump(payload, joblib_path)
        print(f"Main payload saved to: {joblib_path}")
        
    except Exception as e:
        print(f"Standard save triggered (No AE found or error: {e})")
        dump(payload, joblib_path)

def load_multi_pipeline(base_path):
    """ Helper to reload the split model later """
    from tensorflow.keras.models import load_model
    payload = joblib.load(base_path)
    
    keras_path = base_path.replace('.joblib', '_encoder.h5')
    if os.path.exists(keras_path):
        pipeline = payload['model'] if isinstance(payload, dict) else payload
        col_transformer = pipeline.steps[0][1]
        ae_step_wrapper = col_transformer.named_transformers_['ae'].named_steps['autoencoder']
        ae_step_wrapper.model = load_model(keras_path)
        
    return payload

def reconstruct_pipeline(payload, model_path):
    model = payload['model']
    # Ensure consistency in path handling
    weights_path = model_path.replace('.joblib', '_encoder.h5')
    print(f"Looking for encoder weights at: {weights_path}")
    if os.path.exists(weights_path):
        print(f"Found Encoder weights. Reconstructing...")
        try:
            col_transformer = model.steps[0][1]
            ae_pipeline = col_transformer.named_transformers_['ae']
            ae_transformer = ae_pipeline.named_steps['autoencoder']
            # Load back into .encoder attribute
            ae_transformer.encoder = load_model(weights_path)
            print("Autoencoder-Encoder successfully reconstructed.")
        except Exception as e:
            print(f"FATAL: Error during reconstruction: {e}")
            raise e
    else:
        # !!! This causes the silent failure !!!
        raise FileNotFoundError(f"Could not find encoder weights file at: {weights_path}")
    return model

def sort_external_data(dataset, metab_path, prot_path, external_cohort_data, datadir):
    '''
    Dataset is metab, multi_infl_1, multi_infl_1+2, multi_prot_all, prot_infl_1, prot_infl_1+2, prot_prot_all
    Cohort data will always be the standard all_cc_me_cp.csv.gz
    Omics data will vary
    Cols need to match what was used in training
    '''
    data_cc = external_cohort_data.copy()
    if dataset.startswith('metab') :
        print('Loading metabolomics data...')
        data_metab = pd.read_csv(metab_path)
        data_metab = data_metab.rename(columns={'Participant ID': 'eid',
                                                'Spectrometer': 'spectrometer'})
        data_all = data_cc.merge(data_metab, on='eid', how='inner', validate='m:1')
    elif dataset.startswith('prot') or dataset.startswith('multi'):
        # there is a proteomics column called comp so chnage original
        data_cc.rename(columns={'comp': 'comp_orig'}, inplace=True)
        print('Loading proteomics data...')
        data_prot = pd.read_csv(prot_path)
        data_prot = data_prot.rename(columns={'Participant ID': 'eid',
                                              'Spectrometer': 'spectrometer'})
        # inflammation panels
        olink_key_inflammation = pd.read_csv(os.path.join(datadir, 'olink_explore_inflammation.csv'), header=None)
        olink_key_inflammation_ii = pd.read_csv(os.path.join(datadir, 'olink_explore_inflammation_ii.csv'), header=None)
        columns_inflammation = [col.lower() for col in olink_key_inflammation[0] if col.lower() in data_prot.columns]
        columns_inflammation_ii = [col.lower() for col in olink_key_inflammation_ii[0] if col.lower() in data_prot.columns]
        if 'infl_1+2' in dataset:
            data_prot = data_prot[['eid'] + columns_inflammation + columns_inflammation_ii]
        elif 'infl_1' in dataset:
            data_prot = data_prot[['eid'] + columns_inflammation]
        elif 'all' in dataset:
            pass # use all proteomics columns
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        if dataset.startswith('prot'):
            data_all = data_cc.merge(data_prot, on='eid', how='inner', validate='m:1')
        elif dataset.startswith('multi'):
            print('Loading metabolomics data...')
            data_metab = pd.read_csv(metab_path)
            data_metab = data_metab.rename(columns={'Participant ID': 'eid',
                                                    'Spectrometer': 'spectrometer'}) 
            data_multiomic = pd.merge(data_prot, data_metab, on='eid', how='inner')
            data_all = data_cc.merge(data_multiomic, on='eid', how='inner', validate='m:1')
    print(f'External data shape after merging: {data_all.shape}')
    return data_all

def predict_external_probs(model_path, payload_path, external_data_df):
    """
    Modified to handle the two-part model (joblib + h5).
    """
    print(f"Loading model from {model_path}...")
    payload = load(payload_path)
    
    # --- RECONSTRUCTION STEP ---
    # Re-link the Keras model to the sklearn pipeline
    model = reconstruct_pipeline(payload, model_path)
    
    expected_cols = payload['expected_features']
    
    missing_cols = [col for col in expected_cols if col not in external_data_df.columns]
    
    if len(missing_cols) > 0:
        raise ValueError(
            f"ERROR: The external dataset is missing {len(missing_cols)} columns "
            f"required by the model.\nMissing examples: {missing_cols[:5]}"
        )
        
    X_external = external_data_df[expected_cols].copy()
    
    print("Generating predictions (this may take a moment if running AE on CPU)...")
    probs = model.predict_proba(X_external)[:, 1] 
    
    return probs
