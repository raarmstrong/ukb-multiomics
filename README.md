# UK Biobank Multiomics Prediction Pipelines

This repository contains code and pipelines for machine learning prediction of postoperative complications using multiomics data from the UK Biobank. The code accompanies the following publication: Armstrong et al

## Overview

The repository provides scripts for:
- Data preprocessing and merging of clinical and omics datasets
- Feature extraction using autoencoders
- Model training and cross-validation for metabolomics, proteomics, and multiomics data
- Model evaluation
- Both direct postoperative modelling and transfer learning from the non-postoperative domain are implemented

All scripts are configurable via a YAML file (`config.yaml`) specifying data directories, results directories, and model output locations.

## Required inputs

The scripts require case-control data on postoperative complications with accompanying clinical covariates which are extracted from UK Biobank. Scripts detailing this process can be found in the following repositories: https://doi.org/10.1101/2025.09.01.25334224 and https://doi.org/10.64898/2025.12.11.25342055.

Metabolomic and protoemic data were QC'd using the metaboprep pipeline with default parameters: https://mrcieu.github.io/metaboprep/.

## Directory Structure

```
1_baseline_model_metab.py
1_baseline_model_prot.py
1_baseline_model_multiomic.py
2_omics_model_metab.py
2_omics_model_prot.py
2_omics_model_multiomic.py
2_omics_model_multiomic_late.py
Classifiers.py
NeuralNetClasses.py
Preprocessors.py
transfer/
    1_train_metab.py
    1_train_prot.py
    1_train_multi.py
    2_predict_all.py
    3_final_model_all.py
```

## Script Descriptions

### Baseline and Omics Model Scripts (Direct Postoperative Modelling)

As reported in the accompanying paper, baseline models using only routinely-available clinical covariates were trained first in the subset of participants with each omics dataset. Omic models, with additional metabolomic and/or proteomic features, were then trained in the same subset of participants.

- **baseline_model_metab.py**: Baseline model for metabolomics data.
- **baseline_model_prot.py**: Baseline model for proteomics data.
- **baseline_model_multiomic.py**: Baseline model for combined multiomics data.
- **omics_model_metab.py**: Omics model for metabolomics data with autoencoder feature extraction.
- **omics_model_prot.py**: Omics model for proteomics data with autoencoder feature extraction.
- **omics_model_multiomic.py**: Omics model for multiomics data (metabolomics + proteomics) with autoencoder feature extraction.
- **omics_model_multiomic_late.py**: Late-fusion multiomics model with autoencoder feature extraction.
- **Classifiers.py**: Contains classifier objects and cross-validation strategies wrapped in helper functions.
- **NeuralNetClasses.py**: Contains the custom scikit-learn autoencoder transformer used for feature extraction.
- **Preprocessors.py**: Contains preprocessors for different data types wrapped in helper functions.

### Transfer Learning and Prediction Scripts 

These scripts relate to the transfer learning component of the paper. Scripts 1a-c train omics models on the non-postoperative cohort, predicting non-postoperative outcomes. Script 2 then uses those models to generate predicted probabilities of the non-postoperative outcome in the postoperative cohort. Script 3 then trains parsimonious 'transfer models', using the predicted probabilities as features.

- **transfer/1_train_metab.py**: Non-postoperative model training pipeline for metabolomics data .
- **transfer/1_train_prot.py**: Non-postoperative model training pipeline for proteomics data.
- **transfer/1_train_multi.py**: Non-postoperative model training pipeline for multiomics data.
- **transfer/2_predict.py**: Predicts non-postoperative complication probabilities on the surgical cohort using trained models from Scripts 1a-c with autoencoder feature extraction.
- **transfer/3_final_model_all.py**: Trains new 'transfer models' using predicted probabilities from Script 2 as features to predict postoperative outcomes in the postoperative cohort.
- **transfer/utils.py**: Helper functions for pipeline and data import/export/reconstruction.

## Configuration

All scripts require a `config.yaml` file specifying:
- `datadir`: Path to input data directory
- `resultsdir`: Path to output results directory
- `modeldir`: Path to save trained models

Example `config.yaml`:
```yaml
datadir: /path/to/data
resultsdir: /path/to/results
modeldir: /path/to/models
```

## Usage

1. Prepare your `config.yaml` file with correct paths.
2. Run the desired script with the `--config` argument, e.g.:
   ```bash
   python3 baseline_model_metab.py --config config.yaml
   ```
3. For transfer learning and prediction scripts, see the comments in each script for usage details and required arguments.

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn, tensorflow, joblib, pyyaml

## Citation

If you use this code, please cite the accompanying preprint:

> https://doi.org/10.64898/2026.03.10.26348039
