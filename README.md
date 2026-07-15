# UK Biobank Multiomics Prediction Pipelines

[![DOI](https://zenodo.org/badge/1178011987.svg)](https://doi.org/10.5281/zenodo.18979068)

This repository contains code and pipelines for machine learning prediction of postoperative complications using multiomics data from the UK Biobank. The code accompanies the following preprint: Using multiomic data to predict postoperative complications after major surgery in the UK Biobank cohort. medRxiv 2026.03.10.26348039; doi: https://doi.org/10.64898/2026.03.10.26348039.

## Overview

The repository provides scripts for:
- Data preprocessing and merging of clinical and omics datasets
- Feature extraction using autoencoders
- Model training and cross-validation for metabolomics, proteomics, and multiomics data
- Model evaluation and post-hoc analysis (DCA)
- Both direct postoperative modelling and transfer learning from the non-postoperative domain are implemented

All scripts are configurable via a YAML file (`config.yaml`) specifying data directories, results directories, and model output locations.

## Required inputs

The scripts require case-control data on postoperative complications with accompanying clinical covariates which are extracted from UK Biobank. 

Scripts detailing this process can be found in the following repositories:
- Delirium: https://github.com/raarmstrong/gwas-postop-delirium (https://doi.org/10.5281/zenodo.18196798)
- Other outcomes: https://github.com/raarmstrong/gwas-postop-complications (https://doi.org/10.5281/zenodo.17901910).

Metabolomic and protoemic data were QC'd using the metaboprep pipeline with default parameters: https://mrcieu.github.io/metaboprep/.

## Directory Structure

```
1_baseline_model_metab.py
1_baseline_model_prot.py
1_baseline_model_multi.py
2_omics_model_metab.py
2_omics_model_prot.py
2_omics_model_multi.py
2_omics_model_multi_late.py
3_omics_only_model_metab.py
3_omics_only_model_prot.py
3_omics_only_model_multi.py
3_omics_only_model_multi_late.py
Classifiers.py
NeuralNetClasses.py
Preprocessors.py
evaluation/
    generate_oof_metab.py
    generate_oof_prot.py
    generate_oof_multi.py
    dca.py
    plot_dca_oof.py
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

- **1_baseline_model_metab.py**: Baseline model for participants in the metabolomics dataset.
- **1_baseline_model_prot.py**: Baseline model for participants in the proteomics dataset.
- **1_baseline_model_multi.py**: Baseline model for participants in the combined multiomics dataset.
- **2_omics_model_metab.py**: Omics model for metabolomics data with autoencoder feature extraction.
- **2_omics_model_prot.py**: Omics model for proteomics data with autoencoder feature extraction.
- **2_omics_model_multi.py**: Omics model for multiomics data (metabolomics + proteomics) with autoencoder feature extraction - early integration.
- **2_omics_model_multi_late.py**: Omics model for multiomics data (metabolomics + proteomics) with autoencoder feature extraction - late integration.
- **3_omics_only_model_*.py**: Models trained using exclusively omic data (no clinical features).
- **Classifiers.py**: Contains classifier objects and cross-validation strategies wrapped in helper functions.
- **NeuralNetClasses.py**: Contains the custom scikit-learn autoencoder transformer used for feature extraction.
- **Preprocessors.py**: Contains preprocessors for different data types wrapped in helper functions.

### Transfer Learning and Prediction Scripts 

The `transfer/` directory relates to the transfer learning component of the paper. Script 1 trains models for each omic feature set on the non-postoperative cohort, predicting non-postoperative outcomes. Script 2 then uses those models to generate predicted probabilities of the non-postoperative outcome in the postoperative cohort. Script 3 then trains parsimonious 'transfer models', using the predicted probabilities as features.

- **transfer/1_train_metab.py**: Non-postoperative model training pipeline for metabolomics data.
- **transfer/1_train_prot.py**: Non-postoperative model training pipeline for proteomics data.
- **transfer/1_train_multi.py**: Non-postoperative model training pipeline for multiomics data.
- **transfer/2_predict_all.py**: Predicts non-postoperative complication probabilities on the surgical cohort using trained models from Scripts 1a-c with autoencoder feature extraction.
- **transfer/3_final_model_all.py**: Trains new 'transfer models' using predicted probabilities from Script 2 as features to predict postoperative outcomes in the postoperative cohort.
- **transfer/utils.py**: Helper functions for pipeline and data import/export/reconstruction.

### Evaluation and Post-hoc Analysis

The `evaluation/` directory contains tools for validating and evaluating model performance:
- **evaluation/generate_oof_*.py**: Generates out-of-fold (OOF) predictions for various omic tiers to use in decision-curve analysis (DCA).
- **evaluation/dca.py**: Modular utility for calculating Net Benefit and plotting Decision Curves.
- **evaluation/plot_dca_oof.py**: Generates DCA plots using OOF predictions.

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

## Key Requirements

Including versions used in these analysis:

- Python 3.6.8
- imblearn 0.8.1
- pandas 1.1.5
- numpy 1.19.5
- scikit-learn 0.24.2
- tensorflow 2.6.2

## Citation

If you use this code, please cite the accompanying preprint:

> https://doi.org/10.64898/2026.03.10.26348039

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Statement on the Use of AI

Generative AI (Gemini CLI v0.38.2, Model Auto: `gemini-3.1-pro` / `gemini-3-flash`) was used strictly to sanitise this code prior to public release:

- removing any sensitive identifiers and hardcoded filepaths
- removing specific dates
- removing commented-out code which is no longer used
- adding explanatory comments to improve code documentation and readability, without altering the underlying structure or function.

Code diffs were manually reviewed to verify code integrity. Generative AI was not used in code generation or manuscript production.
