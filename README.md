# Customer Churn Prediction (UTKRISHTA 2026, AI/ML)

## Problem
Predict the probability that a telecom customer will churn.

## Data
train.csv (labeled), validation.csv. Target: Churn. 26.5% churn rate.

## Method
Pipeline: fill blanks, scale numbers, one-hot encode text, then a model.
Dropped: customerID (ID), gender, PhoneService, MultipleLines (lowest EDA association).
Split: 80/20 stratified, seed 42.

## Experiments
Runs 1-5, logged in results/role2_run_log.csv.
Final model: Logistic Regression, class_weight=balanced.

## Results
Holdout (20% of train): ROC-AUC 0.8489, PR-AUC 0.6620
Separate validation.csv: ROC-AUC 0.8409, PR-AUC 0.6361

## Files
notebooks/ code, results/predictions.csv scores, models/ saved model.

## How to run
Open the notebook in Google Colab, mount Drive, and run top to bottom.
scikit-learn version is in models/FINAL_metadata.json.

## Limitations
Single validation split. Scores are risk rankings, not certainties.
Predictions are for <file name>, because no separate test file was provided.
