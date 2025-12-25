# Corporate Bankruptcy Prediction using Machine Learning

This repository contains the complete analytical workflow developed for an MSc Business Analytics dissertation focused on predicting corporate bankruptcy using machine learning and ensemble modelling techniques. The project evaluates statistical, machine-learning, and ensemble-based approaches under conditions of severe class imbalance.

## Project Overview

Corporate bankruptcy prediction is a critical problem in financial risk management due to the asymmetric cost of misclassification and the highly imbalanced nature of bankruptcy data. This project develops and compares multiple predictive models, with a particular focus on improving minority-class (bankrupt firm) detection.

The analysis follows an end-to-end, reproducible pipeline implemented in Python using Jupyter notebooks.

## Dataset

- Observations: 6,819 firms  
- Target variable: `Bankrupt?` (binary classification)  
- Predictors: 95 firm-level financial ratios  
- Data type: Fully numerical, no missing values  

A detailed data dictionary is provided separately as part of the project documentation.

## Methodology

The analytical workflow includes the following steps:

- Data loading, inspection, and validation  
- Exploratory data analysis and descriptive statistics  
- Correlation-based feature screening  
- Exploratory Principal Component Analysis (PCA) for visualisation  
- Stratified train–test splitting  
- Feature scaling using standardisation  
- Class imbalance handling using SMOTETomek  
- Model development and evaluation  

## Models Implemented

Baseline models:
- Logistic Regression  
- Random Forest  
- XGBoost  
- Deep Neural Network (Keras + SciKeras)

Ensemble models:
- Stacking (Random Forest + XGBoost)  
- Stacking (Random Forest + XGBoost + Logistic Regression)  
- Stacking (XGBoost + Logistic Regression, Random Forest meta-model)  
- Soft Voting Ensemble  

## Model Evaluation

Model performance is evaluated using metrics suitable for imbalanced classification:

- F1 Score (primary metric)  
- ROC-AUC  
- Precision and Recall  
- Confusion Matrices  

Visual diagnostics include ROC curves, confusion matrices, PCA plots, and model comparison charts.

## Results Summary

Ensemble-based approaches, particularly stacking models, outperform individual baseline models in terms of minority-class detection. The stacking ensemble combining Random Forest and XGBoost achieves the highest F1 Score and ROC-AUC, demonstrating improved balance between precision and recall for bankrupt firms.

## Repository Structure

- Jupyter notebooks for data preprocessing, modelling, and evaluation  
- Visualisations and statistical outputs supporting the analysis  
- Fully documented code with inline comments and Markdown explanations  

## Reproducibility

All analyses were conducted in Python using standard machine-learning libraries. Random states are fixed where applicable to ensure reproducibility. The repository serves as the technical companion to the dissertation.

## Author
Student
MSc Business Analytics (University of Greenwich) – Dissertation Project
