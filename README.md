# STAT 441 Final Project

## Overview

The goal of this project was to answer the question "How religious are you?", where the response falls into one of five categories: "no answer", "very important", "quite important", "not important", or "not at all important". Our team's final submission is part of a Kaggle competition; more details can be found here: https://www.kaggle.com/competitions/w2024-kaggle-contest/overview.

## Methodology

- Use lasso regression for variable selection
- Run each classifier on the resulting dataset
- Compare cross-validation results
- Use best model for test set prediction

## To-Do

- Implement cross-validation for each model
- Predict on test set with model with best logloss
- Submit test set predictions

## Models

- Logistic Regression
- XGBoost
- SVM
- MLP
