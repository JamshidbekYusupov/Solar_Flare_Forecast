# 🌞 Solar Flare Forecasting — Machine Learning Project

## Overview
This project builds a **solar flare classification model** using data scraped from:

https://www.spaceweatherlive.com/en/solar-activity/top-50-solar-flares/year/

Goal: predict solar flare activity using engineered temporal and regional features with tree-based ML models.

---

## Dataset
- Source: SpaceWeatherLive (Top 50 solar flares by year)
- Data collected via web scraping
- Includes timestamps, region info, and activity indicators.

---

## Feature Engineering

### Added Features
- Event duration (minutes)
- Cyclical time encoding:
  - `start_hour_sin`
  - `start_hour_cos`
- Seasonal indicators:
  - Quarter
  - Day of year
- Region activity proxy:
  - Frequency-based region activity level

These capture periodicity, seasonality, and regional activity patterns.

---

## Feature Selection
Using Random Forest feature importance: for the features that skewness is larger then 0.70

## Models Trained

Tree-based algorithms were selected due to:

Multilabel classification targets

Nonlinear relationships in solar activity

Robustness to skewed data and outliers

## Models used:

Decision Tree

Random Forest

XGBoost

LightGBM

## Hyperparameter tuning performed using:

GridSearchCV

## Explainability (xAI)

Model interpretability done using SHAP:

SHAP summary plots

SHAP waterfall plots

## Error Analysis  

![alt text](image.png)

Model errors were analyzed using visualization techniques to:

Identify misclassification patterns

Detect systematic weaknesses

Improve model reliability
