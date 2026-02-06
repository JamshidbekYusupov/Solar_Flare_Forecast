import sys
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv(r'C:\Solar-Flare-Forecast\Data\Engineered_Data\Engineered_Data.csv')

# df = pd.read_csv(data_path)

target = 'Class'

# Models with hyperparameters
models = {
    'Random_Forest': RandomForestClassifier(),
    'Desicion_Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    # 'CatBoost': CatBoostClassifier(),
    'LightGBM': LGBMClassifier()
}

rf_param_grid = {
    'n_estimators': [100, 150, 200],  # Number of trees
    'max_depth': [None, 10, 15, 20],  # Max depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'max_features': ['auto', 'sqrt', 'log2'],  # Max features for splitting nodes
    'class_weight': [None, 'balanced'],  # Handling imbalanced classes
}

dt_param_grid = {
    'max_depth': [None, 10, 20, 30],  # Max depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'max_features': ['auto', 'sqrt', 'log2'],  # Max features for splitting nodes
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'splitter': ['best', 'random'],  # Splitter strategy
    'class_weight': [None, 'balanced'],  # Handling imbalanced classes
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Step size for each iteration
    'max_depth': [3, 5, 7],  # Depth of the tree
    'scale_pos_weight': [1, 2, 5],  # For unbalanced classes
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term
    'reg_lambda': [1, 1.5, 2],  # L2 regularization term
}

catboost_param_grid = {
    'iterations': [500, 1000, 1500],  # Number of boosting iterations
    'learning_rate': [0.01, 0.05, 0.1],  # Step size for each iteration
    'depth': [5, 7, 10],  # Depth of the trees
    'l2_leaf_reg': [3, 5, 10],  # L2 regularization
    'class_weights': [None, 'balanced'],  # Handling imbalanced classes
}

lgbm_param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.8, 1.0],
}

param_grids = {
    'Random_Forest': rf_param_grid,
    'Desicion_Tree': dt_param_grid,
    'XGBoost': xgb_param_grid,
    # 'CatBoost': catboost_param_grid,
    'LightGBM': lgbm_param_grid
}

sys.path.append(r'C:\Solar-Flare-Forecast')

from src.model_pipeline import auto_pipeline

for name, model in models.items():

    au = auto_pipeline(df, target=target, model=model, model_name=name, params=param_grids[name])

    au.feature_prepare()

    au.tuning()

    au.fit()

    au.model_saving()

    au.prediction()

    au.evaluvation()

    au.metrics_saving()

    au.save_preprocessed_data()