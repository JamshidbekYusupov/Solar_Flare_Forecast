import sys, os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

sys.path.append(r'C:\Solar-Flare-Forecast')

data_path = r'Data/Raw_Data/Raw_Data.csv'

df = pd.read_csv(data_path)

# X = df.drop(columns='Class')
# y = df['Class']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from src.feature_engineering import Feature_Eng

fe = Feature_Eng(df=df)
# fe = Feature_Eng(X_train= X_train, X_test=X_test, y_train=y_train)

fe.feature_adding()
fe.feature_selection()
fe.feature_transform()
fe.data_saving()