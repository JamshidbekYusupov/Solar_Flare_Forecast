import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

log_path = r'C:\Solar-Flare-Forecast\Logging\engineering.log'

logging.basicConfig(
    filename= log_path,
    filemode='a',
    level= logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)
# logging.propogate = False
logging.info('Feature enginerring has started')

class Feature_Eng:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_transform = None
        # self.X_train = X_train
        # self.X_test = X_test
        # self.y_train = y_train
        # self.y_test = y_test
        # self.df_transform = None
        # self.X_test_transformation = None
    
    def feature_adding(self):

        try:
        # Duration of an event

            self.df['duration_min'] = abs(self.df['end_hour']*60 + self.df['end_minute'])
            # self.X_train['duration_min'] = abs((self.X_train['end_hour']*60 + self.X_train['end_minute'] 
            #                                     - self.X_train['start_hour']*60 - self.X_train['start_minute'] ))
            # self.X_test['duration_min'] = abs((self.X_test['end_hour']*60 + self.X_test['end_minute'] 
            #                                    - self.X_test['start_hour']*60 - self.X_test['start_minute'] ))
            
            # Cyclical time encoding
            self.df['start_hour_sin'] = np.sin(2*np.pi*self.df['start_hour']/24)
            # self.X_train['start_hour_sin'] = np.sin(2*np.pi*self.X_train['start_hour']/24)
            # self.X_train['start_hour_cos'] = np.cos(2*np.pi*self.X_train['start_hour']/24)

            self.df['start_hour_cos'] = np.cos(2*np.pi*self.df['start_hour']/24)
            # self.X_test['start_hour_sin'] = np.sin(2*np.pi*self.X_test['start_hour']/24)
            # self.X_test['start_hour_cos'] = np.cos(2*np.pi*self.X_test['start_hour']/24)

            # Season / quarter

            self.df['quarter'] = ((self.df['Month']-1)//3)+1
            # self.X_train['quarter'] = ((self.X_train['Month']-1)//3)+1
            # self.X_test['quarter'] = ((self.X_test['Month']-1)//3)+1

            # Day-of-year
            self.df['day_of_year'] = pd.to_datetime(self.df[['Year','Month','Day']]).dt.dayofyear
            # self.X_train['day_of_year'] = pd.to_datetime(self.X_train[['Year','Month','Day']]).dt.dayofyear
            # self.X_test['day_of_year'] = pd.to_datetime(self.X_test[['Year','Month','Day']]).dt.dayofyear

            # Region frequency / activity level

            region_counts = self.df['Region'].value_counts()

            self.df['region_activity'] = self.df['Region'].map(region_counts)
            # self.X_train['region_activity'] = self.X_train['Region'].map(region_counts)
            # self.X_test['region_activity'] = self.X_test['Region'].map(region_counts)
            logging.info(f'Feature Engineering is DONE, added features:[region_activity, day_of_year, quarter, start_hour_cos, start_hour_sin , duration_min]')
            return self

        except Exception as e:
            logging.error(f'ERROR while feature adding:{e}')
            raise

    def feature_selection(self):

        try:

            model = RandomForestClassifier()
            model.fit(self.df.drop(columns='Class'), self.df['Class'])

            # Muhim featurelarni aniqlash
            feature_importances = model.feature_importances_

            # Muhimlilik darajasi boyicha sortirofka qilib olish
            important_features = pd.Series(feature_importances, index=self.df.drop(columns='Class').columns)
            important_features = important_features[important_features > 0.05].index.to_list()

            # self.X_train = self.X_train[important_features]
            # self.X_test = self.X_test[important_features]
            logging.info(f'Feature selection if DONE with {model}')
            return self
        
        except Exception as e:
            logging.error(f'ERROR while feature selection: {e}')
            raise
    
    def feature_transform(self):

        try:

            # selecting numerical features
            num_features = self.df.select_dtypes(include=np.number).columns
            skewness = self.df[num_features].skew()

            # selectiong skewed features
            skewed_features = skewness[abs(skewness) > 0.70].index.tolist()

            # Log transformation

            # self.df_transform = self.X_train.copy()
            # self.X_test_transformation = self.X_test.copy()
            self.df_transform = self.df.copy()

            for col in skewed_features:
                if (self.df_transform[col] >= 0).all():
                    self.df_transform[col] = np.log1p(self.df_transform[col])
                    # self.X_test_transformation[col] = np.log1p(self.X_test_transformation[col])
            logging.info(f'Feature Transformation is done for skewed features: {skewed_features}')
            return self
        
        except Exception as e:
            logging.error(f'ERROR while feature transformation; {e}')
    
    def data_saving(self):

        try:

            # Saving data to Data folder inside Engineered Folder
            out_dir = r'C:\Solar-Flare-Forecast\Data\Engineered_Data'
            os.makedirs(out_dir, exist_ok=True)
            file_path = os.path.join(out_dir, 'Engineered_Data.csv')
            self.df_transform.to_csv(file_path, index = False)

            # out_dir = r'C:\Solar-Flare-Forecast\Data\Engineered_Data'
            # os.makedirs(out_dir, exist_ok=True)
            # file_path = os.path.join(out_dir, 'X_test_eng.csv')
            # self.X_test_transformation.to_csv(file_path, index = False)
            logging.info(f'Data is saved SUCCESSFULLY at:{out_dir}')
            return self
        
        except Exception as e:
            logging.error(f'ERROR while saving dataset')
            raise