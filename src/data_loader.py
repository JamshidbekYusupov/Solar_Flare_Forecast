import pandas as pd
import numpy as np
import os
import logging


log_path = r'C:\Solar-Flare-Forecast\Logging\data_loaader.log'

logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
)

logging.info("Data Loading Started")

class data_loading:

    def __init__(self, path1, path2):
        self.df1 = pd.read_csv(path1)
        self.df2 = pd.read_csv(path2)
        self.df = None
        
    
    def join_dataset(self):

        try:

            self.df1.drop(columns=['start_time', 'end_date'], inplace=True)
            self.df2.drop(columns=['Unnamed: 0','Unnamed: 7'], inplace=True)
            new_columns = ['Class', 'date', 'Region', 'Start', 'Maximum', 'End', 'Year']
            self.df2.columns = new_columns

            self.df = pd.merge(self.df1, self.df2, how='inner')

            logging.info(f'Dataset has been loaded and merged')

            return self

        except Exception as e:
            logging.error(f'Error while merging or loading dataset')
            raise
    
    def date_handling(self):

        try:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['Year'] = self.df['date'].dt.year
            self.df['Month'] = self.df['date'].dt.month
            self.df['Day'] = self.df['date'].dt.day
            self.df = self.df.drop(columns='date')

            t = pd.to_datetime(self.df['Start'], format='%H:%M', errors='coerce')
            self.df['start_hour'] = t.dt.hour
            self.df['start_minute'] = t.dt.minute
            self.df = self.df.drop(columns='Start')

            t = pd.to_datetime(self.df['End'], format='%H:%M', errors='coerce')
            self.df['end_hour'] = t.dt.hour
            self.df['end_minute'] = t.dt.minute
            self.df = self.df.drop(columns='End')

            self.df.drop(columns='Maximum', inplace=True)

            logging.info('Date handling is done SUCCESSFULLY')
            return self
        
        except Exception as e:
            logging.error(r'ERROR while handling dates of dataset')
            raise
    
    def class_replcaing(self):

        try:

            #Classlarni bosh xarfini olish (M2.1 --> M)
            self.df['Class'] = self.df['Class'].astype(str).str[0]
            # self.df[self.df.Region == '-'] = 3576
            self.df.loc[self.df['Region'] == '-', 'Region'] = '3576'
            self.df.loc[self.df.speed == '-----', 'speed'] = '8040'
            self.df.loc[self.df.speed == 'DIM', 'speed'] = '8040'
            self.df.loc[self.df.speed == 'EP', 'speed'] = '8040'

            logging.info('Target values are handled SUCCESSFULLY')
            return self
        except Exception as e:
            logging.error('ERROR while hadling target values')
            raise
    
    def data_saving(self):
        try:

            out_path = r'C:\Solar-Flare-Forecast\Data\Raw_Data'
            os.makedirs(out_path, exist_ok=True)
            path = os.path.join(out_path, 'Raw_Data.csv')
            self.df.to_csv(path, index = False)
            logging.info('Data is saved SUCCESSFULLY')
            return self
        except Exception as e:
            logging.error('ERROR while saving dataset')
            raise

