import sys
import os
sys.path.append(r'C:\Solar-Flare-Forecast')


path1 = r'C:\Solar-Flare-Forecast\Data\Scraped_Data\nasa_data.csv'
path2 = r'C:\Solar-Flare-Forecast\Data\Scraped_Data\sw_data.csv'

from src.data_loader import data_loading

dl = data_loading(path1=path1, path2=path2)

dl.join_dataset().date_handling().class_replcaing().data_saving()