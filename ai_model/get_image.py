import pandas as pd
import random



def get_image_path():
    data = pd.read_csv('test_data.csv')
    idx = random.randint(0, len(data))
    test_data = (data.iloc[idx]['Path'], data.iloc[idx]['Label'])
    return test_data

