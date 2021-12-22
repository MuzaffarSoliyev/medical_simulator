import pandas as pd
import random


def get_image_path(mode_type):
    if mode_type == 1:
        data = pd.read_csv('test_data.csv')
    else:
        data = pd.read_csv('brain_test.csv')
    idx = random.randint(0, len(data))
    test_data = (data.iloc[idx]['Path'], data.iloc[idx]['Label'])
    return test_data
