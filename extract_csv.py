import pandas as pd
import numpy as np

def extract_csv():
    all_data = pd.read_csv('data/train_binary.csv', header=0)
    data = all_data.values
    # print(data)
    # nb_data = data[:, 1:]
    # lables = data[:, 0:1]
    # print(data.dtype)
    p_data = [example for example in data[:, 0] == 1]
    print(p_data)



extract_csv()