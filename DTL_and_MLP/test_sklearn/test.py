import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv("iris.csv")

inputs = data.drop('species', axis = 1)
target = data['species']

kf = KFold(n_splits=10)
kf.get_n_splits(inputs)

print(kf)

for train_index, test_index in kf.split(inputs):
    print("TRAIN:", train_index, "TEST:", test_index)
