import pandas as pd
from Agglomerative import *
data = pd.read_csv("iris.csv")
target = data.species
inputs = data.drop('species', axis = 1)
agglo = Agglomerative(n_clusters=2)
agglo.fit(inputs)
