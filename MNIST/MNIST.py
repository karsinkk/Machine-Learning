import numpy as np
import pandas as pd


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("./Data/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("./Data/test.csv").values

