# Data Preprocessing Template

# Importing the libraries ##Always add this
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #takes all the rows, takes all columns but the last one.
y = dataset.iloc[:, 3].values #select the third column and their values.

#Splitting the dataset into the training set and test set ####Always add this
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

#feature scaling ##Always add this
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

