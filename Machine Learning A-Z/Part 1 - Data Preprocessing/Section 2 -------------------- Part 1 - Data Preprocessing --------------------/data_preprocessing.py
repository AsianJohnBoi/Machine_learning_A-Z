# Data Preprocessing Template

# Importing the libraries ##Always add this
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset ####Always add this
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #takes all the rows, takes all columns but the last one.
y = dataset.iloc[:, 3].values #select the third column and their values.

#Taking care of missing data
from sklearn.impute import SimpleImputer#preprocess data set, imputer is allowing us to take care of missing data.
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', fill_value = None)
imputer = imputer.fit(X[:,1:3])#taking columns one and two
X[:,1:3] = imputer.transform(X[:, 1:3])
#running the code above will compute the average of the column to fill the missing data

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
columntransformer = ColumnTransformer([
    ("Countries", OneHotEncoder(categories='auto'), [0]) 
], remainder='passthrough')
X = columntransformer.fit_transform(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the training set and test set ####Always add this
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

#feature scaling ##Always add this
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

