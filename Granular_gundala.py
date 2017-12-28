##exercise 1
import pandas as pd
from pandas import read_csv
import numpy
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
df = pd.read_csv('IAvalues.txt') ## read csv file
## I think the first two columns are indexes of the dataframe
df.shape[0] - df.dropna().shape[0] ## determine number of NaN values
df['rootznaws'].unique() ## to understand if it is a class based or value based
df = df.fillna(df.mean(), inplace=True)
del df['Unnamed: 0']
del df['Unnamed: 0.1'] ## assuming these two are index columns

y = df['droughty'].values
X = df.values
##fill na with mean values

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
transformed_X = imputer.fit_transform(X)

# using an LDA model with K-fold cross validation (5 -fols)
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=5)
result1 = cross_val_score(model, transformed_X, y, cv=kfold,scoring='accuracy')

print(result1.mean())
##0.945563567964



###exercise2
pdf = pd.read_csv('muaggatt.txt')
newdf = pd.concat([df,pdf], axis='mukey')
y = newdf['droughty'].values
X = newdf.values
##fill na with mean values

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
transformed_X = imputer.fit_transform(X)

# using an LDA model with K-fold cross validation (5 -fols)
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=5)
result2 = cross_val_score(model, transformed_X, y, cv=kfold, scoring='accuracy')

print(result2.mean())
###0.945563567964




