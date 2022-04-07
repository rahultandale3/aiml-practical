import pandas as pd

import fsspec
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("E:\\pvg college study\\6th sem\\aiml\\practical\\aiml3.csv")
X = data.iloc[:,1:20]
y = data.iloc[:,-1]
#print(X,y)
bestfeatures = SelectKBest(score_func=chi2, k=13)
fit = bestfeatures.fit(X,y)
#print(fit)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#print (dfscores , dfcolumns)
#concat two data frames for better visulization

featurescores = pd.concat([dfcolumns , dfscores], axis =1 )
featurescores.columns = ['Specs','Score'] #naming the data frames columns
print(featurescores.nlargest(10,'Score'))


