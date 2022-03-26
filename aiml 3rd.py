import pandas as pd

# import fsspec

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

data = pd.read_csv("E:\pvg college study\6th sem\aiml\practical\text.xls")


X=data.iloc [:,1:20]

y = data.iloc [:,-1]

print (x, y)

bestfeatures = SelectKBest(score_func=chi2, k=13)

fit= bestfeatures.fit (x, y)

print (fit)

dfscores =pd.DataFrame (fit.scores_)

dfcolumns = pd.DataFrame (X.columns)

#concat two dataframes for better visualization

featureScores = pd.concat([dfcolumns, dfscores], axis=1)

featureScores.columns = ['Specs', 'Score'] #naming the dataframe columns

print (featureScores.nlargest (10, 'Score')) #print 10 best features
