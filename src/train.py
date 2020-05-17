#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline


# global vars
result = { 'D': 1, 'H': 2, 'A': 3}

# read the data
df = pd.read_csv('datasets/Bundesliga_Results.csv')

# remove cols
cleaned_df = df.copy().dropna().drop(['Div', 'Season', 'Date'], axis=1)
cleaned_df['FTR'] = cleaned_df['FTR'].map(result)
cleaned_df['HTR'] = cleaned_df['HTR'].map(result)

le = LabelEncoder()
le.fit(cleaned_df['HomeTeam'].unique())

cleaned_df['HomeTeam'] = le.transform(cleaned_df['HomeTeam'])
cleaned_df['AwayTeam'] = le.transform(cleaned_df['AwayTeam'])

# shuffle
shuffeled_df = sklearn.utils.shuffle(cleaned_df)

# drop unneeded
X = shuffeled_df.drop(['FTR', 'FTHG', 'FTAG', 'HTR'], axis=1).values

y = shuffeled_df['FTR'].values

test_size = 100

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]


print("Train data")
clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto',probability=True))
clf.fit(X_train, y_train)

print("Test data")
clf.score(X_test, y_test)

print("Save everthing")
from joblib import dump, load
dump(clf, 'model.sklearn')
dump(le, 'team_encoder')
dump(result, 'result_dict')
