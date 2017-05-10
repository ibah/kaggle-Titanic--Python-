# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:15:53 2017
@author: msiwek
Kaggle - Titanic competition

Topic: fast summary of best ideas to build a running model

Models: RandomForest
Ensembling: none
Tuning: GridSearch CV
CV: default 3-fold
Inspriation: https://www.kaggle.com/zlatankr/titanic-random-forest-82-78 by Zlatan Kremonic

Plan
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer
http://scikit-learn.org/stable/auto_examples/preprocessing/plot_function_transformer.html#sphx-glr-auto-examples-preprocessing-plot-function-transformer-py
# define a transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
def preprocessing(df):
    if 'PassengerId' in df:
        df.drop('PassengerId', axis=1, inplace=True)
    return df
pipeline = make_pipeline(
        FunctionTransformer(preprocessing))
pipeline.transform(train)
preprocessing(train).info()
train.info()
"""

print(__doc__)

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# I work interactively in Spyder on different platforms.
# The code below allows me to set the proper working directory for each platform.
import os
os.chdir(r'/home/michal/Dropbox/cooperation/_python/Titanic/models')
os.chdir(r'D:\data\Dropbox\cooperation\_python\Titanic\models')
os.chdir(r'G:\Dropbox\cooperation\_python\Titanic\models')

# loading data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train = train.drop('PassengerId', axis=1)
train_y = train.pop('Survived')
test_id = test.pop('PassengerId')

# preprocessing

def preprocess(df):
    # df = all_data
    df['Name_Length'] = df.Name.apply(len)
    df['Name_Title'] = df.Name.apply(lambda x: re.sub('(.*, )|(\..*)','',x))
    df['Age_Null_Flag'] = df.Age.apply(lambda x: 1 if pd.isnull(x) else 0)
    tmp = df.groupby(['Name_Title', 'Pclass']).Age
    df['Age'] = tmp.transform(lambda x: x.fillna(x.mean()))
    df.Age.fillna(df.Age.mean(), inplace=True)  # one record with NA remained, check why
    df['Fam_Size'] = np.where((df.SibSp + df.Parch) == 0 , 'Solo',
                     np.where((df.SibSp + df.Parch) <= 3,'Nuclear',
                               'Big'))
    df['Ticket_Letter'] = df.Ticket.apply(lambda x: str(x)[0])
    df['Ticket_Letter'] = np.where(
            df.Ticket_Letter.isin(['1', '2', '3', 'S', 'P', 'C', 'A']), df.Ticket_Letter,
                                   np.where(df.Ticket_Letter.isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
    df['Ticket_Len'] = df.Ticket.apply(len)
    df['Cabin_Letter'] = df.Cabin.apply(lambda x: str(x)[0])
    df['tmp'] = df.Cabin.apply(lambda x: str(x).split(' ')[-1][1:])
    df.tmp.replace('an', np.NaN, inplace = True)
    df['tmp'] = df.tmp.apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
    df['Cabin_number'] = pd.qcut(df.tmp,3)
    df = pd.concat((df, pd.get_dummies(df.Cabin_number, prefix = 'Cabin_number')), axis = 1)
    df['Embarked'] = df.Embarked.fillna('S')
    df.Fare.fillna(df.Fare.mean(), inplace = True)
    object_columns = ['Sex', 'Embarked', 'Ticket_Letter', 'Cabin_Letter', 'Name_Title', 'Fam_Size']
    dummies = pd.get_dummies(df[object_columns])
    df = pd.concat([df, dummies], axis=1)
    df.drop(object_columns, axis=1, inplace=True)
    df.drop(['Name','SibSp','Parch','Ticket','Cabin','tmp','Cabin_number'], axis=1, inplace=True)
    return df

train_n = len(train)
all_data = train.append(test)
tmp = preprocess(all_data)
train = tmp[:train_n]
test = tmp[train_n:]

# model tuning and fitting
rf = RandomForestClassifier(max_features='auto',
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)
params = {"min_samples_leaf" : [1, 5, 10],
         "min_samples_split" : [2, 5, 10, 15],
         "n_estimators": [500, 800, 1200]}
gs = GridSearchCV(rf, params, n_jobs=-1)
gs.fit(train, train_y)
gs.best_score_
gs.best_params_

# making predictions

pred = gs.predict(test)
pred = pd.DataFrame({'PassengerId': test_id, 'Survived': pred})
pred.to_csv('submission.csv', index = False)
