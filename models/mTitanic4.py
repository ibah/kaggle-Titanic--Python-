#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:03:06 2017
@author: michal
Kaggle - Titanic competition

Topic: fast summary of best ideas to build a running model

Models: RandomForest, XGBoost
Ensembling: none -> XGBoost?
Tuning: GridSearch CV
CV: ?

Preprocessing: ?

Conclusions: ?

Plan: ?


http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer
http://scikit-learn.org/stable/auto_examples/preprocessing/plot_function_transformer.html#sphx-glr-auto-examples-preprocessing-plot-function-transformer-py
"""

print(__doc__)

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt


# I usually work interactively in Spyder on different platforms.
# The code below allows me to set the proper working directory for each platform.
import os
os.chdir(r'/home/michal/Dropbox/cooperation/_python/Titanic/models')
os.chdir(r'D:\data\Dropbox\cooperation\_python\Titanic\models')
os.chdir(r'G:\Dropbox\cooperation\_python\Titanic\models')

# Loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Preprocessing
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






