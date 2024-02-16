#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score


# In[ ]:


data = pd.read_csv('AllDataStatefinalfinalfinal.csv')
data


# In[ ]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(data, groups=data['state_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=data.iloc[train_inds].copy()
test=data.iloc[test_inds].copy()


# In[ ]:


print(train.shape)
test.shape

#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])

#create x test and train data
x_test = test.iloc[:,-42:]
x_train = train.iloc[:,-42:]

#array of the group column
group_train=np.array(train['state_group'])


from xgboost import XGBRegressor


xgb = XGBRegressor()

grid_values = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]}
gkf=GroupKFold(n_splits=5).split(x_train, y_train,group_train )
grid_xgb = GridSearchCV(xgb, param_grid = grid_values, scoring = 'neg_root_mean_squared_error', cv=gkf)

result=grid_xgb.fit(x_train, y_train )
print(grid_xgb.best_params_)

