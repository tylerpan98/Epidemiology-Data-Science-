#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from xgboost import plot_importance
from matplotlib import pyplot


# In[100]:


#read in state data
df=pd.read_csv("Desktop/AllDataCountyfinalfinalfinal.csv")


# In[101]:


df.head()


# In[102]:


df.shape


# In[103]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['county_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[104]:


print(train.shape)
test.shape


# In[105]:


#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])


# In[106]:


#create x test and train data
x_test = test.iloc[:,-34:]
x_train = train.iloc[:,-34:]


# In[107]:


#array of the group column
group_train=np.array(train['county_group'])


# In[85]:


#grid search for best parameters using group kfold to keep the states grouped together in testing and training.
#score to minimize is RMSE

rf = RandomForestRegressor()
grid_values = {'n_estimators' : [ 30, 40, 50,60,70,80,90], 'max_depth': [10,20,30,40,50,60], 'min_samples_leaf':[10,20,30,40,50]}
gkf=GroupKFold(n_splits=5).split(x_train, y_train,group_train )
grid_rf = GridSearchCV(rf, param_grid = grid_values, scoring = 'neg_root_mean_squared_error', cv=gkf)
result=grid_rf.fit(x_train, y_train )
print(grid_rf.best_params_)


# In[86]:


#predicting test data using best parameters from grid search
tuned_preds = grid_rf.predict(x_test)


# In[87]:


#evaluation metrics of predictions using best parameters from grid search

print("MAE:", mean_absolute_error(y_test, tuned_preds))
print("MSE:", mean_squared_error(y_test, tuned_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tuned_preds)))
#mape = np.mean(np.abs((y_test - tuned_preds) / y_test))


# In[108]:


#model with best parameters
rf = RandomForestRegressor(max_depth = 30, n_estimators = 70)


# In[109]:


rf.fit(x_train, y_train)


# In[110]:


preds=rfnone.predict(x_test)


# In[111]:


#evaluation metrics of predictions using best parameters from grid search

print("MAE:", mean_absolute_error(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, preds)))
#mape = np.mean(np.abs((y_test - preds) / y_test))
#print("MAPE", mape)


# In[112]:


#cross validation of results using best parameters on training data
gkf2=GroupKFold(n_splits=10)
rftuned = RandomForestRegressor(max_depth = 30, n_estimators = 70)
scores=cross_val_score(rftuned, x_train, y_train, scoring ='neg_root_mean_squared_error', cv=gkf2, groups=group_train)
print(scores)


# In[93]:


#feature importance
#rf.feature_importances_
#sorted_indices = np.argsort(importances)[::-1] 
#print(rfnone.feature_importances_)
#print(sorted_indices)


# In[94]:


#print(x_train.columns[sorted_indices])


# In[113]:


#feature importace chart
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)

plt.show()


# In[97]:


#add predictions to test df
test['preds'] = preds.tolist()


# In[98]:


#export to csv
test.to_csv('Downloads/rferrors.csv')


# In[ ]:




