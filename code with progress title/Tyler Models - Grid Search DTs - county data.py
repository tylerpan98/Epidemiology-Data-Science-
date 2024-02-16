#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


# In[109]:


#read in state data
df=pd.read_csv("Desktop/AllDataCountyfinalfinalfinal.csv")


# In[110]:


df.head()


# In[111]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['county_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[112]:


#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])


# In[113]:


#create x test and train data
x_test = test.iloc[:,-34:]
x_train = train.iloc[:,-34:]


# In[114]:


#array of the group column
group_train=np.array(train['county_group'])


# In[8]:


regtree = DecisionTreeRegressor(random_state=0)


# In[12]:


regtree.fit(x_train, y_train)
regtree.score(x_test, y_test)


# In[13]:


preds = regtree.predict(x_test)


# In[16]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, preds)))


# In[18]:


plt.scatter(y_test,preds)


# In[19]:


diff = y_test - preds
sns.displot(diff)


# In[ ]:


#do not run
rf = RandomForestRegressor()
grid_values = {'n_estimators' : [ 30, 40, 50,60,70,80,90], 'max_depth': [10,20,30,40,50,60], 'min_samples_leaf':[10,20,30,40,50]}
gkf=GroupKFold(n_splits=5).split(x_train, y_train,group_train )
grid_rf = GridSearchCV(rf, param_grid = grid_values, scoring = 'neg_root_mean_squared_error', cv=gkf)
result=grid_rf.fit(x_train, y_train )
print(grid_rf.best_params_)


# In[36]:


regtree = DecisionTreeRegressor(random_state=0)
param = {'max_leaf_nodes': [10,20, 30, 40, 50,60,70 ], 'min_samples_split': [2,3,4,5,6], 'splitter':['best','random']}
#param = {'max_leaf_nodes': [3,6,9,12,15], 'min_samples_split': [2,3,4,5,6]}
gkf=GroupKFold(n_splits=5).split(x_train, y_train,group_train )
grid = GridSearchCV(regtree, param_grid = param, scoring = 'neg_root_mean_squared_error', cv=gkf)
result=grid.fit(x_train, y_train )
print(grid.best_params_)


# In[37]:


#predicting test data using best parameters from grid search
tuned_preds = grid.predict(x_test)


# In[38]:


#evaluation metrics of predictions using best parameters from grid search

print("MAE:", mean_absolute_error(y_test, tuned_preds))
print("MSE:", mean_squared_error(y_test, tuned_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tuned_preds)))


# In[39]:


#ensemble modeling Extra Regressor Tree 


# In[40]:


xtree = ExtraTreeRegressor(random_state=0)


# In[41]:


xtree.fit(x_train, y_train)


# In[42]:


y_pred = xtree.predict(x_test)


# In[43]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", math.sqrt(mean_squared_error(y_test, y_pred)))


# In[44]:


xtree.score(x_test, y_test)


# In[51]:


xtree = ExtraTreeRegressor(random_state=0)
param = {'max_leaf_nodes': [10,20, 30, 40, 50,60,70 ], 'min_samples_split': [2,3,4,5,6], 'splitter':['best','random']}
#param = {'max_leaf_nodes': [3,6,9,12,15], 'min_samples_split': [2,3,4,5,6]}
gkf=GroupKFold(n_splits=5).split(x_train, y_train,group_train )
gridxtree = GridSearchCV(xtree, param_grid = param, scoring = 'neg_root_mean_squared_error', cv=gkf)
result=gridxtree.fit(x_train, y_train )
print(gridxtree.best_params_)


# In[52]:


#predicting test data using best parameters from grid search
tuned_pred_xtree = gridxtree.predict(x_test)


# In[53]:


#evaluation metrics of predictions using best parameters from grid search

print("MAE:", mean_absolute_error(y_test, tuned_pred_xtree))
print("MSE:", mean_squared_error(y_test, tuned_pred_xtree))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tuned_pred_xtree)))


# In[ ]:




