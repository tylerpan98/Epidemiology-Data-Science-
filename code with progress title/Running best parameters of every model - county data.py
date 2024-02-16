#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.svm import SVR
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn import model_selection


# In[2]:


#read in state data
df=pd.read_csv("Desktop/AllDataCountyfinalfinalfinal.csv")


# In[3]:


features = df.iloc[:,[12,13,14, 20,21,22,23,24,25,26,27,28,29,30,31,32, 41,42,43,44,45,46]]
features.head()


# In[4]:



correlations = features.corr()

fig, ax = plt.subplots(figsize=(15,16))
sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .80})
plt.savefig('Downloads/county_corr_matrix.png')
plt.show();
    


# In[6]:


#create split using group shuffle to keep state together
gss=GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)

#column to group by
split = gss.split(df, groups=df['county_group'])

#splitting testing and training indexes
train_inds, test_inds = next(split)

#test and train dfs split by state
train=df.iloc[train_inds].copy()
test=df.iloc[test_inds].copy()


# In[7]:


#create y test & train data
y_test = np.array(test['diff_weeks'])
y_train = np.array(train['diff_weeks'])

#create x test and x train data
x_test = test.iloc[:,-34:]
x_train = train.iloc[:,-34:]

x_test_no_correlated = x_test.drop(['Mutation Fitness', 'Number of business establishments', 'Total people in poverty (%)', 'Population_Density'], axis=1)
x_train_no_correlated = x_train.drop(['Mutation Fitness', 'Number of business establishments', 'Total people in poverty (%)','Population_Density'], axis=1)


# In[8]:


#array of the group column
group_train=np.array(train['county_group'])


# In[9]:


#baseline model decision tree
regtree = DecisionTreeRegressor(random_state=0)
regtree.fit(x_train_no_correlated, y_train)
tree_preds = regtree.predict(x_test_no_correlated)


# In[10]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, tree_preds))
print("MSE:", mean_squared_error(y_test, tree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, tree_preds)))
print("Score", regtree.score(x_test_no_correlated, y_test))


# In[11]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(regtree, x_train_no_correlated, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[12]:


#ensemble modeling Extra Regressor Tree 


# In[13]:


xtree = ExtraTreeRegressor(random_state=0)
xtree.fit(x_train_no_correlated, y_train)
xtree_preds = xtree.predict(x_test_no_correlated)


# In[14]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xtree_preds))
print("MSE:", mean_squared_error(y_test, xtree_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_preds)))
print("Score", xtree.score(x_test_no_correlated, y_test))


# In[15]:


xtree_tuned = ExtraTreeRegressor(max_leaf_nodes= 60, min_samples_split= 3, splitter= 'best')
xtree_tuned.fit(x_train_no_correlated, y_train)
xtree_tuned_preds = xtree_tuned.predict(x_test_no_correlated)


# In[16]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(xtree_tuned, x_train_no_correlated, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[17]:


#evaluation metrics of model on test data
print("MAE:", mean_absolute_error(y_test, xtree_tuned_preds))
print("MSE:", mean_squared_error(y_test, xtree_tuned_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xtree_tuned_preds)))
print("Score", xtree_tuned.score(x_test_no_correlated, y_test))


# In[18]:


#random forest


# In[19]:


rf = RandomForestRegressor()
rf.fit(x_train_no_correlated, y_train)
rf_preds=rf.predict(x_test_no_correlated)


# In[20]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("MSE:", mean_squared_error(y_test, rf_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds)))
print("Score", rf.score(x_test_no_correlated, y_test))


# In[21]:


rf_tuned=RandomForestRegressor(max_depth = 30, n_estimators= 70)
rf_tuned.fit(x_train_no_correlated, y_train)
rf_preds_tuned = rf_tuned.predict(x_test_no_correlated)


# In[22]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(rf_tuned, x_train_no_correlated, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[23]:


#evaluation metrics of model on test data
print("MAE:", mean_absolute_error(y_test, rf_preds_tuned))
print("MSE:", mean_squared_error(y_test, rf_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, rf_preds_tuned)))
print("Score", rf_tuned.score(x_test_no_correlated, y_test))


# In[24]:


diff_not_tuned = y_test - rf_preds
plt.hist(diff_not_tuned, bins=13)


# In[25]:


diff = y_test - rf_preds_tuned
plt.hist(diff, bins=13)


# In[26]:


#feature importance for rf model
importances = rf_tuned.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train_no_correlated.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train_no_correlated.shape[1]), x_train_no_correlated.columns[sorted_indices], rotation=90)
plt.show()


# In[27]:


print(rf_tuned.feature_importances_)
print(sorted_indices)


# In[26]:


#XGBoost


# In[28]:


xgb = XGBRegressor()
xgb.fit(x_train_no_correlated, y_train)
xgb_preds=xgb.predict(x_test_no_correlated)


# In[29]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds))
print("MSE:", mean_squared_error(y_test, xgb_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds)))
print("Score", xgb.score(x_test_no_correlated, y_test))


# In[30]:


xgb_tuned = XGBRegressor(colsample_bytree = 1, gamma = 2, max_depth = 3, min_child_weight = 1, subsample = 1)
xgb_tuned.fit(x_train_no_correlated, y_train)
xgb_preds_tuned=xgb_tuned.predict(x_test_no_correlated)


# In[31]:


#cross validation of metrics using training data
gkf2=GroupKFold(n_splits=10)
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(xgb_tuned, x_train_no_correlated, y_train, scoring = scores, cv=gkf2, groups=group_train)
print(scores)


# In[32]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, xgb_preds_tuned))
print("MSE:", mean_squared_error(y_test, xgb_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, xgb_preds_tuned)))
print("Score", xgb_tuned.score(x_test_no_correlated, y_test))


# In[33]:


#feature importance for xgb tuned model
importances = xgb_tuned.feature_importances_
sorted_indices = np.argsort(importances)[::-1] 
plt.title('Feature Importance')
plt.bar(range(x_train_no_correlated.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(x_train_no_correlated.shape[1]), x_train_no_correlated.columns[sorted_indices], rotation=90)
plt.show()


# In[34]:


print(xgb_tuned.feature_importances_)
print(sorted_indices)


# In[33]:


#svm


# In[59]:


scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train_no_correlated)
x_test_scaled = scale.fit_transform(x_test_no_correlated)


# In[60]:


svr=SVR()
svr.fit(x_train_scaled, y_train)
svr_preds = svr.predict(x_test_scaled)


# In[61]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds))
print("MSE:", mean_squared_error(y_test, svr_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds)))
print("Score", svr.score(x_test_scaled, y_test))


# In[62]:


svr_tuned = SVR(kernel = 'poly', gamma = 'scale', degree = 4)
svr_tuned.fit(x_train_scaled, y_train)
svr_pred_tuned = svr_tuned.predict(x_test_scaled)


# In[63]:


#cross validation of results using best parameters on training data
gkf2=GroupKFold(n_splits=10)
pipe = make_pipeline(scale, SVR(degree = 1, gamma = 'scale', kernel = 'poly'))
scores = ['neg_root_mean_squared_error','neg_mean_squared_error', 'neg_mean_absolute_error' ]
scores=model_selection.cross_validate(pipe, x_train_no_correlated, y_train, scoring =scores, cv=gkf2, groups=group_train)
print(scores)


# In[70]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_pred_tuned))
print("MSE:", mean_squared_error(y_test, svr_pred_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_pred_tuned)))
print("Score", svr_tuned.score(x_test_scaled, y_test))


# In[260]:


#trying features selection using random forest for SVM


# In[261]:


from sklearn.feature_selection import SelectFromModel


# In[262]:


sel = SelectFromModel(rf, max_features = 20)
sel.fit(x_train_no_correlated, y_train)


# In[263]:


sel.get_support()


# In[264]:


selected_feat= x_train_no_correlated.columns[(sel.get_support())]
len(selected_feat)


# In[265]:


print(selected_feat)


# In[266]:


x_train_features = x_train_no_correlated[["S1 Mutations", "21A (Delta)", "Number of residents",
       "1st_Quarter_first", "2nd_Quarter_first", "Omicron",
       "% Fair or Poor Health", 'Unempl rate']]


# In[267]:


x_test_feature_scale = x_test_no_correlated[["S1 Mutations", "21A (Delta)", "Number of residents",
       "1st_Quarter_first", "2nd_Quarter_first", "Omicron",
       "% Fair or Poor Health",'Unempl rate']]


# In[268]:


x_train_feature_scaled = scale.fit_transform(x_train_features)
x_test_feature_scaled = scale.fit_transform(x_test_feature_scale)


# In[269]:


svr_tuned = SVR(kernel = 'poly', gamma = 'scale', degree = 4)
svr_tuned.fit(x_train_feature_scaled, y_train)
svr_preds_tuned = svr_tuned.predict(x_test_feature_scaled)


# In[270]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_test, svr_preds_tuned))
print("MSE:", mean_squared_error(y_test, svr_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_test, svr_preds_tuned)))


# In[40]:


test['DT_preds'] = tree_preds.tolist()
test['xtree_preds_tuned']= xtree_tuned_preds.tolist()
test['svr_preds_tuned'] = svr_pred_tuned.tolist()
test['xgb_preds_tuned'] = xgb_preds_tuned.tolist()
test['rf_preds_tuned'] = rf_preds_tuned.tolist()


# In[41]:


test.to_csv('Downloads/DataWithPredictionsCounty.csv')


# In[42]:


#run best tuned model on training data


# In[43]:


#xtree
xtree_train_tuned_preds = xtree_tuned.predict(x_train_no_correlated)


# In[49]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_train, xtree_train_tuned_preds))
print("MSE:", mean_squared_error(y_train, xtree_train_tuned_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_train, xtree_train_tuned_preds)))
print("Score", xtree_tuned.score(x_train_no_correlated, y_train))


# In[50]:


#baseline decision tree
tree_train_preds = regtree.predict(x_train_no_correlated)


# In[51]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_train, tree_train_preds))
print("MSE:", mean_squared_error(y_train, tree_train_preds))
print("RMSE:", math.sqrt(mean_squared_error(y_train, tree_train_preds)))
print("Score", regtree.score(x_train_no_correlated, y_train))


# In[55]:


#random forest
rf_train_preds_tuned = rf_tuned.predict(x_train_no_correlated)


# In[56]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_train, rf_train_preds_tuned))
print("MSE:", mean_squared_error(y_train, rf_train_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_train, rf_train_preds_tuned)))
print("Score", rf_tuned.score(x_train_no_correlated, y_train))


# In[57]:


#XGBoost
xgb_train_preds_tuned=xgb_tuned.predict(x_train_no_correlated)


# In[58]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_train, xgb_train_preds_tuned))
print("MSE:", mean_squared_error(y_train, xgb_train_preds_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_train, xgb_train_preds_tuned)))
print("Score", xgb_tuned.score(x_train_no_correlated, y_train))


# In[65]:


#SVR
svr_train_pred_tuned = svr_tuned.predict(x_train_scaled)


# In[66]:


#evaluation metrics
print("MAE:", mean_absolute_error(y_train, svr_train_pred_tuned))
print("MSE:", mean_squared_error(y_train, svr_train_pred_tuned))
print("RMSE:", math.sqrt(mean_squared_error(y_train, svr_train_pred_tuned)))
print("Score", svr_tuned.score(x_train_scaled, y_train))


# In[67]:



train['xtree_preds_tuned']= xtree_train_tuned_preds.tolist()
train['svr_preds_tuned'] = svr_train_pred_tuned.tolist()
train['xgb_preds_tuned'] = xgb_train_preds_tuned.tolist()
train['rf_preds_tuned'] = rf_train_preds_tuned.tolist()


# In[68]:


train.to_csv('Downloads/TrainDataWithPredictionsCounty.csv')


# In[ ]:




