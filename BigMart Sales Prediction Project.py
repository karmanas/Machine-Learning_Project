#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install klib')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_data = pd.read_csv(r"C:\Users\Manas Ranjan Kar\OneDrive\Documents\BigMart Sales Data\Train (1).csv") 
test_data = pd.read_csv(r"C:\Users\Manas Ranjan Kar\OneDrive\Documents\BigMart Sales Data\Test (1).csv")


# In[4]:


train_data.head()


# In[5]:


train_data.isnull().sum()


# In[6]:


test_data.isnull().sum()


# In[7]:


train_data.info()


# In[8]:


train_data.describe()


# In[9]:


train_data['Item_Weight'].describe()


# In[10]:


train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(),inplace = True)
test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(),inplace = True)


# In[11]:


train_data.isnull().sum()


# In[12]:


train_data['Item_Weight'].describe()


# In[13]:


train_data['Outlet_Size'].value_counts()


# In[14]:


train_data['Outlet_Size'].mode()


# In[15]:


train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0],inplace=True)
test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0],inplace=True)


# In[16]:


train_data.isnull().sum()


# In[17]:


train_data.drop(['Item_Identifier','Outlet_Identifier'],axis = 1, inplace = True)
test_data.drop(['Item_Identifier','Outlet_Identifier'],axis = 1, inplace = True)


# In[18]:


train_data.head()


# In[19]:


plt.figure(figsize=(6,6))
sns.distplot(train_data["Item_MRP"])
plt.show()


# In[20]:


plt.figure(figsize=(6,6))
sns.distplot(train_data["Item_Weight"])
plt.show()


# In[21]:


plt.figure(figsize=(6,6))
sns.countplot(x= "Outlet_Establishment_Year" ,data = train_data)
plt.show()


# In[22]:


plt.figure(figsize = (5,5))
sns.countplot(x = "Item_Fat_Content" , data = train_data)
plt.show()


# In[23]:


plt.figure(figsize=(24,8))
sns.countplot(x= "Item_Type" ,data = train_data)
plt.show()


# In[24]:


train_data['Item_Fat_Content'].value_counts()


# In[25]:


train_data.replace({'Item_Fat_Content' : {'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace = True)


# In[26]:


train_data['Item_Fat_Content'].value_counts()


# In[ ]:





# In[ ]:





# ## EDA PART

# In[ ]:





# ### 2. EDA WITH pandas-profilling

# In[29]:


import pandas_profiling


# In[30]:


from pandas_profiling import ProfileReport


# In[31]:


profile = ProfileReport(train_data, title="Pandas Profiling Report")


# In[32]:


profile


# In[ ]:





# In[33]:


plt.figure(figsize = (10,8))
sns.heatmap(train_data.corr(),annot=True)
plt.show()


# In[ ]:





# ### 3. EDA WITH klib LIBRARY

# In[34]:


import klib


# In[35]:


klib.cat_plot(train_data)


# In[36]:


klib.corr_mat(train_data)


# In[37]:


klib.corr_plot(train_data)


# In[38]:


klib.dist_plot(train_data)


# In[39]:


klib.missingval_plot(train_data)


# In[40]:


klib.data_cleaning(train_data).head()


# In[41]:


klib.clean_column_names(train_data).head()


# In[42]:


train_data.info()


# In[43]:


train_data = klib.convert_datatypes(train_data)
train_data.info()


# In[44]:


klib.pool_duplicate_subsets(train_data)


# In[ ]:





# ### Preprocessing task before Model Building

# In[45]:


train_data.head()


# In[ ]:





# ### 1. Label Encoding

# In[46]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()


# In[47]:


train_data['item_fat_content'] = le.fit_transform(train_data['item_fat_content'])
train_data['item_type'] = le.fit_transform(train_data['item_type'])
train_data['outlet_size'] = le.fit_transform(train_data['outlet_size'])
train_data['outlet_location_type'] = le.fit_transform(train_data['outlet_location_type'])
train_data['outlet_type'] = le.fit_transform(train_data['outlet_type'])


# In[48]:


train_data.head()


# In[ ]:





# ### 2. splitting data into train and test

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x = train_data.drop('item_outlet_sales', axis = 1)
y = train_data['item_outlet_sales'] 


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.2)


# In[52]:


x_train


# ### 3. Standarization

# In[53]:


x.describe()


# #### Note: The data is in very big range so we need to keep them in a small range or we can say that we need to keep them close by normalising data so that our model converge better ::-->> Feature Scaling

# In[54]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[55]:


x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)
x_train_std


# In[56]:


x_test_std


# In[57]:


y_train


# In[58]:


y_test


# In[ ]:





# In[59]:


import joblib


# In[60]:


joblib.dump(sc,r'C:\Users\Manas Ranjan Kar\PROJECTS\BigMart Sales Prediction Project\models\sc.sav')


# ## Model Building

# In[ ]:





# In[61]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[62]:


lr.fit(x_train_std,y_train)


# In[63]:


x_test.head()


# In[64]:


y_pred_lr=lr.predict(x_test_std)


# In[65]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,accuracy_score


# In[66]:


print(r2_score(y_test,y_pred_lr))
print(mean_absolute_error(y_test,y_pred_lr))
print(np.sqrt(mean_squared_error(y_test,y_pred_lr)))


# In[ ]:





# In[67]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)


# In[68]:


y_pred_svr = regressor.predict(x_test)


# In[69]:


print(r2_score(y_test,y_pred_svr))
print(mean_absolute_error(y_test,y_pred_svr))
print(np.sqrt(mean_squared_error(y_test,y_pred_svr)))


# In[ ]:





# In[70]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 100, random_state = 0)
RF.fit(x_train, y_train)


# In[71]:


y_pred_RF = RF.predict(x_test)


# In[72]:


print(r2_score(y_test,y_pred_RF))
print(mean_absolute_error(y_test,y_pred_RF))
print(np.sqrt(mean_squared_error(y_test,y_pred_RF)))


# In[73]:


joblib.dump(RF,r'C:\Users\Manas Ranjan Kar\PROJECTS\BigMart Sales Prediction Project\models\RF.sav')


# In[ ]:





# ## Hyper Parameter Tuning

# In[74]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
grid = dict(n_estimators=n_estimators)

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

grid_search_forest.fit(x_train_std, y_train)

# summarize results
print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[75]:


grid_search_forest.best_params_


# In[76]:


grid_search_forest.best_score_


# In[77]:


Y_pred_rf_grid=grid_search_forest.predict(x_test_std)


# In[78]:


r2_score(y_test,Y_pred_rf_grid)


# In[ ]:





# In[ ]:




