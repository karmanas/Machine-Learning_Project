#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r"C:\Users\Manas Ranjan Kar\Downloads\Customer Churn Prediction.csv")


# In[3]:


data.head(2)


# In[4]:


data.shape


# In[5]:


data.columns.values


# In[6]:


data.dtypes


# In[7]:


data.describe()


# ##### SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not propoer
# 
# ##### 75% customers have tenure less than 55 months
# 
# #### Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# In[8]:


data['Churn'].value_counts().plot(kind='barh', figsize=(10, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[9]:


len(data['Churn'])


# In[10]:


data['Churn'].value_counts()


# In[11]:


100*data['Churn'].value_counts()/len(data['Churn'])


# #### Data is highly imbalanced, ratio = 73:27
# #### So I analyse the data with other features while taking the target values separately to get some insights.

# In[12]:


data.info()#(verbose = True)


# In[13]:


missing = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# ## Data Cleaning

# #### Create a copy of base data for manupulation & processing

# In[14]:


telco_data = data.copy()


# #### Total Charges should be numeric amount.So convert it to numerical data type

# In[15]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()


# As we can see there are 11 missing values in TotalCharges column. Let's check these records

# In[16]:


telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# #### Missing Value Treatement

# Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

# In[17]:


telco_data.dropna(how = 'any', inplace = True)


# Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[18]:


print(telco_data['tenure'].max())


# In[19]:


labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# In[20]:


telco_data['tenure_group'].value_counts()


# #### Remove columns not required for processing

# In[21]:


telco_data.drop(columns= ['customerID','tenure'], axis=1,inplace=True)
telco_data.head(2)


# ## Data Exploration

# Plot distibution of individual predictors by churn

# ### Univariate Analysis

# In[22]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


# Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0

# In[23]:


telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)


# In[24]:


telco_data.head()


# Convert all the categorical variables into dummy variables

# In[25]:


telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()


# Relationship between Monthly Charges and Total Charges

# In[26]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# This graph shows that total Charges increase as Monthly Charges increase - as expected.

#  #### Churn by Monthly Charges and Total Charges

# In[27]:


Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Purple", shade = True)
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Mth, color="Green", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# Insight is Churn is high when Monthly Charges ar high

# #### Build a corelation of all predictors with 'Churn'

# In[28]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[29]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(), cmap="Paired")


# ### Bivariate Analysis

# In[30]:


new_df1_target0=telco_data.loc[telco_data["Churn"]==0]
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]


# In[31]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[32]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[33]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[34]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[35]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# In[36]:


telco_data_dummies.to_csv('tel_churn.csv')


# ## Model Building

# In[37]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# In[38]:


df=pd.read_csv("tel_churn.csv")
df.head(2)


# In[39]:


df=df.drop('Unnamed: 0',axis=1)


# In[41]:


x=df.drop('Churn',axis=1)
x.head()


# In[44]:


y=df['Churn']
y.head()


# #### Train Test Split

# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# ### Decision Tree Classifier

# In[46]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[47]:


model_dt.fit(x_train,y_train)


# In[48]:


y_pred=model_dt.predict(x_test)
y_pred


# In[49]:


model_dt.score(x_test,y_test)


# In[50]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# As you can see that the accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
# Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
# Hence, moving ahead to call SMOTEENN (UpSampling + ENN)

# In[53]:


sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(x,y)


# In[54]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)


# In[55]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[56]:


model_dt_smote.fit(xr_train,yr_train)


# In[57]:


yr_predict = model_dt_smote.predict(xr_test)


# In[59]:


model_score_r = model_dt_smote.score(xr_test, yr_test)


# In[61]:


print(model_score_r)


# In[63]:


print(metrics.classification_report(yr_test, yr_predict))


# In[64]:


print(metrics.confusion_matrix(yr_test, yr_predict))


# ### Random Forest Classifier

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[67]:


model_rf.fit(x_train,y_train)


# In[68]:


y_pred=model_rf.predict(x_test)


# In[69]:


model_rf.score(x_test,y_test)


# In[70]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[72]:


sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x,y)


# In[73]:


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)


# In[74]:


model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[75]:


model_rf_smote.fit(xr_train1,yr_train1)


# In[76]:


yr_predict1 = model_rf_smote.predict(xr_test1)


# In[77]:


model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)


# In[78]:


print(model_score_r1)


# In[79]:


print(metrics.classification_report(yr_test1, yr_predict1))


# In[80]:


print(metrics.confusion_matrix(yr_test1, yr_predict1))


# In[ ]:





# ### Pickling the model

# In[81]:


import pickle


# In[82]:


filename = 'model.sav'


# In[83]:


pickle.dump(model_rf_smote, open(filename, 'wb'))


# In[84]:


load_model = pickle.load(open(filename, 'rb'))


# In[85]:


model_score_r1 = load_model.score(xr_test1, yr_test1)


# In[87]:


model_score_r1


# In[ ]:




