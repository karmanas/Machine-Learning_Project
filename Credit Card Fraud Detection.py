#!/usr/bin/env python
# coding: utf-8

# ### Importing the Dependencies

# In[32]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


credit_card_data = pd.read_csv(r'C:\Users\Manas Ranjan Kar\Downloads\creditcard.csv\creditcard.csv')


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


# dataset informations
credit_card_data.info()


# In[6]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[7]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[34]:


fig, ax = plt.subplots(figsize = (6,4))
ax = sns.countplot(x= 'Class', data = credit_card_data)
plt.tight_layout()


# ## Dataset is highly unblanced
# 
# 

# In[8]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


credit_card_data.groupby('Class').mean()


# #### Under-Sampling

# #### Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

# #### Number of Fraudulent Transactions --> 492

# In[13]:


legit_sample = legit.sample(n=492)


# ### Concatenating two DataFrames

# In[14]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# #### Splitting the data into Features & Targets

# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[20]:


print(X)


# In[21]:


print(Y)


# #### Split the data into Training data & Testing Data
# 
# 

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# ## Model Training

# ### Logistic Regression

# In[24]:


model = LogisticRegression()


# In[25]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# ### Model Evaluation

# #### Accuracy Score

# In[26]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[27]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[28]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# ### Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)


# In[75]:


y_pred = classifier.predict(X_test)


# In[76]:


test_data_accuracy = accuracy_score(y_pred, Y_test)


# In[77]:


test_data_accuracy


# In[78]:


Y_pred = classifier.predict(X_train)


# In[79]:


test_data_accuracy = accuracy_score(Y_pred, Y_train)


# In[80]:


test_data_accuracy


# ### Decision Tree

# In[82]:


from sklearn.tree import DecisionTreeClassifier


# In[86]:


DT = DecisionTreeClassifier(max_depth = 8, criterion = 'entropy')
DT.fit(X_train, Y_train)
dt_yhat = DT.predict(X_test)


# In[87]:


print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(Y_test, dt_yhat)))


# In[88]:


dt_xhat = DT.predict(X_train)


# In[90]:


print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(Y_train, dt_xhat)))


# In[ ]:




