#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/harlfoxem/housesalesprediction

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


house_data = pd.read_csv('datasets/kc_house_data.csv')

house_data.head()


# In[4]:


house_data.drop(['id', 'lat', 'long', 'zipcode'], inplace=True, axis=1)

house_data.head()


# In[5]:


house_data.info()


# In[6]:


house_data['date'] = pd.to_datetime(house_data['date'])
house_data['house_age'] = house_data['date'].dt.year - house_data['yr_built']

house_data.drop('date', inplace=True, axis=1)
house_data = house_data.drop('yr_built', axis=1)


# In[7]:


house_data['renovated'] = house_data['yr_renovated'].apply(lambda x:0 if x == 0 else 1)

house_data.drop('yr_renovated', inplace=True, axis=1)


# In[11]:


house_data[['renovated', 'house_age']].sample(10)


# In[12]:


house_data.to_csv('datasets/house_data_processed.csv', index=False)


# In[28]:


sns.lmplot('sqft_living', 'price', house_data)


# In[30]:


sns.lmplot('house_age', 'price', house_data)


# In[31]:


sns.lmplot('floors', 'price', house_data)


# ### Scaling dataset and one feature for simple linear regression

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[14]:


X = house_data[['sqft_living']]

y = house_data['price']


# In[15]:


X.head()


# In[16]:


y.head()


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2)


# In[33]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


# In[34]:


X_test = scaler.transform(X_test)


# In[37]:


linear_regression = LinearRegression()

model = linear_regression.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[38]:


df = pd.DataFrame({'test': y_test, 'predicted': y_pred})

df.sample(10)


# ### Regression line

# In[40]:


plt.figure(figsize=(10, 8))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c='r')

plt.show()


# In[42]:


print("Training score : ", linear_regression.score(X_train, y_train))


# In[43]:


from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)

print("Testing score : ", score)


# In[44]:


theta_0 = linear_regression.coef_
theta_0


# In[45]:


intercept = linear_regression.intercept_
intercept


# In[47]:


plt.subplots(figsize=(14,8))

plt.plot(y_pred, label="Prediction")
plt.plot(y_test.values, label="Actual")

plt.legend()

plt.show()


# In[48]:


import statsmodels.api as sm


# ### Adding a constant

# In[52]:


X_train[:5]


# In[54]:


X_train = sm.add_constant(X_train)

X_train[:5]


# In[56]:


model = sm.OLS(y_train, X_train).fit()

print(model.summary())


# In[57]:


theta_0, intercept


# In[ ]:




