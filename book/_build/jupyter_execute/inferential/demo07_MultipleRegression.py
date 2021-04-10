#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yellowbrick')


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[39]:


house_data = pd.read_csv('datasets/house_data_processed.csv')

house_data.head()


# In[40]:


house_data.shape


# In[41]:


target = house_data['price']

features = house_data.drop('price', axis=1)


# In[42]:


features.columns


# In[43]:


from yellowbrick.target import FeatureCorrelation

feature_names = list(features.columns)


# In[44]:


visualizer = FeatureCorrelation(labels = feature_names)

visualizer.fit(features, target)

visualizer.poof()


# ### Select K-Best features to predict price of houses

# In[75]:


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression


# In[76]:


select_univariate = SelectKBest(f_regression, k=5).fit(features, target)


# In[73]:


features_mask = select_univariate.get_support()

features_mask


# In[74]:


selected_columns = features.columns[features_mask]

selected_columns


# In[49]:


selected_features = features[selected_columns]

selected_features.head()


# In[50]:


selected_features.describe()


# In[51]:


from sklearn.preprocessing import scale

X = pd.DataFrame(data=scale(selected_features), columns=selected_features.columns)

y = target


# In[52]:


X.describe()


# In[53]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2)


# In[55]:


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)


# In[56]:


y_pred = linear_regression.predict(X_test)


# In[58]:


df = pd.DataFrame({'test': y_test, 'Predicted': y_pred})

df.head()


# In[59]:


from sklearn.metrics import r2_score

score = linear_regression.score(X_train, y_train)
r2score = r2_score(y_test, y_pred)


# In[60]:


print('Score: {}'.format(score))
print('r2_score: {}'.format(r2score))


# In[61]:


linear_regression.coef_


# In[62]:


linear_regression.intercept_


# In[63]:


import statsmodels.api as sm


# ### Fit and predict

# In[64]:


X_train = sm.add_constant(X_train)


# In[65]:


model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_train)

print(model.summary())


# In[68]:


linear_regression.intercept_


# In[67]:


linear_regression.coef_


# In[ ]:




