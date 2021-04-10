#!/usr/bin/env python
# coding: utf-8

# ## Bivariate Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #### Import Data

# In[2]:


automobile_data = pd.read_csv('datasets/auto-mpg.csv')

automobile_data.head(10)


# In[3]:


automobile_data.shape


# In[4]:


automobile_data = automobile_data.replace('?', np.nan)

automobile_data = automobile_data.dropna()


# In[5]:


automobile_data.shape


# In[6]:


automobile_data.drop(['origin', 'car name'], axis=1, inplace=True)

automobile_data.head()


# In[7]:


automobile_data['model year'] = '19' + automobile_data['model year'].astype(str)

automobile_data.sample(5)


# In[8]:


import datetime

automobile_data['age'] = datetime.datetime.now().year -     pd.to_numeric(automobile_data['model year'])


# In[9]:


automobile_data.drop(['model year'], axis=1, inplace=True)

automobile_data.sample(5)


# In[10]:


automobile_data.dtypes


# In[11]:


automobile_data['horsepower'] = pd.to_numeric(automobile_data['horsepower'], errors='coerce')

automobile_data.describe()


# In[12]:


automobile_data.to_csv('datasets/automobile_data_processed.csv', index=False)


# In[13]:


get_ipython().system('ls datasets/')


# ### Bivariate data analysis

# In[14]:


automobile_data.plot.scatter(x='displacement', y='mpg', figsize=(12, 8))

plt.show()


# In[15]:


automobile_data.plot.scatter(x='horsepower', y='mpg', figsize=(12, 8))

plt.show()


# In[16]:


automobile_data.plot.hexbin(x='acceleration', y='mpg', gridsize=20, figsize=(12, 8))

plt.show()


# In[17]:


automobile_grouped = automobile_data.groupby(['cylinders']).mean()[['mpg', 'horsepower', 
                                                                    'acceleration', 'displacement']]

automobile_grouped


# In[18]:


automobile_grouped.plot.line(figsize=(12, 8))

plt.show()


# ### Multivariate data analysis

# In[19]:


fig, ax = plt.subplots()

automobile_data.plot(x='horsepower', y='mpg', 
                     kind='scatter', s=60, c='cylinders', 
                     cmap='magma_r', title='Automobile Data', 
                     figsize=(12, 8), ax=ax)

plt.show()


# In[20]:


fig, ax = plt.subplots()

automobile_data.plot(x='acceleration', y='mpg', 
                     kind='scatter', s=60, c='cylinders', 
                     cmap='magma_r', title='Automobile Data', 
                     figsize=(12, 8), ax=ax)

plt.show()


# In[21]:


fig, ax = plt.subplots()

automobile_data.plot(x='displacement', y='mpg', 
                     kind='scatter', s=60, c='cylinders', 
                     cmap='viridis', title='Automobile Data', 
                     figsize=(12, 8), ax=ax)

plt.show()


# In[22]:


automobile_data['acceleration'].cov(automobile_data['mpg'])


# In[23]:


automobile_data['acceleration'].corr(automobile_data['mpg'])


# In[24]:


automobile_data['horsepower'].cov(automobile_data['mpg'])


# In[25]:


automobile_data['horsepower'].corr(automobile_data['mpg'])


# In[26]:


automobile_data['horsepower'].cov(automobile_data['displacement'])


# In[27]:


automobile_data['horsepower'].corr(automobile_data['displacement'])


# ### Covariance

# In[28]:


automobile_data_cov = automobile_data.cov()

automobile_data_cov


# ### Correlation

# In[29]:


automobile_data_corr = automobile_data.corr()

automobile_data_corr


# In[30]:


plt.figure(figsize=(12, 8))

sns.heatmap(automobile_data_corr, annot=True)


# ## Linear Regression

# In[31]:


mpg_mean = automobile_data['mpg'].mean()

mpg_mean


# In[32]:


horsepower_mean = automobile_data['horsepower'].mean()

horsepower_mean


# #### Calculate the terms needed for the numerator and denominator of beta

# In[33]:


automobile_data['horsepower_mpg_cov'] = (automobile_data['horsepower'] - horsepower_mean) *                                         (automobile_data['mpg'] - mpg_mean)

automobile_data['horsepower_var'] = (automobile_data['horsepower'] - horsepower_mean)**2


# In[34]:


automobile_data['horsepower_mpg_cov']


# In[35]:


automobile_data['horsepower_var']


# #### Calculate beta and alpha

# In[36]:


beta = automobile_data['horsepower_mpg_cov'].sum() / automobile_data['horsepower_var'].sum()

print(f'beta = {beta}')


# In[37]:


alpha = mpg_mean - (beta * horsepower_mean)

print(f'alpha = {alpha}')


# In[38]:


y_pred = alpha + beta * automobile_data['horsepower']

print(y_pred)


# In[39]:


automobile_data.plot(x='horsepower', y='mpg', 
                     kind='scatter', s=50, figsize=(12, 8))

plt.plot(automobile_data['horsepower'], y_pred, color='red')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




