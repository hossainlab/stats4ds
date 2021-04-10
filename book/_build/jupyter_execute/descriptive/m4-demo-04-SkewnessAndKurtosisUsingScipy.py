#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


sp_data = pd.read_csv('datasets/sp500_1987.csv')

sp_data.head(5)


# In[3]:


sp_data = sp_data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)


# In[4]:


sp_data.shape


# In[5]:


sp_data.head()


# In[6]:


sp_data['Date'] = pd.to_datetime(sp_data['Date'])


# In[7]:


sp_data.dtypes


# In[8]:


sp_data = sp_data.sort_values(by='Date')

sp_data.head(10)


# In[9]:


sp_data.describe()


# In[10]:


plt.figure(figsize=(12, 8))

sns.lineplot(x='Date', y='Adj Close', data=sp_data)

plt.title('SP Data')


# In[11]:


sp_data['Returns'] = sp_data['Adj Close'].pct_change()

sp_data.head(10)


# In[12]:


sp_data.dropna(inplace=True)


# In[13]:


sp_data.head(10)


# In[14]:


sp_data.count()


# #### Outliers

# In[15]:


plt.figure(figsize=(12, 8))

sns.boxplot(sp_data['Returns'], orient='v')

plt.title('SP Data')


# ### Skewness and Kurtosis
# 
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.robust_skewness.html
# 
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.robust_kurtosis.html

# In[16]:


stats.skew(sp_data['Returns'])


# In[17]:


stats.kurtosis(sp_data['Returns'])


# #### Remove Oct 19, 1987 when there was a huge fall

# In[18]:


sp_data_without_oct19 = sp_data[sp_data['Date'] != '1987-10-19']

sp_data_without_oct19.count()


# In[19]:


plt.figure(figsize=(12, 8))

sns.boxplot(sp_data_without_oct19['Returns'], orient='v')

plt.title('SP Data')


# In[20]:


stats.skew(sp_data_without_oct19['Returns'])


# In[21]:


stats.kurtosis(sp_data_without_oct19['Returns'])


# In[ ]:





# In[ ]:





# In[ ]:




