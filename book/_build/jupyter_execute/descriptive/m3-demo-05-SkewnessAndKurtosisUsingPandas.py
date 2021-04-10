#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.head()


# In[3]:


height_weight_data.drop('Index', inplace=True, axis=1)

height_weight_data.shape


# In[4]:


height_weight_data[['Height']].plot(kind = 'hist',
                                    title = 'Height', figsize=(12, 8))


# In[5]:


height_weight_data[['Weight']].plot(kind = 'hist',
                                    title = 'weight', figsize=(12, 8))


# In[6]:


height_weight_data[['Height']].plot(kind = 'kde',
                                    title = 'Height', figsize=(12, 8))


# In[7]:


height_weight_data[['Weight']].plot(kind = 'kde',
                                    title = 'Weight', figsize=(12, 8))


# ### Skewness
# 
# It is the degree of distortion from the normal distribution. It measures the lack of symmetry in data distribution.
# 
# -- Positive Skewness means when the tail on the right side of the distribution is longer or fatter. The mean and median will be greater than the mode.    
# 
# -- Negative Skewness is when the tail of the left side of the distribution is longer or fatter than the tail on the right side. The mean and median will be less than the mode.  

# In[8]:


height_weight_data['Height'].skew()


# In[9]:


height_weight_data['Weight'].skew()


# In[10]:


listOfSeries = [pd.Series(['Male', 400, 300], index=height_weight_data.columns ), 
                pd.Series(['Female', 660, 370], index=height_weight_data.columns ), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ), 
                pd.Series(['Female', 770, 210], index=height_weight_data.columns ),
                pd.Series(['Male', 880, 203], index=height_weight_data.columns )]


# In[11]:


height_weight_updated = height_weight_data.append(listOfSeries , ignore_index=True)

height_weight_updated.tail()


# In[12]:


height_weight_updated[['Height']].plot(kind = 'hist', bins=100,
                                       title = 'Height', figsize=(12, 8))


# In[13]:


height_weight_updated[['Weight']].plot(kind = 'hist', bins=100,
                                       title = 'weight', figsize=(12, 8))


# In[14]:


height_weight_updated[['Height']].plot(kind = 'kde',
                                       title = 'Height', figsize=(12, 8))


# In[15]:


height_weight_updated[['Weight']].plot(kind = 'kde',
                                       title = 'weight', figsize=(12, 8))


# In[16]:


height_weight_updated['Height'].skew()


# In[17]:


height_weight_updated['Weight'].skew()


# ### Kurtosis
# 
# It is actually the measure of outliers present in the distribution.
# 
# -- High kurtosis in a data set is an indicator that data has heavy tails or outliers.       
# -- Low kurtosis in a data set is an indicator that data has light tails or lack of outliers.

# In[18]:


height_weight_data['Height'].kurtosis()


# In[19]:


height_weight_data['Weight'].kurtosis()


# In[20]:


height_weight_updated['Height'].kurtosis()


# In[21]:


height_weight_updated['Weight'].kurtosis()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




