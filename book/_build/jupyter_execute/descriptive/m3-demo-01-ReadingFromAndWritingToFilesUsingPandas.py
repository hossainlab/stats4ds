#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python --version')


# #### Installation of pandas

# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


get_ipython().system('pip install numpy')


# #### Installation of matplotlib

# In[4]:


get_ipython().system('pip install matplotlib')


# In[5]:


get_ipython().system('pip install seaborn')


# #### Pandas version

# In[6]:


import pandas as pd

pd.__version__


# #### Matplotlib version

# In[7]:


import matplotlib

matplotlib.__version__


# #### Numpy version

# In[8]:


import numpy as np

np.__version__


# ### Seaborn Version

# In[9]:


import seaborn

seaborn.__version__


# <b>Dataset</b> https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
# * Gender - Gender of the customer
# * Age - Age of the customer
# * Annual Income (k$) - Annual Income of the customee
# * Spending Score (1-100) - Score assigned by the mall based on customer behavior and spending nature

# #### Reading data using pandas python

# In[10]:


## TODO: While recording please show the file structure first


# In[11]:


mall_data = pd.read_json('datasets/Mall_Customers.json')

mall_data.head(5)


# In[12]:


average_income = mall_data['annual_income'].mean()

average_income


# In[13]:


mall_data['above_average_income'] = (mall_data['annual_income'] - average_income) > 0

mall_data.sample(5)


# * We can save it in any forms like normal text file, csv or json. It will save it in the location where you prefer as well

# In[14]:


mall_data.to_csv('datasets/mall_data_processed.csv', index=False)


# In[15]:


get_ipython().system('ls datasets/')


# #### Read the csv file for the clarification

# In[16]:


mall_data_updated = pd.read_csv('datasets/mall_data_processed.csv', index_col=0)

mall_data_updated.head(5)


# In[17]:


## TODO: While recording please show the files after each save


# In[18]:


mall_data.to_json('datasets/mall_data_column_oriented.json', orient='columns')


# In[19]:


mall_data.to_json('datasets/mall_data_records_oriented.json', orient='records')


# In[20]:


mall_data.to_json('datasets/mall_data_index_oriented.json', orient='index')


# In[21]:


mall_data.to_json('datasets/mall_data_values_oriented.json', orient='values')


# In[22]:


get_ipython().system('ls datasets/')


# In[ ]:




