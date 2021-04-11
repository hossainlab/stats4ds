#!/usr/bin/env python
# coding: utf-8

# ## Univariate and Bivariate Analysis

# <b>Dataset</b>: https://www.kaggle.com/mustafaali96/weight-height

# The variables used are:
#     * Height
#     * Weight
#     * Gender

# ### Import libraries 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ### Load and read dataset

# In[2]:


data = pd.read_csv('datasets/weight-height.csv')

data.head()


# In[3]:


data.shape


# ### Mean, median, mode, min, max and quantiles

# In[4]:


data.describe()


# ## Univariate Analysis

# #### Gender count

# In[5]:


sns.countplot(data['Gender'])

plt.show


# ### Considering only Height

# In[6]:


height = data['Height']

height.head()


# In[7]:


height.shape


# #### Histogram-plot

# In[8]:


plt.figure(figsize=(12, 8))

height.plot(kind = 'hist',
            title = 'Height Histogram')


# #### Box-plot

# In[9]:


plt.figure(figsize=(12, 8))

height.plot(kind = 'box',
            title = 'Height Box-plot')


# #### KDE distribution for height

# In[10]:


height.plot(kind = 'kde',
            title = 'Height KDE', figsize=(12, 8))


# #### Analysis

# As we can see we have a high count for height in the range 60 to 75.

# ### Considering only weight

# In[11]:


weight = data['Weight']

weight.head()


# In[12]:


weight.shape


# #### Histogram-plot

# In[13]:


plt.figure(figsize=(12, 8))

weight.plot(kind = 'hist',
            title = 'Weight Histogram')


# #### Box-plot

# In[14]:


plt.figure(figsize=(12, 8))

weight.plot(kind = 'box',
            title = 'Weight Box-plot')


# #### KDE distribution for height

# In[15]:


plt.figure(figsize=(12, 8))

weight.plot(kind = 'kde',
            title = 'Weight KDE')


# In[ ]:





# ## Bivariate Analysis

# #### Considering both height and weight

# In[16]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x = "Height", y = "Weight", data=data)


# In[17]:


plt.figure(figsize=(12, 8))

sns.scatterplot(x = "Height", y = "Weight", hue='Gender', data=data)


# In[18]:


gender_groupby = data.groupby('Gender', as_index=False)

gender_groupby.head()


# In[19]:


gender_groupby.describe().T


# ### Distribution plots

# #### Considering both gender and height

# In[20]:


sns.FacetGrid(data, hue = 'Gender', height = 5)              .map(sns.distplot, 'Height')              .add_legend()


# #### Considering both gender and weight

# In[21]:


sns.FacetGrid(data, hue = 'Gender', height = 5)    .map(sns.distplot, 'Weight').add_legend()


# ### Violin Plot

# #### Gender vs Height

# In[22]:


plt.figure(figsize=(12, 8))

sns.boxplot(x = 'Gender', y ='Height', data = data)


# #### Gender vs Weight

# In[23]:


plt.figure(figsize=(12, 8))

sns.boxplot(x = 'Gender', y ='Weight', data = data)


# In[24]:


plt.figure(figsize=(12, 8))

sns.violinplot(x = 'Gender', y ='Height', data = data)


# In[25]:


plt.figure(figsize=(12, 8))

sns.violinplot(x = 'Gender', y ='Weight', data = data)


# ### Multivariate Analysis

# In[26]:


sns.pairplot(data, hue = 'Gender', height = 4)


# In[ ]:




