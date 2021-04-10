#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


automobile_data = pd.read_csv('datasets/automobile_data_processed.csv')

automobile_data.sample(5)


# In[3]:


automobile_data.shape


# In[4]:


automobile_data.boxplot('mpg', figsize=(12, 8))

plt.title('Miles per gallon')


# In[5]:


automobile_data['mpg'].plot.kde(figsize=(12, 8))

plt.xlabel('mpg')

plt.title('Density plot for MPG')
plt.show()


# In[6]:


plt.figure(figsize=(12, 8))

sns.boxplot(x='cylinders', y='mpg', data=automobile_data)

plt.show()


# In[7]:


plt.figure(figsize=(12, 8))

sns.violinplot(x='cylinders', y='mpg', data=automobile_data, inner=None)
sns.swarmplot(x='cylinders', y='mpg', data=automobile_data, color='w')

plt.show()


# In[8]:


cylinder_stats = automobile_data.groupby(['cylinders'])['mpg'].agg(['mean', 'count', 'std'])

cylinder_stats


# In[9]:


ci95_high = []

ci95_low = []


# In[10]:


for i in cylinder_stats.index:
    
    mean, count, std = cylinder_stats.loc[i]
    
    ci95_high.append(mean + 1.96 * (std / math.sqrt(count)))
    ci95_low.append(mean - 1.96 * (std / math.sqrt(count)))


# In[11]:


cylinder_stats['ci95_HIGH'] = ci95_high
cylinder_stats['ci95_LOW'] = ci95_low

cylinder_stats


# In[12]:


cylinders = 4

cylinders4_df = automobile_data.loc[automobile_data['cylinders'] == cylinders]

cylinders4_df.sample(10)


# In[13]:


plt.figure(figsize=(12, 8))

sns.distplot(cylinders4_df['mpg'], rug=True, kde=True, hist=False)

plt.show()


# In[14]:


plt.figure(figsize=(12, 8))

sns.distplot(cylinders4_df['mpg'], rug=True, kde=True, hist=False)

plt.stem([cylinder_stats.loc[cylinders]['mean']], 
         [0.07], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean')

plt.stem([cylinder_stats.loc[cylinders]['ci95_LOW']], 
         [0.07], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI High')

plt.stem([cylinder_stats.loc[cylinders]['ci95_HIGH']], 
         [0.07], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI Low')

plt.xlabel('mpg')
plt.legend()
plt.show()


# In[15]:


cylinders = 6

cylinders6_df = automobile_data.loc[automobile_data['cylinders'] == cylinders]

cylinders6_df.sample(10)


# In[16]:


plt.figure(figsize=(12, 8))

sns.distplot(cylinders6_df['mpg'], rug=True, kde=True, hist=False)

plt.show()


# In[17]:


plt.figure(figsize=(12, 8))

sns.distplot(cylinders6_df['mpg'], rug=True, kde=True, hist=False)

plt.stem([cylinder_stats.loc[cylinders]['mean']], 
         [0.16], linefmt = 'C1', 
         markerfmt = 'C1', label = 'mean')

plt.stem([cylinder_stats.loc[cylinders]['ci95_LOW']], 
         [0.16], linefmt = 'C2', 
         markerfmt = 'C2', label = '95% CI High')

plt.stem([cylinder_stats.loc[cylinders]['ci95_HIGH']], 
         [0.16], linefmt = 'C3', 
         markerfmt = 'C3', label = '95% CI Low')

plt.xlabel('mpg')
plt.legend()
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





# In[ ]:




