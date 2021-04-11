#!/usr/bin/env python
# coding: utf-8

# # Descriptive Statistics: The Central Tendency Using Pandas

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')

height_weight_data.head(5)


# In[3]:


height_weight_data.drop('Index', inplace=True, axis=1)


# In[4]:


height_weight_data.shape


# In[5]:


height_weight_data.isnull().sum()


# In[6]:


min_height = height_weight_data['Height'].min()

min_height


# In[7]:


max_height = height_weight_data['Height'].max()

max_height


# In[8]:


min_weight = height_weight_data['Weight'].min()

min_weight


# In[9]:


max_weight = height_weight_data['Weight'].max()

max_weight


# In[10]:


range_of_height = max_height - min_height

range_of_height


# In[11]:


range_of_weight = max_weight - min_weight

range_of_weight


# In[12]:


weight = height_weight_data['Weight']

weight.head()


# In[13]:


sorted_weight = weight.sort_values().reset_index(drop=True)

sorted_weight.head()


# In[14]:


def mean(data):
    num_elements = len(data)
    print('Number of elements: ', num_elements)
    
    weight_sum = data.sum()
    print('Sum: ', weight_sum)
    
    return weight_sum / num_elements


# In[15]:


def median(data):
    num_elements = len(data)
    
    if(num_elements % 2 == 0):
        return  (data[(num_elements / 2) - 1] + data[(num_elements / 2)]) / 2 
    
    else: 
        return (data[((num_elements + 1) / 2) - 1])


# In[16]:


mean(height_weight_data['Weight'])


# In[17]:


weight_mean = height_weight_data['Weight'].mean()

weight_mean


# In[18]:


median(height_weight_data['Weight'])


# In[19]:


median(sorted_weight)


# In[20]:


weight_median = height_weight_data['Weight'].median()

weight_median


# In[21]:


plt.figure(figsize=(12, 8))

height_weight_data['Weight'].hist(bins=30)

plt.axvline(weight_mean, color='r', label='mean')

plt.legend()


# In[22]:


plt.figure(figsize=(12, 8))

height_weight_data['Weight'].hist(bins=30)

plt.axvline(weight_median, color='g', label='median')

plt.legend()


# In[23]:


plt.figure(figsize=(12, 8))

plt.bar(height_weight_data['Height'], height_weight_data['Weight'])

plt.axhline(weight_mean, color='r', label='mean')
plt.axhline(weight_median, color='b', label='median')

plt.show()


# In[24]:


height_weight_data


# In[25]:


listOfSeries = [pd.Series(['Male', 205, 460], index=height_weight_data.columns ), 
                pd.Series(['Female', 202, 390], index=height_weight_data.columns ), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ), 
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 200, 490], index=height_weight_data.columns )]


# In[26]:


height_weight_updated = height_weight_data.append(listOfSeries , ignore_index=True)

height_weight_updated.tail()


# In[27]:


updated_weight_mean = height_weight_updated['Weight'].mean()

updated_weight_mean


# In[28]:


weight_mean


# In[29]:


updated_weight_median = height_weight_updated['Weight'].median()

updated_weight_median


# In[30]:


weight_median


# In[31]:


plt.figure(figsize=(12, 8))

plt.bar(height_weight_updated['Height'], height_weight_updated['Weight'])

plt.axhline(updated_weight_mean, color='r', label='mean')
plt.axhline(updated_weight_median, color='b', label='median')

plt.show()


# In[32]:


plt.figure(figsize=(12, 8))

height_weight_data['Weight'].hist(bins=20)

plt.axvline(updated_weight_mean, color='r', label='mean')
plt.axvline(updated_weight_median, color='g', label='median')

plt.legend()


# In[33]:


plt.figure(figsize=(12, 8))

height_weight_updated['Weight'].hist(bins=100)

plt.axvline(updated_weight_mean, color='r', label='mean')
plt.axvline(updated_weight_median, color='g', label='median')

plt.legend()


# In[34]:


height_counts = {}

for p in height_weight_data['Height']:
    if p not in height_counts:
        height_counts[p] = 1
    else:
        height_counts[p] += 1


# In[35]:


height_counts


# In[36]:


plt.figure(figsize=(25, 10))

x_range = range(len(height_counts))

plt.bar(x_range, list(height_counts.values()), align='center')
plt.xticks(x_range, list(height_counts.keys()))

plt.xlabel('Height')
plt.ylabel('Count')

plt.show()


# In[37]:


count = 0 
size = 0

for s, c in height_counts.items():
    if count < c:
        count = c
        size = s
        
print('Size: ', size, '\nFrequency: ', count)


# In[38]:


height_weight_data['Height'].mode()


# In[39]:


height_weight_data['Weight'].mode()


# In[40]:


height_mean = height_weight_data['Height'].mean()

height_median = height_weight_data['Height'].median()

height_mode = height_weight_data['Height'].mode().values[0]


# In[41]:


plt.figure(figsize=(12, 8))

height_weight_data['Height'].hist()

plt.axvline(height_mean, color='r', label='mean')
plt.axvline(height_median, color='g', label='median')
plt.axvline(height_mode, color='y', label='mode')

plt.legend()


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




