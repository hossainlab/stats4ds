#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ### Gender voice dataset
# #### Source : https://www.mldata.io/dataset-details/gender_voice/
# meanfreq	-	mean frequency (in kHz)
# 
# sd	        -	standard deviation of frequency
# 
# median   	- median frequency (in kHz)
# 
# IQR	        -	interquantile range (in kHz)
# 
# sp.ent	    -	spectral entropy
# 
# centroid	-	frequency centroid (see specprop)
# 
# minfun	    -	minimum fundamental frequency measured across acoustic signal
# 
# label	    -	Predictor class, male or female

# In[2]:


data = pd.read_csv('datasets/gender_voice_dataset.csv')

data.sample(10)


# In[6]:


data[['label']].sample(10)


# In[7]:


data.shape


# ### Describing values

# In[8]:


data.describe().T


# ### Label encoding 

# In[9]:


from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
data['label'] = labelEncoder.fit_transform(data['label'].astype(str))

data['label'].sample(10)


# In[11]:


data.boxplot(by ='label', column =['meanfreq'], grid = False, figsize=(10, 8))


# In[12]:


data.boxplot(by ='label', column =['dfrange'], grid = False, figsize=(10, 8))


# ### Spilting the data into train and test data

# In[13]:


from sklearn.model_selection import train_test_split

features = data.drop('label', axis=1)
target = data['label']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2) 


# In[14]:


x_train.shape , y_train.shape


# In[15]:


x_test.shape , y_test.shape 


# ### Logistic Regression Classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 

# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logistic_model = LogisticRegression(penalty='l2', solver='liblinear')

logistic_model.fit(x_train, y_train)


# In[20]:


y_pred = logistic_model.predict(x_test)


# In[21]:


confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)


# In[22]:


print("Training score : ", logistic_model.score(x_train, y_train))


# ### Accuracy, precision, recall scores

# In[24]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)


# In[25]:


from yellowbrick.target import FeatureCorrelation

feature_names = list(features.columns)


# In[26]:


visualizer = FeatureCorrelation(labels = feature_names)

visualizer.fit(features, target)

visualizer.poof()


# In[27]:


from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest


# In[44]:


## TODO: While recording, re-record from here first with chi2, then with f_classif and then with mutual_info_classif

select_univariate = SelectKBest(chi2, k=4).fit(features, target)


# In[45]:


features_mask = select_univariate.get_support()

features_mask


# In[46]:


selected_columns = features.columns[features_mask]

selected_columns


# In[47]:


selected_features = features[selected_columns]

selected_features.head()


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size =.2)


# In[49]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(x_train, y_train)


# In[50]:


y_pred = logistic_model.predict(x_test)


# In[51]:


acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)


# In[ ]:




