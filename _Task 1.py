#!/usr/bin/env python
# coding: utf-8

# # This is a Supervised machine learning model for predicting a score of a student based on the number of hours he/she studies

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df


# In[4]:


#The %matplotlib inline command is a magic command in Jupyter notebooks that allows the resulting plots or visualizations to be displayed directly below the code cell that generated them. It is used to enable the inline plotting mode for Matplotlib in Jupyter environments.

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(df.Hours,df.Scores,color='red',marker='+')


# In[5]:


X = df.drop('Scores',axis='columns')
X


# In[6]:


Y = df.Scores
Y


# In[7]:


reg = linear_model.LinearRegression()
reg.fit(X,Y)


# In[8]:


reg.predict([[9.25]])


# In[9]:


reg.coef_


# In[10]:


reg.intercept_


# In[11]:


line = reg.coef_*X + reg.intercept_

plt.scatter(X,Y)
plt.plot(X,line)
plt.show()


# # Now solving by test,train and split method and see the difference

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,)


# In[13]:


X_train


# In[14]:


X_test


# In[15]:


Y_train


# In[16]:


Y_test


# In[17]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train)


# In[18]:


clf.predict(X_test)


# In[19]:


Y_test


# In[20]:


clf.score(X_test,Y_test)


# In[23]:


clf.predict([[9.25]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




