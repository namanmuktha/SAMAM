#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import category_encoders as ce
accuracy=[]
model=[]


# In[2]:


ch_data=pd.read_csv('psyc.csv')


# In[3]:





# In[4]:





# In[5]


# In[6]:



# In[7]:





# In[8]:





# In[9]:


from sklearn.preprocessing import LabelEncoder
disorder_column = ch_data['Personality']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(disorder_column)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
ch_data['Personality'] = encoded_values


# In[10]:




# In[11]:





# In[12]:


encoder=ce.OrdinalEncoder(["gender"])


# In[13]:


ch_data=encoder.fit_transform(ch_data)


# In[14]:




# In[15]:


corr=ch_data.corr()


# In[16]:





# In[17]:


# Set a seed for reproducibility
np.random.seed(123)

# Get the number of rows in your dataset
num_rows = len(ch_data)

# Shuffle the indices of the rows
shuffled_indices = np.random.permutation(num_rows)

# Define the percentage for the training set
train_percentage = 0.8

# Calculate the number of rows for the training set
num_train_rows = int(train_percentage * num_rows)

# Split the indices into training and testing sets
train_indices = shuffled_indices[:num_train_rows]
test_indices = shuffled_indices[num_train_rows:]

# Create the training and testing sets
data_train = ch_data.iloc[train_indices]  # For a DataFrame
data_test = ch_data.iloc[test_indices]    # For a DataFrame


# In[18]:




# In[19]:





# In[20]:


X_train=data_train.drop(["Personality"],axis=1)
y_train=data_train["Personality"]
X_test=data_test.drop(["Personality"],axis=1)
y_test=data_test["Personality"]




# In[22]:


svc=SVC(kernel="poly",C=4,gamma='scale')
svc.fit(X_train,y_train)
y_svc_p=svc.predict(X_test)



# In[23]:





# In[24]:





# In[25]:




# In[26]:





# In[27]:




import streamlit as st


# In[30]:


st.title('You are 74 perecnt related to ')
st.sidebar.header('User Input Features')


# In[31]:


feature1 = st.sidebar.slider('gender ', min_value=1, max_value=2, value=1)
feature2 = st.sidebar.slider('age ', min_value=2, max_value=100, value=50)
feature3 = st.sidebar.slider('openness ', min_value=1, max_value=10, value=3)
feature4 = st.sidebar.slider('neuroticism ', min_value=1, max_value=10, value=3)
feature5=st.sidebar.slider('conscientiousness ', min_value=1, max_value=10, value=3)
feature6=st.sidebar.slider('agreeableness ', min_value=1, max_value=10, value=3)
feature7=st.sidebar.slider('extraversion ', min_value=1, max_value=10, value=3)


# In[32]:


# In[40]:


input_data = pd.DataFrame({
    'gender ': [feature1],
    'age ': [feature2],
    'openness ': [feature3],
    'neuroticism ': [feature4],
    'conscientiousness ': [feature5],
    'agreeableness ': [feature6],
    'extraversion ': [feature7]
})


# In[41]:


prediction=svc.predict(input_data)


# In[42]:

def get_key_from_value(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None  # Return None if the value is not found

key_for_value = get_key_from_value(label_mapping, prediction[0])





st.write(key_for_value)


# In[43]:




# 
