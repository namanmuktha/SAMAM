#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')
import category_encoders as ce
accuracy=[]
model=[]


# In[2]:


ch_data=pd.read_csv('psyc.csv')


# In[3]:


ch_data.head()


# In[4]:


ch_data.describe()


# In[5]:


ch_data.shape


# In[6]:


ch_data.nunique()


# In[7]:


ch_data.isna().sum()


# In[8]:


ch_data.head()


# In[9]:


from sklearn.preprocessing import LabelEncoder
disorder_column = ch_data['Personality']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(disorder_column)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
ch_data['Personality'] = encoded_values


# In[10]:


ch_data.dtypes


# In[11]:


label_mapping


# In[12]:


encoder=ce.OrdinalEncoder(["gender"])


# In[13]:


ch_data=encoder.fit_transform(ch_data)


# In[14]:


ch_data


# In[15]:


corr=ch_data.corr()


# In[16]:


corr["Personality"].sort_values()


# In[17]:


data_train,data_test=train_test_split(ch_data,test_size=0.2,random_state=123)


# In[18]:


data_train


# In[19]:


data_test


# In[20]:


X_train=data_train.drop(["Personality"],axis=1)
y_train=data_train["Personality"]
X_test=data_test.drop(["Personality"],axis=1)
y_test=data_test["Personality"]


# In[21]:


knn=KNeighborsClassifier(n_neighbors=11,weights="distance",algorithm="kd_tree")
knn.fit(X_train,y_train)
y_knn_p=knn.predict(X_test)
print(accuracy_score(y_test,y_knn_p))
print(confusion_matrix(y_test,y_knn_p))


# In[22]:


svc=SVC(kernel="poly",C=4,gamma='scale')
svc.fit(X_train,y_train)
y_svc_p=svc.predict(X_test)
print(accuracy_score(y_test,y_svc_p))
print(confusion_matrix(y_test,y_svc_p))


# In[23]:


nb=MultinomialNB()
nb.fit(X_train,y_train)
y_nb_p=nb.predict(X_test)
print(accuracy_score(y_test,y_nb_p))
print(confusion_matrix(y_test,y_nb_p))


# In[24]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_p=dt.predict(X_test)
print(accuracy_score(y_test,y_p))
print(confusion_matrix(y_test,y_p))


# In[25]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_p1=rfc.predict(X_test)
print(accuracy_score(y_test,y_p1))
print(confusion_matrix(y_test,y_p1))


# In[26]:


from sklearn.model_selection import GridSearchCV


# In[27]:


svm_model = SVC()

# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5)

# Fit the model with different hyperparameter combinations
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Parameters: {best_params}')
print(f'Accuracy on Test Set: {accuracy:.2f}')


# In[28]:


dt_classifier = DecisionTreeClassifier()

# Define the parameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model with different hyperparameter combinations
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Parameters: {best_params}')
print(f'Accuracy on Test Set: {accuracy:.2f}')


# In[29]:


import streamlit as st


# In[30]:


st.title('Machine Learning Model Deployment')
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


X_train.columns


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


st.write(prediction[0])


# In[43]:


get_ipython().system('streamlit run c.py')


# In[44]:


get_ipython().system('streamlit run --server.enableCORS=false --server.port=8501 character.ipynb')


# 
