#IMPORTING REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#   LOAD THE DATASET
df = pd.read_csv("gene_expression.csv")
df

#  BASIC DATA EXPLORATION
df.shape

df.isnull().sum()

df.info()

df.describe()

#   CHECK TARGET COLUMN (Cancer Present)

df["Cancer Present"].value_counts()

df.head()

df.tail()
#   DATA VISUALIZATION (Gene One vs Gene Two)

plt.scatter(x=df["Gene One"],y=df["Gene Two"])
plt.show()

#   SEPARATING FEATURES AND TARGET
x = df.iloc[0:,0:2]
y = df.iloc[0:,2]
print(y)
y

#   TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

x_train

x_test

y_test
#   TRAINING KNN MODEL

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=5)

model=kn.fit(x_train,y_train)
model

#   PREDICTION ON TRAIN DATA

y_predict = model.predict(x_train)
y_predict
#   MODEL EVALUATION (Accuracy & Confusion Matrix)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_train,y_predict)

arr = confusion_matrix(y_train,y_predict)
print(arr)
print(y_test.shape)

#   SAVE MODEL USING PICKLE

import pickle
# ---- save model ----
with open("knn_project","wb") as file:   
    pickle.dump(model,file)
# ---- load model ----

with open("knn_project","rb") as file: 
    cancer_prediction = pickle.load(file)   
    
# ---- sample prediction ----

sample = np.array([[6.1,6.6]])
cancer_prediction.predict(sample)


import streamlit as st       

st.title("The Cancer prediction Model")

st.write("The Cancer Prediction for Knn algorithm")
st.write

x = st.slider("Select a value", 0, 100, 25)
st.write("You selected:", x)
#   STREAMLIT APPLICATION

import streamlit as st
import numpy as np
import pickle as pi

# ------------------------------
# Load your saved KNN model
# ------------------------------
with open("knn_project", "rb") as file:
    model = pi.load(file)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="KNN Predictor", page_icon="🔮", layout="centered")

st.title("🔍 KNN Prediction App")
st.write("Enter the values below to get predictions from your trained KNN model.")

# --- Input fields (Example: 2 features) ---

f1 = st.number_input("Enter value for Feature 1", value=0.0, step=0.1)
f2 = st.number_input("Enter value for Feature 2", value=0.0, step=0.1)

# Combine input as numpy array
sample = np.array([[f1, f2]])

# --- Predict Button ---
if st.button("Predict"):
    pred = model.predict(sample)
    st.success(f"🎯 Predicted Output: **{pred[0]}**")
#   FOOTER 
st.write("---")
st.caption("Made with ❤️ using Streamlit")




