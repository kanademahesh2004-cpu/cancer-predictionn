import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\admin\\Downloads\\gene_expression.csv")
df

df.shape

df.isnull().sum()

df.info()

df.describe()

df["Cancer Present"].value_counts()

df.head()

df.tail()

plt.scatter(x=df["Gene One"],y=df["Gene Two"])
plt.show()

sns.scatterplot(data=df,x=df["Gene One"],y=df["Gene Two"],hue="Cancer Present")
# plt.xlim(2,8)
# plt.ylim(2,6)
plt.show()


x = df.iloc[0:,0:2]
y = df.iloc[0:,2]
print(y)


y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

x_train

x_test

y_test

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=5)

model=kn.fit(x_train,y_train)
model

y_predict = model.predict(x_train)
y_predict

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_train,y_predict)

arr = confusion_matrix(y_train,y_predict)
print(arr)
print(y_test.shape)

import pickle

with open("knn_project","wb") as file:    # to store your model
    pickle.dump(model,file)

with open("knn_project","rb") as file: 
    cancer_prediction = pickle.load(file)    # with this you can load your model anytime

sample = np.array([[6.1,6.6]])
cancer_prediction.predict(sample)

import streamlit as st       

st.title("The Cancer prediction Model")

st.write("The Cancer Prediction for Knn algorithm")
st.write

x = st.slider("Select a value", 0, 100, 25)
st.write("You selected:", x)

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

st.write("---")
st.caption("Made with ❤️ using Streamlit")




