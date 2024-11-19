import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
@st.cache_data  # Cache the dataset loading function
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    # Map target to species names
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

# Train and cache the model
@st.cache_resource  # Cache the trained model so it's not retrained every time
def train_model(df):
    X = df.iloc[:, :-1]  # Features
    y = df['species']    # Target variable
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Model Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit UI
st.title('Iris Flower Classification App')

st.header('User Input Measurements')

# Create input fields for user to input flower measurements
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Load data and train model
df = load_data()
model, accuracy = train_model(df)

# Prepare the input data as a DataFrame for prediction
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_input_df = pd.DataFrame(user_input, columns=df.columns[:-1])

# Prediction
predicted_species = model.predict(user_input_df)
st.write(f"Predicted Species: {predicted_species[0]}")

# Display Model Accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
