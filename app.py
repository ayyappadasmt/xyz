import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Set up the page config
st.set_page_config(page_title="Weather Predictor", layout="wide")

# Main title and subheading
st.markdown("<h1 style='text-align: center; color: yellow; font-family:Times New Roman;'>WEATHER PREDICTOR</h1>", unsafe_allow_html=True)
st.subheader("created by AYYAPPADAS M.T.")

# Explanation of Naive Bayes and its application in the model
st.markdown("""
    <p style='text-align: justify; color: white; font-size: 18px;'>
    The <strong>Naive Bayes</strong> algorithm is a probabilistic machine learning model that is based on the Bayes Theorem. It is called "naive" because it assumes that the features are independent of each other, which may not always be the case in real-world data. Despite this assumption, Naive Bayes performs well in various applications such as spam detection, sentiment analysis, and weather prediction. The algorithm is simple and computationally efficient, making it a good choice for large datasets with multiple features.
    </p>
    
    <p style='text-align: justify; color: white; font-size: 18px;'>
    In this application, we have built a weather prediction model using the <strong>Gaussian Naive Bayes</strong> algorithm. Based on the inputs provided (precipitation, temperature, and wind speed), the model predicts the weather category such as drizzle, fog, rain, snow, or sun.
    </p>
    """, unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("Input Weather Data")

# Load the dataset
df = pd.read_csv("seattle-weather.csv")
df.drop(["date"], axis=1, inplace=True)

# Encode the weather column to numerical values
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])

# Prepare X and y
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Split and standardize the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Input fields for the weather conditions (text inputs instead of sliders)
precipitation = st.sidebar.text_input('Precipitation (mm):', '0.0')
max_temp = st.sidebar.text_input('Max Temperature (°C):', '50')
min_temp = st.sidebar.text_input('Min Temperature (°C):', '30')
wind = st.sidebar.text_input('Wind Speed (km/h):', '5.0')

# Validate the input to ensure proper data types
try:
    precipitation = float(precipitation)
    max_temp = float(max_temp)
    min_temp = float(min_temp)
    wind = float(wind)

    # Prediction button
    if st.sidebar.button("Predict Weather"):
        # Input features for prediction
        input_data = np.array([[precipitation, max_temp, min_temp, wind]])
        
        # Standardize the input using the scaler
        input_data = scaler.transform(input_data)
        
        # Predict weather
        prediction = gnb.predict(input_data)
        
        # Map prediction to weather categories
        weather_labels = ['Drizzle', 'Fog', 'Rainy', 'Snow', 'Sunny']
        predicted_weather = weather_labels[prediction[0]]
        
        # Display the predicted weather
        st.markdown(f"<h2 style='text-align: center; color: #FFD700; font-family: Times New Roman;'>Predicted Weather: {predicted_weather}</h2>", unsafe_allow_html=True)

except ValueError:
    st.sidebar.error("Please enter valid numeric values!")

# Add additional styling for a clean and modern look
st.markdown("""
    <style>
    /* Black background */
    .css-1aumxhk {background-color: black;}

    /* Headings */
    h1, h2 {
        font-family: 'Times New Roman';
        
    }

    /* Sidebar input styling */
    .stTextInput input {
        border-radius: 5px;
        box-shadow: inset 3px 3px 5px rgba(0, 0, 0, 0.1), inset -3px -3px 5px rgba(255, 255, 255, 0.6);
    }

    /* 3D effect for button */
    .stButton button {
        background-color: #FF6347;
        border: none;
        border-radius: 10px;
        font-size: 20px;
        color: white;
        padding: 10px 20px;
        text-shadow: 1px 1px 5px #000000;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2), 3px 3px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }

    .stButton button:hover {
        transform: scale(1.05);
        background-color: #FF4500;
    }

    /* Modify text color inside the app */
    h3, p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: white;'>Provide the weather data on the left and press Predict!</h3>", unsafe_allow_html=True)
