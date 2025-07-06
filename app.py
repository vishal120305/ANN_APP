import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
local_css("style.css")

# App title and description
st.title("⚡ ANN Power Plant Energy Output Predictor")
st.markdown("""
This app uses an Artificial Neural Network to predict the energy output of a combined cycle power plant 
based on ambient variables like temperature, pressure, and humidity.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Function to get user input
def user_input_features():
    ambient_temp = st.sidebar.slider('Ambient Temperature (T)', 1.81, 37.11, 20.0)
    ambient_pressure = st.sidebar.slider('Ambient Pressure (AP)', 992.89, 1033.30, 1013.0)
    relative_humidity = st.sidebar.slider('Relative Humidity (RH)', 25.56, 100.16, 70.0)
    exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum (V)', 25.36, 81.56, 50.0)
    
    data = {
        'Ambient Temperature': ambient_temp,
        'Ambient Pressure': ambient_pressure,
        'Relative Humidity': relative_humidity,
        'Exhaust Vacuum': exhaust_vacuum,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Load and preprocess data
@st.cache_data
def load_data():
    # For deployment, you'll need to host this file online or include it in the repo
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    df = pd.read_excel("Folds5x2_pp.xlsx")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Train the model (cached to avoid reloading on every interaction)
@st.cache_resource
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Build the ANN model
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='linear'))
    ann.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    ann.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
    
    return ann, X_test, y_test

# Main app functionality
def main():
    # Load data and model
    ann, X_test, y_test = train_model()
    
    # Show raw data option
    if st.checkbox("Show raw data"):
        X, y = load_data()
        df = pd.DataFrame(X, columns=['AT', 'V', 'AP', 'RH'])
        df['PE'] = y
        st.dataframe(df.head(100))
    
    # Get user input
    st.sidebar.subheader("Make a Prediction")
    input_df = user_input_features()
    
    # Display user input
    st.subheader("User Input Parameters")
    st.write(input_df)
    
    # Predict on user input
    prediction = ann.predict(input_df.values)
    
    st.subheader("Prediction")
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Predicted Energy Output: <span>{prediction[0][0]:.2f} MW</span></h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model evaluation section
    st.subheader("Model Evaluation")
    
    # Make predictions on test set
    y_pred = ann.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{rmse:.2f}")
    with col3:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    
    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred.flatten(), alpha=0.6, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Energy Output (MW)')
    ax.set_ylabel('Predicted Energy Output (MW)')
    ax.set_title('Actual vs Predicted Values')
    st.pyplot(fig)
    
    # Feature importance (simplified)
    st.subheader("Feature Importance")
    st.markdown("""
    While neural networks don't provide direct feature importance measures like tree-based models,
    we can examine the model's sensitivity to each input feature:
    """)
    
    # Create a sensitivity analysis
    base_input = input_df.values[0]
    features = ['Ambient Temp', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity']
    sensitivities = []
    
    for i in range(4):
        perturbed_input = base_input.copy()
        perturbed_input[i] *= 1.1  # 10% increase
        pred_change = ann.predict([perturbed_input])[0][0] - prediction[0][0]
        sensitivities.append(abs(pred_change))
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=features, y=sensitivities, palette="viridis", ax=ax2)
    ax2.set_title('Model Sensitivity to 10% Increase in Each Feature')
    ax2.set_ylabel('Change in Predicted Energy Output (MW)')
    st.pyplot(fig2)

if __name__ == '__main__':
    main()