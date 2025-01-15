import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Model Evaluation App")

# Upload button for the trained model
st.header("Upload Trained Model")
model_file = st.file_uploader("Upload the Python file containing the trained model:", type=["py"])

if model_file is not None:
    try:
        # Import the trained model dynamically
        exec(model_file.read().decode("utf-8"), globals())
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Upload button for training data
st.header("Upload Training Data")
train_file = st.file_uploader("Upload your training dataset (CSV):", type=["csv"], key="train")

# Upload button for testing data
st.header("Upload Testing Data")
test_file = st.file_uploader("Upload your testing dataset (CSV):", type=["csv"], key="test")

if train_file is not None and test_file is not None:
    try:
        # Read the data
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Ensure dependent variable exists
        dependent_var = st.text_input("Enter the name of the dependent variable (target):")

        if dependent_var:
            if dependent_var not in train_data.columns or dependent_var not in test_data.columns:
                st.error("Dependent variable not found in the datasets. Please check your input.")
            else:
                # Separate features and target
                X_train = train_data.drop(columns=[dependent_var])
                y_train = train_data[dependent_var]

                X_test = test_data.drop(columns=[dependent_var])
                y_test = test_data[dependent_var]

                # Predict and evaluate
                if 'model' in globals():
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Calculate metrics
                    r2_train = r2_score(y_train, y_train_pred)
                    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

                    r2_test = r2_score(y_test, y_test_pred)
                    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

                    # Display metrics
                    st.subheader("Training Data Evaluation")
                    st.write(f"R²: {r2_train:.4f}")
                    st.write(f"RMSE: {rmse_train:.4f}")

                    st.subheader("Testing Data Evaluation")
                    st.write(f"R²: {r2_test:.4f}")
                    st.write(f"RMSE: {rmse_test:.4f}")
                else:
                    st.error("Model not loaded. Please upload the trained model.")
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
