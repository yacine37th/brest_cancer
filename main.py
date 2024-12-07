# import streamlit as st
# import numpy as np
# import pickle

# # Load models and scaler
# logistic_model = pickle.load(open('lr_model.pkl', 'rb'))
# rf_model = pickle.load(open('rf_model.pkl', 'rb'))
# scaler = pickle.load(open('scal.pkl', 'rb'))

# # Title
# st.title("Breast Cancer Prediction App")

# # Sidebar for input features
# st.sidebar.header("Patient Input Features")
# # def user_input_features():
# #     feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']  # Replace with your dataset feature names
# #     features = []
# #     for feature in feature_names:
# #         features.append(st.sidebar.number_input(f'{feature}', value=0.0))
# #     return np.array(features).reshape(1, -1)

# def user_input_features():
#     # List of feature names as per your dataset
#     feature_names = [
#         'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#         'smoothness_mean', 'compactness_mean', 'concavity_mean',
#         'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#         'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#         'fractal_dimension_se', 'radius_worst', 'texture_worst',
#         'perimeter_worst', 'area_worst', 'smoothness_worst',
#         'compactness_worst', 'concavity_worst', 'concave points_worst',
#         'symmetry_worst', 'fractal_dimension_worst'
#     ]
    
#     # Create inputs for all features
#     features = []
#     for feature in feature_names:
#         features.append(st.sidebar.number_input(f'{feature}', value=0.00000))
    
#     # Convert inputs into a NumPy array reshaped to match the model input
#     return np.array(features).reshape(1, -1)

# input_features = user_input_features()

# # Predict button
# if st.sidebar.button('Predict'):
#     # Scale the input
#     scaled_features = scaler.transform(input_features)
    
#     # Logistic Regression Prediction
#     logistic_prediction = logistic_model.predict(scaled_features)
#     logistic_result = "Malignant" if logistic_prediction[0] == 1 else "Benign"

#     # Random Forest Prediction
#     rf_prediction = rf_model.predict(scaled_features)
#     rf_result = "Malignant" if rf_prediction[0] == 1 else "Benign"

#     # Display results
#     st.write("### Prediction Results:")
#     st.write(f"Logistic Regression Prediction: **{logistic_result}**")
#     st.write(f"Random Forest Prediction: **{rf_result}**")
import streamlit as st
import numpy as np
import pickle

# Load models and scaler
logistic_model = pickle.load(open('lr_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scal.pkl', 'rb'))

# Title
st.title("Breast Cancer Prediction App")

# Sidebar for input features
st.sidebar.header("Patient Input Features")

def user_input_features():
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]

    # Create a single text input for all features
    input_text = st.sidebar.text_area(
        "Enter all features as comma-separated values", 
        placeholder="Enter values for the features, separated by commas"
    )

    if input_text:
        try:
            # Convert input text into a list of floats
            features = np.array([float(value.strip()) for value in input_text.split(',')])

            # Check if the correct number of features is provided
            if len(features) != len(feature_names):
                st.sidebar.error(f"Please provide exactly {len(feature_names)} values.")
                return None

            # Reshape to match model input
            return features.reshape(1, -1)

        except ValueError:
            st.sidebar.error("Please ensure all inputs are valid numbers.")
            return None
    
    return None

input_features = user_input_features()

# Predict button
if st.sidebar.button('Predict'):
    if input_features is not None:
        # Scale the input
        scaled_features = scaler.transform(input_features)

        # Logistic Regression Prediction
        logistic_prediction = logistic_model.predict(scaled_features)
        logistic_result = "Malignant" if logistic_prediction[0] == 1 else "Benign"

        # Random Forest Prediction
        rf_prediction = rf_model.predict(scaled_features)
        rf_result = "Malignant" if rf_prediction[0] == 1 else "Benign"

        # Display results
        st.write("### Prediction Results:")
        st.write(f"Logistic Regression Prediction: **{logistic_result}**")
        st.write(f"Random Forest Prediction: **{rf_result}**")
    else:
        st.write("Please provide valid input values for prediction.")
