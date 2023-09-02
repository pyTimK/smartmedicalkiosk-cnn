import numpy as np
import tensorflow as tf

# Load the pre-trained models
loaded_sbp_model = tf.keras.models.load_model("model/sbp_model.keras")
loaded_dbp_model = tf.keras.models.load_model("model/dbp_model.keras")

def predict_blood_pressure(age, weight, height, temp, heart_rate, spo2):
    # Create an input array with the provided parameters
    input_data = np.array([[age, weight, height, temp, heart_rate, spo2]])
    
    # Reshape input data for Conv1D model input with shape (1, 6, 1)
    input_data_reshaped = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    
    # Use the loaded models to predict SBP and DBP
    predicted_sbp = loaded_sbp_model.predict(input_data_reshaped)[0][0]
    predicted_dbp = loaded_dbp_model.predict(input_data_reshaped)[0][0]
    
    return predicted_sbp, predicted_dbp

# Example usage
# age = 45
# weight = 75
# height = 170
# temp = 36.7
# heart_rate = 85
# spo2 = 97

# age= 45
# weight= 72
# height= 170
# temp= 36.8
# SBP= 130
# DBP= 82
# heart_rate= 112
# spo2= 97

# predicted_sbp, predicted_dbp = predict_blood_pressure(age, weight, height, temp, heart_rate, spo2)
# print("Predicted SBP:", predicted_sbp)
# print("Predicted DBP:", predicted_dbp)