import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

print("Loading Data...")
# Sample data (replace this with your actual data)
with open("data.json", "r") as f:
    raw_data = json.load(f)
    data = raw_data["data"]


# Extract input features and target values
X = np.array([[d["age"], d["weight"], d["height"], d["temp"], d["heart_rate"], d["spo2"]] for d in data])
y_sbp = np.array([d["SBP"] for d in data])
y_dbp = np.array([d["DBP"] for d in data])


# Split the dataset into training and testing sets
X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(X, y_sbp, y_dbp, test_size=0.2, random_state=42)

print(X_train)


print("Creating CNN models...")


def create_sbp_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for SBP prediction
    return model

def create_dbp_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for DBP prediction
    return model

input_shape = (X_train.shape[1], 1)  # Shape for Conv1D input

# Create SBP model
sbp_model = create_sbp_model(input_shape)
sbp_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Create DBP model
dbp_model = create_dbp_model(input_shape)
dbp_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("Train and evaluate SBP model")

# Reshape input data for Conv1D
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train SBP model
sbp_model.fit(X_train_reshaped, y_sbp_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Train DBP model
dbp_model.fit(X_train_reshaped, y_dbp_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate models
sbp_loss, sbp_mae = sbp_model.evaluate(X_test_reshaped, y_sbp_test, verbose=0)
dbp_loss, dbp_mae = dbp_model.evaluate(X_test_reshaped, y_dbp_test, verbose=0)

print("SBP Model - Loss:", sbp_loss, "MAE:", sbp_mae)
print("DBP Model - Loss:", dbp_loss, "MAE:", dbp_mae)

# After training the models
print("Saving models")

# Save SBP model
sbp_model.save("model/sbp_model.keras")

# Save DBP model
dbp_model.save("model/dbp_model.keras")