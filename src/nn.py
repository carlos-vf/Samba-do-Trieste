import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from preprocessing import Preprocess, FILE_PATH
    
data = pd.read_csv(FILE_PATH)

preprocess = Preprocess(data)
preprocess.preprocess_data()
processed_data = preprocess.get_data()

X_train, y_train, X_test, y_test = preprocess.split_test_train()

print(X_train.head(1))
print(y_train.head(1))
print(X_test.head(1))
print(y_test.head(1))

# Convert DataFrame to numpy arrays
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Define a simple feed-forward neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# Final layer with ReLU to ensure non-negative output
model.add(Dense(1, activation='relu'))

# Compile the model with mean squared error loss
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train_scaled, y_train_np,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Make predictions on test data
predictions = model.predict(X_test_scaled)
# Convert predictions to positive integers (rounding)
predictions = np.round(predictions.flatten()).astype(int)

# Display sample predictions
print("Sample Predictions:", predictions[:5])

