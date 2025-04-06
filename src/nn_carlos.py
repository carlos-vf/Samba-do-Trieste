from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import numpy as np


def neural_network(X_train, y_train, X_val, y_val, X_test, y_test):

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Neural Network Architecture
    model = models.Sequential()

    # Input layer (shape is the number of features)
    model.add(layers.Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(layers.Dropout(0.2))  # Dropout for regularization
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))  # Dropout for regularization
    model.add(layers.Dense(32, activation='relu'))

    # Output layer (single neuron for regression, no activation function)
    model.add(layers.Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss=custom_loss)

    # Train the model
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Make predictions
    y_pred = np.round(model.predict(X_test))

    return y_pred




def custom_loss(reference, proposal):
    # Avoid division by zero in reference by replacing 0 with 1
    new_reference = tf.where(reference == 0, 1.0, reference)
    
    # Calculate the loss
    loss = tf.reduce_mean(tf.square(reference - proposal) / new_reference)
    
    return loss
