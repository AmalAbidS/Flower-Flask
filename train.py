import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from roboflow import Roboflow
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Roboflow API setup
rf = Roboflow(api_key="E4m7B8OuxL1fsHGNcT8E")
project = rf.workspace("asas-xloki").project("227-yc1vm")
version = project.version(1)
dataset = version.download("folder")

# Define dataset paths
train_dir = os.path.join(dataset.location, 'train')  # Adjust if necessary
test_dir = os.path.join(dataset.location, 'test')    # Adjust if necessary

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    horizontal_flip=True  # Randomly flip images horizontally
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize for test data

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), # Resize images
    batch_size=32,
    class_mode='sparse'     # Use sparse categorical labels
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False  # Ensure we get the true labels in order for evaluation
)

# Print class indices
print("Class indices:", train_data.class_indices)

# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')  # Number of classes
])

# Print the model summary
model.summary()

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_data,
                    epochs=20,  # Increase number of epochs
                    validation_data=test_data)

# Evaluate the Model
loss, accuracy = model.evaluate(test_data)
print(f"Test loss: {loss:.2f}")
print(f"Test accuracy: {accuracy:.2f}")

# Save the Model
model.save('model/my_model.h5')
