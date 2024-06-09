import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import psutil
import argparse

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Disable TensorRT and oneDNN optimizations


# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Path to dataset
dataset_path = './test'  # Correct path to your dataset

# Load and preprocess face dataset
data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    dataset_path,  # Ensure this path is correct
    target_size=(224, 224),  # Match pre-trained model input size
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    dataset_path,  # Ensure this path is correct
    target_size=(224, 224),  # Match pre-trained model input size
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Debugging prints
print(f"Number of training samples: {train_data.samples}")
print(f"Number of validation samples: {val_data.samples}")

if train_data.samples == 0 or val_data.samples == 0:
    raise ValueError("Training or validation data is empty. Check the dataset path and structure.")

print_memory_usage()

# Function to create model with transfer learning
def create_model(optimizer='adam', dropout_rate=0.5):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification for face ID
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--user_id', type=str, required=True, help='User ID for the model')
args = parser.parse_args()
username = args.user_id

# Train initial model
model = create_model()
history = model.fit(train_data, epochs=1, validation_data=val_data)
print("Initial model training complete")

# Save the initial model
model_save_path = f"{username}.h5"
model.save(model_save_path)
print("Initial model saved as {model_save_path}")

print_memory_usage()

# Hyperparameter optimization
#model = KerasClassifier(model=create_model, optimizer='adam', dropout_rate=0.5, epochs=2, batch_size=32)
'''
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.5, 0.6],
    'epochs': [10, 20],
    'batch_size': [32, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(train_data)

print("Grid search complete")

best_params = grid_result.best_params_
print(f"Best parameters: {best_params}")

print_memory_usage()

# Save the best model from GridSearchCV
best_model = grid_result.best_estimator_.model_
best_model.save("best_model.h5")
print("Best model saved as best_model.h5")
'''
# Evaluate the best model
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Validation accuracy: {test_accuracy}")

print_memory_usage()