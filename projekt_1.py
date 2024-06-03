import os

# Onemogočanje TensorRT in oneDNN optimizacij
os.environ['TF_TRT_MODE'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Preverjanje različic
print(f"TensorFlow version: {tf.__version__}")

# Priprava podatkov
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalizacija podatkov
train_images = train_images / 255.0
test_images = test_images / 255.0

# Razdelitev podatkov na učni in validacijski del
val_images = train_images[:5000]
val_labels = train_labels[:5000]
train_images = train_images[5000:]
train_labels = train_labels[5000:]

print_memory_usage()

# Izbor modela
def create_model(optimizer='adam', filters=32):
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(filters, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(filters*2, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Učenje modela
model = create_model()
history = model.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels), batch_size=32)
print("Initial model training complete")

print_memory_usage()

# Optimizacija hiperparametrov
model = KerasClassifier(model=create_model, optimizer='adam', filters=32, epochs=1, batch_size=32)

param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'filters': [32, 64],
    'epochs': [10, 20],  # zmanjšano število epoch za hitrejše testiranje
    'batch_size': [32, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_images, train_labels)
print("Grid search complete")

best_params = grid_result.best_params_
print(f"Best parameters: {best_params}")

print_memory_usage()

# Evaluacija modela
test_loss, test_accuracy = grid_result.best_estimator_.model_.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")

print_memory_usage()
