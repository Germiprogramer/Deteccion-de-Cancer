from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import cv2
import seaborn as sns


from sklearn.metrics import precision_score, recall_score, confusion_matrix


# Generador de imágenes con aumentos de datos
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes desde las carpetas
train_generator = train_datagen.flow_from_directory(
    'train/',  # Ruta de las imágenes de entrenamiento
    target_size=(224, 224),  # Tamaño al que se redimensionarán las imágenes
    batch_size=32,
    class_mode='binary'  # Clasificación binaria (benigno/maligno)
)

test_generator = test_datagen.flow_from_directory(
    'test/',  # Ruta de las imágenes de prueba
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Creación del modelo CNN
model = Sequential()

# Primera capa de convolución y pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa de convolución y pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa de convolución y pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar la matriz
model.add(Flatten())

# Capa densa totalmente conectada
model.add(Dense(128, activation='relu'))

# Capa de salida (binaria)
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # número de lotes por época
    epochs=10,  # número de épocas
    validation_data=test_generator,
    validation_steps=50
)

# Evaluar el modelo en los datos de prueba
scores = model.evaluate(test_generator)
print(f"Precisión en test: {scores[1]*100}%")


y_pred = (model.predict(test_generator) > 0.5).astype("int32")
precision = precision_score(test_generator.classes, y_pred)
print(f"Precisión (Precision): {precision:.2f}")



recall = recall_score(test_generator.classes, y_pred)
print(f"Sensibilidad (Recall): {recall:.2f}")


cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.ylabel('Actual')
plt.xlabel('Predicho')
plt.title('Matriz de Confusión')
plt.show()


def grad_cam(input_model, image, layer_name):
    # Procesamiento para generar mapa de calor
    pass  # Aquí implementarías el Grad-CAM
