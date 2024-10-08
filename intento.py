from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import cv2
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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


# Hacer predicciones sobre las imágenes de prueba
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convertir probabilidades a 0 (benigno) y 1 (maligno)

# Obtener las etiquetas reales del test set
y_true = test_generator.classes  # Las etiquetas verdaderas (0 para benigno, 1 para maligno)


# Comparar las predicciones con las etiquetas reales
correct_predictions = np.where(y_pred == y_true)[0]  # Índices de las predicciones correctas
incorrect_predictions = np.where(y_pred != y_true)[0]  # Índices de las predicciones incorrectas

# Mostrar resultados
print(f"Total de imágenes de prueba: {len(y_true)}")
print(f"Imágenes correctamente clasificadas: {len(correct_predictions)}")
print(f"Imágenes incorrectamente clasificadas: {len(incorrect_predictions)}")

# Detallar algunos resultados (puedes ajustar cuántos mostrar)
for i in correct_predictions[:5]:  # Muestra hasta 5 predicciones correctas
    print(f"Imagen {i}: Correcto - Etiqueta real: {y_true[i]}, Predicción: {y_pred[i]}")

for i in incorrect_predictions[:5]:  # Muestra hasta 5 predicciones incorrectas
    print(f"Imagen {i}: Incorrecto - Etiqueta real: {y_true[i]}, Predicción: {y_pred[i]}")
