from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# Generador de imágenes con más aumentos de datos
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.3,  # Incremento de shear
    zoom_range=0.3,   # Mayor zoom range para más variedad
    rotation_range=30,  # Rotación de las imágenes
    horizontal_flip=True,
    vertical_flip=True  # Agregar flips verticales
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes desde las carpetas
train_generator = train_datagen.flow_from_directory(
    'train/',  
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary'  
)

test_generator = test_datagen.flow_from_directory(
    'test/',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Creación del modelo CNN mejorado
model = Sequential()

# Primera capa de convolución, batch normalization y pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa de convolución, batch normalization y pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa de convolución, batch normalization y pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Cuarta capa de convolución y pooling
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar la matriz
model.add(Flatten())

# Capa densa con Dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida (binaria)
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo con un learning rate ajustado
optimizer = Adam(learning_rate=0.001)  # Aumentar ligeramente el learning rate

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback para detener el entrenamiento si la validación no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  
    epochs=3,  # Mantener 3 épocas
    validation_data=test_generator,
    validation_steps=50,
    callbacks=[early_stopping]  # Añadir early stopping
)

# Evaluar el modelo en los datos de prueba
scores = model.evaluate(test_generator)
print(f"Precisión en test: {scores[1]*100}%")

y_pred = (model.predict(test_generator) > 0.5).astype("int32")
precision = precision_score(test_generator.classes, y_pred)
print(f"Precisión (Precision): {precision:.2f}")

recall = recall_score(test_generator.classes, y_pred)
print(f"Sensibilidad (Recall): {recall:.2f}")

# Matriz de confusión
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.ylabel('Actual')
plt.xlabel('Predicho')
plt.title('Matriz de Confusión')
plt.show()

# Predicciones detalladas
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)

# Obtener las etiquetas reales del test set
y_true = test_generator.classes

y_pred = y_pred.flatten()

correct_predictions = np.where(y_pred == y_true)[0]
incorrect_predictions = np.where(y_pred != y_true)[0]

print(f"Total de imágenes de prueba: {len(y_true)}")
print(f"Imágenes correctamente clasificadas: {len(correct_predictions)}")
print(f"Imágenes incorrectamente clasificadas: {len(incorrect_predictions)}")

for i in correct_predictions[:5]:
    print(f"Imagen {i}: Correcto - Etiqueta real: {y_true[i]}, Predicción: {y_pred[i]}")

for i in incorrect_predictions[:5]:
    print(f"Imagen {i}: Incorrecto - Etiqueta real: {y_true[i]}, Predicción: {y_pred[i]}")
