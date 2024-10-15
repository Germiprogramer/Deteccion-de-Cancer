import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Definir directorios de imágenes de entrenamiento y prueba
train_dir = 'train/'
test_dir = 'test/'

# Parámetros del modelo
ROWS, COLS = 224, 224  # Tamaño de las imágenes
BATCH = 32  # Tamaño del batch
N_CLASS = 2  # Número de clases (benigno y maligno)
EPOCHS = 10  # Número de épocas
LR = 0.0001  # Learning rate

# Preprocesamiento y aumento de datos para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(ROWS, COLS),
    batch_size=BATCH,
    class_mode='categorical',  # Cambia a 'binary' si tienes solo dos clases
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(ROWS, COLS),
    batch_size=BATCH,
    class_mode='categorical',  # Cambia a 'binary' si tienes solo dos clases
    shuffle=False
)

# Construcción del modelo CNN con Dropout
model = Sequential()

# Primera capa de convolución + MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(ROWS, COLS, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
model.add(Dropout(0.25))

# Segunda capa de convolución + MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
model.add(Dropout(0.25))

# Tercera capa de convolución + MaxPooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
model.add(Dropout(0.5))

# Aplanamiento
model.add(Flatten())

# Capa densa
model.add(Dense(512, activation='relu'))

# Dropout en la capa densa
model.add(Dropout(0.5))

# Capa de salida con 'softmax' para clasificación multiclase
model.add(Dense(N_CLASS, activation='softmax'))

# Compilación del modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=LR),
    metrics=['accuracy']
)

# Resumen del modelo
model.summary()

# EarlyStopping para detener el entrenamiento si el rendimiento no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamiento del modelo
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=[early_stopping]
)

# Evaluación del modelo
test_loss, test_acc = model.evaluate(test_gen)
print(f"Accuracy en test: {test_acc * 100:.2f}%")

# Métricas adicionales
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes

# Reporte de clasificación y matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_true, y_pred))

print("Reporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

# Gráfica de la precisión y la pérdida
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.show()

plot_metrics(history)
