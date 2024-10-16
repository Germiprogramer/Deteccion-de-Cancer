import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix


# Definir directorios de imágenes de entrenamiento y prueba
train_dir = 'train/'
test_dir =  'test/'

# Parámetros del modelo
ROWS, COLS = 224, 224  # Tamaño de las imágenes
BATCH = 32  # Tamaño del batch
N_CLASS = 2  # Número de clases (benigno y maligno)
EPOCHS = 7  # Número de épocas para entrenar
LR = 0.001  # Learning rate
'''
ROWS, COLS: El tamaño al que se redimensionan las imágenes (224x224 píxeles).
BATCH: Cantidad de imágenes que se procesan juntas en cada iteración.
N_CLASS: Número de clases, en este caso 2 (Benigno y Maligno).
EPOCHS: Número de pasadas completas sobre el conjunto de entrenamiento.
LR: Tasa de aprendizaje, que controla la magnitud del ajuste de pesos en cada paso de optimización.
'''

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
'''
rescale: Normaliza los valores de píxeles de 0-255 a 0-1.
rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, 
horizontal_flip: Parámetros para aplicar aumento de datos. 
Esto ayuda a que el modelo generalice mejor al entrenar con variaciones de las imágenes originales.
fill_mode='nearest': Rellena los píxeles que se generan fuera de los límites de la imagen cuando se aplica 
alguna transformación (rotación, desplazamiento, etc.).
'''

test_datagen = ImageDataGenerator(rescale=1./255)
#Para las imágenes de prueba, solo se aplica una normalización (rescale), sin aumento de datos.


train_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(ROWS, COLS),
    batch_size=BATCH,
    class_mode='categorical',  # Cambia a 'binary' si tienes solo dos clases
    shuffle=True
)
'''
flow_from_directory: Genera lotes de imágenes con etiquetas desde una carpeta. Aquí, las imágenes 
se redimensionan a 224x224, el batch size es de 32, y las clases se codifican en una 
representación categorical (one-hot encoded).
shuffle=True: Las imágenes se mezclan aleatoriamente en cada época.
'''

test_gen = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(ROWS, COLS),
    batch_size=BATCH,
    class_mode='categorical',  # Cambia a 'binary' si tienes solo dos clases
    shuffle=False
)
'''
Similar al generador de entrenamiento, pero en este caso no se mezclan las imágenes de prueba.
'''

# Construcción del modelo CNN
model = Sequential()

# Primera capa de convolución + MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(ROWS, COLS, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
Conv2D: Primera capa convolucional, con 32 filtros, cada uno de tamaño 3x3. Se utiliza la función de activación ReLU.
input_shape=(224, 224, 3): Define el tamaño de las imágenes de entrada (224x224 píxeles, con 3 canales para RGB).
MaxPooling2D: Reduce la dimensionalidad espacial (reduce la imagen en un factor de 2x2) para disminuir 
la cantidad de parámetros y evitar el sobreajuste.
'''


# Segunda capa de convolución + MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

'''
# Tercera capa de convolución + MaxPooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''


# Aplanamiento
model.add(Flatten())
'''
Flatten: Aplana la salida de las capas convolucionales en un vector unidimensional 
que se puede alimentar a una capa totalmente conectada (densa).
'''

# Capa densa
model.add(Dense(300, activation='relu')) #512
'''
Dense: Capa totalmente conectada con 512 neuronas y activación ReLU.

'''

# Capa de salida con 'softmax' para clasificación multiclase o 'sigmoid' para binaria
model.add(Dense(N_CLASS, activation='softmax'))
'''
Capa de salida con 2 neuronas (una para cada clase). 
Softmax se usa para clasificación multiclase (o binaria).
'''



# Compilación del modelo
model.compile(
    loss='categorical_crossentropy',  # Si tienes 2 clases, puedes cambiarlo a 'binary_crossentropy'
    optimizer=Adam(learning_rate=LR),
    metrics=['accuracy']
)
'''
loss='categorical_crossentropy': Función de pérdida para problemas de clasificación multiclase. 
Si fuera binario, se podría usar binary_crossentropy.
Adam: Optimizador popular que ajusta los pesos.
metrics=['accuracy']: Métrica utilizada para evaluar el rendimiento del modelo.

'''


# Resumen del modelo
model.summary()

# Entrenamiento del modelo
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen
)
'''
Entrena el modelo durante un número de épocas determinado. 
Usa los datos de entrenamiento y evalúa en el conjunto de prueba después de cada época.
'''


# Evaluación del modelo
test_loss, test_acc = model.evaluate(test_gen)
print(f"Accuracy en test: {test_acc * 100:.2f}%")
#Evalúa el modelo en el conjunto de prueba. Calcula la precisión y la pérdida.




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