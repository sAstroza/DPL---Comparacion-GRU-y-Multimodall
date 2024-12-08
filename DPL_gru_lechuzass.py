import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# Cargar el dataset
data = pd.read_csv('lechuzasdataset.csv')

# Selección de características (inputs) y objetivo (target)
X = data[['Radiacion', 'Temperatura']]  # Variables de entrada
y = data['Potencia'].values  # Variable objetivo

# Escalar las variables de entrada y salida
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Red neuronal basada en GRU
model = Sequential()

# Capa GRU
model.add(GRU(64, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False))

# Capa de salida
model.add(Dense(1))

# Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Ajuste de hiperparámetros y entrenamiento con early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Reshape de los datos para ser adecuados para la capa GRU (3D)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Entrenamiento del modelo
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_val_reshaped, y_val), callbacks=[early_stopping])

# Medir el tiempo de evaluación
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
start_time_eval = time.time()  # Iniciar temporizador de evaluación
y_pred = model.predict(X_test_reshaped)
end_time_eval = time.time()  # Detener temporizador de evaluación
evaluation_time = end_time_eval - start_time_eval
print(f'Tiempo de evaluación: {evaluation_time:.2f} segundos')

# Evaluar el desempeño utilizando las métricas solicitadas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R²: {r2}')
