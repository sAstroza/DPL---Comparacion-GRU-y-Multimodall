import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar el conjunto de datos
data = pd.read_csv('lechuzasdataset.csv')

# Selección de características (inputs) y objetivo (target)
X_radiacion = data[['Radiacion']].values
X_temperatura = data[['Temperatura']].values
X_temperatura_panel = data[['Temperatura panel']].values
y_potencia = data['Potencia'].values

# Escalar los datos
scaler_X = StandardScaler()
X_radiacion_scaled = scaler_X.fit_transform(X_radiacion)
X_temperatura_scaled = scaler_X.fit_transform(X_temperatura)
X_temperatura_panel_scaled = scaler_X.fit_transform(X_temperatura_panel)

scaler_y = StandardScaler()
y_potencia_scaled = scaler_y.fit_transform(y_potencia.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y prueba
X_radiacion_train, X_radiacion_test, X_temperatura_train, X_temperatura_test, X_temperatura_panel_train, X_temperatura_panel_test, y_train, y_test = train_test_split(
    X_radiacion_scaled, X_temperatura_scaled, X_temperatura_panel_scaled, y_potencia_scaled, test_size=0.2, random_state=42)

# Definición del modelo multimodal
def create_multimodal():
    input_radiacion = Input(shape=(1,))
    input_temperatura = Input(shape=(1,))
    input_temperatura_panel = Input(shape=(1,))

    # Procesamiento de cada entrada por separado
    dense_radiacion = Dense(32, activation='relu')(input_radiacion)
    dense_radiacion = Dropout(0.2)(dense_radiacion)

    dense_temperatura = Dense(32, activation='relu')(input_temperatura)
    dense_temperatura = Dropout(0.2)(dense_temperatura)

    dense_temperatura_panel = Dense(32, activation='relu')(input_temperatura_panel)
    dense_temperatura_panel = Dropout(0.2)(dense_temperatura_panel)

    # Concatenación de todas las ramas
    concatenated = concatenate([dense_radiacion, dense_temperatura, dense_temperatura_panel])

    # Capa de salida
    output = Dense(1)(concatenated)

    # Definir y compilar el modelo
    model = Model(inputs=[input_radiacion, input_temperatura, input_temperatura_panel], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

# Crear el modelo
model = create_multimodal()

# Medir el tiempo de entrenamiento
import time
start_time_train = time.time()  # Iniciar temporizador de entrenamiento
history = model.fit(
    [X_radiacion_train, X_temperatura_train, X_temperatura_panel_train],
    y_train,
    epochs=50,  # Aumentado a 50 épocas
    batch_size=32,
    validation_split=0.2
)
end_time_train = time.time()  # Detener temporizador de entrenamiento
training_time = end_time_train - start_time_train
print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

# Medir el tiempo de evaluación
start_time_eval = time.time()  # Iniciar temporizador de evaluación
y_pred = model.predict([X_radiacion_test, X_temperatura_test, X_temperatura_panel_test])
end_time_eval = time.time()  # Detener temporizador de evaluación
evaluation_time = end_time_eval - start_time_eval
print(f'Tiempo de evaluación: {evaluation_time:.2f} segundos')

# Calcular la pérdida en el conjunto de prueba
loss = model.evaluate([X_radiacion_test, X_temperatura_test, X_temperatura_panel_test], y_test)
print(f'Pérdida en el conjunto de prueba: {loss}')

# Calcular las métricas adicionales
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar las métricas adicionales
print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'Error Absoluto Medio (MAE): {mae}')
print(f'Coeficiente de Determinación (R²): {r2}')