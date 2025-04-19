# App Web con Streamlit para Predicción de Radiación Solar
# Esta app permite subir archivos CSV, hacer predicciones con un modelo LSTM entrenado, 
# evaluar métricas y descargar resultados.
# Código principal de la app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from keras.models import load_model
from io import BytesIO

# Configurar la página de Streamlit
st.set_page_config(page_title='Predicción Radiación Solar', layout='wide')

# Título de la aplicación
st.title('☀️ Predicción de Radiación Solar ALLSKY_SFC_SW_DWN')

# Widget para subir archivo CSV
uploaded_file = st.file_uploader('Sube tu archivo CSV con la columna ALLSKY_SFC_SW_DWN')

# Widget para seleccionar el número de pasos hacia atrás (lookback)
n_steps = st.slider('Selecciona el número de pasos hacia atrás (lookback)', 1, 100, 24)

# Si se ha subido un archivo
if uploaded_file is not None:
    # Leer el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(uploaded_file)
    
    # Extraer la columna 'ALLSKY_SFC_SW_DWN' y convertirla a un array NumPy
    series = df['ALLSKY_SFC_SW_DWN'].values.reshape(-1, 1)

    # Importar MinMaxScaler para escalar los datos
    from sklearn.preprocessing import MinMaxScaler
    
    # Crear un objeto MinMaxScaler
    scaler = MinMaxScaler()
    
    # Escalar los datos al rango [0, 1]
    series_scaled = scaler.fit_transform(series)

    # Función para crear secuencias de datos para el modelo LSTM
    def create_sequences(data, steps):
        """
        Crea secuencias de datos para el modelo LSTM.

        Args:
            data: El array de datos.
            steps: El número de pasos hacia atrás (lookback).

        Returns:
            Una tupla con dos arrays: X (datos de entrada) e y (datos de salida).
        """
        X, y = [], []
        for i in range(steps, len(data)):
            X.append(data[i-steps:i])  # Secuencia de 'steps' valores anteriores
            y.append(data[i])          # Valor actual a predecir
        return np.array(X), np.array(y)

    # Crear las secuencias de datos
    X, y = create_sequences(series_scaled, n_steps)

    # Cargar el modelo LSTM pre-entrenado
    model = load_model('modelo_lstm_radiacion.keras')
    
    # Hacer predicciones con el modelo
    y_pred = model.predict(X)

    # Invertir el escalado para obtener las predicciones en la escala original
    y_inv = scaler.inverse_transform(y)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Calcular las métricas de evaluación
    rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))
    mae = mean_absolute_error(y_inv, y_pred_inv)
    r2 = r2_score(y_inv, y_pred_inv)

    # Mostrar las métricas en la aplicación
    st.subheader('📊 Métricas de Evaluación')
    st.write(f'**RMSE:** {rmse:.2f}')
    st.write(f'**MAE:** {mae:.2f}')
    st.write(f'**R²:** {r2:.2f}')

    # Comparar con Promedio Móvil y Regresión Lineal
    y_ma = pd.Series(y_inv.flatten()).rolling(window=n_steps).mean().fillna(method='bfill')
    lr = LinearRegression().fit(np.arange(len(y_inv)).reshape(-1,1), y_inv)
    y_lr = lr.predict(np.arange(len(y_inv)).reshape(-1,1))

    # Crear un gráfico de comparación
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_inv, label='Real')
    ax.plot(y_pred_inv, label='LSTM')
    ax.plot(y_ma, label='Promedio Móvil')
    ax.plot(y_lr, label='Regresión Lineal')
    ax.legend()
    ax.set_title('Comparación: Real vs Predicciones')
    
    # Mostrar el gráfico en la aplicación
    st.pyplot(fig)

    # Crear un DataFrame con las predicciones
    pred_df = pd.DataFrame({
        'Real': y_inv.flatten(),
        'LSTM': y_pred_inv.flatten(),
        'Media_Movil': y_ma.values,
        'Regresion_Lineal': y_lr.flatten()
    })

    # Convertir el DataFrame a CSV
    csv = pred_df.to_csv(index=False)
    
    # Botón para descargar el CSV con las predicciones
    st.download_button('⬇️ Descargar predicciones CSV', csv, 'predicciones.csv', 'text/csv')