# App Web con Streamlit para Predicción de Radiación Solar

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO
from datetime import datetime
from io import BytesIO, StringIO

# Configuración inicial de la página
st.set_page_config(
    page_title='Predicción Radiación Solar', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# Estilos 
def set_custom_style():
    st.markdown(f"""
    <style>
        /* Fondo principal */
        .stApp {{
            background: url("https://raw.githubusercontent.com/jaimearce/mi-app-lstm-radiacion/main/fondo.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Tipografía y títulos */
        h1, h2, h3, h4, h5, h6 {{
            color: #1B1F23 !important;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* Texto general */
        .stMarkdown, .stText, .css-1aumxhk, .css-1v3fvcr {{
            color: #222222 !important;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* Contenedores principales */
        .block-container {{
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}

        /* Botones */
        .stButton>button {{
            background-color: #007ACC !important;
            color: white !important;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #005A99 !important;
            transform: translateY(-1px);
        }}

        /* Métricas y cuadros */
        .stMetric {{
            background-color: #E6F0FA !important;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }}

        /* DataFrame y alertas */
        .stDataFrame, .stAlert {{
            background-color: #FAFAFA !important;
            border-radius: 10px;
            border: 1px solid #ddd;
        }}

        /* Barra lateral */
        .css-1aumxhk {{
            background-color: #f0f2f6 !important;
        }}

        /* Inputs */
        .stSelectbox, .stTextInput, .stFileUploader {{
            background-color: #ffffff !important;
            border: 1px solid #ccc;
            border-radius: 6px;
        }}

        /* Hover de tarjetas */
        .block-container:hover {{
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }}
    </style>
    """, unsafe_allow_html=True)

set_custom_style()

# Título principal
st.title('Predicción de Radiación Solar')

# --------------------------------------------
# Sección de Ayuda y Descripción
# --------------------------------------------
with st.expander("Cómo usar esta app", expanded=False):
    st.markdown("""
    **Guía de uso:**
    
    1. **Selecciona datos**: Elige una ubicación predeterminada o sube tu propio archivo CSV.
    2. **Configura el modelo**: Ajusta el parámetro de lookback (número de pasos hacia atrás).
    3. **Analiza resultados**: Revisa las predicciones por intervalos, métricas y recomendaciones.
    
    **Requisitos del archivo CSV:**
    - Debe contener la columna `ALLSKY_SFC_SW_DWN` con los valores de radiación solar.
    - Opcionalmente puede incluir una columna de fechas.
    
    **Novedades en esta versión:**
    - Predicciones para 1h, 3h, 6h y 12h
    - Interpretación de condiciones solares
    - Recomendaciones prácticas para sistemas fotovoltaicos
    """)

# --------------------------------------------
# Sección de Créditos Académicos
# --------------------------------------------
with st.expander("Creditos Academicos", expanded=False):
    st.markdown("""

**Software desarrollado para la Universidad Francisco de Paula Santander**  
**Curso Integrador II - Ingeniería Electrónica**  
**Autores**: Jaime Arce, Johan Salazar, Angel Hernández, Frank Portillo  
**Asesor**: MSc. IE. Darwin O. Cardoso S. | **Coasesora**: MSc. IE. Oriana A. Lopez B.  
**Año**: 2025  
**Versión**: 2.1.0 | Última actualización: """ + datetime.now().strftime("%Y-%m-%d") + """  
""")

# --------------------------------------------
# Funciones Auxiliares
# --------------------------------------------
@st.cache_data
def load_data(url):
    """Carga datos desde una URL con caché para mejor rendimiento"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

def create_sequences(data, steps):
    """Crea secuencias para el modelo LSTM"""
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def interpret_radiation(value):
    """Traduce el valor de radiación a condiciones solares"""
    if isinstance(value, (np.ndarray, list)):
        value = value[0] if len(value) > 0 else 0
    
    if value < 100:
        return "Nublado - Baja producción solar"
    elif 100 <= value < 300:
        return "Parcialmente nublado - Producción moderada"
    elif 300 <= value < 600:
        return "Mayormente soleado - Buena producción"
    else:
        return "Soleado - Excelente producción solar"

def get_recommendation(avg_radiation):
    """Genera recomendaciones basadas en la radiación promedio"""
    if avg_radiation < 100:
        return {
            "color": "red",
            "message": "Baja producción solar esperada",
            "recommendations": [
                "Considerar usar energía almacenada en baterías",
                "Limitar cargas no esenciales",
                "Verificar estado del sistema"
            ]
        }
    elif 100 <= avg_radiation < 300:
        return {
            "color": "orange",
            "message": "Producción solar moderada",
            "recommendations": [
                "Buen momento para cargas estándar",
                "Cargar baterías si es necesario",
                "Monitorear producción"
            ]
        }
    else:
        return {
            "color": "green",
            "message": "Alta producción solar esperada",
            "recommendations": [
                "Excelente momento para cargas pesadas",
                "Cargar baterías al máximo",
                "Considerar excedentes para inyección a red"
            ]
        }

def show_prediction_card(title, value, delta=None, interpretation=""):
    """Muestra una tarjeta de predicción profesional"""
    delta_html = f"<div style='color: #666; font-size: 14px;'>{delta}</div>" if delta else ""
    
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    ">
        <h3 style="color: #003366; margin-top: 0;">{title}</h3>
        <div style="font-size: 24px; font-weight: bold; color: #0066cc;">{value}</div>
        {delta_html}
        <div style="margin-top: 10px; padding: 10px; background: #f5f9ff; border-radius: 5px;">
            {interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)

def plot_radiation_area(pred, time_steps=24):
    """
    Gráfico para radiación solar 
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Convertir a 1D si es necesario
        pred = np.array(pred).flatten()  # Esto asegura 1D
        
        plt.rcParams.update({
            'axes.facecolor': '#f0f2f6',
            'figure.facecolor': '#f0f2f6',
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'text.color': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.color': '#dddddd'
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Asegurar que los datos sean 1D
        ax.plot(pred[:time_steps], marker='o', linestyle='-', color='#007ACC', label='Predicción')
        ax.fill_between(range(time_steps), pred[:time_steps], alpha=0.2, color='#007ACC')
        
        ax.set_title("Predicción de Radiación Solar (Próximas Horas)")
        ax.set_xlabel("Horas")
        ax.set_ylabel("Radiación Solar (W/m²)")
        ax.grid(True)
        ax.legend()
        
        plt.close(fig)
        return fig

    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        fig = plt.figure(figsize=(12, 6))
        plt.close(fig)
        return fig
    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        fig = plt.figure(figsize=(12, 6))
        plt.close(fig)
        return fig

# --------------------------------------------
# Pestañas Principales
# --------------------------------------------
tab1, tab2 = st.tabs(["Predicción", "Resultados"])

with tab1:
    st.subheader("Carga de Datos")
    
    # Opciones de datos predefinidos
    dataset_option = st.selectbox("Selecciona una ubicación de datos (NASA POWER)", [
        "Seleccionar...", 
        "Datos_Centro_Cucuta_1h",
        "Datos_Torcoroma2_Cucuta", 
        "Datos_LomaDeBolivar_Cucuta",
        "Datos_AntoniaSantos_Cucuta",
        "Patios_Centro",
        "Datos_Aeropuerto_filtrado"
    ], help="Datos de radiación solar de estaciones meteorológicas")
    
    # URLs de datos (ejemplo)
    urls = {
        "Datos_Centro_Cucuta_1h":"https://www.dropbox.com/scl/fi/ksmju96igz34ic19ht2ea/Datos_Centro_Cucuta_1h.csv?rlkey=6u1tkv4gr0en0vulrmt9ek3y5&st=2fxqn6be&dl=1",
        "Datos_Torcoroma2_Cucuta":"https://www.dropbox.com/scl/fi/gd2z0zjaemwhed950tzhr/Datos_Torcoroma2_Cucuta.csv?rlkey=saq2y5g0fb0auh722cxf9z87q&st=tvkobdpj&dl=1",
        "Datos_LomaDeBolivar_Cucuta":"https://www.dropbox.com/scl/fi/k5pv3oq58lile65ukqltf/Datos_LomaDeBolivar_Cucuta.csv?rlkey=igekok84zwwjtcejrivgggjjo&st=085ps83b&dl=1",
        "Datos_AntoniaSantos_Cucuta":"https://www.dropbox.com/scl/fi/ckemqwwwh98zw1uuaxklo/Datos_AntoniaSantos_Cucuta.csv?rlkey=1bhwj7q64jgdjzfk5qbt89x77&st=abuqlrkh&dl=1",
        "Patios_Centro":"https://www.dropbox.com/scl/fi/jwld88o2z7okv81k8dcb5/Patios_Centro.csv?rlkey=yuryueanbp2rgt2x6qiue0bhz&st=3vs1cp7a&dl=1",
        "Datos_Aeropuerto_filtrado":"https://www.dropbox.com/scl/fi/l0z0z8itycfps6687dziz/Datos_Aeropuerto_filtrado.csv?rlkey=ss6cdmll2qb4cnrbpotljizxg&st=m13g3dui&dl=1"
    }

    df = None

    # Carga de datos predefinidos
    if dataset_option != "Seleccionar...":
        with st.spinner(f'Cargando datos de {dataset_option}...'):
            df = load_data(urls.get(dataset_option))
            if df is not None:
                st.success(f"Datos cargados: {dataset_option}")
                st.session_state['data_source'] = dataset_option

    # Opción para subir archivo personalizado
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'], 
                                   help="El archivo debe contener la columna ALLSKY_SFC_SW_DWN")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'ALLSKY_SFC_SW_DWN' not in df.columns:
                st.error("El archivo debe contener la columna 'ALLSKY_SFC_SW_DWN'")
            else:
                st.success("Archivo cargado correctamente")
                st.session_state['data_source'] = "Archivo personalizado"
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")

    # Configuración del modelo
    if df is not None:
        st.subheader("Configuración del Modelo")
        n_steps = st.selectbox(
            'Número de pasos hacia atrás (lookback)',
            options=[1, 6, 12, 24, 48, 72],
            index=3,
            help="Determina cuántos puntos anteriores usará el modelo para cada predicción de 1 - 100"
        )
        
        # Mostrar vista previa de datos
        with st.expander("Vista previa de los datos"):
            st.dataframe(df.head(), height=150)
            st.write(f"Total de registros: {len(df)}")

        # Procesamiento y predicción
        if st.button("Ejecutar Predicción", type="primary"):
            if 'ALLSKY_SFC_SW_DWN' not in df.columns:
                st.error("Los datos no contienen la columna requerida")
            else:
                with st.spinner('Procesando datos...'):
                    # Normalización
                    series = df['ALLSKY_SFC_SW_DWN'].values.reshape(-1, 1)
                    scaler = MinMaxScaler()
                    series_scaled = scaler.fit_transform(series)
                    
                    # Crear secuencias
                    X, y = create_sequences(series_scaled, n_steps)
                    
                    # Cargar modelo
                    try:
                        model = load_model('modelo_lstm_radiacion.keras')
                        
                        # Barra de progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Realizando predicción...")
                        y_pred = model.predict(X)
                        progress_bar.progress(50)
                        
                        # Inversión de la normalización
                        y_inv = scaler.inverse_transform(y)
                        y_pred_inv = scaler.inverse_transform(y_pred)
                        progress_bar.progress(80)
                        
                        # Cálculo de métricas adicionales
                        y_ma = pd.Series(y_inv.flatten()).rolling(window=n_steps).mean().fillna(method='bfill')
                        lr = LinearRegression().fit(np.arange(len(y_inv)).reshape(-1,1), y_inv)
                        y_lr = lr.predict(np.arange(len(y_inv)).reshape(-1,1))
                        progress_bar.progress(100)
                        
                        # Guardar resultados en session state
                        st.session_state.update({
                            "y_inv": y_inv,
                            "y_pred_inv": y_pred_inv,
                            "y_ma": y_ma,
                            "y_lr": y_lr,
                            "n_steps": n_steps,
                            "scaler": scaler
                        })
                        
                        st.success("Predicción completada!")
                        
                    except Exception as e:
                        st.error(f"Error en la predicción: {str(e)}")
                        st.error("Asegúrese que el archivo 'modelo_lstm_radiacion.keras' está en el directorio correcto")

with tab2:
    st.subheader("Resultados de la Predicción")
    
    if "y_inv" not in st.session_state:
        st.info("Realiza una predicción en la pestaña anterior para ver los resultados")
    else:
        # Recuperar datos de la sesión
        y_inv = st.session_state["y_inv"]
        y_pred_inv = st.session_state["y_pred_inv"]
        y_ma = st.session_state["y_ma"]
        y_lr = st.session_state["y_lr"]
        n_steps = st.session_state["n_steps"]
        scaler = st.session_state.get("scaler", None)
        
        # Mostrar predicciones para diferentes lapsos
        st.subheader("Predicción para Próximas Horas")
        
        cols = st.columns(4)
        time_intervals = [1, 3, 6, 12]
        
        for i, hours in enumerate(time_intervals):
            with cols[i]:
                if hours <= len(y_pred_inv):
                    pred_values = y_pred_inv[:hours]
                    avg_pred = np.mean(pred_values)
                    current_value = y_inv[0][0] if len(y_inv) > 0 else 0
                    
                    # Calcular cambio porcentual
                    if current_value != 0:
                        delta_pct = ((avg_pred - current_value) / current_value) * 100
                        delta_text = f"{delta_pct:.1f}%"
                        delta_color = "inverse" if delta_pct < 0 else "normal"
                    else:
                        delta_text = " "
                        delta_color = "off"
                    
                    # Mostrar tarjeta de predicción
                    show_prediction_card(
                        title=f"Próximas {hours} hora{'s' if hours > 1 else ''}",
                        value=f"{avg_pred:.1f} W/m²",
                        delta=delta_text,
                        interpretation=interpret_radiation(avg_pred))
                    
                else:
                    st.warning(f"No hay suficientes datos para {hours} horas")
        
        # Mostrar métricas de rendimiento
        st.subheader("Métricas de Rendimiento del Modelo")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))
            st.metric("RMSE (Error Cuadrático Medio)", f"{rmse:.2f}", 
                     help="Medida de la diferencia entre valores predichos y reales")
        
        with col2:
            mae = mean_absolute_error(y_inv, y_pred_inv)
            st.metric("MAE (Error Absoluto Medio)", f"{mae:.2f}",
                     help="Error promedio absoluto entre predicciones y valores reales")
        
        with col3:
            r2 = r2_score(y_inv, y_pred_inv)
            st.metric("R² (Coeficiente de Determinación)", f"{r2:.2f}",
                     help="Proporción de la varianza explicada por el modelo (0-1)")
        
        # Gráfico de resultados mejorado
        st.subheader("Distribución Horaria de Radiación")
        fig = plot_radiation_area(y_pred_inv, time_steps=n_steps)
        st.pyplot(fig)
                
        # Recomendaciones basadas en la predicción
        st.subheader("Recomendaciones para Sistemas Solares")
        avg_next_6h = np.mean(y_pred_inv[:6]) if len(y_pred_inv) >= 6 else np.mean(y_pred_inv)
        recommendation = get_recommendation(avg_next_6h)
        
        # Mostrar alerta según el nivel de radiación
        if recommendation["color"] == "red":
            st.error(f"""
            **{recommendation['message']}**  
            {interpret_radiation(avg_next_6h)}
            """)
        elif recommendation["color"] == "orange":
            st.warning(f"""
            **{recommendation['message']}**  
            {interpret_radiation(avg_next_6h)}
            """)
        else:
            st.success(f"""
            **{recommendation['message']}**  
            {interpret_radiation(avg_next_6h)}
            """)
        
        # Mostrar recomendaciones en lista
        st.markdown("**Acciones recomendadas:**")
        for item in recommendation["recommendations"]:
            st.markdown(f"- {item}")
        
        # Predicción extendida (opcional)
        with st.expander("Predicción extendida para las próximas 24 horas"):
            if len(y_pred_inv) >= 24:
                extended_pred = y_pred_inv[:24]
                extended_fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(extended_pred, color='#0066cc', linestyle='-', linewidth=2)
                ax.set_title("Predicción de radiación solar - Próximas 24 horas", fontsize=12)
                ax.set_xlabel("Horas", fontsize=10)
                ax.set_ylabel("Radiación (W/m²)", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.3)
                st.pyplot(extended_fig)
                
                # Resumen estadístico
                st.markdown(f"""
                **Resumen estadístico (24 horas):**
                - Promedio: {np.mean(extended_pred):.1f} W/m²
                - Mínimo: {np.min(extended_pred):.1f} W/m² (hora {np.argmin(extended_pred) + 1})
                - Máximo: {np.max(extended_pred):.1f} W/m² (hora {np.argmax(extended_pred) + 1})
                """)
            else:
                st.warning("No hay suficientes datos para mostrar la predicción de 24 horas")
        
        # Descarga de resultados
        st.subheader("Exportar Resultados")
        pred_df = pd.DataFrame({
            'Real': y_inv.flatten(),
            'Prediccion_LSTM': y_pred_inv.flatten(),
            'Media_Movil': y_ma.values,
            'Tendencia_Lineal': y_lr.flatten()
        })
        
        # Convertir a CSV
        csv = pred_df.to_csv(index=False).encode('utf-8')
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Descargar como CSV",
                data=csv,
                file_name='prediccion_radiacion.csv',
                mime='text/csv',
                help="Descarga los resultados en formato CSV"
            )
        
        with col2:
            # Crear una figura nueva para la descarga
            fig_descarga = plt.figure(figsize=(12, 6))
            plt.plot(y_pred_inv[:24], color='#0066cc', linewidth=2)
            plt.title("Predicción de Radiación Solar")
            plt.xlabel("Horas")
            plt.ylabel("Radiación (W/m²)")
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Guardar la figura en un buffer de memoria
            buffer = BytesIO()
            fig_descarga.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
            plt.close(fig_descarga)  # Cerrar la figura para liberar memoria
            buffer.seek(0)  # Rebobinar el buffer al inicio
            
            # Botón de descarga
            st.download_button(
                label="Descargar Gráfico",
                data=buffer,
                file_name='prediccion_radiacion.png',
                mime='image/png'
            )
        
        # Explicación técnica
        with st.expander("Detalles Técnicos"):
            st.markdown("""
            **Modelo LSTM utilizado:**
            - Arquitectura: 2 capas LSTM con 50 neuronas cada una
            - Función de activación: tanh
            - Dropout: 0.2 para regularización
            - Optimizador: Adam
            - Función de pérdida: Error Cuadrático Medio (MSE)
            
            **Preprocesamiento:**
            - Normalización Min-Max (0-1)
            - Secuencias de {} pasos temporales
            
            **Interpretación de valores de radiación:**
            - < 100 W/m²: Condiciones nubladas
            - 100-300 W/m²: Parcialmente nublado
            - 300-600 W/m²: Mayormente soleado
            - > 600 W/m²: Soleado
            """.format(n_steps))
