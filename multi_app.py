import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output # No se usan directamente en multi_app.py, pero son comunes en apps Dash
import plotly.express as px # No se usa directamente en multi_app.py, pero común en apps Dash
import joblib
import requests
import os
from datetime import datetime
import numpy as np

# --- Carga de Datos y Modelo (Global) ---
# Estos recursos se cargan al iniciar la aplicación.
# Para modelos y/o datasets extremadamente grandes que superen la RAM,
# se necesitarían estrategias más avanzadas (ej. bases de datos externas, Dask, etc.).

# ID del archivo del Modelo en Google Drive
file_id_modelo = "1wrdWPjF47w7IEf0WkRWMLTVgPWqT3Jpf"
drive_url_modelo = f"https://drive.google.com/uc?export=download&id={file_id_modelo}"
modelo_path = "modelo_forest.pkl" # Nombre local del archivo del modelo

# Descargar el modelo si no existe localmente
if not os.path.exists(modelo_path):
    print("Descargando modelo desde Google Drive...")
    try:
        r = requests.get(drive_url_modelo, stream=True)
        r.raise_for_status() # Lanza un error si la descarga no fue exitosa (código 4xx o 5xx)
        with open(modelo_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Modelo descargado.")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el modelo: {e}")
        print("Asegúrate de que el ID del archivo sea correcto y que el archivo sea accesible públicamente.")
        # Si el despliegue es en Render, podría fallar aquí si el archivo no se descarga.
        # Considera poner un modelo pequeño directamente en el repositorio con Git LFS si es viable.
    except Exception as e:
        print(f"Error inesperado al procesar el modelo: {e}")

try:
    modelo_forest = joblib.load(modelo_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # En un entorno de producción, aquí podrías querer que la aplicación no inicie
    # o que inicie en un modo degradado. Para Render, esto probablemente causaría un fallo.
    modelo_forest = None # Establecer a None para evitar errores si la carga falla

# Cargar los datos (DataFrame principal)
file_id_df = "1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
url_df = f"https://drive.google.com/uc?export=download&id={file_id_df}"

try:
    df = pd.read_csv(url_df)
    print("Datos descargados y cargados exitosamente.")

    # --- Optimización y Preprocesamiento de Datos ---
    # Estas funciones y columnas son necesarias globalmente si varias páginas las usan.
    def clasificar_edad(edad):
        if edad < 13: return "Niño"
        elif edad < 19: return "Adolescente"
        elif edad < 30: return "Joven"
        elif edad < 61: return "Adulto"
        elif edad < 200: return "Adulto mayor"
    df['Rango de Edad'] = df['EDAD'].apply(clasificar_edad).astype('category') # Convertir a categoría

    def clasificar_dias(dias):
        if dias < 10: return "0-9"
        elif dias < 20: return "10-19"
        elif dias < 30: return "20-29"
        elif dias < 40: return "30-39"
        elif dias < 50: return "40-49"
        elif dias < 60: return "50-59"
        elif dias < 70: return "60-69"
        elif dias < 80: return "70-79"
        elif dias < 90: return "80-89"
        else: return "90+"
    df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias).astype('category') # Convertir a categoría

    # Asegurar que las columnas sean categóricas para optimizar memoria y evitar warnings
    df['DIA_SOLICITACITA'] = pd.to_datetime(df['DIA_SOLICITACITA'], errors='coerce')
    df['MES'] = df['DIA_SOLICITACITA'].dt.to_period('M').astype(str).astype('category')
    df['ATENDIDO'] = df['ATENDIDO'].astype('category')
    df['PRESENCIAL_REMOTO'] = df['PRESENCIAL_REMOTO'].astype('category')
    df['ESPECIALIDAD'] = df['ESPECIALIDAD'].astype('category')
    df['SEGURO'] = df['SEGURO'].astype('category')
    df['SEXO'] = df['SEXO'].astype('category')

    # Calcular el uso de memoria del DataFrame
    memoria_df_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memoria del DataFrame (después de optimización): {memoria_df_mb:.2f} MB")

except requests.exceptions.RequestException as e:
    print(f"Error al descargar los datos: {e}")
    print("Asegúrate de que el ID del archivo de datos sea correcto y que el archivo sea accesible públicamente.")
    df = pd.DataFrame() # Crear un DataFrame vacío para que la app no falle al iniciar
except Exception as e:
    print(f"Error inesperado al cargar o procesar los datos: {e}")
    df = pd.DataFrame() # Crear un DataFrame vacío


# --- Flask Server y Configuración de la Aplicación Dash Principal ---
server = Flask(__name__)

# Servir archivos estáticos (como tu logo.png) desde la carpeta 'assets'
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('assets', path)

# Ruta principal de Flask (página de inicio HTML)
@server.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bienvenido al Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f6f8;
                text-align: center;
                padding: 50px;
                color: #333;
            }
            h2 {
                color: #2c3e50;
            }
            .logo {
                width: 80px;
                height: auto;
                margin-bottom: 20px;
            }
            .container {
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                display: inline-block;
                max-width: 600px;
                width: 100%;
                animation: fadeIn 1s ease-in-out;
            }
            .links {
                margin-top: 30px;
            }
            a {
                display: inline-block;
                margin: 0 15px;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s ease;
            }
            a:hover {
                background-color: #2980b9;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="/static/logo.png" alt="Logo Hospital" class="logo">
            <h2>Bienvenido al Dashboard de Gestión Hospitalaria</h2>
            <p>Selecciona una opción para acceder a los dashboards o al simulador de tiempos de espera.</p>
            <div class="links">
                <a href="/dash/edad/">Análisis por Edad</a>
                <a href="/dash/espera/">Tiempos de Espera</a>
                <a href="/dash/modalidad/">Modalidad de Atención</a>
                <a href="/dash/asegurados/">Análisis por Seguro</a>
                <a href="/dash/tiempo/">Análisis de Tiempos</a>
                <a href="/dash/simulador/">Simulador de Tiempos</a>
            </div>
        </div>
    </body>
    </html>
    """)

# Inicialización de la aplicación Dash
app = dash.Dash(
    __name__,
    server=server,              # Vincula Dash a tu servidor Flask existente
    url_base_pathname="/dash/", # Todas las rutas de Dash comenzarán con /dash/
    use_pages=True,             # ¡Habilita la carga perezosa y el enrutamiento basado en archivos!
    assets_folder=os.path.join(os.getcwd(), 'assets'), # Si tienes assets específicos de Dash
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'] # Opcional: una hoja de estilos externa
)

# El layout principal de la aplicación Dash utiliza dash.page_container
# para mostrar el contenido de las páginas registradas dinámicamente.
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), # Necesario para manejar cambios de URL
    dash.page_container # Contenedor donde Dash insertará el layout de la página actual
])

# Variables globales adicionales que tus páginas de Dash podrían necesitar
# Por ejemplo, si tienes diccionarios de especialidades o fechas actuales
dia_actual = datetime.now().day
semana_anio = datetime.now().isocalendar()[1]
especialidades_dict = {
    'Cardiología': 1, 'Pediatría': 2, 'Dermatología': 3, 'Oftalmología': 4,
    'Ginecología': 5, 'Traumatología': 6, 'Neurología': 7, 'Otorrinolaringología': 8,
    'Urología': 9, 'Psiquiatría': 10
}


# Este bloque solo se ejecuta cuando ejecutas el script directamente (por ejemplo, python multi_app.py)
# Gunicorn no usa este bloque; usa `multi_app:server` para iniciar.
if __name__ == '__main__':
    # Obtener el puerto del entorno (para Render) o usar 8050 por defecto (para desarrollo local)
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
