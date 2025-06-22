import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import requests
import os
from datetime import datetime
import numpy as np

# --- Data and Model Loading (Keep this global for now, optimize later if needed) ---
# It's still loaded at startup, but the *pages* themselves will be loaded on demand.
# For truly large datasets/models, you might move these into the page files or
# use a caching mechanism (e.g., Redis) or Dask if data doesn't fit in memory.

# ID del archivo en Google Drive (Modelo)
file_id_modelo = "1wrdWPjF47w7IEf0WkRWMLTVgPWqT3Jpf"
drive_url_modelo = f"https://drive.google.com/uc?export=download&id={file_id_modelo}"
modelo_path = "modelo_forest.pkl"

if not os.path.exists(modelo_path):
    print("Descargando modelo desde Google Drive...")
    r = requests.get(drive_url_modelo, stream=True)
    with open(modelo_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Modelo descargado.")

modelo_forest = joblib.load(modelo_path)

# Cargar los datos (DataFrame principal)
file_id_df = "1PWTw-akWr59Gu7MoHra5WXMKwllxK9bp"
url_df = f"https://drive.google.com/uc?export=download&id={file_id_df}"
df = pd.read_csv(url_df)

# Clasificación por edad y días (estas funciones y columnas son necesarias globalmente si varias páginas las usan)
def clasificar_edad(edad):
    if edad < 13: return "Niño"
    elif edad < 19: return "Adolescente"
    elif edad < 30: return "Joven"
    elif edad < 61: return "Adulto"
    elif edad < 200: return "Adulto mayor"
df['Rango de Edad'] = df['EDAD'].apply(clasificar_edad).astype('category') # Convert to category for memory and future warning

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
df['RANGO_DIAS'] = df['DIFERENCIA_DIAS'].apply(clasificar_dias).astype('category') # Convert to category

# Asegurar que MES y ATENDIDO sean categóricas para evitar warnings y optimizar
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

# --- Flask Server and Main Dash App Setup ---
server = Flask(__name__)

# Serve static files (like your logo.png)
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('assets', path) # Assuming 'assets' folder for static files

# Home route (index.html)
@server.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Bienvenido</title>
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
                display: inline-block
