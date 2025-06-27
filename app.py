import pandas as pd
from flask import Flask, render_template_string, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import joblib # Para cargar el modelo guardado con joblib
import requests # Para descargar el modelo desde la URL
import io # Para manejar el contenido binario del modelo en memoria
from datetime import datetime # Para obtener la fecha actual (día y semana_del_año)
import os # Para trabajar con rutas de archivos, especialmente para la carpeta 'static'

print("--- Iniciando la aplicación Dash/Flask (app.py) ---")

# URLs de los recursos
HF_MODEL_URL = "https://huggingface.co/themasterdrop/simulador_citas_modelo/resolve/main/modelo_forest.pkl?download=true"


# Diccionario de especialidades (tal cual lo proporcionaste)
especialidades_dic = {
    17: 'GERIATRIA', 16: 'GASTROENTEROLOGIA', 13: 'ENDOCRINOLOGIA',
    51: 'PSIQUIATRIA', 2: 'CARDIOLOGIA', 61: 'UROLOGIA', 50: 'PSICOLOGIA',
    6: 'CIRUGIA GENERAL', 34: 'NEUROLOGIA', 20: 'HEMATOLOGIA',
    26: 'MEDICINA INTERNA', 42: 'OFTALMOLOGIA', 54: 'REUMATOLOGIA',
    4: 'CIRUGIA PLASTICA Y QUEMADOS', 33: 'NEUROCIRUGIA',
    48: 'PEDIATRIA GENERAL', 27: 'NEFROLOGIA',
    35: 'NEUROLOGIA PEDIATRICA', 40: 'OBSTETRICIA', 29: 'NEUMOLOGIA',
    43: 'ONCOLOGIA GINECOLOGIA', 28: 'NEONATOLOGIA', 21: 'INFECTOLOGIA',
    0: 'ADOLESCENTE', 18: 'GINECOLOGIA', 10: 'DERMATOLOGIA',
    8: 'CIRUGIA PEDIATRICA', 56: 'TRAUMATOLOGIA', 47: 'PATOLOGIA MAMARIA',
    46: 'OTORRINOLARINGOLOGIA', 12: 'ECOGRAFIA',
    25: 'MEDICINA FÍSICA Y REHABILITACIÓN', 31: 'NEUMOLOGIA PEDIATRICA',
    44: 'ONCOLOGIA MEDICA', 5: 'CIRUGIA CABEZA Y CUELLO',
    7: 'CIRUGIA MAXILO-FACIAL', 19: 'GINECOLOGIA DE ALTO RIESGO',
    36: 'NEUROPSICOLOGIA', 52: 'PUERPERIO',
    59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS', 3: 'CARDIOLOGIA PEDIATRICA',
    41: 'ODONTOLOGIA', 53: 'RADIOTERAPIA', 9: 'CIRUGIA TORAXICA',
    37: 'NUTRICION - ENDOCRINOLOGIA', 57: 'TUBERCULOSIS',
    38: 'NUTRICION - MEDICINA', 22: 'INFECTOLOGIA PEDIATRICA',
    30: 'NEUMOLOGIA FUNCION RESPIRATORIA', 39: 'NUTRICION - PEDIATRICA',
    14: 'ENDOCRINOLOGIA PEDIATRICA', 55: 'SALUD MENTAL ', 23: 'INFERTILIDAD',
    45: 'ONCOLOGIA QUIRURGICA', 32: 'NEUMOLOGIA TEST DE CAMINATA',
    49: 'PLANIFICACION FAMILIAR', 24: 'MEDICINA ALTERNATIVA',
    1: 'ANESTESIOLOGIA', 11: 'DERMATOLOGIA PEDIATRICA',
    58: 'TUBERCULOSIS PEDIATRICA', 62: 'ZPRUEBA',
    60: 'URODINAMIA', 15: 'ENDOCRINOLOGIA TUBERCULOSIS'
}

# --- Carga del Modelo de Machine Learning (joblib) ---
print("--- Iniciando descarga y carga del modelo desde Hugging Face (app.py) ---")
modelo_forest = None # Inicializar a None en caso de error

try:
    response = requests.get(HF_MODEL_URL)
    response.raise_for_status() # Lanza una excepción para errores HTTP

    model_bytes = io.BytesIO(response.content)
    modelo_forest = joblib.load(model_bytes) # Carga el modelo con joblib
    print("¡Modelo cargado con éxito desde Hugging Face URL!")

except requests.exceptions.RequestException as e:
    print(f"ERROR al descargar el modelo desde Hugging Face: {e}")
    print("Por favor, verifica la URL del modelo o tu conexión a internet.")
except Exception as e:
    print(f"ERROR inesperado al cargar el modelo con joblib: {e}")
    print("Asegúrate de que el archivo .pkl fue guardado correctamente con joblib y es compatible.")

print("-" * 40)


# --- Configuración del Servidor Flask Compartido ---
server = Flask(__name__)

# Ruta para servir archivos estáticos (como el logo.png)
@server.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(server.root_path, 'static'), filename)

# Ruta raíz con enlaces a todas las aplicaciones Dash
@server.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Bienvenido</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif; /* Usando Inter */
                background-color: #eef2f6; /* Fondo más suave */
                text-align: center;
                padding: 50px 20px;
                color: #333;
                margin: 0;
            }
            h2 {
                color: #2c3e50;
                font-weight: 700; /* Más audaz */
                margin-bottom: 25px;
            }
            p {
                color: #555;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .logo {
                width: 100px; /* Logo un poco más grande */
                height: auto;
                margin-bottom: 25px;
                border-radius: 15px; /* Bordes redondeados */
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Sombra sutil */
            }
            .container {
                background-color: #ffffff;
                padding: 50px 30px; /* Mayor padding */
                border-radius: 15px; /* Bordes más redondeados */
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); /* Sombra más pronunciada */
                display: inline-block;
                max-width: 650px; /* Ancho ligeramente mayor */
                width: 100%;
                animation: fadeIn 0.8s ease-out; /* Animación más suave */
                box-sizing: border-box;
            }
            .links {
                margin-top: 30px;
                display: flex; /* Usar flexbox para alinear botones */
                flex-wrap: wrap; /* Permitir que los botones se envuelvan */
                justify-content: center; /* Centrar botones */
                gap: 15px; /* Espacio entre botones */
            }
            a {
                flex-grow: 1; /* Permite que los enlaces crezcan */
                max-width: 280px; /* Controla el ancho máximo de los enlaces */
                margin: 0; /* Eliminar margen para usar gap */
                padding: 15px 30px; /* Mayor padding */
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); /* Degradado */
                color: white;
                text-decoration: none;
                border-radius: 8px; /* Bordes redondeados */
                font-weight: 600; /* Texto más audaz */
                font-size: 1.05em;
                transition: all 0.3s ease; /* Transición para todas las propiedades */
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            a:hover {
                background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); /* Degradado inverso al hover */
                transform: translateY(-3px) scale(1.02); /* Efecto 3D sutil */
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            /* Media Queries para responsividad */
            @media (max-width: 768px) {
                body { padding: 30px 15px; }
                .container { padding: 30px 20px; max-width: 95%; }
                h2 { font-size: 1.6em; margin-bottom: 20px; }
                p { font-size: 1em; margin-bottom: 25px; }
                .logo { width: 80px; margin-bottom: 20px; }
                .links { gap: 10px; }
                a { padding: 12px 20px; font-size: 0.95em; max-width: 100%; }
            }
            @media (max-width: 480px) {
                body { padding: 20px 10px; }
                .container { padding: 20px 15px; }
                h2 { font-size: 1.4em; }
                .logo { width: 60px; }
                a { font-size: 0.9em; padding: 10px 15px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img src="https://placehold.co/100x100/A8DADC/2F4F4F?text=App+Logo" alt="Logo de la Institución" class="logo">
            <h2>Bienvenido al Hospital Virtual</h2>
            <p>Accede a herramientas y visualizaciones para una mejor gestión y predicción de citas.</p>
            <div class="links">
                <a href="/simulador/">Simulador de Tiempo de Espera</a>
                <!-- Puedes añadir más enlaces aquí si extiendes el proyecto -->
            </div>
        </div>
    </body>
    </html>
    """)

# Definir el bloque de estilos CSS para el simulador como una cadena de Python
simulador_css_styles = '''
    .button-predict:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(40,167,69,0.3);
        border-bottom: 3px solid #1c7430;
    }
    .button-predict:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-bottom: none;
    }
    /* Estilos para el dropdown de Dash */
    .Select-control, .Select-menu-outer {
        border-radius: 8px !important;
        border-color: #cce7ff !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.08) !important;
    }
    .Select-control {
        height: 45px !important;
        font-size: 16px !important;
    }
    .Select-placeholder, .Select--single > .Select-control .Select-value {
        line-height: 45px !important;
    }
    .Select-option.is-focused {
        background-color: #e0f2f7 !important;
    }
    .Select-option.is-selected {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
    }
    .back-button a:hover {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(243,156,18,0.3);
        border-bottom: 3px solid #b76610;
    }
    .back-button a:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-bottom: none;
    }
    /* Media Queries para el simulador */
    @media (max-width: 768px) {
        .dash-app-container { padding: 20px 10px; }
        .dash-app-container h1 { font-size: 1.8em; margin-bottom: 20px; }
        .dash-app-container > div { padding: 30px 20px; margin: 30px auto; max-width: 95%; }
        .dash-app-container label { font-size: 0.9em; margin-bottom: 5px; }
        .dash-app-container .input-field, .dash-app-container .dropdown-field {
            padding: 10px;
            font-size: 15px;
        }
        .dash-app-container .Select-control {
            height: 40px !important;
            font-size: 15px !important;
        }
        .dash-app-container .Select-placeholder, .dash-app-container .Select--single > .Select-control .Select-value {
            line-height: 40px !important;
        }
        .dash-app-container .button-predict, .back-button a {
            padding: 12px 25px;
            font-size: 16px;
        }
        .dash-app-container #sim-output-prediction { font-size: 20px; margin-top: 30px; }
        .dash-app-container #sim-output-warning { font-size: 16px; margin-top: 15px; padding: 8px; }
    }
    @media (max-width: 480px) {
        .dash-app-container h1 { font-size: 1.5em; }
        .dash-app-container > div { padding: 25px 15px; margin: 20px auto; }
        .dash-app-container .button-predict, .back-button a {
            padding: 10px 20px;
            font-size: 14px;
        }
        .dash-app-container #sim-output-prediction { font-size: 18px; }
        .dash-app-container #sim-output-warning { font-size: 14px; }
    }
'''


# --- App: Simulador de Tiempo de Espera ---
simulador_app = dash.Dash(__name__, server=server, url_base_pathname='/simulador/')

simulador_app.layout = html.Div([
    # Bloque de estilos generales para el simulador
    html.Style(simulador_css_styles), # Aquí se pasa la variable con los estilos
    html.H1("Simulador de Tiempo de Espera Estimado", style={'color': '#2c3e50', 'marginBottom': '30px', 'fontWeight': '700'}),
    html.Div([
        html.Label("Edad:", style={'display': 'block', 'marginBottom': '8px', 'fontWeight': 'bold', 'color': '#444'}),
        dcc.Input(id='sim-input-edad', type='number', value=30, min=0, max=120, className="input-field",
                  style={'width': 'calc(100% - 20px)', 'padding': '12px', 'borderRadius': '8px', 'border': '1px solid #cce7ff', 'marginBottom': '20px', 'fontSize': '16px', 'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.08)'}),

        html.Label("Especialidad:", style={'display': 'block', 'marginBottom': '8px', 'fontWeight': 'bold', 'color': '#444'}),
        dcc.Dropdown(
            id='sim-input-especialidad',
            options=[{'label': v, 'value': k} for k, v in especialidades_dic.items()],
            value=17, # Valor por defecto (GERIATRIA) o el que prefieras
            placeholder="Selecciona una especialidad",
            className="dropdown-field",
            style={'marginBottom': '30px', 'borderRadius': '8px', 'border': '1px solid #cce7ff', 'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.08)'}
        ),

        html.Button('Predecir Tiempo de Espera', id='sim-predict-button', n_clicks=0, className="button-predict",
                     style={
                         'backgroundColor': '#28a745', 'color': 'white', 'padding': '15px 35px', 'border': 'none',
                         'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '18px', 'fontWeight': '600',
                         'transition': 'all 0.3s ease', 'boxShadow': '0 5px 15px rgba(40,167,69,0.2)',
                         'outline': 'none', 'borderBottom': '3px solid #1e7e34' # Efecto 3D
                     }),
        
        # Div para mostrar la predicción
        html.Div(id='sim-output-prediction', style={'marginTop': '40px', 'fontSize': '26px', 'fontWeight': 'bold', 'color': '#0056b3', 'textShadow': '1px 1px 2px rgba(0,0,0,0.05)'}),
        
        # Div para mostrar la advertencia
        html.Div(id='sim-output-warning', style={'marginTop': '20px', 'fontSize': '20px', 'color': '#dc3545', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#f8d7da', 'border': '1px solid #f5c6cb', 'borderRadius': '8px', 'display': 'none'})

    ], style={'padding': '40px', 'border': '1px solid #d1ecf1', 'borderRadius': '15px', 'maxWidth': '600px', 'margin': '50px auto', 'backgroundColor': '#ffffff', 'boxShadow': '0 10px 30px rgba(0,0,0,0.1)'}),

    html.Br(),
    html.Div(dcc.Link('Volver a la Página Principal', href='/', style={
        'display': 'inline-block', 'marginTop': '40px', 'padding': '15px 30px',
        'background': 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)',
        'color': 'white', 'textDecoration': 'none', 'borderRadius': '8px',
        'fontSize': '18px', 'fontWeight': '600', 'transition': 'all 0.3s ease',
        'boxShadow': '0 5px 15px rgba(243,156,18,0.2)', 'outline': 'none', 'borderBottom': '3px solid #c07712'
    }), className="back-button"),
], className="dash-app-container", style={'textAlign': 'center', 'fontFamily': 'Inter', 'backgroundColor': '#eef2f6', 'padding': '40px 20px', 'minHeight': '100vh', 'boxSizing': 'border-box'})


@simulador_app.callback(
    [Output('sim-output-prediction', 'children'),
     Output('sim-output-warning', 'children'),
     Output('sim-output-warning', 'style')],
    Input('sim-predict-button', 'n_clicks'),
    State('sim-input-edad', 'value'),
    State('sim-input-especialidad', 'value'),
    prevent_initial_call=True
)
def predecir(n_clicks, edad, especialidad_cod_input):
    if n_clicks is None or n_clicks == 0:
        return "", "", {'display': 'none'}

    if modelo_forest is None:
        return "❌ Error: El modelo de predicción no se pudo cargar. Contacta al administrador.", "", {'display': 'none'}
    if edad is None or especialidad_cod_input is None:
        return "⚠️ Por favor, ingrese la edad y seleccione una especialidad para la predicción.", "", {'display': 'none'}

    today = datetime.now()
    dia = today.day
    semana_del_año = today.isocalendar()[1]

    input_data = pd.DataFrame([[
        especialidad_cod_input,
        edad,
        dia,
        semana_del_año
    ]],
    columns=[
        'ESPECIALIDAD_cod',
        'EDAD',
        'día',
        'semana_del_año'
    ])

    try:
        predicted_days = modelo_forest.predict(input_data)[0]
        predicted_days_rounded = max(0, round(predicted_days))

        nombre_especialidad = especialidades_dic.get(especialidad_cod_input, "Especialidad Desconocida")

        prediction_text = f"Especialidad: {nombre_especialidad} — Tiempo estimado de espera: ➡️ **{predicted_days_rounded} días**."
        
        warning_text = "⚠️ La espera procede a ser de un mes o más, se recomienda tomar precauciones."
        warning_style = {'marginTop': '20px', 'fontSize': '20px', 'color': '#dc3545', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#f8d7da', 'border': '1px solid #f5c6cb', 'borderRadius': '8px', 'display': 'block'}
        
        if predicted_days_rounded < 30:
            warning_text = ""
            warning_style['display'] = 'none'
        
        return prediction_text, warning_text, warning_style

    except Exception as e:
        return f"❌ Error al realizar la predicción: {e}. Asegúrate de que los datos de entrada coincidan con lo que el modelo espera.", "", {'display': 'none'}


# --- Punto de Entrada para Gunicorn y Desarrollo Local ---
application = server 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    server.run(host='0.0.0.0', port=port, debug=True)
