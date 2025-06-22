import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from dash import register_page
from datetime import datetime
import numpy as np

# Import global variables (modelo_forest, hoy, dia_actual, semana_anio)
from multi_app import modelo_forest, hoy, dia_actual, semana_anio

# Define especialidades mapping (could also be imported if it's large)
especialidades_dict = {
    17: 'GERIATRIA', 16: 'GASTROENTEROLOGIA', 13: 'ENDOCRINOLOGIA', 51: 'PSIQUIATRIA',
    2: 'CARDIOLOGIA', 61: 'UROLOGIA', 50: 'PSICOLOGIA', 6: 'CIRUGIA GENERAL',
    34: 'NEUROLOGIA', 20: 'HEMATOLOGIA', 26: 'MEDICINA INTERNA', 42: 'OFTALMOLOGIA',
    54: 'REUMATOLOGIA', 4: 'CIRUGIA PLASTICA Y QUEMADOS', 33: 'NEUROCIRUGIA',
    48: 'PEDIATRIA GENERAL', 27: 'NEFROLOGIA', 35: 'NEUROLOGIA PEDIATRICA',
    40: 'OBSTETRICIA', 29: 'NEUMOLOGIA', 43: 'ONCOLOGIA GINECOLOGIA',
    28: 'NEONATOLOGIA', 21: 'INFECTOLOGIA', 0: 'ADOLESCENTE', 18: 'GINECOLOGIA',
    10: 'DERMATOLOGIA', 8: 'CIRUGIA PEDIATRICA', 56: 'TRAUMATOLOGIA',
    47: 'PATOLOGIA MAMARIA', 46: 'OTORRINOLARINGOLOGIA', 12: 'ECOGRAFIA',
    25: 'MEDICINA FÍSICA Y REHABILITACIÓN', 31: 'NEUMOLOGIA PEDIATRICA',
    44: 'ONCOLOGIA MEDICA', 5: 'CIRUGIA CABEZA Y CUELLO', 7: 'CIRUGIA MAXILO-FACIAL',
    19: 'GINECOLOGIA DE ALTO RIESGO', 36: 'NEUROPSICOLOGIA', 52: 'PUERPERIO',
    59: 'UNIDAD DEL DOLOR Y CUIDADOS PALIATIVOS', 3: 'CARDIOLOGIA PEDIATRICA',
    41: 'ODONTOLOGIA', 53: 'RADIOTERAPIA', 9: 'CIRUGIA TORAXICA',
    37: 'NUTRICION - ENDOCRINOLOGIA', 57: 'TUBERCULOSIS',
    38: 'NUTRICION - MEDICINA', 22: 'INFECTOLOGIA PEDIATRICA',
    30: 'NEUMOLOGIA FUNCION RESPIRATORIA', 39: 'NUTRICION - PEDIATRICA',
    14: 'ENDOCRINOLOGIA PEDIATRICA', 55: 'SALUD MENTAL ', 23: 'INFERTILIDAD',
    45: 'ONCOLOGIA QUIRURGICA', 32: 'NEUMOLOGIA TEST DE CAMINATA',
    49: 'PLANIFICACION FAMILIAR', 24: 'MEDICINA ALTERNATIVA',
    1: 'ANESTESIOLOGIA', 11: 'DERMATOLOGIA PEDIATRICA',
    58: 'TUBERCULOSIS PEDIATRICA', 62: 'ZPRUEBA', 60: 'URODINAMIA',
    15: 'ENDOCRINOLOGIA TUBERCULOSIS'
}

register_page(
    __name__,
    path='/simulador/',
    name='Simulador de Tiempo de Espera',
    title='Dashboard - Simulador'
)

layout = html.Div([
    html.H2("Simulador de Tiempo de Espera de Citas"),

    html.Label("Especialidad:"),
    dcc.Dropdown(
        id='input-especialidad',
        options=[{'label': v, 'value': k} for k, v in especialidades_dict.items()], # Corrected: label is name, value is ID
        value=1,
        placeholder="Selecciona una especialidad"
    ),

    html.Label("Edad:"),
    dcc.Input(id='input-edad', type='number', value=30),

    html.Label("Día:"),
    dcc.Input(id='input-dia', type='number', value=dia_actual),

    html.Label("Semana del año:"),
    dcc.Input(id='input-semana_anio', type='number', value=semana_anio),

    html.Br(),
    html.Button("Predecir", id='btn-predecir', n_clicks=0),
    html.Div(id='output-prediccion')
])

@dash.callback(
    Output('output-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    Input('input-especialidad', 'value'),
    Input('input-edad', 'value'),
    Input('input-dia', 'value'),
    Input('input-semana_anio', 'value'),
)
def predecir(n_clicks, especialidad, edad, dia, semana_anio):
    if n_clicks > 0:
        if edad is None or edad < 0 or edad > 120:
            return "Edad no válida."
        if especialidad is None or dia is None or semana_anio is None:
            return "Por favor, completa todos los campos."

        # Asegúrate de que los valores de entrada sean números enteros
        try:
            especialidad = int(especialidad)
            edad = int(edad)
            dia = int(dia)
            semana_anio = int(semana_anio)
        except ValueError:
            return "Valores de entrada no válidos. Asegúrate de que sean números."

        entrada = [[
            especialidad, edad, dia, semana_anio
        ]]
        prediccion = modelo_forest.predict(entrada)[0]
        nombre_especialidad = especialidades_dict.get(especialidad, "Desconocida")
        return html.Div([
            html.P(f"Especialidad: {nombre_especialidad}"),
            html.P(f"Tiempo estimado de espera: {prediccion:.2f} días")
        ])

    return ""
