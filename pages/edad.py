import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import register_page

# Import df and other global variables from multi_app.py
# This is crucial for lazy loading.
from multi_app import df

# Registrar la página
# path_template allows for dynamic routing (e.g., /dash/edad/<some_id>)
# but for simple pages, just path is fine.
register_page(
    __name__,
    path='/edad/',  # This will be the URL: /dash/edad/
    name='Distribución por Edad',
    title='Dashboard - Edad'
)

layout = html.Div([
    html.H1("Distribución por Rango de Edad"),
    dcc.Graph(id='histogram-edad', figure=px.histogram(
        df,
        x='Rango de Edad',
        category_orders={'Rango de Edad': ["Niño", "Adolescente", "Joven", "Adulto", "Adulto mayor"]},
        title='Distribución de edades de los pacientes del hospital María Auxiliadora',
        labels={'Rango de Edad': 'Rango de Edad'},
        template='plotly_white'
    )),
    dcc.Graph(id='pie-chart-edad', figure=px.pie(
        names=[], values=[], title="Seleccione una barra en el histograma"
    ))
])

@dash.callback(
    Output('pie-chart-edad', 'figure'),
    Input('histogram-edad', 'clickData')
)
def update_pie_chart_edad(clickData):
    if clickData is None:
        # Asegúrate de que el DataFrame vacío tenga las columnas necesarias
        df_empty = pd.DataFrame(columns=['ESPECIALIDAD', 'CUENTA'])
        return px.pie(df_empty, names='ESPECIALIDAD', values='CUENTA', title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    filtered_df = df[df['Rango de Edad'] == selected_range].copy()

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    # Añade observed=False aquí para evitar el FutureWarning si 'ESPECIALIDAD_AGRUPADA' es categórica
    grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts(observed=False).reset_index()
    grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de edad '{selected_range}'",
        height=600
    )
