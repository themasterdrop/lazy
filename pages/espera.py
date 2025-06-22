import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import register_page

from multi_app import df # Import the shared DataFrame

register_page(
    __name__,
    path='/espera/',
    name='Tiempos de Espera',
    title='Dashboard - Espera'
)

layout = html.Div([
    html.H1("Distribución por Tiempo de Espera"),
    dcc.Graph(id='histogram-espera', figure=px.histogram(
        df,
        x='RANGO_DIAS',
        category_orders={'RANGO_DIAS': ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]},
        title='Distribución de la Cantidad de Pacientes según su Tiempo de Espera',
        labels={'RANGO_DIAS': 'Rango de Días'},
        template='plotly_white'
    )),
    dcc.Graph(id='pie-chart-espera', figure=px.pie(
        names=[], values=[], title="Seleccione una barra en el histograma"
    ))
])

@dash.callback(
    Output('pie-chart-espera', 'figure'),
    Input('histogram-espera', 'clickData')
)
def update_pie_chart_espera(clickData):
    if clickData is None:
        df_empty = pd.DataFrame(columns=['ESPECIALIDAD', 'CUENTA'])
        return px.pie(df_empty, names='ESPECIALIDAD', values='CUENTA', title="Seleccione una barra en el histograma", height=500)

    selected_range = clickData['points'][0]['x']
    filtered_df = df[df['RANGO_DIAS'] == selected_range].copy()

    top_especialidades = filtered_df['ESPECIALIDAD'].value_counts().nlargest(5)
    filtered_df['ESPECIALIDAD_AGRUPADA'] = filtered_df['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    grouped = filtered_df['ESPECIALIDAD_AGRUPADA'].value_counts(observed=False).reset_index()
    grouped.columns = ['ESPECIALIDAD', 'CUENTA']
    return px.pie(
        grouped,
        names='ESPECIALIDAD',
        values='CUENTA',
        title=f"Top 5 Especialidades para el rango de espera '{selected_range}' días",
        height=600
    )
