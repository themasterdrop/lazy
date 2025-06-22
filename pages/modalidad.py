import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import register_page

from multi_app import df # Import the shared DataFrame

register_page(
    __name__,
    path='/modalidad/',
    name='Modalidad de Atención',
    title='Dashboard - Modalidad'
)

layout = html.Div([
    html.H1("Distribución por Modalidad de Cita"),
    dcc.Graph(id='pie-modalidad', figure=px.pie(
        df,
        names='PRESENCIAL_REMOTO',
        title='Distribución de Citas: Remotas vs Presenciales',
        template='plotly_white'
    )),
    dcc.Graph(id='bar-especialidad-modalidad', figure=px.bar(
        pd.DataFrame(columns=['ESPECIALIDAD', 'DIFERENCIA_DIAS']),
        x='ESPECIALIDAD',
        y='DIFERENCIA_DIAS',
        title="Seleccione una modalidad en el gráfico de pastel"
    ))
])

@dash.callback(
    Output('bar-especialidad-modalidad', 'figure'),
    Input('pie-modalidad', 'clickData')
)
def update_bar_modalidad(clickData):
    if clickData is None:
        # DataFrame vacío con las columnas esperadas para este gráfico.
        df_empty = pd.DataFrame(columns=['ESPECIALIDAD', 'DIFERENCIA_DIAS'])
        return px.bar(df_empty, x='ESPECIALIDAD', y='DIFERENCIA_DIAS', title="Seleccione una modalidad en el gráfico de pastel")

    modalidad = clickData['points'][0]['label']
    filtered_df = df[df['PRESENCIAL_REMOTO'] == modalidad]
    # Añade observed=False aquí.
    mean_wait = filtered_df.groupby('ESPECIALIDAD', observed=False)['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)

    return px.bar(
        mean_wait,
        x='ESPECIALIDAD',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por Especialidad ({modalidad})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
    )
