import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import register_page

from multi_app import df # Import the shared DataFrame

register_page(
    __name__,
    path='/asegurados/',
    name='Estado del Seguro',
    title='Dashboard - Asegurados'
)

layout = html.Div([
    html.H1("Distribución por Estado del Seguro"),
    dcc.Graph(id='pie-seguro', figure=px.pie(
        df.dropna(subset=['SEGURO']), # Asegúrate de que 'SEGURO' no tenga NaNs si es un problema
        names='SEGURO',
        title='Distribución de Pacientes: Asegurados vs No Asegurados',
        template='plotly_white'
    )),
    dcc.Graph(id='bar-espera-seguro', figure=px.bar(
        pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS']),
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title="Seleccione una opción en el gráfico de pastel"
    ))
])

@dash.callback(
    Output('bar-espera-seguro', 'figure'),
    Input('pie-seguro', 'clickData')
)
def update_bar_seguro(clickData):
    if clickData is None or 'label' not in clickData['points'][0]:
        df_empty = pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS'])
        return px.bar(df_empty, x='SEXO', y='DIFERENCIA_DIAS', title="Seleccione una opción en el gráfico de pastel")

    seguro = clickData['points'][0]['label']
    filtered_df = df[df['SEGURO'] == seguro]

    if filtered_df.empty:
        df_empty = pd.DataFrame(columns=['SEXO', 'DIFERENCIA_DIAS'])
        return px.bar(df_empty, x='SEXO', y='DIFERENCIA_DIAS', title=f"No hay datos para {seguro}")

    # Añade observed=False aquí.
    mean_wait = filtered_df.groupby('SEXO', observed=False)['DIFERENCIA_DIAS'].mean().reset_index()
    mean_wait = mean_wait.sort_values(by='DIFERENCIA_DIAS', ascending=False)

    fig = px.bar(
        mean_wait,
        x='SEXO',
        y='DIFERENCIA_DIAS',
        title=f"Media de Días de Espera por SEXO ({seguro})",
        labels={'DIFERENCIA_DIAS': 'Días de Espera'},
        template='plotly_white'
    )

    y_min = mean_wait['DIFERENCIA_DIAS'].min() - 1 if not mean_wait.empty else 0
    y_max = mean_wait['DIFERENCIA_DIAS'].max() + 1 if not mean_wait.empty else 10
    fig.update_yaxes(range=[y_min, y_max])

    return fig
