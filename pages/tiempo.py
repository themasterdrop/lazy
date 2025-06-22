import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import register_page

from multi_app import df # Import the shared DataFrame

# Asegúrate de que 'MES' sea categórica y ordenada para que el gráfico lineal tenga el orden correcto
# Esto ya se debería haber hecho en multi_app.py
citas_por_mes = df.groupby('MES', observed=False).size().reset_index(name='CANTIDAD_CITAS')
citas_por_mes['MES_SORT'] = pd.to_datetime(citas_por_mes['MES']) # Para ordenar correctamente en el gráfico
citas_por_mes = citas_por_mes.sort_values('MES_SORT')


register_page(
    __name__,
    path='/tiempo/',
    name='Línea de Tiempo',
    title='Dashboard - Tiempo'
)

layout = html.Div([
    html.H1("Citas Agendadas por Mes"),
    dcc.Graph(
        id='grafico-lineal',
        figure=px.line(citas_por_mes, x='MES', y='CANTIDAD_CITAS', markers=True,
                       title='Cantidad de Citas por Mes',
                       # Asegúrate de que el eje X se ordene correctamente si MES no es categórico ordenado
                       category_orders={'MES': citas_por_mes['MES'].tolist()})
    ),
    html.Div([
        dcc.Graph(id='grafico-pie-especialidades'),
        dcc.Graph(id='grafico-pie-atencion')
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around'}) # Para que se muestren lado a lado
])

@dash.callback(
    [Output('grafico-pie-especialidades', 'figure'),
     Output('grafico-pie-atencion', 'figure')],
    [Input('grafico-lineal', 'clickData')]
)
def actualizar_graficos(clickData):
    if clickData is None:
        df_empty_especialidad = pd.DataFrame(columns=['ESPECIALIDAD', 'CUENTA'])
        df_empty_atencion = pd.DataFrame(columns=['ATENDIDO', 'COUNT']) # o el nombre de tu columna de conteo
        return (px.pie(df_empty_especialidad, names='ESPECIALIDAD', values='CUENTA', title="Seleccione un mes"),
                px.pie(df_empty_atencion, names='ATENDIDO', title="Seleccione un mes"))

    mes_seleccionado_str = clickData['points'][0]['x']
    # Dash ClickData a veces devuelve el valor crudo, que ya es el string de MES si se configuró bien
    # Si 'MES' es datetime, tendrías que convertirlo. Pero si es str de Period, ya está bien.
    df_mes = df[df['MES'] == mes_seleccionado_str]

    top_especialidades = df_mes['ESPECIALIDAD'].value_counts().nlargest(5)
    df_mes['ESPECIALIDAD_AGRUPADA'] = df_mes['ESPECIALIDAD'].apply(
        lambda x: x if x in top_especialidades.index else 'Otras'
    )

    # Añade observed=False
    grouped_especialidades = df_mes['ESPECIALIDAD_AGRUPADA'].value_counts(observed=False).reset_index()
    grouped_especialidades.columns = ['ESPECIALIDAD', 'CUENTA']
    grouped_especialidades = grouped_especialidades.sort_values(by='CUENTA', ascending=False)

    # Añade observed=False
    grouped_atencion = df_mes['ATENDIDO'].value_counts(observed=False).reset_index(name='COUNT')

    fig_especialidades = px.pie(grouped_especialidades, names='ESPECIALIDAD', values="CUENTA", title=f'Distribución de Especialidades en {mes_seleccionado_str}')
    fig_atencion = px.pie(grouped_atencion, names='ATENDIDO', values='COUNT', title=f'Estado de Atención en {mes_seleccionado_str}') # Usar 'COUNT' para values

    return fig_especialidades, fig_atencion
