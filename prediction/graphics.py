import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import random
from decouple import config



from plotly.subplots import make_subplots

from prediction.conecta import *

from django.core.management import call_command



# Realizar Query
# Carrega tabela municipios
# municipios = pd.read_csv('/home/eltonss/Documents/MEC/data/municipio.csv');
municipios = pd.read_json(get_municipios())
# Carrega tabela dados_ia_apa
# data_apa = pd.read_csv('/home/eltonss/Documents/MEC/data/dados_ia_apa.csv');
data_apa = pd.read_json(get_dados_ia_apa());

data_apa.head();

dict_uf = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
           25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
           43: 'RS', 50: 'MS',
           51: 'MT', 52: 'GO', 53: 'DF', };
data_apa = data_apa.replace({'cod_estado': dict_uf});

dict_municipio = {};
for code in set(data_apa['codigo_cidade']):
    for codigo, nome in zip(municipios.id_municipio, municipios.nome):
        if (str(code) == str(codigo)):
            dict_municipio[code] = nome;
data_apa = data_apa.replace({'codigo_cidade': dict_municipio})

# Remover Duplicatas
data_apa_duplicate = data_apa.drop_duplicates(subset=['nome', 'level_pontuacao', 'created_at']);
# result = data_apa_duplicate.groupby(['nome','level_pontuacao','created_at'],group_keys=True).apply(lambda x: x);
# print('Check duplicidade: ', result[['nome','level_pontuacao','created_at']].duplicated().any());

NIVEL_ORTOGRAFICO = 'A: NÍVEL ORTOGRÁFICO';
NIVEL_ALFABETICO = 'B: NÍVEL ALFABÉTICO';
NIVEL_SILABICO = 'D: NÍVEL SILÁBICO';



def con_db_caio():
    conn = psycopg2.connect(host=config('PREDICTION_DB_HOST', ''), database=config('PREDICTION_DB_NAME', ''), user=config('PREDICTION_DB_USER', ''), password=config('PREDICTION_DB_PASSWORD', ''))
    return conn


def con_db_caio2(query):
    conn = psycopg2.connect(host=config('PREDICTION_DB_HOST', ''), database=config('PREDICTION_DB_NAME', ''),
        user=config('PREDICTION_DB_USER', ''), password=config('PREDICTION_DB_PASSWORD', ''))
    cursor = conn.cursor()
    conn.autocommit = True
    cursor.execute(query)
    result = cursor.fetchone()
    
    return result


def graph_bar_pontuacao_uf():
    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', 'level_pontuacao']).agg(
        {'level_pontuacao': ['count']}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado', 'Pontuacao', 'Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    fig = px.bar(
        data_estado_pontuacao,
        x='Estado',
        y='Valor',
        color='Pontuacao',
        color_discrete_map={'A': 'green', 'B': 'yellow', 'D': 'red'},
        labels={'Valor': '<b>Valor</b>', 'Estado': '<b>Estado</b>', },
        title='<b>Pontuação por UF</b>'
    )

    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_bar_proporcao_pontuacao_uf():
    cross_tab_prop = pd.crosstab(index=data_apa_duplicate['cod_estado'], columns=data_apa_duplicate['level_pontuacao'],
                                 normalize='index')

    cross_tab_prop.reset_index(inplace=True)

    fig = px.bar(
        cross_tab_prop,
        x='cod_estado',
        y=['A', 'B', 'D'],
        color_discrete_map={'A': 'green', 'B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por UF</b>',
        labels={'variable': 'Nivel', 'value': '<b>Proporção da Pontuação(%)</b>', 'cod_estado': '<b>Estado</b>', }
    )

    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_bar_proporcao_pontuacao_cidades(uf):
    data_apa_duplicate_uf = data_apa_duplicate[data_apa_duplicate['cod_estado'] == uf]
    cross_tab_prop_cidade = pd.crosstab(
        index=data_apa_duplicate_uf['codigo_cidade'],
        columns=data_apa_duplicate_uf['level_pontuacao'],
        normalize='index'
    )
    cross_tab_prop_cidade.reset_index(inplace=True)

    cross_tab_prop_cidade['codigo_cidade'] = cross_tab_prop_cidade['codigo_cidade'].apply(str)
    fig = px.bar(
        cross_tab_prop_cidade,
        x='codigo_cidade',
        y=['A', 'B', 'D'],
        color_discrete_map={'A': 'green', 'B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por Cidade</b>',
        labels={'variable': 'Nivel', 'value': '<b>Proporção da Pontuação(%)</b>', 'codigo_cidade': '<b>Cidade</b>', }
    )

    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_bar_proporcao_pontuacao_escolas(uf, cidade):
    filtro = (data_apa_duplicate['cod_estado'] == uf) & (data_apa_duplicate['codigo_cidade'] == cidade)
    data_apa_duplicate_uf_cidade = data_apa_duplicate[filtro]

    cross_tab_prop_cidade_escola = pd.crosstab(
        index=data_apa_duplicate_uf_cidade['escola'],
        columns=data_apa_duplicate_uf_cidade['level_pontuacao'],
        normalize='index'
    )
    cross_tab_prop_cidade_escola.reset_index(inplace=True)
    # cross_tab_prop_cidade_escola
    fig = px.bar(
        cross_tab_prop_cidade_escola,
        x='escola',
        y=['A', 'B', 'D'],
        color_discrete_map={'A': 'green', 'B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por Escola</b>',
        labels={'variable': 'Nivel', 'value': '<b>Proporção da Pontuação(%)</b>', 'escola': '<b>Escola</b>', }
    )

    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_bar_valor_pontuacao_agregado():
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1)

    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', "level_pontuacao"]).agg(
        {"level_pontuacao": ["count"]}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado', 'Pontuacao', 'Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'A'].sort_values('Estado')
    c1, c2 = st.t.interval(confidence=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))

    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='A', marker_color='green'), 1, 1)

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'B'].sort_values('Estado')
    c1, c2 = st.t.interval(confidence=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='B', marker_color='yellow'), 1, 1)

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'D'].sort_values('Estado')
    c1, c2 = st.t.interval(confidence=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='D', marker_color='red'), 1, 1)

    fig['layout'].update(height=900, width=850, title='<b>Pontuação agrupada por Estado </b>', )

    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_bar_valor_por_pontuacao_segmentado():
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Pontuação A', 'Pontuação B', 'Pontuação D'),
                        vertical_spacing=0.1)

    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', "level_pontuacao"]).agg(
        {"level_pontuacao": ["count"]}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado', 'Pontuacao', 'Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'A'].sort_values('Estado')
    c1, c2 = st.t.interval(alpha=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))

    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='A', marker_color='green'), 1, 1)
    fig.add_hline(y=c1, line_dash='dot', row=1, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot', row=1, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']), row=1, col=1, annotation_text='Média', annotation_position='right')

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'B'].sort_values('Estado')
    c1, c2 = st.t.interval(alpha=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='B', marker_color='yellow'), 2, 1)

    fig.add_hline(y=c1, line_dash='dot', row=2, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot', row=2, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']), row=2, col=1, annotation_text='Média', annotation_position='right')

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao'] == 'D'].sort_values('Estado')
    c1, c2 = st.t.interval(alpha=0.90, df=len(data['Valor']) - 1, loc=np.mean(data['Valor']),
                           scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'], name='D', marker_color='red'), 3, 1)
    fig.add_hline(y=c1, line_dash='dot', row=3, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot', row=3, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']), row=3, col=1, annotation_text='Média', annotation_position='right')

    fig['layout'].update(height=900, width=850, title='<b>Pontuação Por Estado</b>', )
    fig.update_layout(legend_title='<b>Nível</b>', title_x=0.5)

    fig.data[0].name = NIVEL_ORTOGRAFICO
    fig.data[1].name = NIVEL_ALFABETICO
    fig.data[2].name = NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig


def graph_test():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')

    return fig



def grafico_risco_escola_dimensoes_barras():
    query = """select * from public.dimensoes_est de
    inner join escolas.aluno a 
    on de.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    df = pd.read_sql(query, con_db_caio())

    df_dim = pd.DataFrame()
    df_dim = pd.DataFrame()
    df_for = ['E_ESCC', 'E_PROFC', 'E_FAMC', 'E_COMC', 'E_ESTC']

    for x in df_for:
        # print(x)
        df_temp = df.groupby(x)['id'].count()
        df_temp = df_temp.reset_index()
        df_temp.columns = ['Classificação', 'Quant. Estudantes']
        df_temp['Dimensão'] = x
        df_temp['Porcentagem'] = (df_temp['Quant. Estudantes'] / df_temp['Quant. Estudantes'].sum()) * 100
        frames = [df_dim, df_temp]
        df_dim = pd.concat(frames)

    df_dim = df_dim.sort_values(by=['Classificação', 'Porcentagem'], ascending=[False, False])

    df_dim['Risco'] = df_dim['Classificação']
    df_dim['Risco'] = df_dim['Risco'].astype(str)
    df_dim['Risco'] = df_dim['Risco'].replace('1', 'Risco Baixo')
    df_dim['Risco'] = df_dim['Risco'].replace('2', 'Risco Médio')
    df_dim['Risco'] = df_dim['Risco'].replace('3', 'Risco Alto')
    df_dim
    fig = px.bar(df_dim, x="Dimensão", y="Quant. Estudantes", color='Risco')
    return fig


def grafico_risco_escola_dimensoes_barras2():
    query = """select * from public.dimensoes_est de
    inner join escolas.aluno a 
    on de.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    df = pd.read_sql(query, con_db_caio())

    df_temp = df.groupby('E_ESCC')['id'].count()
    df_temp = df_temp.reset_index()
    df_temp.columns = ['Classificação', 'Quant. Estudantes']
    df_temp['Porcentagem'] = (df_temp['Quant. Estudantes'] / df_temp['Quant. Estudantes'].sum()) * 100
    df_temp

    df_temp['class'] = 'R' + df_temp['Classificação'].astype(str)
    i = 1
    for index, row in df_temp.iterrows():
        # print(row['class'])
        if (i == 1):
            c1 = row['class']
            i = i + 1
        else:
            c2 = row['class']

    df_dim_graph = pd.DataFrame(columns=['Dimensão', 'R1', 'R2', 'R3'])

    df_for = ['E_ESCC', 'E_PROFC', 'E_FAMC', 'E_COMC', 'E_ESTC']

    for x in df_for:
        # print(x)
        df_temp = df.groupby(x)['id'].count()
        df_temp = df_temp.reset_index()
        df_temp.columns = ['Classificação', 'Quant. Estudantes']
        df_temp['Dimensao'] = x
        df_temp['Porcentagem'] = (df_temp['Quant. Estudantes'] / df_temp['Quant. Estudantes'].sum()) * 100
        # df_dim2.loc[index,nomeColuna] = df_reun['Result'].iloc[0]
        df_temp['class'] = 'R' + df_temp['Classificação'].astype(str)
        if (len(df_temp) == 3):
            c1 = 'R1'
            c2 = 'R2'
            c3 = 'R3'
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                c2: df_temp['Porcentagem'].iloc[1],
                                                c3: df_temp['Porcentagem'].iloc[2]}, ignore_index=True)
        elif (len(df_temp) == 2):

            i = 1
            for index, row in df_temp.iterrows():
                # print(row['class'])
                if (i == 1):
                    c1 = row['class']
                    i = i + 1
                else:
                    c2 = row['class']
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                c2: df_temp['Porcentagem'].iloc[1],
                                                }, ignore_index=True)

        else:
            for index, row in df_temp.iterrows():
                c1 = row['class']
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                }, ignore_index=True)
    df_dim_graph
    df_dim_graph = df_dim_graph.fillna(0)
    df_dim_graph = df_dim_graph.sort_values('R3', ascending=True)
    df_dim_graph.round(2)

    def trunc(values, decs=0):
        return np.trunc(values * 10 ** decs) / (10 ** decs)

    x_data = df_dim_graph[["R3", "R2", "R1"]].to_numpy()
    x_data = trunc(x_data, decs=2)
    x_data[np.isnan(x_data)] = 0

    y_data = list(df_dim_graph['Dimensão'])

    # --
    top_labels = ['Risco Alto', 'Risco Médio', 'Risco Baixo']

    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
              'rgba(190, 192, 213, 1)']

    # x_data = [[8.06, 12.90, 79.03,],
    #          [8.06, 27.41, 64.51,],
    #          [9.67, 24.19, 66.12,],
    #          [16.12, 11.29, 72.58,],
    #         [33.87, 33.87, 32.23,]]

    # y_data = ['Estudante<br>Estudante',
    #          'Escudante<br>Profissionais da' +
    #          '<br>Escola', 'Estudante' +
    #          '<br>Familia',
    #          'Estudante<br>Comunidade', 'Estudante<br>Escola']

    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        if (xd[0] != 0):
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text=str(xd[0]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
        else:
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text='',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))

        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            if (xd[i] != 0):
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i] / 2), y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
            else:
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i] / 2), y=yd,
                                        text='',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i] / 2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

    fig.update_layout(annotations=annotations)

    return fig


def grafico_risco_escola_fatores_barras():
    query = """select * from public.fatores_est fe
    inner join escolas.aluno a 
    on fe.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    df = pd.read_sql(query, con_db_caio())

    df_dim = pd.DataFrame()
    df_dim = pd.DataFrame()

    df_col = df.loc[:, df.columns.str.startswith("E_")]
    df_col = df_col.loc[:, df_col.columns.str.endswith("C")]
    df_for = list(df_col)
    # df_for = ['E_ESCC', 'E_PROFC', 'E_FAMC', 'E_COMC', 'E_ESTC']

    for x in df_for:
        # print(x)
        df_temp = df.groupby(x)['id'].count()
        df_temp = df_temp.reset_index()
        df_temp.columns = ['Classificação', 'Quant. Estudantes']
        df_temp['Dimensão'] = x
        df_temp['Porcentagem'] = (df_temp['Quant. Estudantes'] / df_temp['Quant. Estudantes'].sum()) * 100
        frames = [df_dim, df_temp]
        df_dim = pd.concat(frames)

    df_dim = df_dim.sort_values(by=['Classificação', 'Porcentagem'], ascending=[False, False])

    df_dim['Risco'] = df_dim['Classificação']
    df_dim['Risco'] = df_dim['Risco'].astype(str)
    df_dim['Risco'] = df_dim['Risco'].replace('1', 'Risco Baixo')
    df_dim['Risco'] = df_dim['Risco'].replace('2', 'Risco Médio')
    df_dim['Risco'] = df_dim['Risco'].replace('3', 'Risco Alto')
    df_dim
    fig = px.bar(df_dim, x="Dimensão", y="Quant. Estudantes", color='Risco')
    return fig


def grafico_risco_escola_fatores_barras2():
    query = """select * from public.fatores_est fe
    inner join escolas.aluno a 
    on fe.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    df = pd.read_sql(query, con_db_caio())
    df

    df_col = df.loc[:, df.columns.str.startswith("E_")]
    df_col = df_col.loc[:, df_col.columns.str.endswith("C")]
    df_col = df_col.reset_index()
    df_col = df_col.astype('int64')
    #remover linha abaixo posteriormente
    df_col.replace(0, 1, inplace=True)
    df_col

    df_dim_graph = pd.DataFrame(columns=['Dimensão', 'R1', 'R2', 'R3'])
    df_for = list(df_col)

    for x in df_for:
        # print(x)
        df_temp = df_col[x].value_counts()
        df_temp = df_temp.reset_index()
        df_temp.columns = ['Classificação', 'Quant. Estudantes']
        df_temp['Dimensao'] = x
        df_temp['Porcentagem'] = (df_temp['Quant. Estudantes'] / df_temp['Quant. Estudantes'].sum()) * 100
        # df_dim2.loc[index,nomeColuna] = df_reun['Result'].iloc[0]
        df_temp['class'] = 'R' + df_temp['Classificação'].astype(str)
        if (len(df_temp) == 3):
            c1 = 'R1'
            c2 = 'R2'
            c3 = 'R3'
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                c2: df_temp['Porcentagem'].iloc[1],
                                                c3: df_temp['Porcentagem'].iloc[2]}, ignore_index=True)
        elif (len(df_temp) == 2):

            i = 1
            for index, row in df_temp.iterrows():
                # print(row['class'])
                if (i == 1):
                    c1 = row['class']
                    i = i + 1
                else:
                    c2 = row['class']
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                c2: df_temp['Porcentagem'].iloc[1],
                                                }, ignore_index=True)

        else:
            for index, row in df_temp.iterrows():
                c1 = row['class']
            df_dim_graph = df_dim_graph.append({'Dimensão': x,
                                                c1: df_temp['Porcentagem'].iloc[0],
                                                }, ignore_index=True)
    df_dim_graph
    df_dim_graph = df_dim_graph.fillna(0)
    df_dim_graph = df_dim_graph.sort_values('R3', ascending=True)
    df_dim_graph.drop(df_dim_graph.head(1).index,inplace=True)
    df_dim_graph.round(2)

    def trunc(values, decs=0):
        return np.trunc(values * 10 ** decs) / (10 ** decs)

    x_data = df_dim_graph[["R3", "R2", "R1"]].to_numpy()
    x_data = trunc(x_data, decs=2)

    y_data = list(df_dim_graph['Dimensão'])

    top_labels = ['Risco Alto', 'Risco Médio', 'Risco Baixo']

    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
              'rgba(190, 192, 213, 1)']

    # x_data = [[8.06, 12.90, 79.03,],
    #          [8.06, 27.41, 64.51,],
    #          [9.67, 24.19, 66.12,],
    #          [16.12, 11.29, 72.58,],
    #         [33.87, 33.87, 32.23,]]

    # y_data = ['Estudante<br>Estudante',
    #          'Escudante<br>Profissionais da' +
    #          '<br>Escola', 'Estudante' +
    #          '<br>Familia',
    #          'Estudante<br>Comunidade', 'Estudante<br>Escola']

    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        if (xd[0] != 0):
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text=str(xd[0]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
        else:
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text='',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))

        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            if (xd[i] != 0):
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i] / 2), y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
            else:
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i] / 2), y=yd,
                                        text='',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i] / 2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

    fig.update_layout(annotations=annotations)

    return fig


def grafico_radar_sap():
    query = """SELECT * FROM dados_sap.fatores_escola where aluno = 'E M E F PROFESSORA DALILA LEAO'"""
    df = pd.read_sql(query, con_db_caio())

    df1 = pd.DataFrame([list(df['value'])], columns=list(df['fatores_dimensões']))
    df1

    arr = df["fatores_dimensões"].to_numpy()
    unique_arr = np.unique(arr)
    unique_arr

    mun = [random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2)]
    mun = [*mun, mun[0]]
    est = [random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2)]
    est = [*est, est[0]]

    dimensoes = list(df["fatores_dimensões"])
    dimensoes = [*dimensoes, dimensoes[0]]

    esc = df1.values.flatten().tolist()
    esc = [*esc, esc[0]]
    esc

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=esc, theta=dimensoes, name='Escola 1'),
            go.Scatterpolar(r=mun, theta=dimensoes, name='Municipio'),
            go.Scatterpolar(r=est, theta=dimensoes, name='Estado'),
            # go.Scatterpolar(r=restaurant_3, theta=categories, name='Restaurant 3')
        ],
        layout=go.Layout(
            title=go.layout.Title(text='E M E F PROFESSORA DALILA LEAO'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    return fig

def texto_sap_quant_est_esc():
    query = """SELECT  count(DISTINCT id_aluno)  FROM public.dimensoes_est"""
    row = con_db_caio2(query)
    texto = 'Atualmente existem ' +  str(row[0]) +  ' estudantes diferentes cadastrados no SAP '
    query = """select count(distinct(e.cod_escola)) from public.dimensoes_est de
inner join escolas.aluno a 
on de.id_aluno = a.id_aluno 
inner join escolas.turma t 
on a.id_turma = t.id_turma 
inner join escolas.escola e 
on t.id_escola = e.cod_escola;"""
    row = con_db_caio2(query)
    texto = texto + 'de ' +  str(row[0]) +  ' escolas diferentes.'
    return texto

def texto_apa_quant_est_esc():
    query = """select count(distinct a.aluno_id) qtd, coalesce(c.nome,'Total') as ciclo 
                from digitalizacoes_firebase.avaliacao a
                left join digitalizacoes_firebase.ciclo c 
                on a.ciclo_id = c.id 
                where 1=1
                and a.arquivos is not null
                and c.id <> 'KIlqTBMSWL1Qi0wmIjJr'
                group by rollup(2)
                order by 1 desc
                ;"""
    row = con_db_caio2(query)
    texto = 'Atualmente existem ' +  str(row[0]) +  ' estudantes distintos que digitalizaram no APA.'
    return texto

def table_apa_ciclo():
    #quantidade de digitalizações por ciclo:
    df = pd.read_json(get_apa_ciclo());
    df.columns = ['Quant. de Digitalizações', 'Ciclo']
    return df





def velocimetro_fator():
    #dataframe para os indices fatores medio baixo e medio alto
    d = {'Fator': ['E_ESC1', 'E_ESC2', 'E_PROF1', 'E_PROF2', 'E_FAM1', 'E_FAM2', 'E_COM1', 'E_COM2', 'E_COM3', 'E_EST1',
               'E_EST2', 'E_EST3'], 'Medio_Baixo': [3.66, 2.01, 2.34, 2.34, 3.01, 3.34, 2.01, 2.01, 2.01, 2.01, 2.34, 2.01], 
     'Medio_Alto': [5.33, 3.33, 4.33, 4.00, 5.00, 5.00, 3.66, 3.66, 3.66, 4.00, 4.00, 3.33]}
    fatores = pd.DataFrame(data = d)
    query = """select * from public.fatores_est fe
    inner join escolas.aluno a 
    on fe.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    #df = pd.read_sql(con_db_caio2(query))
    df = pd.read_sql(query, con_db_caio())
    #df = connection(query)

    df = df.loc[:,~df.columns.duplicated()].copy()
    
    cross_tab_prop = df[['id_aluno','nome_aluno','id_turma', 'escola', 'uf', 'municipio','cod_escola','nome_turma','E_ESC1V', 'E_ESC2V', 'E_PROF1V', 'E_PROF2V',
       'E_FAM1V', 'E_FAM2V', 'E_COM1V', 'E_COM2V', 'E_COM3V', 'E_EST1V',
       'E_EST2V', 'E_EST3V', 'E_ESC1C', 'E_ESC2C', 'E_PROF1C', 'E_PROF2C',
       'E_FAM1C', 'E_FAM2C', 'E_COM1C', 'E_COM2C', 'E_COM3C', 'E_EST1C',
       'E_EST2C', 'E_EST3C']]

    cross_tab_prop.set_index('id_aluno')
    #cross_tab_prop
    
    filter_column = cross_tab_prop.columns.str.startswith("E_") | cross_tab_prop.columns.str.startswith("id") | cross_tab_prop.columns.str.startswith("cod")|cross_tab_prop.columns.str.startswith("nome")|cross_tab_prop.columns.str.startswith("escola") |cross_tab_prop.columns.str.startswith("uf")|cross_tab_prop.columns.str.startswith("municipio")
    df_col = cross_tab_prop.loc[:,filter_column]
    filter_column = df_col.columns.str.endswith("V") | df_col.columns.str.startswith("id") | df_col.columns.str.startswith("cod")|df_col.columns.str.startswith("nome")|df_col.columns.str.startswith("escola") |df_col.columns.str.startswith("uf")|df_col.columns.str.startswith("municipio")
    df_col = df_col.loc[:, filter_column]
    #df_col.info()
    
    df_col['E_ESCV'] = (df_col['E_ESC2V'] +df_col['E_ESC2V'])/2
    df_col['E_PROFV'] = (df_col['E_PROF1V'] +df_col['E_PROF1V'])/2
    df_col['E_FAMV'] = (df_col['E_FAM1V'] +df_col['E_FAM1V'])/2
    df_col['E_COMV'] = (df_col['E_COM1V'] +df_col['E_COM2V']+df_col['E_COM3V'])/3
    df_col['E_ESTV'] = (df_col['E_EST1V'] +df_col['E_EST2V']+df_col['E_EST3V'])/3
    
    nomeEscola = 'E M E F PROFESSORA DALILA LEAO'
    filter = cross_tab_prop['escola']==nomeEscola
    escola_fator = cross_tab_prop[filter]
    
    
    #filter = (cross_tab_prop['escola']=='E M E F PROFESSORA DALILA LEAO') & (cross_tab_prop['id_turma']==1064)
    #turma_fator = cross_tab_prop[filter]

    
    #filter = (cross_tab_prop['escola']=='E M E F PROFESSORA DALILA LEAO') & (cross_tab_prop['id_turma']==1064) & (cross_tab_prop['nome_aluno']=='Jeferson Oliveira dos Santos')
    #aluno_fator = cross_tab_prop[filter]

    
    cols = escola_fator.loc[:, escola_fator.columns.str.startswith("E_")]
    cols = cols.loc[:, cols.columns.str.endswith("V")]
    r = 1
    c = 1
    n = 1
    tracer = {}
    dictionary_name = {'E_ESC1V': 'Condições Materiais da Escola',
                              'E_ESC2V':  'Condições Materiais do Estudante',
                              'E_PROF1V': 'Inflexibilidade<br>Pedagógica',
                              'E_PROF2V': 'Qualidade<br>Pedagógica',
                              'E_FAM1V':  'Suporte<br>Familiar',
                              'E_FAM2V':  'Gravidez-Parentalidade<br>Atividades Domésticas de Cuidado',
                              'E_COM1V':  'Medidas Socioeducativas<br>Contextos de Violência',
                              'E_COM2V':  'Distanciamento<br>Escola-Comunidade',
                              'E_COM3V':  'Acessibilidade<br>Frequência Escolar',
                              'E_EST1V':  'Significados da Escolarização',
                              'E_EST2V':  'Aspectos Emocionais e Afetivos',
                              'E_EST3V':  'Reprovações e Distorção Idade-Série',
                              }

    for (columnName, columnData) in cols.iteritems():
        
        cols[columnName] = cols[columnName].astype(np.float16)
        mean = cols[columnName].mean()
        
        nameC = columnName
        nameC = nameC.replace('V',"")
        f2 = fatores[fatores['Fator'] == nameC]
        
        
        tracer[n] = go.Indicator(
            mode = "gauge+number+delta",
            value = mean,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': dictionary_name[columnName], 'font': {'size': 18}},
            #reference é pode ser a média geral
            delta = {'reference': f2['Medio_Baixo'].values[0], 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 7], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "red",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, f2['Medio_Baixo'].values[0]], 'color': 'green'},
                    {'range': [f2['Medio_Baixo'].values[0], f2['Medio_Alto'].values[0]], 'color': 'yellow'},
                    {'range': [f2['Medio_Alto'].values[0], 7], 'color': 'red'}
                    
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 490}})
        r = 1
        c = 1
        n = n + 1



    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[[{'type' : 'indicator'}, {'type' : 'indicator'}, {'type' : 'indicator'}],
              [{'type' : 'indicator'}, {'type' : 'indicator'}, {'type' : 'indicator'}],
               [{'type' : 'indicator'}, {'type' : 'indicator'}, {'type' : 'indicator'}],
               [{'type' : 'indicator'}, {'type' : 'indicator'}, {'type' : 'indicator'}],

        ],
        )
    for i in range (1, 13):
            if(c%3 != 0):
                #fig.append_trace(trace1, row=1, col=1)
                #fig.append_trace(trace, row=r, col=c)
                #r = r + 1
                #print ('if, r: ', r, ' C: ', c)
                fig.append_trace(tracer[i], row=r, col=c)
                c = c + 1

            else:
                #print ('else, r: ', r, ' C: ', c)
                fig.append_trace(tracer[i], row=r, col=c)
                c = 1
                r = r + 1

                #fig.append_trace(trace, row=r, col=c)

    fig.update_layout(height=1000, width=1200,title_y=0.99,title_x=0.5, title_text="<b>Fatores da {0}</b>".format(nomeEscola))
    #fig.show()
    return fig


def velocimetro_dimensao():
    # dataframe para os indices fatores medio baixo e medio alto
    d = {'Dimensao': ['E-ESC', 'E-PROF', 'E-FAM', 'E-COM', 'E-EST'],
         'Medio_Baixo': [2.84, 2.67, 3.34, 2.45, 2.44],
         'Medio_Alto': [4, 3.66, 4.33, 3.55, 3.41]}
    dimensoes = pd.DataFrame(data=d)
    query = """select * from public.dimensoes_est de
    inner join escolas.aluno a 
    on de.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola"""
    # df = pd.read_sql(con_db_caio2(query))
    df = pd.read_sql(query, con_db_caio())

    df = df.loc[:, ~df.columns.duplicated()].copy()

    cross_tab_prop = df[
        ['id_aluno', 'nome_aluno', 'id_turma', 'escola', 'uf', 'municipio', 'cod_escola', 'nome_turma',
         'E_ESCV', 'E_PROFV', 'E_FAMV', 'E_COMV', 'E_ESTV',
         'E_ESCC', 'E_PROFC', 'E_FAMC', 'E_COMC', 'E_ESTC', ]]

    cross_tab_prop.set_index('id_aluno')
    # cross_tab_prop

    nomeEscola = 'E M E F PROFESSORA DALILA LEAO'
    filter = cross_tab_prop['escola'] == nomeEscola
    escola_dimensao = cross_tab_prop[filter]


    # filter = (cross_tab_prop['escola']=='E M E F PROFESSORA DALILA LEAO') & (cross_tab_prop['id_turma']==1064)
    # turma_dimensao = cross_tab_prop[filter]


    # filter = (cross_tab_prop['escola']=='E M E F PROFESSORA DALILA LEAO') & (cross_tab_prop['id_turma']==1064) & (cross_tab_prop['nome_aluno']=='Jeferson Oliveira dos Santos')
    # aluno_dimensao = cross_tab_prop[filter]

    dictionary_name = { 'E-ESC':   'Dimensão Estudante-Escola',
                        'E-PROF': 'Dimensão Estudante-Profissionais',
                        'E-FAM' : 'Dimensão Estudante-Familia',
                        'E-COM' : 'Dimensão Estudante-Comunidade',
                        'E-EST' : 'Dimensão Estudante-Estudante'}

    cols = escola_dimensao.loc[:, escola_dimensao.columns.str.startswith("E_")]
    cols = cols.loc[:, cols.columns.str.endswith("V")]
    r = 1
    c = 1
    n = 1
    tracer = {}

    for (columnName, columnData) in cols.iteritems():
        cols[columnName] = cols[columnName].astype(np.float16)
        mean = cols[columnName].mean()
        # print(mean)
        # print('Column Contents : ', columnData.values)
        # print('Column Name : ', columnName)
        nameC = columnName
        nameC = nameC.replace('V', "")
        # print('Column Name whitout C : ', nameC.replace('_', "-"))
        f2 = dimensoes[dimensoes['Dimensao'] == nameC.replace('_', "-")]
        # print(f2)
        columnName = nameC.replace('_', "-")
        tracer[n] = go.Indicator(
            mode="gauge+number+delta",
            value=mean,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': dictionary_name[columnName], 'font': {'size': 18}},
            # reference é pode ser a média geral
            delta={'reference': f2['Medio_Baixo'].values[0], 'increasing': {'color': "RebeccaPurple"}},
            gauge={
                'axis': {'range': [None, 7], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "red",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, f2['Medio_Baixo'].values[0]], 'color': 'green'},
                    {'range': [f2['Medio_Baixo'].values[0], f2['Medio_Alto'].values[0]], 'color': 'yellow'},
                    {'range': [f2['Medio_Alto'].values[0], 7], 'color': 'red'}

                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 490}})
        r = 1
        c = 1
        n = n + 1

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],

               ],
    )
    for i in range(1, 6):
        if (c % 2 != 0):
            # fig.append_trace(trace1, row=1, col=1)
            # fig.append_trace(trace, row=r, col=c)
            # r = r + 1
            # print ('if, r: ', r, ' C: ', c)
            fig.append_trace(tracer[i], row=r, col=c)
            c = c + 1

        else:
            # print ('else, r: ', r, ' C: ', c)
            fig.append_trace(tracer[i], row=r, col=c)
            c = 1
            r = r + 1

            # fig.append_trace(trace, row=r, col=c)

    fig.update_layout(height=1000, width=1200,title_y=0.98,title_x=0.25, title_text="<b>Dimensoes da {0}</b>".format(nomeEscola))
    # fig.show()
    return fig
        
def media_dimensoes():
    #quantidade de digitalizações por ciclo:
    df = pd.read_json(get_dimensoes_geral())
    df = df.groupby(['fatores_dimensões'])['value'].mean()
    #df = df.reset_index()
    df = df.to_frame().reset_index()
    df.columns = ['Dimensão', 'Média Geral']
    return df

def digitalizacoes_apa():

    #inicio = time.time()

    ##############Descritivo Digitalizacoes####################
    query = 'SELECT count(arquivos_separados) FROM digitalizacoes_firebase.avaliacao WHERE arquivos_separados IS NOT NULL group by arquivos_separados;'
    #data_digitalizacoes = connection(query)
    v1 = con_db_caio2(query)
    v1 = len(v1)

    query = 'SELECT count(*) FROM digitalizacoes_firebase.avaliacao WHERE arquivos_separados IS NULL;'
    #data_digitalizacoes = connection(query)
    v2 = con_db_caio2(query)

    #v1 = df_drop_duplicates['arquivos_separados'].count()
    #v2 = df_nan['arquivos_separados'].isna().sum()


    fig = go.Figure(data=[go.Table(
        header=dict(values=list(['Redações Digitalizadas','Redações Não Digitalizadas']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[v1, v2],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title_text="Descritivo das digitalizações das Redações",title_x=0.5) 
    #fim = time.time()
    #print(fim - inicio)
    #fig.show()
    return fig

def dem_quantidades():
    query = 'SELECT * FROM devolutivas_apa.municipio;'
    #municipios =  con_db_caio2(query)
    municipios = pd.read_sql(query, con_db_caio())
    query = 'SELECT * FROM devolutivas_apa.dados_ia_apa;'
    #data_apa = con_db_caio2(query)
    data_apa = pd.read_sql(query, con_db_caio())

    dict_uf= {11:'RO',12:'AC',13:'AM',14:'RR',15:'PA',16:'AP',17:'TO',21:'MA',22:'PI',23:'CE',24:'RN',
                  25:'PB',26:'PE',27:'AL',28:'SE',29:'BA',31:'MG',32:'ES',33:'RJ',35:'SP',41:'PR',42:'SC',43:'RS',50:'MS',
                  51:'MT',52:'GO',53:'DF',}
    data_apa = data_apa.replace({"cod_estado": dict_uf})

    dict_municipio = {}
    for code in set(data_apa['codigo_cidade']):
        for codigo,nome in zip(municipios.id_municipio,municipios.nome):
            if(code==codigo):
                dict_municipio[code] = nome;
    data_apa=data_apa.replace({'codigo_cidade': dict_municipio})
    
    
    #Remover Duplicatas
    data_apa_duplicate = data_apa.drop_duplicates(subset=['nome','level_pontuacao','created_at'])

    result = data_apa_duplicate.groupby(['cod_estado',]).agg({'nome':["nunique"],"turma": ["nunique"],"codigo_cidade": ["nunique"],'escola': ["nunique"]},split_out=4)
    result.reset_index(inplace=True)
    result.columns = ['Estado','Quantidade de Alunos','Quantidade de Turmas','Quantidade de Cidades','Quantidade de Escolas']
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(result.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[result.Estado, result['Quantidade de Alunos'],result['Quantidade de Turmas'], result['Quantidade de Cidades'],result['Quantidade de Escolas']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title_text="Demostrativo da quantidade de Alunos, Turmas, Cidades e Escolas por Estado",title_x=0.5) 
    
    return fig

def dem_quan_pont():
    query = 'SELECT * FROM devolutivas_apa.dados_ia_apa;'
    data_apa = pd.read_sql(query, con_db_caio())


    data_apa_duplicate = data_apa.drop_duplicates(subset=['nome','level_pontuacao','created_at'])

    df = data_apa_duplicate.groupby(['cod_estado']).agg({'codigo_cidade': lambda x: x.nunique(),
                                                         "escola": lambda x: x.nunique(),
                                                          'nome': lambda x: x.nunique()})

    df = df.reset_index()
    df.columns = ['cod_estado','N° de Cidades','N° de Escolas','N° de Estudantes']

    cross_tab_prop = pd.crosstab(index=data_apa_duplicate['cod_estado'],
                                 columns=data_apa_duplicate['level_pontuacao'],
                                 normalize="index")

    cross_tab = pd.crosstab(index=data_apa_duplicate['cod_estado'],columns=data_apa_duplicate['level_pontuacao'],)

    cross_tab_prop.reset_index(inplace=True)

    df_aux = pd.merge(df,cross_tab_prop[['cod_estado','A','B','D']],on='cod_estado', how='left')
    df_aux.rename(columns={'cod_estado': 'Estado',
                           'A':'Pontuação A(%)',
                           'B':'Pontuação B(%)',
                           'D':'Pontuação D(%)'}, inplace=True)

    #Add Linha Brasil
    valorA = cross_tab[['A']].agg('sum', axis=0)[0]
    valorB = cross_tab[['B']].agg('sum', axis=0)[0]
    valorD = cross_tab[['D']].agg('sum', axis=0)[0]
    total = valorA+valorB+valorD
    df_new_row = pd.DataFrame.from_records({'Estado':['Brasil'], 
                            'N° de Cidades':[df[['N° de Cidades']].agg('sum', axis=0)[0]],
                            'N° de Escolas':[df[['N° de Escolas']].agg('sum', axis=0)[0]],
                            'N° de Estudantes':[df[['N° de Estudantes']].agg('sum', axis=0)[0]],
                             'Pontuação A(%)':[valorA/total],
                             'Pontuação B(%)':[valorB/total],
                             'Pontuação D(%)':[valorD/total], 
                           })

    df_aux = pd.concat([df_aux,df_new_row])

    df_aux = df_aux.round(2)

    #Plota Tabela
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_aux.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_aux['Estado'],df_aux['N° de Cidades'], df_aux['N° de Escolas'], df_aux['N° de Estudantes'], 
                           df_aux['Pontuação A(%)'],df_aux['Pontuação B(%)'],df_aux['Pontuação D(%)']],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title_text="Demostrativo Quantidade de Estudantes por Estado considerando <b>Pontuação</b>",title_x=0.5) 
    return fig

def dem_quan_seg():
    query = 'SELECT * FROM devolutivas_apa.dados_ia_apa;'
    data_apa = pd.read_sql(query, con_db_caio())
    data_apa_duplicate = data_apa.drop_duplicates(subset=['nome','level_pontuacao','created_at'])

    cross_tab_segmentacao = pd.crosstab(index=data_apa_duplicate['cod_estado'],
                                 columns=data_apa_duplicate['level_segmentacao'])
    cross_tab_segmentacao.reset_index(inplace=True)
    cross_tab_segmentacao.columns = ['Estado','Pontuação A','Pontuação B','Pontuação C']

    fig = go.Figure(data=[go.Table(header=dict(values=list(cross_tab_segmentacao.columns),
                    fill_color='paleturquoise',
                    align='left'),
                   cells=dict(values=[cross_tab_segmentacao['Estado'],
                                      cross_tab_segmentacao['Pontuação A'],
                                      cross_tab_segmentacao['Pontuação B'],
                                      cross_tab_segmentacao['Pontuação C']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title_text="Demostrativo Quantidade de Alunos por Estado considerando <b>Segmento</b>",title_x=0.5) 
    return fig

def dem_quan_dig_status():
    query = 'SELECT * FROM devolutivas_apa.dados_ia_apa;'
    data_apa = pd.read_sql(query, con_db_caio())
    data_apa_duplicate = data_apa.drop_duplicates(subset=['nome','level_pontuacao','created_at'])

    result = data_apa_duplicate.groupby(['state']).agg({'state':["count"]},split_out=4)
    result.reset_index(inplace=True)
    result.columns = ['Status da Digitalização','Quantidade']
    fig = go.Figure(data=[go.Table(header=dict(values=list(result.columns),
                    fill_color='paleturquoise',
                    align='left'),
                   cells=dict(values=[result['Status da Digitalização'],result['Quantidade']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title_text="Status Digitalizações da Base APA",title_x=0.5) 
    return fig

def desc_estado_sap():
    ##################################Descritivo SAP##############################
    sql_aluno ='SELECT id_aluno, id_turma FROM escolas.aluno'
    sql_turma ='SELECT * FROM escolas.turma'
    #sql_escola ='SELECT * FROM escolas.escola'
    sql_escola = 'SELECT uf, cod_escola, municipio, cod_estado FROM escolas.escola'
    #df = connection(sql_escola)
    #data_aluno = pd.read_csv('/home/eltonss/Documents/MEC/data/aluno.csv');
    data_aluno = pd.read_sql(sql_aluno, con_db_caio())
    #data_turma = pd.read_csv('/home/eltonss/Documents/MEC/data/turma.csv'
    data_turma  = pd.read_sql(sql_turma, con_db_caio())                   
    #data_escola = pd.read_csv('/home/eltonss/Documents/MEC/data/escola.csv');
    data_escola = pd.read_sql(sql_escola, con_db_caio())
    sql_mun = 'SELECT * FROM devolutivas_apa.municipio;'
    municipios =  pd.read_sql(sql_mun, con_db_caio())
    
    #Merge tabela aluno e turma pela chave estrangeira removendo colunas iguais nas tabelas
    cols_to_use = data_turma.columns.difference(data_aluno.columns)
    #print(cols_to_use)
    data_aluno_turma = pd.merge(data_aluno, data_turma[cols_to_use], left_index=True, right_index=True, how='outer')
    #data_aluno_turma = pd.merge(data_aluno, data_turma, on='id_turma')
    data_escola.rename(columns={'cod_escola': 'id_escola',}, inplace=True)
    #Merge tabela escola com dataframe aluno_turma
    cols_to_use = data_escola.columns.difference(data_aluno_turma.columns)
    #print(cols_to_use)
    data_aluno_turma_escola = pd.merge(data_aluno_turma, data_escola[cols_to_use], 
                                       left_index=True, right_index=True, how='outer')

    dict_uf= {11:'RO',12:'AC',13:'AM',14:'RR',15:'PA',16:'AP',17:'TO',21:'MA',22:'PI',23:'CE',24:'RN',
                  25:'PB',26:'PE',27:'AL',28:'SE',29:'BA',31:'MG',32:'ES',33:'RJ',35:'SP',
                  41:'PR',42:'SC',43:'RS',50:'MS',
                  51:'MT',52:'GO',53:'DF',}
    data_aluno_turma_escola = data_aluno_turma_escola.replace({"cod_estado": dict_uf})

    dict_municipio = {}
    for code in set(data_aluno_turma_escola['municipio']):
        for codigo,nome in zip(municipios.id_municipio,municipios.nome):
            if(code==codigo):
                dict_municipio[code] = nome;
    data_aluno_turma_escola=data_aluno_turma_escola.replace({'codigo_cidade': dict_municipio})

    result = data_aluno_turma_escola.groupby(['cod_estado',]).agg({'id_aluno':["nunique"],"id_turma": ["nunique"],"municipio": ["nunique"],'id_escola': ["nunique"]},split_out=4)
    result.reset_index(inplace=True)
    result.columns = ['Estado','Quantidade de Alunos','Quantidade de Turmas','Quantidade de Cidades','Quantidade de Escolas']
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(result.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[result.Estado, result['Quantidade de Alunos'],result['Quantidade de Turmas'], result['Quantidade de Cidades'],result['Quantidade de Escolas']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title_text="Descritivo por Estado da Base Cadastrados Plataforma",title_x=0.5) 

    return fig
