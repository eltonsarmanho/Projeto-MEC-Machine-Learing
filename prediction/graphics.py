import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import random
from decouple import config

from plotly.subplots import make_subplots

from prediction.conecta import get_municipios, get_dados_ia_apa, get_apa_ciclo, get_apa_ciclo2

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


def con_db_caio():
    conn = psycopg2.connect(host=config('PREDICTION_DB_HOST', ''), database=config('PREDICTION_DB_NAME', ''), user=config('PREDICTION_DB_USER', ''), password=config('PREDICTION_DB_PASSWORD', ''))
    return conn


def grafico_risco_escola_dimensoes_barras():
    query = """select * from public.dimensoes_est de
    inner join escolas.aluno a 
    on de.id_aluno = a.id_aluno 
    inner join escolas.turma t 
    on a.id_turma = t.id_turma 
    inner join escolas.escola e 
    on t.id_escola = e.cod_escola WHERE escola = 'E M E F PROFESSORA DALILA LEAO'"""
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
    on t.id_escola = e.cod_escola WHERE escola = 'E M E F PROFESSORA DALILA LEAO'"""
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
    on t.id_escola = e.cod_escola WHERE escola = 'E M E F PROFESSORA DALILA LEAO';"""
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
    on t.id_escola = e.cod_escola WHERE escola = 'E M E F PROFESSORA DALILA LEAO';"""
    df = pd.read_sql(query, con_db_caio())
    df

    df_col = df.loc[:, df.columns.str.startswith("E_")]
    df_col = df_col.loc[:, df_col.columns.str.endswith("C")]
    df_col

    df_dim_graph = pd.DataFrame(columns=['Dimensão', 'R1', 'R2', 'R3'])
    df_for = list(df_col)

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
    query = """SELECT count(DISTINCT aluno) FROM dados_sap.dimensoes_alunos"""
    row = con_db_caio2(query)
    texto = 'Atualmente existem ' +  str(row[0]) +  ' estudantes diferentes cadastrados no SAP '
    query = """SELECT count(DISTINCT aluno) FROM dados_sap.fatores_escola"""
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




def con_db_caio2(query):
    conn = psycopg2.connect(host=config('PREDICTION_DB_HOST', ''), database=config('PREDICTION_DB_NAME', ''),
        user=config('PREDICTION_DB_USER', ''), password=config('PREDICTION_DB_PASSWORD', ''))
    cursor = conn.cursor()
    conn.autocommit = True
    cursor.execute(query)
    result = cursor.fetchone()
    
    return result

