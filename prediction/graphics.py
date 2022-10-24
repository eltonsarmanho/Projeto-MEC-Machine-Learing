# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pandas as pd;
import numpy as np;
import scipy.stats as st;
import plotly.express as px;
import plotly.graph_objects as go;
from plotly.subplots import make_subplots;

from prediction.conecta import get_municipios, get_dados_ia_apa


from django.core.management import call_command
#Realizar Query
#Carrega tabela municipios
# municipios = pd.read_csv('/home/eltonss/Documents/MEC/data/municipio.csv');
municipios = pd.read_json(get_municipios())
#Carrega tabela dados_ia_apa
# data_apa = pd.read_csv('/home/eltonss/Documents/MEC/data/dados_ia_apa.csv');
data_apa = pd.read_json(get_dados_ia_apa());

data_apa.head()


dict_uf= {11:'RO',12:'AC',13:'AM',14:'RR',15:'PA',16:'AP',17:'TO',21:'MA',22:'PI',23:'CE',24:'RN',
              25:'PB',26:'PE',27:'AL',28:'SE',29:'BA',31:'MG',32:'ES',33:'RJ',35:'SP',41:'PR',42:'SC',43:'RS',50:'MS',
              51:'MT',52:'GO',53:'DF',};
data_apa = data_apa.replace({'cod_estado': dict_uf});

dict_municipio = {};
for code in data_apa['codigo_cidade']:
    for codigo,nome in zip(municipios.id_municipio,municipios.nome):
        if(str(code)==str(codigo)):
            dict_municipio[code] = nome;
data_apa=data_apa.replace({'codigo_cidade': dict_municipio});


#Remover Duplicatas
data_apa_duplicate = data_apa.drop_duplicates(subset=['nome','level_pontuacao','created_at']);
result = data_apa_duplicate.groupby(['nome','level_pontuacao','created_at'],group_keys=True).apply(lambda x: x);
print('Check duplicidade: ', result[['nome','level_pontuacao','created_at']].duplicated().any());

NIVEL_ORTOGRAFICO = 'A: NÍVEL ORTOGRÁFICO';
NIVEL_ALFABETICO = 'B: NÍVEL ALFABÉTICO';
NIVEL_SILABICO = 'D: NÍVEL SILÁBICO';

def graph_bar_pontuacao_uf():
    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', 'level_pontuacao']).agg({'level_pontuacao': ['count']}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado','Pontuacao','Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    fig = px.bar(
        data_estado_pontuacao,
        x='Estado',
        y='Valor',
        color='Pontuacao',
        color_discrete_map={'A': 'green','B': 'yellow', 'D': 'red'}, 
        labels={'Valor':'<b>Valor</b>','Estado':'<b>Estado</b>',},
        title='<b>Pontuação por UF</b>'
    )

    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)

    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)
    
    return fig

def graph_bar_proporcao_pontuacao_uf():
    cross_tab_prop = pd.crosstab(index=data_apa_duplicate['cod_estado'], columns=data_apa_duplicate['level_pontuacao'], normalize='index')

    cross_tab_prop.reset_index(inplace=True)

    fig = px.bar(
        cross_tab_prop,
        x='cod_estado',
        y=['A', 'B', 'D'],
        color_discrete_map={'A': 'green','B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por UF</b>',
        labels={'variable':'Nivel','value':'<b>Proporção da Pontuação(%)</b>','cod_estado':'<b>Estado</b>',}
    )

    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)
    
    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)
    
    return fig

def graph_bar_proporcao_pontuacao_cidades(uf):
    data_apa_duplicate_uf = data_apa_duplicate[data_apa_duplicate['cod_estado']==uf]
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
        color_discrete_map={'A': 'green','B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por Cidade</b>',
        labels={'variable':'Nivel','value':'<b>Proporção da Pontuação(%)</b>','codigo_cidade':'<b>Cidade</b>',}
    )

    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)

    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)
    
    return fig

def graph_bar_proporcao_pontuacao_escolas(uf, cidade):
    filtro = (data_apa_duplicate['cod_estado']==uf) & (data_apa_duplicate['codigo_cidade']==cidade)
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
        color_discrete_map={'A': 'green','B': 'yellow', 'D': 'red'},
        title='<b>Competência Pontuação(%) por Escola</b>',
        labels={'variable':'Nivel','value':'<b>Proporção da Pontuação(%)</b>','escola':'<b>Escola</b>',}
    )

    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)

    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)

    return fig

def graph_bar_valor_pontuacao_agregado():
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1)

    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', "level_pontuacao"]).agg(
        {"level_pontuacao": ["count"]}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado', 'Pontuacao', 'Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='A'].sort_values('Estado')
    c1,c2 = st.t.interval(confidence=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))

    fig.add_trace(go.Bar(x=data['Estado'],y=data['Valor'],name='A',marker_color='green'),1, 1)

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='B'].sort_values('Estado')
    c1,c2 = st.t.interval(confidence=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'],name='B',marker_color='yellow'),1, 1)

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='D'].sort_values('Estado')
    c1,c2 = st.t.interval(confidence=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'],name='D',marker_color='red'),1, 1)

    fig['layout'].update(height=900, width=850, title='<b>Pontuação agrupada por Estado </b>',)
    
    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)

    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)
    
    return fig

def graph_bar_valor_por_pontuacao_segmentado():
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Pontuação A','Pontuação B', 'Pontuação D'), vertical_spacing=0.1)

    data_estado_pontuacao = data_apa_duplicate.groupby(['cod_estado', "level_pontuacao"]).agg(
        {"level_pontuacao": ["count"]}, split_out=4)
    data_estado_pontuacao = data_estado_pontuacao.reset_index()
    data_estado_pontuacao.columns = ['Estado', 'Pontuacao', 'Valor']
    data_estado_pontuacao = data_estado_pontuacao.sort_values(by=['Pontuacao'])

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='A'].sort_values('Estado')
    c1,c2 = st.t.interval(alpha=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))

    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'],name='A',marker_color='green'),1, 1)
    fig.add_hline(y=c1, line_dash='dot',row=1, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot',row=1, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']),row=1, col=1, annotation_text='Média', annotation_position='right')

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='B'].sort_values('Estado')
    c1,c2 = st.t.interval(alpha=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'],name='B',marker_color='yellow'),2, 1)

    fig.add_hline(y=c1, line_dash='dot',row=2, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot',row=2, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']),row=2, col=1, annotation_text='Média', annotation_position='right')

    data = data_estado_pontuacao[data_estado_pontuacao['Pontuacao']=='D'].sort_values('Estado')
    c1,c2 = st.t.interval(alpha=0.90, df=len(data['Valor'])-1,loc=np.mean(data['Valor']), scale=st.sem(data['Valor']))
    fig.add_trace(go.Bar(x=data['Estado'], y=data['Valor'],name='D',marker_color='red'),3, 1)
    fig.add_hline(y=c1, line_dash='dot',row=3, col=1, annotation_text='Limite Inferior', annotation_position='right')
    fig.add_hline(y=c2, line_dash='dot',row=3, col=1, annotation_text='Limite Superior', annotation_position='right')
    fig.add_hline(y=np.mean(data['Valor']),row=3, col=1, annotation_text='Média', annotation_position='right')

    fig['layout'].update(height=900, width=850, title='<b>Pontuação Por Estado</b>',)
    fig.update_layout(legend_title='<b>Nível</b>',title_x=0.5)

    fig.data[0].name=NIVEL_ORTOGRAFICO
    fig.data[1].name=NIVEL_ALFABETICO
    fig.data[2].name=NIVEL_SILABICO
    fig.update_traces(showlegend=True)
    
    return fig

