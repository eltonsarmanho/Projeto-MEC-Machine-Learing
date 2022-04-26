
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap

pd.set_option("display.max.columns", None)
import sys
import matplotlib.pyplot as plt

def rows(f, chunksize=1024, sep='\n'):
    row = ''
    while (chunk := f.read(chunksize)) != '':  # final do arquivo
        while (i := chunk.find(sep)) != -1:  # Nenhum demilitador identificado
            yield row + chunk[:i]
            chunk = chunk[i + 1:]
            row = ''
        row += chunk
    yield row

def lazyloadData(name):
    print('Method lazyloadData..')
    try:
        count  =  0
        with open(name, encoding="ISO-8859-1") as f:
            for r in rows(f):
                if(count < 1):
                    cols = r.split('|')
                    df = pd.DataFrame(columns=cols)
                    #print(df)
                else:
                    #print(r.split('|'))
                    df.loc[len(df)] = r.split('|')
                count = count + 1
                if(count > 10000): break
        drop_columns= ['NU_ANO_CENSO','NU_ANO','NU_IDADE_REFERENCIA','NU_IDADE',
                       'TP_NACIONALIDADE','CO_PAIS_ORIGEM','CO_UF_NASC',
                       'CO_MUNICIPIO_NASC','CO_UF_END','CO_MUNICIPIO_END',
                       'IN_RECURSO_CD_AUDIO','IN_RECURSO_NENHUM','IN_AEE_OPTICOS_NAO_OPTICOS',
                       'IN_AEE_ENRIQ_CURRICULAR','TP_ETAPA_ENSINO','TP_MEDIACAO_DIDATICO_PEDAGO',
                       'CO_REGIAO', 'CO_MICRORREGIAO','CO_UF','CO_MUNICIPIO',
                       'CO_DISTRITO','TP_DEPENDENCIA','TP_LOCALIZACAO','TP_CATEGORIA_ESCOLA_PRIVADA',
                       'IN_CONVENIADA_PP','TP_CONVENIO_PODER_PUBLICO','IN_MANT_ESCOLA_PRIVADA_EMP',
                       'IN_MANT_ESCOLA_PRIVADA_ONG','IN_MANT_ESCOLA_PRIVADA_OSCIP',
                       'IN_MANT_ESCOLA_PRIV_ONG_OSCIP','IN_MANT_ESCOLA_PRIVADA_SIND',
                       'IN_MANT_ESCOLA_PRIVADA_SIST_S','IN_MANT_ESCOLA_PRIVADA_S_FINS',
                       'IN_RECURSO_INTERPRETE','IN_RECURSO_TRANSCRICAO','IN_RECURSO_LEDOR',
                       'IN_SUPERDOTACAO','IN_DEF_MULTIPLA','IN_SURDOCEGUEIRA','IN_RECURSO_PROVA_PORTUGUES',
                       'TP_LOCAL_RESID_DIFERENCIADA','IN_RECURSO_AMPLIADA_18','IN_RECURSO_AMPLIADA_24',
                       'IN_AEE_LINGUA_PORTUGUESA','IN_AEE_INFORMATICA_ACESSIVEL','IN_AEE_CAA',
                       'IN_AEE_SOROBAN','IN_AEE_VIDA_AUTONOMA','IN_AEE_DESEN_COGNITIVO',
                       'IN_AEE_MOBILIDADE','TP_OUTRO_LOCAL_AULA','TP_RESPONSAVEL_TRANSPORTE',
                       'TP_REGULAMENTACAO','TP_LOCALIZACAO_DIFERENCIADA','TP_UNIFICADA','NU_DIAS_ATIVIDADE',
                       'TP_TIPO_ATENDIMENTO_TURMA','TP_TIPO_LOCAL_TURMA',
                       'NU_DUR_ATIV_COMP_OUTRAS_REDES','NU_DUR_ATIV_COMP_MESMA_REDE','NU_DURACAO_TURMA',
                       'NU_DUR_AEE_OUTRAS_REDES','NU_DUR_AEE_MESMA_REDE','CO_CURSO_EDUC_PROFISSIONAL']
        dataset_reduce = df.drop(columns=drop_columns, axis=1)
        print("Dimensionality reduced from {} to {}.".format(df.shape[1], dataset_reduce.shape[1]))

        dataset_reduce = transformData(dataset_reduce)
        dataset_reduce.to_csv('/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/matricula_norte_update.csv',
                                  sep='\t', encoding='utf-8', index=False)
        return  dataset_reduce;
    except:
         print("Oops!", sys.exc_info()[0], "occurred.");

def transformData(dataset_reduce):
    mapping = {'1': True, '0': False}
    dt = dataset_reduce.filter(like='IN_', axis=1)
    columns_boleana = pd.DataFrame(dt.select_dtypes(['object'])).columns
    for item in columns_boleana:
        dataset_reduce[item].map(mapping)

    dt = dataset_reduce.filter(like='QT_', axis=1)
    columns_int = pd.DataFrame(dt.select_dtypes(['object'])).columns
    for item in columns_int:
        # print("Sobre Coluna %s " % item)
        dataset_reduce[item] = pd.to_numeric(dataset_reduce[item], errors='coerce')
        # print(dataset_reduce[item].dtype)
    # for item in dataset_reduce.columns:
    #    print(dataset_reduce[item].describe())
    return  dataset_reduce

def loadData(name):

    print('Method loadData..')
    try:
        dataset = pd.read_csv(name, encoding="ISO-8859-1", delimiter='|',
                              index_col=False, dtype='unicode')
        filetype = 'CSV'
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
    print(dataset.columns)
    print(dataset.shape)

    drop_columns = [ 'NU_ANO_CENSO',
                     'CO_ORGAO_REGIONAL','IN_VINCULO_OUTRO_ORGAO',
                     'IN_CONVENIADA_PP','TP_CONVENIO_PODER_PUBLICO',
                     'IN_MANT_ESCOLA_PRIVADA_EMP','IN_MANT_ESCOLA_PRIVADA_ONG',
                     'IN_MANT_ESCOLA_PRIVADA_OSCIP','IN_MANT_ESCOLA_PRIV_ONG_OSCIP',
                     'IN_MANT_ESCOLA_PRIVADA_SIND','IN_MANT_ESCOLA_PRIVADA_SIST_S',
                     'IN_MANT_ESCOLA_PRIVADA_S_FINS','TP_REGULAMENTACAO',
                     'TP_RESPONSAVEL_REGULAMENTACAO','TP_OCUPACAO_PREDIO_ESCOLAR',
                     'IN_LOCAL_FUNC_SOCIOEDUCATIVO','IN_LOCAL_FUNC_UNID_PRISIONAL',
                     'IN_LOCAL_FUNC_PRISIONAL_SOCIO','IN_LOCAL_FUNC_GALPAO','TP_OCUPACAO_GALPAO',
                     'IN_LOCAL_FUNC_SALAS_OUTRA_ESC','IN_LOCAL_FUNC_OUTROS','IN_PREDIO_COMPARTILHADO',
                     'TP_INDIGENA_LINGUA','CO_LINGUA_INDIGENA_1','CO_LINGUA_INDIGENA_2','CO_LINGUA_INDIGENA_3',
                     #Variaveis Quantitativas
                     'QT_SALAS_UTILIZADAS_DENTRO','QT_SALAS_UTILIZADAS_FORA','QT_SALAS_UTILIZADAS',
                     'QT_SALAS_UTILIZA_CLIMATIZADAS','QT_SALAS_UTILIZADAS_ACESSIVEIS','QT_EQUIP_DVD',
                     'QT_EQUIP_SOM','QT_EQUIP_TV','QT_EQUIP_LOUSA_DIGITAL','QT_EQUIP_MULTIMIDIA',
                     'CO_DISTRITO',
                     'TP_DEPENDENCIA','TP_LOCALIZACAO','TP_LOCALIZACAO_DIFERENCIADA',
                     'IN_VINCULO_SECRETARIA_EDUCACAO','IN_VINCULO_SEGURANCA_PUBLICA',
                     'IN_VINCULO_SECRETARIA_SAUDE','TP_CATEGORIA_ESCOLA_PRIVADA','CO_ESCOLA_SEDE_VINCULADA',
                     'CO_IES_OFERTANTE','IN_LOCAL_FUNC_PREDIO_ESCOLAR',
                     'IN_ESP_EXCLUSIVA_MEDIO_INTEGR', 'IN_ESP_EXCLUSIVA_MEDIO_NORMAL',
                     'IN_COMUM_EJA_FUND', 'IN_COMUM_EJA_MEDIO', 'IN_COMUM_EJA_PROF',
                     'IN_ESP_EXCLUSIVA_EJA_FUND', 'IN_ESP_EXCLUSIVA_EJA_MEDIO',
                     'IN_ESP_EXCLUSIVA_EJA_PROF', 'IN_COMUM_PROF', 'IN_ESP_EXCLUSIVA_PROF',
                     'TP_REDE_LOCAL','TP_SITUACAO_FUNCIONAMENTO',
                     # Variaveis Temporais
                     'DT_ANO_LETIVO_INICIO','DT_ANO_LETIVO_TERMINO',
                     # Variaveis geograficas
                     'CO_ORGAO_REGIONAL', 'DT_ANO_LETIVO_INICIO', 'DT_ANO_LETIVO_TERMINO', 'CO_REGIAO',
                     'CO_DISTRITO',
                     'TP_DEPENDENCIA', 'TP_LOCALIZACAO',
                     'TP_LOCALIZACAO_DIFERENCIADA','TP_LOCALIZACAO',
                     'CO_ESCOLA_SEDE_VINCULADA','CO_IES_OFERTANTE','TP_REGULAMENTACAO',
                     'TP_RESPONSAVEL_REGULAMENTACAO','TP_PROPOSTA_PEDAGOGICA',
                     'TP_AEE','TP_ATIVIDADE_COMPLEMENTAR'
                     ]
    dataset_reduce = dataset.drop(columns=drop_columns, axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataset.shape[1], dataset_reduce.shape[1]))
    print(dataset_reduce.columns)

    #Transformar colunas
    dataset_reduce = transformData(dataset_reduce)

    for item in dataset_reduce.columns:
        print(dataset_reduce[item].describe())
    dataset_reduce.to_csv('/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/escola_update.csv',
                          sep='\t', encoding='utf-8',index=False)
    return dataset_reduce

def agrupamentoEscola(dataframe,filtered_df):
    columns =  ['QT_PROF_PSICOLOGO','QT_PROF_FONAUDIOLOGO','QT_PROF_ASSIST_SOCIAL',
                'IN_MATERIAL_PED_NENHUM','IN_INTERNET_ALUNOS',
                'IN_ACESSO_INTERNET_COMPUTADOR','IN_ACESSIBILIDADE_INEXISTENTE',
                'IN_ESPECIAL_EXCLUSIVA',
                'IN_ACESSIBILIDADE_RAMPAS','IN_BANHEIRO_PNE','CO_ENTIDADE',
                'CO_MESORREGIAO','CO_MICRORREGIAO','CO_UF','CO_MUNICIPIO','NO_ENTIDADE'
                ]

    df = dataframe[columns]
    df_result = pd.merge(df, filtered_df, on='CO_ENTIDADE')
    return df_result;

def mergeData(url_csv_matricula,url_csv_escola):
    print('Método mergeData')
    #url_csv_escola = "../Dataset/escola_update.csv"
    #url_csv_matricula = "../Dataset/matricula_norte_update.csv"
    #try:
    #    dataset_escola = pd.read_csv(url_csv_escola,delimiter='\t' )
    #    dataset_matricula = pd.read_csv(url_csv_matricula, delimiter='\t')
    #except:
    #    print("Oops!", sys.exc_info()[0], "occurred.")
    print('Shape escola  ' , dataset_escola.shape)
    print('Shape matricula '  , dataset_matricula.shape)

    #Limpeza dos dados
    dataset_matricula.fillna(0,inplace  = True)

    #conta_alunos = dataset_matricula.groupby(['CO_ENTIDADE'],
    #                                         as_index=False).agg({"IN_DEF_INTELECTUAL": ["count"],
    #                                                              "IN_AUTISMO":["sum"]})
    # conta_alunos = dataset_matricula.groupby(['CO_ENTIDADE'],
    #                                         as_index=False).IN_AUTISMO.agg(lambda x:(x==1).sum())

    #Pegar as colunas com IN
    dt = dataset_matricula.filter(like='IN_', axis=1)
    dt.reset_index()
    colunas = list(dt.columns)
    colunas.insert(0,'CO_ENTIDADE')

    #
    estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE'],
                                             as_index=False).agg({lambda x:(x==1).sum()})
    estatistica_alunos_por_escola.reset_index(inplace=True)

    #Padronizar colunas
    colunas_rename = [w.replace('IN_', '') for w in colunas]
    estatistica_alunos_por_escola.columns = colunas_rename

    #Remover Colunas com nenhuma informação ou igual a zero
    filtered_df = estatistica_alunos_por_escola.loc[:, (estatistica_alunos_por_escola != 0).any(axis=0)]
    print("Dimensionality reduced from {} to {}.".format(estatistica_alunos_por_escola.shape[1],
                                                         filtered_df.shape[1]))
    #Check duplicidade
    print("Check duplicidade: ", filtered_df['CO_ENTIDADE'].duplicated().any())
    print('Shape dataframe estatistica Escola ', filtered_df.shape)


    #correlation(filtered_df.drop(columns=['CO_ENTIDADE'], axis=1))
    filter_escola_matricula = dataset_escola['CO_ENTIDADE'].isin(dataset_matricula['CO_ENTIDADE'])

    dataset_escola_filtered = agrupamentoEscola(dataset_escola[filter_escola_matricula],filtered_df)
    print('Shape escola com alunos matriculados ', dataset_escola_filtered.shape)
    #print(dataset_escola_filtered.info())
    #print(dataset_escola_filtered.describe())
    dataset_escola_filtered.to_csv('/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/dataset_escola_filtered.csv',
                          sep=',', encoding='utf-8',index=False)

    # Make the plot
    colunas  = ['CO_ENTIDADE','CO_MICRORREGIAO','CO_UF','CO_MUNICIPIO','REGULAR','NO_ENTIDADE']
    df_aux = dataset_escola_filtered.drop(columns=colunas, axis=1)
    pd.plotting.parallel_coordinates(df_aux, 'CO_MESORREGIAO', colormap=plt.get_cmap("Set2"))
    plt.show()

def correlation(dataframe):
    print("correlation...")

    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    corr = dataframe.corr()
    # Create positive correlation matrix
    corr = dataframe.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    print(to_drop)
    reduced_df = dataframe.drop(to_drop,axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], reduced_df.shape[1]))
    #Insert Column without erro

    # Create and apply mask
    mask = np.triu(np.ones_like(reduced_df.corr(), dtype=bool))
    sns.heatmap(reduced_df.corr(), mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=True, fmt=".2f")
    plt.tight_layout()

    plt.show()
    return  reduced_df;



if __name__ == '__main__':

    url_csv = "../Dataset/matricula_norte.csv"
    dataset_matricula = lazyloadData(url_csv)
    url_csv = "../Dataset/escolas.csv"
    dataset_escola  = loadData(url_csv)


    mergeData(dataset_matricula,dataset_escola)
