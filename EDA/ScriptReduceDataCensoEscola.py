import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap

pd.set_option("display.max.columns", None)
import sys
import matplotlib.pyplot as plt
import time
from dask import dataframe as dd
import glob
import os
from tqdm import tqdm
import itertools
import csv
import os



def runDimensionReduction(url, nameNewFile=None):
    try:
        drop_columns = ['NU_ANO_CENSO',
                        'CO_ORGAO_REGIONAL', 'IN_VINCULO_OUTRO_ORGAO',
                        'IN_CONVENIADA_PP', 'TP_CONVENIO_PODER_PUBLICO',
                        'IN_MANT_ESCOLA_PRIVADA_EMP', 'IN_MANT_ESCOLA_PRIVADA_ONG',
                        'IN_MANT_ESCOLA_PRIVADA_OSCIP', 'IN_MANT_ESCOLA_PRIV_ONG_OSCIP',
                        'IN_MANT_ESCOLA_PRIVADA_SIND', 'IN_MANT_ESCOLA_PRIVADA_SIST_S',
                        'IN_MANT_ESCOLA_PRIVADA_S_FINS', 'TP_REGULAMENTACAO',
                        'TP_RESPONSAVEL_REGULAMENTACAO', 'TP_OCUPACAO_PREDIO_ESCOLAR',
                        'IN_LOCAL_FUNC_SOCIOEDUCATIVO', 'IN_LOCAL_FUNC_UNID_PRISIONAL',
                        'IN_LOCAL_FUNC_PRISIONAL_SOCIO', 'IN_LOCAL_FUNC_GALPAO', 'TP_OCUPACAO_GALPAO',
                        'IN_LOCAL_FUNC_SALAS_OUTRA_ESC', 'IN_LOCAL_FUNC_OUTROS', 'IN_PREDIO_COMPARTILHADO',
                        'TP_INDIGENA_LINGUA', 'CO_LINGUA_INDIGENA_1', 'CO_LINGUA_INDIGENA_2', 'CO_LINGUA_INDIGENA_3',
                        # Variaveis Quantitativas
                        'QT_SALAS_UTILIZADAS_DENTRO', 'QT_SALAS_UTILIZADAS_FORA', 'QT_SALAS_UTILIZADAS',
                        'QT_SALAS_UTILIZA_CLIMATIZADAS', 'QT_SALAS_UTILIZADAS_ACESSIVEIS',
                        'CO_DISTRITO',
                        'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'TP_LOCALIZACAO_DIFERENCIADA',
                        'IN_VINCULO_SECRETARIA_EDUCACAO', 'IN_VINCULO_SEGURANCA_PUBLICA',
                        'IN_VINCULO_SECRETARIA_SAUDE', 'TP_CATEGORIA_ESCOLA_PRIVADA', 'CO_ESCOLA_SEDE_VINCULADA',
                        'CO_IES_OFERTANTE', 'IN_LOCAL_FUNC_PREDIO_ESCOLAR',
                        'IN_ESP_EXCLUSIVA_MEDIO_INTEGR', 'IN_ESP_EXCLUSIVA_MEDIO_NORMAL',
                        'IN_COMUM_EJA_FUND', 'IN_COMUM_EJA_MEDIO', 'IN_COMUM_EJA_PROF',
                        'IN_ESP_EXCLUSIVA_EJA_FUND', 'IN_ESP_EXCLUSIVA_EJA_MEDIO',
                        'IN_ESP_EXCLUSIVA_EJA_PROF', 'IN_COMUM_PROF', 'IN_ESP_EXCLUSIVA_PROF',
                        'TP_REDE_LOCAL', 'TP_SITUACAO_FUNCIONAMENTO',
                        # Variaveis Temporais
                        'DT_ANO_LETIVO_INICIO', 'DT_ANO_LETIVO_TERMINO',
                        # Variaveis geograficas
                        'CO_ORGAO_REGIONAL', 'DT_ANO_LETIVO_INICIO', 'DT_ANO_LETIVO_TERMINO', 'CO_REGIAO',
                        'CO_DISTRITO',
                        'TP_DEPENDENCIA', 'TP_LOCALIZACAO',
                        'TP_LOCALIZACAO_DIFERENCIADA', 'TP_LOCALIZACAO',
                        'CO_ESCOLA_SEDE_VINCULADA', 'CO_IES_OFERTANTE', 'TP_REGULAMENTACAO',
                        'TP_RESPONSAVEL_REGULAMENTACAO', 'TP_PROPOSTA_PEDAGOGICA',
                        'TP_AEE', 'TP_ATIVIDADE_COMPLEMENTAR'
                        ]
        start = time.time()
        #dataframe = dd.read_csv(url, sep='|',  dtype='object')
        #dataset_reduce = pd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns )
        dataset_reduce = pd.read_csv(url, delimiter='|',
                                     encoding="utf-8",usecols=lambda x: x not in drop_columns )

        #print("Dimensionality reduced from {} to {}.".format(dataframe.shape, dataset_reduce.shape))
        dataset_reduce.update(dataset_reduce[['NO_ENTIDADE']].applymap('"{}"'.format))

        end = time.time()
        print("Read csv with Dask: ", (end - start), "sec")
        print(dataset_reduce.head(5))
        print(dataset_reduce.shape)
        # dataset_reduce = transformData(dataframe)
        dataset_reduce.to_csv('../Dataset/' + nameNewFile,sep='\t', encoding='utf-8',index=False)

    except:
        print("Oops!", sys.exc_info(), "occurred.");

def showDataWithDask(url):
    dataframe = dd.read_csv(url,  dtype='object')

    print(dataframe.shape)
    print(dataframe.columns)
    calculateMissingValues(dataframe)

def calculateMissingValues(nyc_data_raw):
    print("Detect missing values.")
    start = time.time()
    missing_values = nyc_data_raw.isnull().sum()
    percent_missing = ((missing_values / nyc_data_raw.index.size) * 100).compute()
    print(percent_missing)
    end = time.time()
    print("Read csv with Dask: ", (end - start), "sec")

if __name__ == '__main__':
    url_csv = '../Dataset/ESCOLAS.CSV'
    runDimensionReduction(url_csv,'escola_update.csv')

    #showDataWithDask('../Dataset/escola_update.csv')