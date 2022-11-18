import sys
import matplotlib.pyplot as plt
import time
import dask.dataframe as dd
import glob
import os
from tqdm import tqdm
import itertools
import csv
import os
import re
import gc
import pandas as pd;

if __name__ == '__main__':
    file_path = "/home/eltonss/Documents/MEC/data/CENSO/matricula_sudeste.CSV"
    dataframe = dd.read_csv(file_path, sep='|', dtype={'CO_MUNICIPIO_END': 'float64',
                                                       'CO_MUNICIPIO_NASC': 'float64',
                                                       'CO_UF_END': 'float64',
                                                       'CO_UF_NASC': 'float64',
                                                       'IN_CONVENIADA_PP': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_EMP': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_ONG': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_OSCIP': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_SIND': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_SIST_S': 'float64',
                                                       'IN_MANT_ESCOLA_PRIVADA_S_FINS': 'float64',
                                                       'IN_MANT_ESCOLA_PRIV_ONG_OSCIP': 'float64',
                                                       'NU_DIAS_ATIVIDADE': 'float64',
                                                       'NU_DURACAO_TURMA': 'float64',
                                                       'NU_DUR_AEE_MESMA_REDE': 'float64',
                                                       'NU_DUR_AEE_OUTRAS_REDES': 'float64',
                                                       'NU_DUR_ATIV_COMP_MESMA_REDE': 'float64',
                                                       'NU_DUR_ATIV_COMP_OUTRAS_REDES': 'float64',
                                                       'TP_CATEGORIA_ESCOLA_PRIVADA': 'float64',
                                                       'TP_ETAPA_ENSINO': 'float64',
                                                       'TP_TIPO_LOCAL_TURMA': 'float64',
                                                       'TP_UNIFICADA': 'float64', 'IN_EJA': 'float64',
                                                       'IN_ESPECIAL_EXCLUSIVA': 'float64',
                                                       'IN_PROFISSIONALIZANTE': 'float64',
                                                       'IN_REGULAR': 'float64', 'TP_ZONA_RESIDENCIAL': 'float64',
                                                       'IN_TRANSPORTE_PUBLICO': 'float64','CO_DISTRITO': 'float64',
                                                        'CO_ENTIDADE': 'float64',
                                                           'CO_MESORREGIAO': 'float64',
                                                           'CO_MICRORREGIAO': 'float64',
                                                           'CO_MUNICIPIO': 'float64',
                                                           'CO_REGIAO': 'float64',
                                                           'CO_UF': 'float64',
                                                           'ID_TURMA': 'float64',
                                                           'IN_EDUCACAO_INDIGENA': 'float64',
                                                           'TP_DEPENDENCIA': 'float64',
                                                           'TP_LOCALIZACAO': 'float64',
                                                           'TP_LOCALIZACAO_DIFERENCIADA': 'float64',
                                                           'TP_MEDIACAO_DIDATICO_PEDAGO': 'float64',
                                                           'TP_OUTRO_LOCAL_AULA': 'float64',
                                                           'TP_REGULAMENTACAO': 'float64',
                                                           'TP_TIPO_ATENDIMENTO_TURMA': 'float64'})

    #print("Dimensionality total {}.".format(dataframe.shape))
    print('Numero de registros {}'.format(dataframe.shape[0].compute()))
    #matriculas = dataframe.groupby(by='ID_MATRICULA').agg({"ID_MATRICULA": ["count"]}, split_out=100).compute()
    #print("Numero de matriculas total {}.".format(len(matriculas)))
    #escolas  = dataframe.groupby(by='CO_ENTIDADE').agg({"CO_ENTIDADE": ["count"]},split_out=24).compute()
    #print("Numero de Escolas total {}.".format(len(escolas)))