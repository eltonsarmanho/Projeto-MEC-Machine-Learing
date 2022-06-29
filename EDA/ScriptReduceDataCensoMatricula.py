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
import re


# Comando para Quebrar CSV
# split -b 800000000 MATRICULA_NORDESTE.CSV /home/eltonss/Documents/data/MATRICULA_SUDESTE.CSV
def runDimensionReduction(url, nameNewFile=None):
    try:
        drop_columns = ['NU_ANO_CENSO', 'NU_ANO', 'NU_IDADE_REFERENCIA', 'NU_IDADE',
                        'TP_NACIONALIDADE', 'CO_PAIS_ORIGEM', 'CO_UF_NASC',
                        'CO_MUNICIPIO_NASC', 'CO_UF_END', 'CO_MUNICIPIO_END',
                        'IN_RECURSO_CD_AUDIO', 'IN_RECURSO_NENHUM', 'IN_AEE_OPTICOS_NAO_OPTICOS',
                        'IN_AEE_ENRIQ_CURRICULAR', 'TP_ETAPA_ENSINO', 'TP_MEDIACAO_DIDATICO_PEDAGO',
                        'CO_REGIAO', 'CO_MICRORREGIAO', 'CO_UF', 'CO_MUNICIPIO',
                        'CO_DISTRITO', 'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'TP_CATEGORIA_ESCOLA_PRIVADA',
                        'IN_CONVENIADA_PP', 'TP_CONVENIO_PODER_PUBLICO', 'IN_MANT_ESCOLA_PRIVADA_EMP',
                        'IN_MANT_ESCOLA_PRIVADA_ONG', 'IN_MANT_ESCOLA_PRIVADA_OSCIP',
                        'IN_MANT_ESCOLA_PRIV_ONG_OSCIP', 'IN_MANT_ESCOLA_PRIVADA_SIND',
                        'IN_MANT_ESCOLA_PRIVADA_SIST_S', 'IN_MANT_ESCOLA_PRIVADA_S_FINS',
                        'IN_RECURSO_TRANSCRICAO', 'IN_RECURSO_LEDOR',
                        'IN_SUPERDOTACAO', 'IN_DEF_MULTIPLA', 'IN_SURDOCEGUEIRA', 'IN_RECURSO_PROVA_PORTUGUES',
                        'TP_LOCAL_RESID_DIFERENCIADA', 'IN_RECURSO_AMPLIADA_18', 'IN_RECURSO_AMPLIADA_24',
                        'IN_AEE_LINGUA_PORTUGUESA', 'IN_AEE_INFORMATICA_ACESSIVEL', 'IN_AEE_CAA',
                        'IN_AEE_SOROBAN', 'IN_AEE_VIDA_AUTONOMA', 'IN_AEE_DESEN_COGNITIVO',
                        'IN_AEE_MOBILIDADE', 'TP_OUTRO_LOCAL_AULA', 'TP_RESPONSAVEL_TRANSPORTE',
                        'TP_REGULAMENTACAO', 'TP_LOCALIZACAO_DIFERENCIADA', 'TP_UNIFICADA', 'NU_DIAS_ATIVIDADE',
                        'TP_TIPO_ATENDIMENTO_TURMA', 'TP_TIPO_LOCAL_TURMA',
                        'NU_DUR_ATIV_COMP_OUTRAS_REDES', 'NU_DUR_ATIV_COMP_MESMA_REDE', 'NU_DURACAO_TURMA',
                        'NU_DUR_AEE_OUTRAS_REDES', 'NU_DUR_AEE_MESMA_REDE', 'CO_CURSO_EDUC_PROFISSIONAL']

        #dataframe = dd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns, )
        dataframe = dd.read_csv(url, sep='\t',usecols=lambda x: x not in drop_columns,dtype='object')
        print("Dimensionality Reduzida {}.".format(dataframe.shape))
        #dataset_reduce = transformData(dataframe)
        dataframe.to_csv('../Dataset/2017/'+nameNewFile+'.csv',sep='\t', encoding='utf-8', index_label='indice',index=False, single_file=True)
        #return dataframe
    except:
        print("Oops!", sys.exc_info(), "occurred.");

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
    return dataset_reduce

def splitFileWithDask(file):
    start = time.time()

    file_path = "../Dataset/2017/"+file+".CSV"
    dataframe = dd.read_csv(file_path,sep='|',dtype={'CO_MUNICIPIO_END': 'float64',
       'CO_MUNICIPIO_NASC': 'float64',
       'CO_UF_END': 'float64',
       'CO_UF_NASC': 'float64',
       'IN_EJA': 'float64',
       'IN_ESPECIAL_EXCLUSIVA': 'float64',
       'IN_PROFISSIONALIZANTE': 'float64',
       'IN_REGULAR': 'float64',
       'NU_DURACAO_TURMA': 'float64',
       'NU_DUR_AEE_MESMA_REDE': 'float64',
       'NU_DUR_AEE_OUTRAS_REDES': 'float64',
       'NU_DUR_ATIV_COMP_MESMA_REDE': 'float64',
       'NU_DUR_ATIV_COMP_OUTRAS_REDES': 'float64',
       'TP_ETAPA_ENSINO': 'float64',
       'TP_UNIFICADA': 'float64'})
    print("Dimensionality total {}.".format(dataframe.shape))

    # set how many file you would like to have
    # in this case 10
    dataframe = dataframe.repartition(npartitions=20)
    dataframe.to_csv("../Dataset/2017/"+file+"_file_*.csv",sep='\t')
    end = time.time()
    print("Split csv: ", (end - start), "sec")

def concatCSV():
    # setting the path for joining multiple files
    files = os.path.join("../Dataset/2017","matricula_reduzido_*.csv")
    CHUNK_SIZE = 1024
    # list of merged files returned
    files = glob.glob(files)
    print(files)
    print("Resultant CSV after joining all CSV files at a particular location...");

    # reading files using read_csv with chunk
    # chunk_container = [pd.read_csv(f, chunksize=CHUNK_SIZE, sep='\t') for f in files]

    # joining files using chunk
    # combined_csv = pd.concat(map(pd.DataFrame, chunk_container), ignore_index=True)
    # combined_csv = [pd.concat(chunk,ignore_index=True) for chunk in chunk_container]

    # joining files using read_csv withou chunk
    # combined_csv = pd.concat(map(pd.read_csv(sep='\t'), files), ignore_index=True)
    combined_csv = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
    combined_csv.to_csv('../Dataset/2017/matricula_reduzido_all.csv',sep='\t', encoding='utf-8', index=False)
    print(combined_csv.shape)

def concatCSVbyRows(filefind,file):
    files = os.path.join("../Dataset/2017",filefind+'*.csv')

    all_files = glob.glob(files)
    print(all_files)
    out_file = '../Dataset/2017/'+file+'.csv';
    with open(out_file, 'w') as outfile:
        for i, filename in enumerate(all_files):
            print(i, filename)
            with open(filename, 'r') as infile:
                for rownum, line in enumerate(infile):
                    if (i != 0) and (rownum == 0):  # Only write header once
                        continue
                    outfile.write(line)

def concatCSVWithDask(findFile,file):
    files = os.path.join("../Dataset/2017", findFile)
    # list of merged files returned
    files = glob.glob(files)
    df_all = dd.concat([dd.read_csv(f, sep='\t',dtype={'IN_EJA': 'float64',
       'IN_ESPECIAL_EXCLUSIVA': 'float64',
       'IN_PROFISSIONALIZANTE': 'float64',
       'IN_REGULAR': 'float64'}) for f in files])
    print(df_all.shape)
    df_all.to_csv('../Dataset/2017/'+file+'.csv',  sep='\t', encoding='utf-8', index=False,single_file=True)

def loadMatriculaWithDask(url, separate=None, newFile=None):
    start = time.time()
    if(separate != None):
        dataframe = dd.read_csv(url,dtype='object' ,sep=separate )
    else: dataframe = dd.read_csv(url,dtype='object',blocksize="10MB" )

    end = time.time()
    print("Read csv without chunks: ", (end - start), "sec")
    print(dataframe.shape)
    print(dataframe.columns)
    # ensure small enough block size for the graph to fit in your memory
    #print(dataframe.shape[0].compute())
    #print(dataframe.head(5))

    calculateMissingValues(dataframe)

def removeColumnWithDask(url, separate=None, newFile=None):
    if (separate != None):
        dataset = dd.read_csv(url, dtype='object', sep=separate)
    else:
        dataset = dd.read_csv(url, dtype='object', blocksize="10MB")

    if newFile != None:
        dataset_reduce = dataset.drop('Unnamed: 0', axis=1)
        dataset_reduce.to_csv('../Dataset/2017/'+newFile+'.csv',single_file=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

def calculateMissingValues(nyc_data_raw):
    print("Detect missing values.")
    start = time.time()
    missing_values = nyc_data_raw.isnull().sum()
    percent_missing = ((missing_values / nyc_data_raw.index.size) * 100).compute()
    print(percent_missing)
    end = time.time()
    print("Read csv with Dask: ", (end - start), "sec")


if __name__ == '__main__':
    #Passo 1: Quebrar os arquivos
    #Arquivos da base de dados
    region = 'norte'
    file = 'MATRICULA_'+region.upper()
    file_out_split='MATRICULA_'+region.upper()+'_REDUZIDO_'
    file_out_split_remove = 'MATRICULA_'+region.upper()+'_REDUZIDO_REMOVE_'
    file_out_merge= 'matricula_reduzido_'+region

    print("Inicia Split File")
    splitFileWithDask(file)
    print("Finaliza Split File")

    # Passo 2: Reduz numero de dimensões
    print("Inicia Redução de Dimensão")
    start = time.time()
    files = os.path.join("../Dataset/2017", file+"_file_*.csv")
    # list of merged files returned
    files = glob.glob(files)
    for f in files:
        n = re.findall(r'\d+', f)[1]
        runDimensionReduction(f,file_out_split+str(n))

    end = time.time()
    print("Finaliza redução em : ", (end - start), "sec")

    # Passo 3: remove index
    print("Inicia Remocao de indice")
    start = time.time()
    files = os.path.join("../Dataset/2017", file_out_split+"*.csv")
    files = glob.glob(files)
    for f in files:
        n = re.findall(r'\d+', f)[1]
        removeColumnWithDask(f,'\t',file_out_split_remove+str(n))
    end = time.time()
    print("Finaliza remoção indice em : ", (end - start), "sec")

    #Passo 4: Realiza Merge dos arquivos - Se caso separou os arquivos
    print("Inicia Merge de Linhas")
    start = time.time()
    concatCSVbyRows(file_out_split_remove,file_out_merge)
    #concatCSVWithDask(findFile,file_out_merge)
    end = time.time()
    print("Finaliza Merge em : ", (end - start), "sec")

