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

def loadData(indice):
    start = time.time()
    dataset_escola = pd.read_csv('../Dataset/escola_update.csv', sep='\t',)
    dataset_matricula  = pd.read_csv('../Dataset/all/matricula_reduzido_all_2019_'+str(indice)+'.csv', low_memory=True, sep='\t')

    end = time.time()
    print("Read csv without Dask: ", (end - start), "sec")
    print(dataset_escola.shape)
    print(dataset_matricula.shape)

    #Tratamento dos dados
    #dataset_matricula = dataset_matricula.fillna(0)
    #dataset_escola.fillna(0)
    #calculateMissingValues(dataset_matricula)
    #calculateMissingValues(dataset_escola)
    return dataset_matricula,dataset_escola;

def calculateMissingValues(nyc_data_raw):
    print("Detect missing values.")
    start = time.time()
    missing_values = nyc_data_raw.isnull().sum()
    percent_missing = ((missing_values / nyc_data_raw.index.size) * 100).compute()
    print(percent_missing)
    end = time.time()
    print("Read csv with Dask: ", (end - start), "sec")

def calculateIndicators(dataset_escola, dataset_matricula,indice):
    start = time.time()
    columns = dataset_matricula.columns;
    filtered = filter(lambda name: name.find("IN_") != -1, columns);

    colunas = list(filtered)
    for coluna in colunas:
        dataset_matricula[coluna] = pd.to_numeric(dataset_matricula[coluna], errors="coerce")
    colunas.insert(0, 'CO_ENTIDADE')
    #print(colunas)
    #print(dataset_matricula.dtypes)

    #estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE']).agg({"IN_DEF_INTELECTUAL": ["sum"]},split_out=4)
    estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE'],as_index=False).agg(["sum"], split_out=8)
    #estatistica_alunos_por_escola = dataset_matricula[colunas].groupby(['CO_ENTIDADE'],as_index=False).agg({lambda x: (x == 1).sum()},split_out=8)
    estatistica_alunos_por_escola = estatistica_alunos_por_escola.reset_index()
    #print(estatistica_alunos_por_escola.head(5))


    # Padronizar colunas
    colunas_rename = [w.replace('IN_', '') for w in colunas]
    estatistica_alunos_por_escola.columns = colunas_rename

    df_result = pd.merge(estatistica_alunos_por_escola,dataset_escola, on='CO_ENTIDADE')

    # Check duplicidade
    print("Check duplicidade: ", df_result['CO_ENTIDADE'].duplicated().any())
    print('Shape dataframe estatistica Escola ', df_result.shape)

    df_result.to_csv('/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/dataset_escola_filtered_'+str(indice)+'.csv',sep='\t', encoding='utf-8',index=False)

    end = time.time()
    print("Running process: ", (end - start), "sec")


def splitFileWithDask():
    file_path = "/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/matricula_reduzido_all_2019.csv"
    df = dd.read_csv(file_path,
    dtype={'CO_ENTIDADE': 'float64',
       'CO_MESORREGIAO': 'float64',
       'ID_MATRICULA': 'float64',
       'ID_TURMA': 'float64',
       'IN_EDUCACAO_INDIGENA': 'float64',
       'IN_EJA': 'float64',
       'IN_ESPECIAL_EXCLUSIVA': 'float64',
       'IN_NECESSIDADE_ESPECIAL': 'float64',
       'IN_PROFISSIONALIZANTE': 'float64',
       'IN_REGULAR': 'float64',
       'IN_TRANSPORTE_PUBLICO': 'float64',
       'NU_MES': 'float64',
       'TP_COR_RACA': 'float64',
       'TP_SEXO': 'float64',
       'TP_ZONA_RESIDENCIAL': 'float64'})
    print(df.shape)
    # set how many file you would like to have
    # in this case 10
    df = df.repartition(npartitions=10)
    df.to_csv("../Dataset/all/matricula_reduzido_all_2019_*.csv",sep='\t', encoding='utf-8', index=False)

def concatCSV():
    # setting the path for joining multiple files
    files = os.path.join("/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/",
                         "dataset_escola_filtered_*.csv")

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
    combined_csv.to_csv('/home/eltonss/PycharmProjects/Projeto-MEC-Machine-Learing/Dataset/out.csv',
                        sep='\t', encoding='utf-8', index=False)
    print(combined_csv.shape)

if __name__ == '__main__':
    #splitFileWithDask()
    for n in range(0,10):
        dataset_matricula,dataset_escola = loadData(n)
        calculateIndicators(dataset_escola, dataset_matricula,n)

    concatCSV()