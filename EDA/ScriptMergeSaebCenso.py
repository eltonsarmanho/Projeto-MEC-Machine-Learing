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



def runDimensionReduction(url):
    try:
        #dataframe = dd.read_csv(url, sep='|',  dtype='object')
        #dataset_reduce = pd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns )
        dataset = pd.read_csv(url, delimiter=',', encoding="utf-8", )
        print(dataset.shape)
        columns = dataset.columns;
        #columns_drop_1 = filter(lambda name: name.find("NIVEL_") != -1, columns);
        #columns_drop_2 = filter(lambda name: name.find("TAXA_") != -1, columns);
        #columns_drop_3 = filter(lambda name: name.find("PC_") != -1, columns);

        columns_selected = list(filter(lambda name: name.find("MEDIA") != -1, columns));
        columns_selected.append('ID_ESCOLA')
        columns_selected.append('ID_SAEB')
        #print(columns_selected)
        #print(dataset.columns.difference(columns_selected))

        #columns_drop = list(columns_drop_1)+list(columns_drop_2)+list(columns_drop_3)
        dataset_reduce = dataset.drop(dataset.columns.difference(columns_selected),axis= 1)
        dataset_reduce.rename(columns={'ID_ESCOLA': 'CO_ENTIDADE'}, inplace=True)

        print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))
        #dataset_reduce.update(dataset_reduce[['NO_ENTIDADE']].applymap('"{}"'.format))

        print(dataset_reduce.columns)
        return dataset_reduce
    except:
        print("Oops!", sys.exc_info(), "occurred.");

def calculateMissingValues(dataframe):
    print("Detect missing values.")
    start = time.time()
    missing_values = dataframe.isnull().sum()
    percent_missing = ((missing_values / dataframe.index.size) * 100)
    print(percent_missing)
    end = time.time()
    print("Read: ", (end - start), "sec")

def merge(censo,dataset_escola_saeb,ano):
    print('Merge Method')
    dataset_censo_escola_matricula = pd.read_csv(censo, delimiter='\t',encoding="utf-8", )
    print(dataset_censo_escola_matricula.shape)
    print("Check duplicidade: ", dataset_censo_escola_matricula['CO_ENTIDADE'].duplicated().any())
    df_result = pd.merge(dataset_escola_saeb,dataset_censo_escola_matricula, on='CO_ENTIDADE')
    print(df_result.shape)
    # Check duplicidade
    print("Check duplicidade: ", df_result['CO_ENTIDADE'].duplicated().any())
    df_result.to_csv('../Dataset/'+str(ano)+'/inep_sabe_merge_'+str(ano)+'.csv', sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    ano = str(2019)
    start = time.time()
    url_csv = '../Dataset/'+ano+'/TS_ESCOLA.csv'
    dataset_escola_saeb = runDimensionReduction(url_csv)
    censo = '../Dataset/'+ano+'/dataset_escola_filtered.csv'
    merge(censo,dataset_escola_saeb,ano)
    end = time.time()
    print("Total Time: ", (end - start), "sec")