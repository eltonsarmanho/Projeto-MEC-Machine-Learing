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


        start = time.time()
        #dataframe = dd.read_csv(url, sep='|',  dtype='object')
        #dataset_reduce = pd.read_csv(url, sep='|', usecols=lambda x: x not in drop_columns )
        dataset = pd.read_csv(url, delimiter=',',
                                     encoding="utf-8", )

        columns = dataset.columns;
        #columns_drop_1 = filter(lambda name: name.find("NIVEL_") != -1, columns);
        #columns_drop_2 = filter(lambda name: name.find("TAXA_") != -1, columns);
        #columns_drop_3 = filter(lambda name: name.find("PC_") != -1, columns);

        columns_selected = list(filter(lambda name: name.find("_EM") != -1, columns));
        columns_selected.append('ID_ESCOLA')
        columns_selected.append('ID_SAEB')
        #print(columns_selected)


        #columns_drop = list(columns_drop_1)+list(columns_drop_2)+list(columns_drop_3)
        dataset_reduce = dataset.drop(dataset.columns.difference(columns_selected),axis= 1)
        dataset_reduce.rename(columns={'ID_ESCOLA': 'CO_ENTIDADE'}, inplace=True)

        print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))
        #dataset_reduce.update(dataset_reduce[['NO_ENTIDADE']].applymap('"{}"'.format))

        end = time.time()
        print("Read csv: ", (end - start), "sec")
        print(dataset_reduce.columns)
        # dataset_reduce = transformData(dataframe)
        #dataset_reduce.to_csv('../Dataset/' + nameNewFile,sep='\t', encoding='utf-8',index=False)
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

def merge(dataset_escola_saeb):
    print('Merge Method')
    start = time.time()
    dataset_censo_escola_matricula = pd.read_csv('../Dataset/escola_matricula.csv', delimiter='\t',encoding="utf-8", )
    print(dataset_censo_escola_matricula.shape)
    df_result = pd.merge(dataset_escola_saeb,dataset_censo_escola_matricula, on='CO_ENTIDADE')
    print(df_result.shape)
    end = time.time()
    df_result.to_csv('../Dataset/inep_sabe_merge_2019_new.csv', sep='\t', encoding='utf-8', index=False)
    print("Read csv in merge: ", (end - start), "sec")

if __name__ == '__main__':
    url_csv = '../Dataset/TS_ESCOLA.csv'
    dataset_escola_saeb = runDimensionReduction(url_csv,'ts_escola_update.csv')
    merge(dataset_escola_saeb)
