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
    file_path = "/home/eltonss/Documents/MEC/data/CENSO/matricula_norte.CSV"
    dataframe = dd.read_csv(file_path, sep='|', assume_missing=True)

    print(dataframe.info())
    #print("Dimensionality total {}.".format(dataframe.shape))
    #print('Numero de registros {}'.format(dataframe.shape[0].compute()))
    #matriculas = dataframe.groupby(by='ID_MATRICULA').agg({"ID_MATRICULA": ["count"]}, split_out=100).compute()
    #print("Numero de matriculas total {}.".format(len(matriculas)))
    #escolas  = dataframe.groupby(by='CO_ENTIDADE').agg({"CO_ENTIDADE": ["count"]},split_out=24).compute()
    #print("Numero de Escolas total {}.".format(len(escolas)))
