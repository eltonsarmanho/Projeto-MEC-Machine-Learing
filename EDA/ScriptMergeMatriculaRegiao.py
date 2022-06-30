
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

def loadMatriculaWithDask(url, separate=None,):
    start = time.time()
    if(separate != None):
        dataframe = dd.read_csv(url,dtype='object' ,sep=separate )
    else: dataframe = dd.read_csv(url,dtype='object',blocksize="10MB" )

    end = time.time()
    print("Read csv without chunks: ", (end - start), "sec")
    print(dataframe.shape)
    print(dataframe.columns)

def concatCSVWithDask(filefind,file,ano):
    # list of merged files returned
    files = os.path.join("../Dataset/"+ano, filefind + '*.csv')

    all_files = glob.glob(files)
    print(all_files)
    out_file = '../Dataset/'+ano+'/' + file + '.csv';
    with open(out_file, 'w') as outfile:
        for i, filename in enumerate(all_files):
            print(i, filename)
            with open(filename, 'r') as infile:
                for rownum, line in enumerate(infile):
                    if (i != 0) and (rownum == 0):  # Only write header once
                        continue
                    outfile.write(line)
if __name__ == '__main__':
    ano = str(2019)
    start = time.time()
    #loadMatriculaWithDask('../Dataset/2017/matricula_reduzido_nordeste.csv',)
    concatCSVWithDask('matricula_reduzido_','matricula_reduzido_all_'+ano,ano)
    end = time.time()
    print("Total time:", (end - start), "sec")
