
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def loadData():
    try:
        dataset_censo = pd.read_csv('../Dataset/dataset_escola_filtered.csv',delimiter=',' )
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

    print(dataset_censo.info())
    #Limpeza de dados
    print("Detect missing values.")
    print(dataset_censo.isna().sum() / len(dataset_censo))
    dataset_censo.dropna(inplace=True)
    
if __name__ == '__main__':
    loadData();
