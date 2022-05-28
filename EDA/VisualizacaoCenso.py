
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
def loadData():
    try:
        dataset_censo = pd.read_csv('../Dataset/inep_sabe_merge_2019.csv',delimiter=',' )
        print(dataset_censo.info())
        # Limpeza de dados
        print("Detect missing values.")
        print(dataset_censo.isna().sum() / len(dataset_censo))
        dataset_censo.dropna(inplace=True)

        return dataset_censo;
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

def analiseFactorial(dataset):
    # Dropping unnecessary columns

    dataset.drop(['gender', 'education', 'age'], axis=1, inplace=True)
def mediaByRegiao(dataset):
    dataset_regiao = dataset.copy()
    dataset_regiao.dropna(subset=['MEDIA_EM_LP','MEDIA_EM_MT',
                                  'NU_PRESENTES_EMT','TAXA_PARTICIPACAO_EMT'], inplace=True)
    dataset_regiao_grouped = dataset_regiao.groupby('ID_REGIAO')
    mean_regiao_ME_MT = dataset_regiao_grouped['MEDIA_EM_MT'].mean()
    mean_regiao_ME_PT = dataset_regiao_grouped['MEDIA_EM_LP'].mean()


    UFs = dataset_regiao_grouped['NU_PRESENTES_EMT'].count().index
    size = dataset_regiao_grouped['NU_PRESENTES_EMT'].count()/10
    plt.scatter(x=UFs, y=mean_regiao_ME_MT,s=size,
             alpha=0.5,label='Média em Matemática no EM')
    #plt.scatter(x=UFs,y=mean_regiao_ME_PT,s=size,
    #     alpha=0.5,label='Média em LP no EM')
    plt.legend( markerscale=0.5, scatterpoints=1, fontsize=10)

    plt.xlabel('Região')
    plt.ylabel('Média')
    plt.show()

    
if __name__ == '__main__':
    dt = loadData();
    mediaByRegiao(dt)