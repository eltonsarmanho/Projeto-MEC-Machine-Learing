
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
    columns = ["ID_SAEB","ID_REGIAO",	'ID_UF',	'ID_MUNICIPIO',	'ID_AREA',	'ID_ESCOLA',
'ID_DEPENDENCIA_ADM','ID_LOCALIZACAO','PC_FORMACAO_DOCENTE_INICIAL',
'PC_FORMACAO_DOCENTE_FINAL','PC_FORMACAO_DOCENTE_MEDIO','NIVEL_SOCIO_ECONOMICO',
'NU_MATRICULADOS_CENSO_5EF','NU_PRESENTES_5EF',	'TAXA_PARTICIPACAO_5EF',
'NIVEL_0_LP5','NIVEL_1_LP5','NIVEL_2_LP5','NIVEL_3_LP5','NIVEL_4_LP5','NIVEL_5_LP5','NIVEL_6_LP5',
'NIVEL_7_LP5','NIVEL_8_LP5','NIVEL_9_LP5','NIVEL_0_MT5','NIVEL_1_MT5','NIVEL_2_MT5','NIVEL_3_MT5','NIVEL_4_MT5',
'NIVEL_5_MT5','NIVEL_6_MT5','NIVEL_7_MT5','NIVEL_8_MT5','NIVEL_9_MT5',
'NIVEL_10_MT5','NU_MATRICULADOS_CENSO_9EF','NU_PRESENTES_9EF','TAXA_PARTICIPACAO_9EF','NIVEL_0_LP9','NIVEL_1_LP9',
'NIVEL_2_LP9','NIVEL_3_LP9','NIVEL_4_LP9','NIVEL_5_LP9','NIVEL_6_LP9',
'NIVEL_7_LP9','NIVEL_8_LP9','NIVEL_0_MT9','NIVEL_1_MT9','NIVEL_2_MT9','NIVEL_3_MT9','NIVEL_4_MT9',
'NIVEL_5_MT9','NIVEL_6_MT9','NIVEL_7_MT9','NIVEL_8_MT9','NIVEL_9_MT9',
'NU_MATRICULADOS_CENSO_EMT','NU_PRESENTES_EMT','TAXA_PARTICIPACAO_EMT','NIVEL_0_LPEMT','NIVEL_1_LPEMT',
'NIVEL_2_LPEMT','NIVEL_3_LPEMT','NIVEL_4_LPEMT','NIVEL_5_LPEMT','NIVEL_6_LPEMT',
'NIVEL_7_LPEMT','NIVEL_8_LPEMT','NIVEL_0_MTEMT','NIVEL_1_MTEMT','NIVEL_2_MTEMT','NIVEL_3_MTEMT','NIVEL_4_MTEMT',
'NIVEL_5_MTEMT','NIVEL_6_MTEMT','NIVEL_7_MTEMT','NIVEL_8_MTEMT',
'NIVEL_9_MTEMT','NIVEL_10_MTEMT','NU_MATRICULADOS_CENSO_EMI','NU_PRESENTES_EMI','TAXA_PARTICIPACAO_EMI',
'NIVEL_0_LPEMI','NIVEL_1_LPEMI','NIVEL_2_LPEMI',
'NIVEL_3_LPEMI','NIVEL_4_LPEMI','NIVEL_5_LPEMI','NIVEL_6_LPEMI','NIVEL_7_LPEMI','NIVEL_8_LPEMI',
'NIVEL_0_MTEMI','NIVEL_1_MTEMI','NIVEL_2_MTEMI','NIVEL_3_MTEMI','NIVEL_4_MTEMI',
'NIVEL_5_MTEMI','NIVEL_6_MTEMI','NIVEL_7_MTEMI','NIVEL_8_MTEMI','NIVEL_9_MTEMI','NIVEL_10_MTEMI']

    reduced_df = dataset.drop(columns, axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataset.shape[1], reduced_df.shape[1]))
    return  reduced_df

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
    analiseFactorial(dt)