import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from empiricaldist import Cdf
from matplotlib import cm
from matplotlib.colors import ListedColormap
import  empiricaldist
import scipy
def loadData():
    try:
        dataset = pd.read_csv("../Dataset/inep_sabe_merge_all.csv",low_memory=False);

        #for key, value in dict_uf.items():
        #    dataset.loc[dataset["ID_UF"] == str(key) , "ID_UF"] = value
        #dataset.to_csv('../Dataset/inep_sabe_merge_all4.csv', index=False)
        return dataset;
    except:
         print("Oops!", sys.exc_info()[0], "occurred.");

def plotDiscriminate(dataset):
    dataset_uf = dataset[dataset['ID_REGIAO'] == 'Norte'].copy()

    # Encontrar a média
    #media = dataset_uf.NECESSIDADE_ESPECIAL.mean()
    # Make a column that denotes which day had highest O3
    dataset_uf['Suporte'] = ['EXISTE' if value1 >0 or value2 >0 or value3 >0 else 'NAO EXISTE'
                                          for value1, value2,value3 in zip(dataset_uf.IN_ESPECIAL_EXCLUSIVA,
                                                           dataset_uf.IN_ACESSIBILIDADE_RAMPAS,
                                                           dataset_uf.IN_BANHEIRO_PNE)]



    # Encode the hue of the points with the O3 generated column
    sns.scatterplot(x='NECESSIDADE_ESPECIAL',
                    y='DEF_FISICA',
                    hue='Suporte',
                    data=dataset_uf)
    plt.show()



    pass;

def plotKernelNotas(dataset):
    dataset.MEDIA_EM_MT.dropna(axis=0,inplace=True)
    dataset_norte = dataset[dataset['ID_REGIAO'] == 'Norte'].copy()
    dataset_regioes = dataset[dataset['ID_REGIAO'] != 'Norte'].copy()
    # Filtrar dataset em região
    sns.distplot(dataset_norte.MEDIA_5EF_MT,
                rug = True,hist=False,
                label='Norte',color='red')

    # Filtrar dataset em região
    sns.distplot(dataset_regioes.MEDIA_5EF_MT,
                hist=False,
                label='Norte', color='blue')
    plt.show()

def beeswarmsUF(dataset):
    dataset.MEDIA_EM_LP.dropna(axis=0, inplace=True)
    #dataset_regiao = dataset[dataset['ID_REGIAO'] != 'Sul'].copy()
    #print("Shape ", dataset_regiao.shape)
    # Filtrar dataset em região
    sns.swarmplot(y=dataset.ID_UF,
                  x=dataset.MEDIA_EM_LP,
                  data=dataset,
                  # Decrease the size of the points to avoid crowding
                  size=1)

    # Give a descriptive title
    plt.title('Média em Língua Portuguesa por Estado no Ensino Médio')
    plt.show()

def violinPlot(dataset):
    dataset_regiao = dataset[dataset['ID_REGIAO'] != 'Sul'].copy()
    dataset_regiao.dropna(subset=['MEDIA_EM_LP','MEDIA_EM_MT'], inplace=True)
    sns.violinplot(x=dataset_regiao.ID_UF,y=dataset_regiao.MEDIA_EM_LP,data=dataset_regiao,inner=None)
    plt.show()

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

def cdfPlot(dataset):
    dataset_regiao = dataset[dataset['ID_REGIAO'] == 'Sul'].copy()
    dataset_regiao.dropna(subset=['QT_PROF_ASSIST_SOCIAL','QT_PROF_PSICOLOGO','QT_PROF_FONAUDIOLOGO'],
                          inplace=True)

    for value in ['QT_PROF_ASSIST_SOCIAL','QT_PROF_PSICOLOGO','QT_PROF_FONAUDIOLOGO']:
        # sort data
        data_sorted = np.sort(dataset_regiao[value])
        # calculate CDF values
        norm_cdf = scipy.stats.norm.cdf(data_sorted)
        # plot CDF
        plt.plot(data_sorted, norm_cdf,label=value,alpha=0.4)

    plt.legend()
    # plot CDF
    plt.xlabel('Quantidade')
    plt.ylabel('CDF')
    plt.show()
def boxPlot(dataset):
    dataset_regiao = dataset.copy()
    dataset_regiao.dropna(subset=['MEDIA_EM_LP','MEDIA_EM_MT'], inplace=True)
    sns.boxplot(x=dataset_regiao.ID_UF,y=dataset_regiao.MEDIA_EM_LP,data=dataset_regiao,whis=10)
    #plt.yscale('log')
    plt.show()

def correlation(dataframe):
    print("correlation...")
    drop_columns = ['ID_SAEB','ID_REGIAO','ID_UF','ID_MUNICIPIO',
                    'ID_AREA','ID_ESCOLA','ID_DEPENDENCIA_ADM','CO_MUNICIPIO','CO_UF',
                    'CO_MESORREGIAO','CO_MICRORREGIAO','CO_UF','CO_MUNICIPIO',
                    'NO_ENTIDADE','NECESSIDADE_ESPECIAL',

                    'ID_LOCALIZACAO','PC_FORMACAO_DOCENTE_INICIAL','PC_FORMACAO_DOCENTE_FINAL',
                     'PC_FORMACAO_DOCENTE_MEDIO','NIVEL_SOCIO_ECONOMICO','NU_MATRICULADOS_CENSO_5EF',
                     'NU_PRESENTES_5EF','TAXA_PARTICIPACAO_5EF','NIVEL_0_LP5','NIVEL_1_LP5',
                     'NIVEL_2_LP5','NIVEL_3_LP5','NIVEL_4_LP5','NIVEL_5_LP5','NIVEL_6_LP5',
                     'NIVEL_7_LP5','NIVEL_8_LP5','NIVEL_9_LP5','NIVEL_0_MT5','NIVEL_1_MT5',
                      'NIVEL_2_MT5','NIVEL_3_MT5','NIVEL_4_MT5','NIVEL_5_MT5','NIVEL_6_MT5',
                      'NIVEL_7_MT5','NIVEL_8_MT5','NIVEL_9_MT5','NIVEL_10_MT5',
                      'NU_MATRICULADOS_CENSO_9EF','NU_PRESENTES_9EF','TAXA_PARTICIPACAO_9EF',
                      'NIVEL_0_LP9','NIVEL_1_LP9','NIVEL_2_LP9','NIVEL_3_LP9','NIVEL_4_LP9',
                      'NIVEL_5_LP9','NIVEL_6_LP9','NIVEL_7_LP9','NIVEL_8_LP9','NIVEL_0_MT9',
                       'NIVEL_1_MT9','NIVEL_2_MT9','NIVEL_3_MT9','NIVEL_4_MT9','NIVEL_5_MT9',
                       'NIVEL_6_MT9','NIVEL_7_MT9','NIVEL_8_MT9','NIVEL_9_MT9',
                       'NU_MATRICULADOS_CENSO_EMT','NU_PRESENTES_EMT','TAXA_PARTICIPACAO_EMT',
                       'NIVEL_0_LPEMT','NIVEL_1_LPEMT','NIVEL_2_LPEMT','NIVEL_3_LPEMT',
                       'NIVEL_4_LPEMT','NIVEL_5_LPEMT','NIVEL_6_LPEMT','NIVEL_7_LPEMT',
                       'NIVEL_8_LPEMT','NIVEL_0_MTEMT','NIVEL_1_MTEMT',	'NIVEL_2_MTEMT',
                        'NIVEL_3_MTEMT','NIVEL_4_MTEMT','NIVEL_5_MTEMT','NIVEL_6_MTEMT',
                        'NIVEL_7_MTEMT','NIVEL_8_MTEMT','NIVEL_9_MTEMT','NIVEL_10_MTEMT',
                        'NU_MATRICULADOS_CENSO_EMI','NU_PRESENTES_EMI','TAXA_PARTICIPACAO_EMI',
                        'NIVEL_0_LPEMI','NIVEL_1_LPEMI','NIVEL_2_LPEMI','NIVEL_3_LPEMI',
                        'NIVEL_4_LPEMI','NIVEL_5_LPEMI','NIVEL_6_LPEMI',
                        'NIVEL_7_LPEMI','NIVEL_8_LPEMI','NIVEL_0_MTEMI','NIVEL_1_MTEMI',
                        'NIVEL_2_MTEMI','NIVEL_3_MTEMI',	'NIVEL_4_MTEMI','NIVEL_5_MTEMI',
                        'NIVEL_6_MTEMI','NIVEL_7_MTEMI','NIVEL_8_MTEMI',	'NIVEL_9_MTEMI',
                        'NIVEL_10_MTEMI','NU_MATRICULADOS_CENSO_EM','NU_PRESENTES_EM',
                        'TAXA_PARTICIPACAO_EM','NIVEL_0_LPEM','NIVEL_1_LPEM',
                        'NIVEL_2_LPEM','NIVEL_3_LPEM',	'NIVEL_4_LPEM',	'NIVEL_5_LPEM',	'NIVEL_6_LPEM',	'NIVEL_7_LPEM',
                        'NIVEL_8_LPEM','NIVEL_0_MTEM',	'NIVEL_1_MTEM',	'NIVEL_2_MTEM',	'NIVEL_3_MTEM',	'NIVEL_4_MTEM',
                        'NIVEL_5_MTEM',	'NIVEL_6_MTEM','NIVEL_7_MTEM','NIVEL_8_MTEM','NIVEL_9_MTEM','NIVEL_10_MTEM',
                         'RECURSO_LABIAL','RECURSO_BRAILLE',
                    'AEE_LIBRAS'  ,  'AEE_BRAILLE' , 'TRANSP_TR_ANIMAL',    'TRANSP_VANS_KOMBI' ,
                    'TRANSP_OUTRO_VEICULO',    'TRANSP_EMBAR_ATE5',
                    'TRANSP_EMBAR_5A15'   , 'TRANSP_EMBAR_15A35'  ,  'TRANSP_EMBAR_35' ,
                        'REGULAR' ,   'EJA','PROFISSIONALIZANTE','EDUCACAO_INDIGENA',
                    'TRANSP_BICICLETA'		,'TRANSP_TR_ANIMAL',	'TRANSP_VANS_KOMBI','IN_ACESSO_INTERNET_COMPUTADOR'

                    ]
    dataframe.drop(columns=drop_columns, axis=1,inplace=True)

    dataframe.dropna(inplace=True,how='all')
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    corr = dataframe.corr()
    # Create positive correlation matrix
    corr = dataframe.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.6)]
    print(to_drop)
    reduced_df = dataframe.drop(to_drop,axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], reduced_df.shape[1]))
    #Insert Column without erro

    # Create and apply mask
    mask = np.triu(np.ones_like(reduced_df.corr(), dtype=bool))
    sns.heatmap(reduced_df.corr(), mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=True, fmt=".2f")
    plt.tight_layout()

    plt.show()
    return  reduced_df;


if __name__ == '__main__':
    dataset = loadData()
    mediaByRegiao(dataset)