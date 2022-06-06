
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import seaborn as sns
sns.set(style="ticks")
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn import preprocessing

import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

def loadData(url):
    try:
        dataset_censo = pd.read_csv(url,delimiter=',' )
        #print(dataset_censo.info())
        # Limpeza de dados

        #dataset_censo.dropna(inplace=True)

        return dataset_censo;
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")


def preprocessamento(dataset):
    # Dropping unnecessary columns
    columns = ["ID_SAEB",'ID_UF',	'ID_MUNICIPIO',	'ID_AREA',	'ID_ESCOLA',
'ID_DEPENDENCIA_ADM','ID_LOCALIZACAO','PC_FORMACAO_DOCENTE_INICIAL',
'PC_FORMACAO_DOCENTE_FINAL','PC_FORMACAO_DOCENTE_MEDIO','NIVEL_SOCIO_ECONOMICO',
'NU_MATRICULADOS_CENSO_5EF','NU_MATRICULADOS_CENSO_EM',
               'MEDIA_EMT_MT',            'TAXA_PARTICIPACAO_EMT','MEDIA_EMT_LP','MEDIA_EMI_LP', 'MEDIA_EMI_MT',
'NIVEL_0_LP5','NIVEL_1_LP5','NIVEL_2_LP5','NIVEL_3_LP5','NIVEL_4_LP5','NIVEL_5_LP5','NIVEL_6_LP5',
'NIVEL_7_LP5','NIVEL_8_LP5','NIVEL_9_LP5','NIVEL_0_MT5','NIVEL_1_MT5','NIVEL_2_MT5','NIVEL_3_MT5','NIVEL_4_MT5',
'NIVEL_5_MT5','NIVEL_6_MT5','NIVEL_7_MT5','NIVEL_8_MT5','NIVEL_9_MT5',
'NIVEL_10_MT5','NU_MATRICULADOS_CENSO_9EF','NU_PRESENTES_9EF','NIVEL_0_LP9','NIVEL_1_LP9',
'NIVEL_2_LP9','NIVEL_3_LP9','NIVEL_4_LP9','NIVEL_5_LP9','NIVEL_6_LP9',
'NIVEL_7_LP9','NIVEL_8_LP9','NIVEL_0_MT9','NIVEL_1_MT9','NIVEL_2_MT9','NIVEL_3_MT9','NIVEL_4_MT9',
'NIVEL_5_MT9','NIVEL_6_MT9','NIVEL_7_MT9','NIVEL_8_MT9','NIVEL_9_MT9',
'NU_MATRICULADOS_CENSO_EMT','NIVEL_0_LPEMT','NIVEL_1_LPEMT',
'NIVEL_2_LPEMT','NIVEL_3_LPEMT','NIVEL_4_LPEMT','NIVEL_5_LPEMT','NIVEL_6_LPEMT',
'NIVEL_7_LPEMT','NIVEL_8_LPEMT','NIVEL_0_MTEMT','NIVEL_1_MTEMT','NIVEL_2_MTEMT','NIVEL_3_MTEMT','NIVEL_4_MTEMT',
'NIVEL_5_MTEMT','NIVEL_6_MTEMT','NIVEL_7_MTEMT','NIVEL_8_MTEMT',
'NIVEL_9_MTEMT','NIVEL_10_MTEMT','NU_MATRICULADOS_CENSO_EMI','NU_PRESENTES_EMI','TAXA_PARTICIPACAO_EMI',
'NIVEL_0_LPEMI','NIVEL_1_LPEMI','NIVEL_2_LPEMI',
'NIVEL_3_LPEMI','NIVEL_4_LPEMI','NIVEL_5_LPEMI','NIVEL_6_LPEMI','NIVEL_7_LPEMI','NIVEL_8_LPEMI',
'NIVEL_0_MTEMI','NIVEL_1_MTEMI','NIVEL_2_MTEMI','NIVEL_3_MTEMI','NIVEL_4_MTEMI',
'NIVEL_5_MTEMI','NIVEL_6_MTEMI','NIVEL_7_MTEMI','NIVEL_8_MTEMI','NIVEL_9_MTEMI','NIVEL_10_MTEMI',
               'NIVEL_0_LPEM', 'NIVEL_1_LPEM', 'NIVEL_2_LPEM',
               'NIVEL_3_LPEM', 'NIVEL_4_LPEM', 'NIVEL_5_LPEM', 'NIVEL_6_LPEM',
               'NIVEL_7_LPEM', 'NIVEL_8_LPEM', 'NIVEL_0_MTEM', 'NIVEL_1_MTEM',
               'NIVEL_2_MTEM', 'NIVEL_3_MTEM', 'NIVEL_4_MTEM', 'NIVEL_5_MTEM',
               'NIVEL_6_MTEM', 'NIVEL_7_MTEM', 'NIVEL_8_MTEM', 'NIVEL_9_MTEM',
               'NIVEL_10_MTEM','CO_MESORREGIAO', 'CO_MICRORREGIAO', 'CO_UF', 'CO_MUNICIPIO',
               'NO_ENTIDADE','EDUCACAO_INDIGENA','PROFISSIONALIZANTE','EJA','ESPECIAL_EXCLUSIVA',
               'TRANSP_BICICLETA','TRANSP_MICRO_ONIBUS',  'TRANSP_TR_ANIMAL',
               'TRANSP_VANS_KOMBI', 'TRANSP_OUTRO_VEICULO', 'TRANSP_EMBAR_ATE5',
               'TRANSP_EMBAR_5A15', 'TRANSP_EMBAR_15A35', 'TRANSP_EMBAR_35','IN_ESPECIAL_EXCLUSIVA',
               'CEGUEIRA','SURDEZ','AUTISMO','RECURSO_LIBRAS','RECURSO_LABIAL','RECURSO_VIDEO_LIBRAS',
               'RECURSO_BRAILLE','AEE_LIBRAS', 'TAXA_PARTICIPACAO_5EF', 'TAXA_PARTICIPACAO_9EF',
               'TAXA_PARTICIPACAO_EM',
               ]

    reduced_df = dataset.drop(columns, axis=1)
    #print(reduced_df.info())
    #print(reduced_df.columns)
    print("Dimensionality reduced from {} to {}.".format(dataset.shape[1], reduced_df.shape[1]))
    reduced_df.drop(reduced_df[reduced_df['QT_PROF_FONAUDIOLOGO'] >= 88888].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['QT_PROF_PSICOLOGO'] >= 88888].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['IN_ACESSO_INTERNET_COMPUTADOR'] >= 2].index, inplace=True)

    reduced_df.drop(reduced_df[reduced_df['NECESSIDADE_ESPECIAL'] >= 60].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['BAIXA_VISAO'] >= 2].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['DEF_AUDITIVA'] >= 2].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['DEF_FISICA'] >= 2].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['DEF_INTELECTUAL'] >= 2].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['TRANSPORTE_PUBLICO'] >= 263].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['TRANSP_ONIBUS'] >= 2].index, inplace=True)
    reduced_df.drop(reduced_df[reduced_df['REGULAR'] >= 400].index, inplace=True)
    print(reduced_df.shape)
    #data = reduced_df['REGULAR']
    #histogram_boxplot(data, bins = 20, title="Plot", xlabel="Valores")
    print("Detect missing values.")
    print(reduced_df.isna().sum() / len(reduced_df))
    #reduced_df_isna = reduced_df.dropna(subset=["MEDIA_EM_MT",'MEDIA_EM_LP','MEDIA_9EF_MT','MEDIA_9EF_LP'])

    #print("length reduced from {} to {}.".format(reduced_df.shape[0], reduced_df_isna.shape[0]))
    #print(reduced_df_isna.isna().sum() / len(reduced_df_isna))

    return  reduced_df


def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """

    sns.set(font_scale=font_scale)
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    sns.boxplot(data, ax=ax_box2,)
    sns.distplot(data, ax=ax_hist2, bins=bins) if bins else sns.distplot(data, ax=ax_hist2)
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title)
    plt.show()

def runAnaliseFactorial(dataframe):
    print("runAnaliseFactorial")
    print(dataframe.info())
    print(dataframe.columns)
    columns = ['ID_REGIAO', 'NU_PRESENTES_5EF', 'NU_PRESENTES_EMT', 'NU_PRESENTES_EM',
       'MEDIA_5EF_LP', 'MEDIA_5EF_MT', 'MEDIA_9EF_LP', 'MEDIA_9EF_MT',
       'MEDIA_EM_LP', 'MEDIA_EM_MT', 'QT_PROF_PSICOLOGO',
       'QT_PROF_FONAUDIOLOGO']
    dataframe_reduce = dataframe.drop(columns, axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], dataframe_reduce.shape[1]))

    chi_square_value, p_value = calculate_bartlett_sphericity(dataframe_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataframe_reduce)
    print('kmo: ',kmo_model)

    # Create factor analysis object and perform factor analysis
    #fa = FactorAnalyzer(n_factors=14, rotation=None)
    #fa.fit(dataframe_reduce)
    # Check Eigenvalues
    #ev, v = fa.get_eigenvalues()

    # Create scree plot using matplotlib
    #plt.scatter(range(1, dataframe_reduce.shape[1] + 1), ev)
    #plt.plot(range(1, dataframe_reduce.shape[1] + 1), ev)
    #plt.title('Scree Plot')
    #plt.xlabel('Factors')
    #plt.ylabel('Eigenvalue')
    #plt.grid()
    #plt.show()


    # Create factor analysis object and perform factor analysis

    fa = FactorAnalyzer(n_factors=3,rotation="varimax")
    fa.fit(dataframe_reduce)
    #print(fa.loadings_)
    fa_loading_df = pd.DataFrame(fa.loadings_,index= dataframe_reduce.columns, columns=['Factor 1', 'Factor 2', 'Factor 3'])
    #print(fa_loading_df)

    # Get variance of each factors
    #print(fa.get_factor_variance())
    fa_variance_df = pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
                                  columns=['Factor 1', 'Factor 2', 'Factor 3'])
    #print(fa_variance_df)

    array = fa.transform(dataframe_reduce)

    factor1 = np.around(array[:, 0], 2)
    dataframe['Acesso Internet'] = factor1

    factor2 = np.around(array[:, 1], 2)
    dataframe['Acessibilidade'] = factor2

    factor3 = np.around(array[:, 2], 2)
    dataframe['Deficiencia Fisica'] = factor3
    return dataframe;

def normalize(X):
    from_min = np.min(X);
    from_max = np.max(X)
    to_max = 1;
    to_min = 0;
    df = (X - np.min(X)) * (to_max - to_min) / (from_max - from_min) + to_min

    return  df;
def mediaByRegiaoComFator(dataset):
    fig, ax = plt.subplots(figsize=(7, 7))
    axins1 = inset_axes(ax, width='10%', height='2%', loc='lower right')

    dataset_regiao = dataset.copy()
    dataset_regiao.dropna(subset=['MEDIA_EM_LP', 'MEDIA_EM_MT',
                                  'NU_PRESENTES_EM'], inplace=True)
    dataset_regiao_grouped = dataset_regiao.groupby('ID_REGIAO')
    mean_regiao_ME_MT = dataset_regiao_grouped['MEDIA_EM_MT'].mean()
    mean_regiao_ME_PT = dataset_regiao_grouped['MEDIA_EM_LP'].mean()

    UFs = dataset_regiao_grouped['NU_PRESENTES_EM'].count().index
    size_presentes = dataset_regiao_grouped['NU_PRESENTES_EM'].count()
    color = dataset_regiao_grouped['Acesso Internet'].sum()*40


    hexbins = ax.hexbin(x=UFs,y=mean_regiao_ME_MT,
                        C=color,label='Média em Matemática no EM',alpha=0.9,
                        bins=20, gridsize=20, cmap='RdBu')
    cmin, cmax = hexbins.get_clim()
    below = 0 * (cmax - cmin) + cmin
    above = 1 * (cmax - cmin) + cmin

    cbar = fig.colorbar(hexbins, cax=axins1, orientation='horizontal', ticks=[below, above])
    cbar.ax.set_xticklabels(['0', '1'])
    axins1.xaxis.set_ticks_position('top')

    plt.show()

def resumeDataframe(dataset,fator):
    dataset_regiao = dataset.copy()

    dataset_regiao.dropna(subset=['MEDIA_EM_LP', 'MEDIA_EM_MT',
                                  'NU_PRESENTES_EM'], inplace=True)
    dataset_regiao_grouped = dataset_regiao.groupby('ID_REGIAO')
    mean_regiao_ME_MT = dataset_regiao_grouped['MEDIA_EM_MT'].mean()
    mean_regiao_ME_PT = dataset_regiao_grouped['MEDIA_EM_LP'].mean()

    UFs = dataset_regiao_grouped['NU_PRESENTES_EM'].count().index
    size_presentes = dataset_regiao_grouped['NU_PRESENTES_EM'].count()

    normalized = normalize(dataset_regiao_grouped[fator].sum()) * 255
    color = [str(1 - (item / 255.)) for item in normalized]
    return UFs, mean_regiao_ME_PT,size_presentes,color

def mediaByRegiaoAnual(dataset_1,dataset_2):
    #fator = 'Deficiencia Fisica'
    fator = 'Acesso Internet'
    #fator = 'Acessibilidade'
    UFs, mean_regiao_ME,size_presentes,color = resumeDataframe(dataset_1,fator)

    plt.scatter(x=UFs,
                y=mean_regiao_ME,s=size_presentes*10,c=color,edgecolors="black",
             alpha=0.5,label='2019')
    UFs, mean_regiao_ME, size_presentes, color = resumeDataframe(dataset_2,fator)
    plt.scatter(x=UFs,
                y=mean_regiao_ME, s=size_presentes * 10, c=color,marker=',', edgecolors="black",
                alpha=0.5,label='2020')
    plt.legend( markerscale=0.5, scatterpoints=1, fontsize=10)
    plt.title('Média em Pt no EM com Fator '+fator)
    plt.ylabel('Média')
    plt.xlabel('Região')

    plt.show()

    
if __name__ == '__main__':
    url_2019 = '../Dataset/inep_sabe_merge_2019.csv'
    url_2020 = '../Dataset/inep_sabe_merge_2020.csv'
    data_1 = loadData(url_2019);
    data_2 = loadData(url_2020);
    dataframe_1 = preprocessamento(data_1)
    dataframe_2 = preprocessamento(data_2)

    dataframe_1 = runAnaliseFactorial(dataframe_1)
    dataframe_2 = runAnaliseFactorial(dataframe_2)

    #mediaByRegiao(dataframe)
    mediaByRegiaoAnual(dataframe_1,dataframe_2)