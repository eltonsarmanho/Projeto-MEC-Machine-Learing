
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
        dataset_censo = pd.read_csv(url,delimiter='\t' )
        print(dataset_censo.shape)
        return dataset_censo;
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")



def preprocessamento(df):
    # Dropping unnecessary columns
    #Pegar 10 valores mais frequentes
    n = 10
    data = df['QT_PROF_PSICOLOGO']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    df['QT_PROF_PSICOLOGO'] = np.where(a > data.mean(), max(mode), a).tolist()

    data = df['IN_ACESSO_INTERNET_COMPUTADOR']
    a = np.array(data.values.tolist())
    df['IN_ACESSO_INTERNET_COMPUTADOR'] = np.where(a >= data.mean(), data.mean(), a).tolist()

    data = df['NECESSIDADE_ESPECIAL']
    a = np.array(data.values.tolist())
    threshould = np.percentile(data, 75)
    df['NECESSIDADE_ESPECIAL'] = np.where(a >= threshould, threshould, a).tolist()

    data = df['BAIXA_VISAO']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = max(mode)
    df['BAIXA_VISAO'] = np.where(a > threshould, threshould, a).tolist()

    data = df['DEF_AUDITIVA']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = max(mode)
    df['DEF_AUDITIVA'] = np.where(a > threshould, threshould, a).tolist()

    data = df['DEF_FISICA']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = max(mode)
    df['DEF_FISICA'] = np.where(a > threshould, threshould, a).tolist()
    data = df['DEF_FISICA']

    data = df['DEF_INTELECTUAL']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = max(mode)
    df['DEF_INTELECTUAL'] = np.where(a > threshould, threshould, a).tolist()

    data = df['TRANSPORTE_PUBLICO']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = data.median()
    df['TRANSPORTE_PUBLICO'] = np.where(a > threshould, threshould, a).tolist()
    data = df['TRANSPORTE_PUBLICO']


    data = df['TRANSP_ONIBUS']
    mode = data.value_counts()[:n].index.tolist()
    a = np.array(data.values.tolist())
    threshould = max(mode)
    df['TRANSP_ONIBUS'] = np.where(a > threshould, threshould, a).tolist()
    data = df['TRANSP_ONIBUS']

    columns_drop = ['IN_REGULAR','IN_SERIE_ANO','REGULAR', 'IN_FUNDAMENTAL_CICLOS','IN_COMUM_FUND_AI']
    df.drop(columns_drop,axis=1,inplace=True)
    #histogram_boxplot(data, bins = 30, title="Plot", xlabel="Valores")
    #print("Detect missing values.")
    #print(df.isna().sum() / len(df))

    return  df

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

def runAnaliseFactorial(dataset):
    print("runAnaliseFactorial")
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)

    columns_categorical.append('CO_ENTIDADE')

    print("Categorical Columns")
    print(columns_categorical)
    drop_columns = columns_numeric
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)


    result = dataset_reduce.isna().mean()
    dataset_reduce = dataset_reduce.loc[:, result < .1]

    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce = dataset_reduce.drop(cols_to_drop, axis=1)

    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);
    dataset_reduce.drop(filtered,axis=1,inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))


    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ',kmo_model)

    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(n_factors=30, rotation=None)
    fa.fit(dataset_reduce)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()

    # Create scree plot using matplotlib
    plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


    # Create factor analysis object and perform factor analysis

    fa = FactorAnalyzer(n_factors=3,rotation="varimax")
    fa.fit(dataset_reduce)
    #print(fa.loadings_)
    fa_loading_df = pd.DataFrame(fa.loadings_,index= dataset_reduce.columns, columns=['Factor 1', 'Factor 2', 'Factor 3'])
    #print(fa_loading_df)

    # Get variance of each factors
    #print(fa.get_factor_variance())
    fa_variance_df = pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
                                  columns=['Factor 1', 'Factor 2', 'Factor 3'])
    #print(fa_variance_df)

    array = fa.transform(dataset_reduce)

    factor1 = np.around(array[:, 0], 2)
    dataset['Acesso Internet'] = factor1

    factor2 = np.around(array[:, 1], 2)
    dataset['Acessibilidade'] = factor2

    factor3 = np.around(array[:, 2], 2)
    dataset['Deficiencia Fisica'] = factor3
    return dataset;

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
                alpha=0.5,label='2017')
    plt.legend( markerscale=0.5, scatterpoints=1, fontsize=10)
    plt.title('Média em Pt no EM com Fator '+fator)
    plt.ylabel('Média')
    plt.xlabel('Região')

    plt.show()

    
if __name__ == '__main__':
    url_2019 = '../Dataset/2019/inep_saeb_merge_2019.csv'
    url_2017 = '../Dataset/2017/inep_saeb_merge_2017.csv'
    data_1 = loadData(url_2019);
    data_2 = loadData(url_2017);

    columns = data_2.columns.difference(data_2.columns)
    print(columns)

    dataframe_1 = preprocessamento(data_1)
    #dataframe_2 = preprocessamento(data_2)

    dataframe_1 = runAnaliseFactorial(data_2)
    #dataframe_2 = runAnaliseFactorial(dataframe_2)

    #mediaByRegiao(dataframe)
    #mediaByRegiaoAnual(dataframe_1,dataframe_2)