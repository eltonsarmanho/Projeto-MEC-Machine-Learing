
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


def reduceNoise(data,threshould,indicator):
    a = np.array(data.values.tolist())
    return np.where(a >= threshould, indicator, a).tolist()

def getValue_upper_whisker_quarile(data):
    median = np.median(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()
    return upper_whisker,upper_quartile

def preprocessamento(df):
    # Dropping unnecessary columns
    #Pegar 10 valores mais frequentes
    n = 10
    data = df['QT_PROF_PSICOLOGO']
    mode = data.value_counts()[:n].index.tolist()
    df['QT_PROF_PSICOLOGO'] = reduceNoise(data,data.mean(),max(mode))

    df['IN_ACESSO_INTERNET_COMPUTADOR'] = reduceNoise(df['IN_ACESSO_INTERNET_COMPUTADOR'],data.mean(),data.mean())

    for parametro in ['NECESSIDADE_ESPECIAL','IN_BIBLIOTECA']:
        data = df[parametro]
        threshould = getValue_upper_whisker_quarile(data)[0]
        indicator = getValue_upper_whisker_quarile(data)[1]
        df[parametro] = reduceNoise(data,threshould,indicator)

    for parametro in ['BAIXA_VISAO','DEF_AUDITIVA','DEF_FISICA','DEF_INTELECTUAL','TRANSPORTE_PUBLICO','TRANSP_ONIBUS']:
        data = df[parametro]
        mode = data.value_counts()[:n].index.tolist()
        threshould = max(mode)
        df[parametro] = reduceNoise(data,threshould,threshould)

    columns_drop = ['IN_REGULAR', 'IN_SERIE_ANO', 'REGULAR', 'IN_FUNDAMENTAL_CICLOS',
                    'IN_BANHEIRO_FUNCIONARIOS', 'IN_COMUM_FUND_AI', 'IN_DORMITORIO_ALUNO',
                    'IN_COMUM_PRE', 'IN_REDES_SOCIAIS', 'IN_PERIODOS_SEMESTRAIS', 'IN_PROFISSIONALIZANTE', 'IN_EJA',
                    'IN_QUADRA_ESPORTES_COBERTA', 'IN_MEDIACAO_PRESENCIAL',
                    'PROFISSIONALIZANTE', 'IN_RESERVA_PPI', 'IN_RESERVA_PUBLICA', 'IN_COMUM_MEDIO_INTEGRADO',
                    'IN_ESGOTO_FOSSA_COMUM', 'IN_COZINHA', 'IN_DESPENSA', 'IN_ALMOXARIFADO', 'IN_BIBLIOTECA',
                    'IN_PATIO_COBERTO', 'IN_SALA_PROFESSOR', 'IN_ACESSIBILIDADE_PISOS_TATEIS', 'IN_INTERNET',
                    'IN_INTERNET_ADMINISTRATIVO', 'IN_INTERNET_APRENDIZAGEM',
                    'IN_TRATAMENTO_LIXO_SEPARACAO', 'IN_TRATAMENTO_LIXO_REUTILIZA', 'IN_TRATAMENTO_LIXO_RECICLAGEM',
                    'IN_TRATAMENTO_LIXO_INEXISTENTE', 'AEE_LIBRAS', 'AEE_BRAILLE', 'IN_EQUIP_IMPRESSORA_MULT']


    df.drop(columns_drop,axis=1,inplace=True)
    #data =  df['IN_BIBLIOTECA']
    #print(data.describe())
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
    dataset_reduce = dataset.copy().drop(columns=columns_categorical, axis=1)
    dataset_reduce = dataset_reduce.astype(float)

    result = dataset_reduce.isna().mean()

    dataset_reduce = dataset_reduce.loc[:, result < .1]
    # Remove colunas com valores iguais em todas as linhas
    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce.drop(cols_to_drop, axis=1, inplace=True)

    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);
    dataset_reduce.drop(filtered, axis=1, inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ',kmo_model)

    # 6 fatores
    fa = FactorAnalyzer(12, rotation="varimax")
    # o objeto tem o método fit para análise do dataframe
    fa.fit(dataset_reduce)
    # Desse extraimos as cargas fatoriais (factor loadings)
    # Observe que fa.loadings_ é um numpy.array. Usamos o método
    # do pandas pd.DataFrame.from_records para convertê-lo em um dataframe
    factorLoadings = pd.DataFrame.from_records(fa.loadings_)

    # Selecionar as linhas acima de 0.6 e abaixo -0.6
    factorLoadings = factorLoadings[factorLoadings.gt(.6).any(axis=1) | factorLoadings.lt(-.6).any(axis=1)]

    array = fa.transform(dataset_reduce)

    factor1 = np.around(array[:, 0], 2)
    dataset['Estrutura'] = factor1

    factor2 = np.around(array[:, 1], 2)
    dataset['PED'] = factor2

    factor3 = np.around(array[:, 4], 2)
    dataset['LibrasBraile'] = factor3

    factor4 = np.around(array[:, 5], 2)
    dataset['PNE'] = factor4

    factor5 = np.around(array[:, 6], 2)
    dataset['Acessibilidade'] = factor5

    factor6 = np.around(array[:, 11], 2)
    dataset['Transporte'] = factor6
    
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
    dataset_regiao.dropna(subset=['MEDIA_3EM_LP', 'MEDIA_3EM_MT',], inplace=True)
    dataset_regiao_grouped = dataset_regiao.groupby('CO_UF')
    mean_regiao_ME_MT = dataset_regiao_grouped['MEDIA_3EM_MT'].mean()
    mean_regiao_ME_PT = dataset_regiao_grouped['MEDIA_3EM_LP'].mean()

    UFs = dataset_regiao_grouped['MEDIA_3EM_MT'].count().index
    size_presentes = dataset_regiao_grouped['MEDIA_3EM_MT'].count()
    color = dataset_regiao_grouped['Transporte'].sum()*40


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

    #dataset_regiao.dropna(subset=['MEDIA_3EM_LP', 'MEDIA_3EM_MT','NU_PRESENTES_EM'], inplace=True)
    dataset_regiao.dropna(subset=['MEDIA_3EM_LP', 'MEDIA_3EM_MT',], inplace=True)
    #dataset_regiao_grouped = dataset_regiao.groupby('ID_REGIAO')
    dataset_regiao_grouped = dataset_regiao.groupby('ID_UF')
    mean_regiao_ME_MT = dataset_regiao_grouped['MEDIA_3EM_MT'].mean()
    mean_regiao_ME_PT = dataset_regiao_grouped['MEDIA_3EM_LP'].mean()

    UFs = dataset_regiao_grouped['ID_UF'].count().index
    print(UFs)
    dict_uf= {11:'RO',12:'AC',13:'AM',14:'RR',15:'PA',16:'AP',17:'TO',21:'MA',22:'PI',23:'CE',24:'RN',
              25:'PB',26:'PE',27:'AL',28:'SE',29:'BA',31:'MG',32:'ES',33:'RJ',35:'SP',41:'PR',42:'SC',43:'RS',50:'MS',
              51:'MT',52:'GO',53:'DF',}
    UFs_nome = [dict_uf[x] for x in UFs]
    print(UFs_nome)
    #size_presentes = dataset_regiao_grouped['NU_PRESENTES_EM'].count()
    size_presentes = 10
    normalized = normalize(dataset_regiao_grouped[fator].sum()) * 255
    color = [str(1 - (item / 255.)) for item in normalized]
    return UFs_nome, mean_regiao_ME_MT,size_presentes,color

def mediaByRegiaoAnual(dataset_1,dataset_2=None):
    #fator = 'Deficiencia Fisica'
    fator = ['Estrutura','PED','LibrasBraile','PNE','Acessibilidade','Transporte'][0]
    #fator = 'Acessibilidade'
    UFs, mean_regiao_ME,size_presentes,color = resumeDataframe(dataset_1,fator)

    plt.scatter(x=UFs, y=mean_regiao_ME,s=size_presentes*10,c=color,edgecolors="black",
             alpha=0.5,label='2019')
    #UFs, mean_regiao_ME, size_presentes, color = resumeDataframe(dataset_2,fator)
    #plt.scatter(x=UFs,y=mean_regiao_ME, s=size_presentes * 10, c=color,marker=',', edgecolors="black",alpha=0.5,label='2017')
    plt.legend( markerscale=0.5, scatterpoints=1, fontsize=10)
    plt.title('Média em Matemática no EM com Fator '+fator)
    plt.ylabel('Média')
    plt.xlabel('Região')

    plt.show()

    
if __name__ == '__main__':
    url_2019 = '../Dataset/2019/inep_saeb_merge_2019.csv'
    url_2017 = '../Dataset/2017/inep_saeb_merge_2017.csv'
    data_1 = loadData(url_2019);
    #data_2 = loadData(url_2017);

    #columns = data_2.columns.difference(data_2.columns)
    #print(columns)

    dataframe_1 = preprocessamento(data_1)
    #dataframe_2 = preprocessamento(data_2)

    dataframe_1 = runAnaliseFactorial(dataframe_1)
    #dataframe_2 = runAnaliseFactorial(dataframe_2)

    #mediaByRegiaoComFator(dataframe_1)
    mediaByRegiaoAnual(dataframe_1)