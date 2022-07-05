import sys
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
import os
from factor_analyzer import FactorAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def load():
    try:
        dataset = pd.read_csv('../Dataset/2019/inep_saeb_merge_2019.csv', delimiter='\t')
        print(dataset.shape)
        print(dataset.columns)

        return dataset;

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

def checkFeasibility(dataset):
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)
    columns_categorical.append('CO_ENTIDADE')

    print("Categorical Columns")
    print(columns_categorical)
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)
    dataset_reduce = dataset_reduce.astype(float)

    result = dataset_reduce.isna().mean()

    dataset_reduce = dataset_reduce.loc[:, result < .1]
    #Remove colunas com valores iguais em todas as linhas
    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce.drop(cols_to_drop, axis=1,inplace=True)

    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);
    dataset_reduce.drop(filtered, axis=1, inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))
    print((dataset_reduce.values < 0).any())
    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

    # Criamos objeto factor_analysis, sem rotação e usando 5 fatores (tentativamente)
    #fa = FactorAnalyzer(40, rotation=None)

    # Aplicamos o método fit (ajuste) desse objeto no dataframe
    #fa.fit(dataset_reduce)

    # Depois desse ajuste podemos coletar os autovetores e autovalores
    #ev, v = fa.get_eigenvalues()
    #print('São ' + str(len(ev)) + ' autovalores:\n', ev)

    # Create scree plot using matplotlib
    #plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    #plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    #plt.title('Scree Plot')
    #plt.xlabel('Factors')
    #plt.ylabel('Eigenvalue')
    #plt.grid()
    #plt.axhline(y=1, c='k')
    #plt.show()

    # 6 fatores
    fa = FactorAnalyzer(30, rotation="varimax")

    # o objeto tem o método fit para análise do dataframe
    fa.fit(dataset_reduce)

    # Desse extraimos as cargas fatoriais (factor loadings)
    # Observe que fa.loadings_ é um numpy.array com shape (25,6). Usamos o método
    # do pandas pd.DataFrame.from_records para convertê-lo em um dataframe
    factorLoadings = pd.DataFrame.from_records(fa.loadings_)

    # Para ver a dataframe gerado:
    #factorLoadings.head(4)
    # Substitue as linhas pelo nomes dos itens
    factorLoadings.index = dataset_reduce.columns
    #print(factorLoadings.describe())
    data_filtered = factorLoadings.copy()

    for c in data_filtered.columns:
        data = data_filtered[abs(data_filtered[c])>0.6]
        print("Fator {} formado por {}.".format(c,data[c].index.values.tolist()))

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=.9)
    sns.heatmap(factorLoadings, linewidths=1, linecolor='#ffffff', cmap="YlGnBu", xticklabels=1, yticklabels=1)
    plt.show()
if __name__ == '__main__':
    dataset = load()
    dataset = preprocessamento(dataset)
    checkFeasibility(dataset)