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
        dataset = pd.read_csv('../Dataset/inep_sabe_merge_2017.csv', delimiter=',',index_col=[0]).reset_index(drop=True)
        print(dataset.shape)
        print(dataset.columns)

        return dataset;

    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
def drop_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1
   not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   outliers_dropped = not_outliers.dropna().reset_index()

   return outliers_dropped

def reduceNoise(data,threshould,indicator):
    a = np.array(data.values.tolist())
    return np.where(a >= threshould, indicator, a).tolist()

def preprocessamento(dataset):
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)

    columns_categorical.append('ID_UF')
    columns_categorical.append('ID_PROVA_BRASIL')
    columns_categorical.append('IN_PREENCHIMENTO_QUESTIONARIO')

    print("Categorical Columns")
    print(columns_categorical)

    # Remove Categorical Columns
    dataset_reduce = dataset.drop(columns=columns_categorical, axis=1)
    dataset_reduce = dataset_reduce.astype(float)

    # Remove as colunas com linhas acima 10% NA
    result = dataset_reduce.isna().mean()
    dataset_reduce = dataset_reduce.loc[:, result < .10]

    # Remove colunas com valores iguais em todas as linhas
    nunique = dataset_reduce.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataset_reduce.drop(cols_to_drop, axis=1, inplace=True)

    # Remove as columnas relacionadas a códigos
    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("CO_") != -1, columns);

    dataset_reduce.drop(filtered, axis=1, inplace=True)

    columns = dataset_reduce.columns;
    filtered = filter(lambda name: name.find("ID_") != -1, columns);
    dataset_reduce.drop(filtered, axis=1, inplace=True)

    filtered = filter(lambda name: name.find("NIVEL_") != -1, columns);
    dataset_reduce.drop(filtered, axis=1, inplace=True)

    #filtered = filter(lambda name: name.find("SINDROME_") != -1, columns);
    #dataset_reduce.drop(filtered, axis=1, inplace=True)

    #columns_drop = ['IN_FUNDAMENTAL_CICLOS']

    #drop =list(set(columns_drop) & set(dataset_reduce.columns))
    #print("List Drop com 2019")
    #print(drop)
    #dataset_reduce.drop(columns_drop, axis=1, inplace=True)

    #print(dataset_reduce.describe())
    #for c in dataset_reduce.columns:
    #    print(dataset_reduce[c].describe());
    #dataset_without_outliers  = drop_outliers_IQR(dataset_reduce)

    stdev_min = 0.75
    dataset_reduce = dataset_reduce.loc[:, dataset_reduce.std() < stdev_min]

    #print(dataset_without_outliers.info())
    #print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

    #Remove lines NA
    #dataset_reduce.dropna(inplace=True)
    dataset_reduce.fillna(0,inplace=True)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape, dataset_reduce.shape))

    return dataset_reduce


def getValue_upper_whisker_quarile(data):
    median = np.median(data)
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()

    return upper_whisker,upper_quartile

def checkFeasibility(dataset_reduce):


    #Teste de Factorability
    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    #Teste de Adequacidade dos dados
    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

    # # Criamos objeto factor_analysis, sem rotação e usando 5 fatores (tentativamente)
    fa = FactorAnalyzer(n_factors=30, rotation="promax")
    #
    # # Aplicamos o método fit (ajuste) desse objeto no dataframe
    fa.fit(dataset_reduce)
    #
    # # Depois desse ajuste podemos coletar os autovetores e autovalores
    ev, v = fa.get_eigenvalues()
    print('São ' + str(len(ev)) + ' autovalores:\n', ev)
    #
    # # Create scree plot using matplotlib
    plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.axhline(y=1, c='k')
    plt.show()

    #eigenvalues, _ = fa.get_eigenvalues()
    # count eigenvalues > 1
    #number_of_factors = sum(eigenvalues > 1)
    #print('Numero de Fatores = {}'.format(number_of_factors))

    #Apos saber Numero de Fatores com eigenvalues > 1
    fa = FactorAnalyzer(12, rotation="varimax")

    # o objeto tem o método fit para análise do dataframe
    fa.fit(dataset_reduce)

    # Desse extraimos as cargas fatoriais (factor loadings)
    # Observe que fa.loadings_ é um numpy.array com shape . Usamos o método
    # do pandas pd.DataFrame.from_records para convertê-lo em um dataframe
    factorLoadings = pd.DataFrame.from_records(fa.loadings_)

    # Para ver a dataframe gerado:
    # factorLoadings.head(4)
    # Substitue as linhas pelo nomes dos itens
    factorLoadings.index = dataset_reduce.columns

    threshould = 0.4
    drop_factos = [c for c in factorLoadings.columns if len(factorLoadings[abs(factorLoadings[c]) > threshould]) == 0]
    factorLoadings.drop(drop_factos, axis=1, inplace=True)
    for c in factorLoadings.columns:
        data = factorLoadings[abs(factorLoadings[c])>threshould]
        print("Fator {} formado por {}.".format(c,data[c].index.values.tolist()))

    #Selecionar as linhas acima de threshould e abaixo -threshould
    factorLoadings = factorLoadings[factorLoadings.gt(threshould).any(axis=1) | factorLoadings.lt(-1*threshould).any(axis=1)]
    print(factorLoadings.shape)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=.9)
    sns.heatmap(factorLoadings, linewidths=1, linecolor='#ffffff', cmap="RdYlGn", xticklabels=1, yticklabels=1)
    plt.show()

    dataset = load();

    array = fa.transform(dataset_reduce)
    factor0 = np.around(array[:, 0], 2)
    dataset['Estrutura'] = factor0

    factor1 = np.around(array[:, 1], 2)
    dataset['TratamentoLixo'] = factor1

    factor2 = np.around(array[:, 2], 2)
    dataset['Internet'] = factor2

    factor3 = np.around(array[:, 10], 2)
    dataset['PCD'] = factor3

    dataset.to_csv('../Dataset/inep_saeb_merge_fatorial_2017.csv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    dataset = load()
    dataset = preprocessamento(dataset)
    checkFeasibility(dataset)