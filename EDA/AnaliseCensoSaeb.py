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
        dataset = pd.read_csv('../Dataset/inep_sabe_merge_2019_new.csv', delimiter='\t')
        print(dataset.shape)
        print(dataset.columns)

        return dataset;

    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

def checkFeasibility(dataset):
    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = list(pd.DataFrame(dataset.select_dtypes(['object'])).columns)
    columns_categorical.append('ID_SAEB')
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
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape, dataset_reduce.shape))
    print(dataset_reduce.columns)

    chi_square_value, p_value = calculate_bartlett_sphericity(dataset_reduce)
    print(chi_square_value, p_value);

    kmo_all, kmo_model = calculate_kmo(dataset_reduce)
    print('kmo: ', kmo_model)

    # Criamos objeto factor_analysis, sem rotação e usando 5 fatores (tentativamente)
    fa = FactorAnalyzer(40, rotation=None)

    # Aplicamos o método fit (ajuste) desse objeto no dataframe
    fa.fit(dataset_reduce)

    # Depois desse ajuste podemos coletar os autovetores e autovalores
    ev, v = fa.get_eigenvalues()
    print('São ' + str(len(ev)) + ' autovalores:\n', ev)

    # Create scree plot using matplotlib
    plt.scatter(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.plot(range(1, dataset_reduce.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

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

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=.9)
    sns.heatmap(factorLoadings, linewidths=1, linecolor='#ffffff', cmap="YlGnBu", xticklabels=1, yticklabels=1)
    plt.show()
if __name__ == '__main__':
    dataframe = load()
    checkFeasibility(dataframe)